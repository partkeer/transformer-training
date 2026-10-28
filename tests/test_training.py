"""Synthetic optimizer tests only. Never downloads or optimizes TinyStories."""

from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from transformer_training.checkpoint import (capture_rng, load_checkpoint,
                                             restore_rng, save_checkpoint)
from transformer_training.config import ModelConfig, TrainingConfig
from transformer_training.data import NextTokenCollator, TinyStoriesLoaders
from transformer_training.evaluation import evaluate, next_token_loss
from transformer_training.model import DecoderOnlyTransformer
from transformer_training.training import (EpochSampler, Trainer, WarmupCosine,
                                           accumulate_gradients, make_optimizer, validate_device)


def synthetic_loaders(batch_size=2, vocab_size=12):
    examples = [{"tokens": tokens} for tokens in
                ([2, 3, 4, 1], [3, 1], [4, 5, 1], [6, 7, 8, 1], [8, 1],
                 [9, 10, 1], [2, 1], [3, 4, 1], [5, 6, 7, 1])]
    collator = NextTokenCollator(4, 1)
    def loader(seed):
        return DataLoader(examples, batch_size=batch_size, collate_fn=collator,
                          generator=torch.Generator().manual_seed(seed))
    return TinyStoriesLoaders(loader(1), loader(2), {
        "sequence_length": 4, "vocab_size": vocab_size, "seed": 42,
        "splits": {"train": {"windows_fingerprint": "synthetic-v1"},
                   "validation": {"windows_fingerprint": "synthetic-v1"}},
    })


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.model_config = ModelConfig(vocab_size=12, sequence_length=4, layers=2,
                                        d_model=12, heads=3, d_ff=24, dropout=0.2)
        self.config = TrainingConfig(device="cpu", batch_size=2, gradient_accumulation_steps=2,
                                     epochs=2, num_workers=0, eval_every=2, checkpoint_every=2,
                                     log_every=2, timing_warmup_steps=1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)

    def test_accumulation_matches_combined_batch_with_unequal_lengths(self):
        model = DecoderOnlyTransformer(replace(self.model_config, dropout=0))
        other = DecoderOnlyTransformer(model.config)
        other.load_state_dict(model.state_dict())
        batches = list(synthetic_loaders().train)
        # Includes the one-row final batch and an incomplete accumulation group.
        for group in (batches[:2], batches[2:]):
            model.zero_grad(set_to_none=True)
            other.zero_grad(set_to_none=True)
            result = accumulate_gradients(model, group)
            combined = {key: torch.cat([batch[key] for batch in group]) for key in group[0]}
            loss = next_token_loss(other(combined["input_ids"], combined["attention_mask"]),
                                   combined["labels"]) / result["supervised_tokens"]
            loss.backward()
            self.assertAlmostEqual(result["loss_sum"] / result["supervised_tokens"], loss.item(), places=6)
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a.grad, b.grad, atol=2e-7, rtol=2e-5)
        self.assertEqual(result["examples"], 5)

    def test_validation_weights_tokens_and_restores_mode(self):
        model = DecoderOnlyTransformer(self.model_config)
        model.train()
        loader = synthetic_loaders().validation
        result = evaluate(model, loader)
        self.assertTrue(model.training)
        model.eval()
        batch = {key: torch.cat([b[key] for b in loader]) for key in next(iter(loader))}
        expected = next_token_loss(model(batch["input_ids"], batch["attention_mask"]),
                                   batch["labels"]).item()
        self.assertEqual(result["supervised_tokens"], int((batch["labels"] != -100).sum()))
        self.assertAlmostEqual(result["loss"], expected / result["supervised_tokens"], places=6)
        evaluate(model, loader)
        self.assertFalse(model.training)
        model.train()
        with self.assertRaisesRegex(ValueError, "no supervised"):
            evaluate(model, [])
        self.assertTrue(model.training)

    def test_schedule_short_runs_and_weight_decay_groups(self):
        model = DecoderOnlyTransformer(self.model_config)
        optimizer = make_optimizer(model, self.config)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), len(list(model.parameters())))
        self.assertTrue(all(p.ndim >= 2 for p in optimizer.param_groups[0]["params"]))
        for total in (1, 2, 3, 100):
            schedule = WarmupCosine(optimizer, self.config, total)
            self.assertGreater(schedule.learning_rate(0), 0)
            self.assertLessEqual(schedule.learning_rate(0), self.config.learning_rate)
            self.assertAlmostEqual(schedule.learning_rate(total - 1), self.config.learning_rate
                                   if total == 1 else self.config.min_learning_rate)
            schedule.step()
            saved = schedule.state_dict()
            restored = WarmupCosine(optimizer, self.config, total)
            restored.load_state_dict(saved)
            self.assertEqual(restored.state_dict(), saved)
        for options in ({"epochs": 0}, {"max_steps": 0}, {"num_workers": -1},
                        {"warmup_ratio": 1}, {"precision": "auto"}, {"device": "mps"},
                        {"learning_rate": float("nan")}, {"beta1": 1}, {"seed": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TrainingConfig(**options)

    def test_device_precision_requests_fail_without_silent_fallback(self):
        for precision in ("fp16", "bf16"):
            with self.assertRaisesRegex(ValueError, "CPU only with fp32"):
                validate_device(replace(self.config, precision=precision))
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA is unavailable"):
                validate_device(TrainingConfig())
        with patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.device", return_value=nullcontext()), \
                patch("torch.cuda.is_bf16_supported", return_value=False):
            with self.assertRaisesRegex(ValueError, "native GPU BF16 support"):
                validate_device(TrainingConfig(precision="bf16"))

    def test_sampler_and_prefetch_independent_consumed_cursor(self):
        order = list(EpochSampler(20, 42, 3))
        self.assertEqual(sorted(order), list(range(20)))
        self.assertEqual(list(EpochSampler(20, 42, 3, 7)), order[7:])
        self.assertNotEqual(order, list(EpochSampler(20, 42, 4)))
        with self.assertRaises(ValueError):
            EpochSampler(20, 42, 0, 21)

    def test_atomic_checkpoint_and_rng_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            state = capture_rng()
            save_checkpoint(path, {"rng": state, "tensor": torch.tensor([1, 2])})
            expected = (random.random(), torch.rand(4))
            payload = load_checkpoint(path)
            restore_rng(payload["rng"])
            self.assertEqual(random.random(), expected[0])
            torch.testing.assert_close(torch.rand(4), expected[1], rtol=0, atol=0)
            with patch("transformer_training.checkpoint.os.replace", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError):
                    save_checkpoint(path, {"tensor": torch.tensor([3])})
            torch.testing.assert_close(load_checkpoint(path)["tensor"], torch.tensor([1, 2]))
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_resume_matches_uninterrupted_with_dropout_and_worker_prefetch(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            for workers in (0, 2):
                with self.subTest(workers=workers):
                    config = replace(self.config, num_workers=workers)
                    full = Trainer(self.model_config, config, synthetic_loaders(), root / f"full-{workers}")
                    full.fit()
                    partial = Trainer(self.model_config, config, synthetic_loaders(), root / f"part-{workers}")
                    result = partial.fit(stop_after_updates=1)
                    self.assertEqual(result["status"], "interrupted")
                    self.assertEqual(partial.state["consumed_examples"], 4)
                    checkpoint = root / f"part-{workers}" / "latest.pt"
                    # Exercise continuation with a different worker count, too.
                    resumed = Trainer(self.model_config, replace(config, num_workers=2 - workers),
                                      synthetic_loaders(), checkpoint.parent, resume=checkpoint)
                    resumed.fit()
                    for a, b in zip(full.model.parameters(), resumed.model.parameters()):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    for key in ("updates", "attempts", "epoch", "consumed_examples",
                                "supervised_tokens", "best_loss"):
                        self.assertEqual(full.state[key], resumed.state[key])
                    self.assertEqual(full.scheduler.state_dict(), resumed.scheduler.state_dict())
                    for a, b in zip(full.optimizer.state.values(), resumed.optimizer.state.values()):
                        for key in a:
                            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
                    self.assertTrue((checkpoint.parent / "best.pt").exists())
                    metrics = [json.loads(line) for line in
                               (checkpoint.parent / "metrics.jsonl").read_text().splitlines()]
                    self.assertTrue(any(row["event"] == "resume" for row in metrics))

    def test_resume_at_epoch_boundary_and_step_cap(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            config = replace(self.config, max_steps=4)
            full = Trainer(self.model_config, config, synthetic_loaders(), root / "full")
            result = full.fit()
            self.assertEqual(full.total_updates, 4)
            self.assertEqual(result["state"]["updates"], 4)
            part = Trainer(self.model_config, config, synthetic_loaders(), root / "part")
            part.fit(stop_after_updates=3)  # Nine rows: three accumulation groups per epoch.
            self.assertEqual(part.state["epoch"], 1)
            self.assertEqual(part.state["consumed_examples"], 0)
            resumed = Trainer(self.model_config, config, synthetic_loaders(), root / "part",
                              resume=root / "part" / "latest.pt")
            resumed.fit()
            for a, b in zip(full.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_incompatible_resume_and_nonempty_run_directory(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory) / "run"
            trainer = Trainer(self.model_config, self.config, synthetic_loaders(), root)
            trainer.fit(stop_after_updates=1)
            checkpoint = root / "latest.pt"
            with self.assertRaisesRegex(ValueError, "not empty"):
                Trainer(self.model_config, self.config, synthetic_loaders(), root)
            for config in (replace(self.config, epochs=3), replace(self.config, learning_rate=1e-3)):
                with self.assertRaisesRegex(ValueError, "Incompatible"):
                    Trainer(self.model_config, config, synthetic_loaders(), root, resume=checkpoint)
            data = synthetic_loaders()
            data.metadata["splits"]["train"]["windows_fingerprint"] = "changed"
            with self.assertRaisesRegex(ValueError, "fingerprints"):
                Trainer(self.model_config, self.config, data, root, resume=checkpoint)

    def test_dry_run_never_imports_dataset_or_creates_output(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "not-created"
            code = (
                "import sys; from transformer_training.cli import main; "
                f"main(['--dry-run', '--output-dir', {str(output)!r}]); "
                "assert 'transformer_training.data' not in sys.modules; "
                "assert 'datasets' not in sys.modules"
            )
            result = subprocess.run([sys.executable, "-c", code], cwd=root, text=True,
                                    capture_output=True, check=True, timeout=60)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["parameters"], 33691776)
            self.assertEqual(payload["effective_batch_size"], 64)
            self.assertFalse(output.exists())

    def test_cli_resume_inherits_configuration_and_data_options(self):
        from transformer_training.cli import main

        def data():
            loaders = synthetic_loaders()
            loaders.metadata.update(revision="dataset-sha", tokenizer_revision="tokenizer-sha",
                                    max_stories=9)
            return loaders

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "run"
            partial = Trainer(self.model_config, self.config, data(), output)
            partial.fit(stop_after_updates=1)
            with patch("transformer_training.data.load_tinystories", return_value=data()) as load:
                self.assertEqual(main(["--resume", str(output / "latest.pt"),
                                       "--log-every=1"]), 0)
            options = load.call_args.kwargs
            self.assertEqual(options["revision"], "dataset-sha")
            self.assertEqual(options["tokenizer_revision"], "tokenizer-sha")
            self.assertEqual(options["max_stories"], 9)
            self.assertEqual(options["sequence_length"], 4)
            self.assertTrue(options["exclude_validation_overlap"])
            self.assertEqual(load_checkpoint(output / "latest.pt")["state"]["updates"], 6)
            self.assertEqual(load_checkpoint(output / "latest.pt")["training_config"]["log_every"], 1)

    def test_cli_rejects_abbreviations_before_loading_checkpoint_or_data(self):
        from transformer_training.cli import main

        for option in (["--batch-s", "3"], ["--learning-r=0.01"]):
            with self.subTest(option=option), redirect_stderr(io.StringIO()), \
                    patch("transformer_training.checkpoint.load_checkpoint") as checkpoint, \
                    patch("transformer_training.data.load_tinystories") as data:
                with self.assertRaises(SystemExit) as error:
                    main(["--resume", "unused.pt", *option])
                self.assertEqual(error.exception.code, 2)
                checkpoint.assert_not_called()
                data.assert_not_called()

    def test_nonfinite_loss_stops_before_update_and_validation_restores_mode(self):
        model = DecoderOnlyTransformer(self.model_config)
        with torch.no_grad():
            model.lm_head.weight.fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite training loss"):
            accumulate_gradients(model, list(synthetic_loaders().train)[:1])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        model.train()
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite validation loss"):
            evaluate(model, synthetic_loaders().validation)
        self.assertTrue(model.training)

    def test_skipped_update_consumes_data_without_advancing_schedule(self):
        # Simulate the scaler contract on CPU; this is not a claim of CUDA AMP validation.
        class SkipFirstScaler:
            def __init__(self):
                self.scale_value, self.calls = 2.0, 0

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def get_scale(self):
                return self.scale_value

            def step(self, optimizer):
                if self.calls:
                    optimizer.step()

            def update(self):
                if self.calls == 0:
                    self.scale_value /= 2
                self.calls += 1

            def state_dict(self):
                return {"scale": self.scale_value, "calls": self.calls}

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.model_config, self.config, synthetic_loaders(),
                              Path(directory) / "run")
            trainer.scaler = SkipFirstScaler()
            trainer.fit(stop_after_updates=1)
            self.assertEqual(trainer.state["attempts"], 2)
            self.assertEqual(trainer.state["updates"], 1)
            self.assertEqual(trainer.state["skipped_updates"], 1)
            self.assertEqual(trainer.state["consumed_examples"], 8)
            self.assertEqual(trainer.scheduler.completed_updates, 1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable; GPU checks not performed")
class CudaTests(unittest.TestCase):
    def test_supported_precision_forward_backward(self):
        model_config = ModelConfig(vocab_size=12, sequence_length=4, layers=1,
                                   d_model=12, heads=3, d_ff=24, dropout=0)
        precisions = ["fp32", "fp16"]
        if torch.cuda.is_bf16_supported(including_emulation=False):
            precisions.append("bf16")
        for precision in precisions:
            with self.subTest(precision=precision):
                config = TrainingConfig(precision=precision)
                device = validate_device(config)
                model = DecoderOnlyTransformer(model_config).to(device)
                scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
                optimizer = make_optimizer(model, config)
                result = accumulate_gradients(model, list(synthetic_loaders().train)[:2],
                                              device=device, precision=precision, scaler=scaler)
                scaler.unscale_(optimizer)
                self.assertGreater(result["supervised_tokens"], 0)
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
