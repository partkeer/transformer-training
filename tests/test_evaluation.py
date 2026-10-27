"""Offline quality evaluation: synthetic models/tokenizers, never TinyStories downloads."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from datasets import Dataset, DatasetDict, load_dataset_builder
import torch
from torch import nn

from transformer_training.checkpoint import save_checkpoint
from transformer_training.config import ModelConfig, TrainingConfig
from transformer_training.data import (NextTokenCollator, create_dataloaders,
                                       create_validation_loader, load_tinystories_validation,
                                       _load_validation_split)
from transformer_training.evaluation import evaluate
from transformer_training.evaluation_cli import main
from transformer_training.generation import DEFAULT_PROMPTS, generate
from transformer_training.model import DecoderOnlyTransformer
from transformer_training.quality import (evaluate_checkpoint, loss_metrics,
                                         verify_validation_metadata)
from transformer_training.training import CHECKPOINT_VERSION, data_signature
from test_data import fixture_tokenizer


REVISION = "a" * 40
TOKENIZER_REVISION = "b" * 40


def quality_splits():
    return DatasetDict({
        "train": Dataset.from_dict({"text": ["a b", "f g", "h i"]}),
        "validation": Dataset.from_dict({"text": ["a b c d e", "f", "", "h i", "j"]}),
    })


def checkpoint_fixture(path, max_stories=3, *, training_device="cpu", precision="fp32"):
    tokenizer = fixture_tokenizer()
    loaders = create_dataloaders(quality_splits(), tokenizer, sequence_length=12,
                                batch_size=2, max_stories=max_stories)
    loaders.metadata.update(dataset="roneneldan/TinyStories", revision=REVISION,
                            tokenizer="openai-community/gpt2",
                            tokenizer_revision=TOKENIZER_REVISION)
    model_config = ModelConfig(vocab_size=12, sequence_length=12, layers=1,
                               d_model=12, heads=3, d_ff=24)
    config = TrainingConfig(device=training_device, precision=precision, batch_size=2,
                            num_workers=2)
    model = DecoderOnlyTransformer(model_config)
    payload = {
        "version": CHECKPOINT_VERSION, "model_config": asdict(model_config),
        "training_config": asdict(config), "data_signature": data_signature(loaders.metadata),
        "model": model.state_dict(), "state": {"updates": 7},
    }
    save_checkpoint(path, payload)
    return payload, model, tokenizer


def local_validation(**options):
    revision = options.pop("revision")
    tokenizer_revision = options.pop("tokenizer_revision")
    result = create_validation_loader(quality_splits()["validation"], fixture_tokenizer(),
                                      **options)
    result.metadata.update(dataset="roneneldan/TinyStories", revision=revision,
                           tokenizer="openai-community/gpt2",
                           tokenizer_revision=tokenizer_revision)
    return result


class FixedLogits(nn.Module):
    def __init__(self, values, context=8):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.config = SimpleNamespace(sequence_length=context, vocab_size=len(values))

    def forward(self, ids, attention_mask=None):
        if torch.is_grad_enabled():
            raise AssertionError("Evaluation/generation must disable gradients")
        return self.values.expand(*ids.shape, -1)


class SingleThreadTests(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)


class ValidationLoaderTests(SingleThreadTests):
    def test_shared_preparation_matches_training_batches_and_provenance(self):
        tokenizer = fixture_tokenizer()
        tokenizer.enable_padding(length=20, pad_id=1)
        tokenizer.enable_truncation(max_length=1)
        for limit in (None, 3, 99):
            with self.subTest(limit=limit):
                source = quality_splits()
                both = create_dataloaders(source, tokenizer, sequence_length=2,
                                          batch_size=2, max_stories=limit,
                                          exclude_validation_overlap=True)
                # No DatasetDict or training split is passed to this API.
                with patch("transformer_training.data.audit_overlap",
                           side_effect=AssertionError("Validation must not audit train")):
                    only = create_validation_loader(source["validation"], tokenizer,
                                                     sequence_length=2, batch_size=2,
                                                     max_stories=limit)
                self.assertEqual(both.metadata["splits"]["validation"],
                                 only.metadata["splits"]["validation"])
                self.assertEqual(len(both.validation), len(only.validation))
                for expected, actual in zip(both.validation, only.validation):
                    for key in expected:
                        torch.testing.assert_close(expected[key], actual[key])
                self.assertIsNone(only.tokenizer.truncation)
                self.assertIsNone(only.tokenizer.padding)
        self.assertIsNotNone(tokenizer.truncation)

    def test_hub_loader_requests_only_validation_and_verifies_cache(self):
        split = quality_splits()["validation"]
        with tempfile.TemporaryDirectory() as directory:
            tokenizer_path = Path(directory) / "tokenizer.json"
            fixture_tokenizer().save(str(tokenizer_path))
            with patch("transformer_training.data.hf_hub_download", return_value=str(tokenizer_path)), \
                    patch("transformer_training.data._load_validation_split", return_value=split) as load, \
                    patch("transformer_training.data._check_dataset_revision") as check:
                result = load_tinystories_validation(revision=REVISION,
                                                     tokenizer_revision=TOKENIZER_REVISION)
            load.assert_called_once_with(REVISION)
            self.assertEqual(list(check.call_args.args[0]), ["validation"])
            self.assertEqual(result.metadata["revision"], REVISION)
            with patch("transformer_training.data.hf_hub_download", return_value=str(tokenizer_path)), \
                    patch("transformer_training.data._load_validation_split", return_value=split):
                with self.assertRaisesRegex(RuntimeError, "Cannot verify validation"):
                    load_tinystories_validation(revision=REVISION,
                                                tokenizer_revision=TOKENIZER_REVISION)

    def test_fresh_preparation_never_reads_train_and_preserves_builder_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_path = root / "validation.parquet"
            quality_splits()["validation"].to_parquet(validation_path)
            # Deliberately unreadable as Parquet: preparation would fail if the
            # train file were passed through to a download/prepare operation.
            train_path = root / "train.parquet"
            train_path.write_bytes(b"This training file must never be read")
            data_files = {"train": [str(train_path)], "validation": [str(validation_path)]}
            standard = load_dataset_builder("parquet", data_files=data_files,
                                            cache_dir=str(root / "standard"))
            isolated = load_dataset_builder("parquet", data_files=data_files,
                                            cache_dir=str(root / "isolated"))
            expected_fingerprint = standard._get_dataset_fingerprint("validation")
            with patch("transformer_training.data.load_dataset_builder",
                       side_effect=[standard, isolated]) as load:
                result = _load_validation_split(REVISION)
            self.assertEqual(len(result), 5)
            self.assertEqual(result._fingerprint, expected_fingerprint)
            self.assertEqual(list(isolated.config.data_files), ["validation"])
            self.assertEqual(list(isolated.info.splits), ["validation"])
            self.assertFalse((Path(standard.cache_dir) / "dataset_info.json").exists())
            self.assertTrue(load.call_args.kwargs["cache_dir"].endswith("validation-only"))
            # Prepared caches are read directly; no preparation is required.
            with patch("transformer_training.data.load_dataset_builder", return_value=isolated), \
                    patch("transformer_training.data._check_dataset_revision"), \
                    patch.object(isolated, "download_and_prepare",
                                 side_effect=AssertionError("Must reuse prepared validation")):
                cached = _load_validation_split(REVISION)
            self.assertEqual(cached._fingerprint, result._fingerprint)
            # Offline Hub lookup raises ConnectionError when the normal cache
            # is absent; a separately prepared evaluation cache is still usable.
            with patch("transformer_training.data.load_dataset_builder",
                       side_effect=[ConnectionError("No normal cache"), isolated]):
                fallback = _load_validation_split(REVISION)
            self.assertEqual(fallback._fingerprint, result._fingerprint)

    def test_empty_and_invalid_validation_errors(self):
        for texts, message in ((["", "  "], "no usable stories"),
                               ([None], "string text")):
            with self.subTest(texts=texts), self.assertRaisesRegex(ValueError, message):
                create_validation_loader(Dataset.from_dict({"text": texts}), fixture_tokenizer())
        for options in ({"batch_size": 0}, {"num_workers": -1}, {"max_stories": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_validation_loader(quality_splits()["validation"], fixture_tokenizer(), **options)

    def test_loss_and_perplexity_match_manual_valid_token_calculation(self):
        values = [0.0, 1.5, -0.5, 2.0]
        model = FixedLogits(values)
        examples = [{"tokens": ids} for ids in ([2, 3, 1], [3, 1], [2, 2, 3, 1])]
        collate = NextTokenCollator(4, 1)
        batches = [collate(examples[:2]), collate(examples[2:])]
        result = loss_metrics(evaluate(model, batches))
        # Each real target contributes once, including each story's EOS.
        targets = [3, 1, 1, 2, 3, 1]
        log_normalizer = math.log(sum(math.exp(x) for x in values))
        expected = sum(log_normalizer - values[token] for token in targets) / len(targets)
        self.assertEqual(result["supervised_tokens"], len(targets))
        self.assertAlmostEqual(result["loss"], expected, places=6)
        self.assertAlmostEqual(result["perplexity"], math.exp(expected), places=5)
        self.assertFalse(result["perplexity_overflow"])
        self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError, "no supervised tokens"):
            evaluate(model, [])
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite validation"):
            evaluate(FixedLogits([float("nan")] * 4), batches)
        overflow = loss_metrics({"loss": 1000.0, "supervised_tokens": 1})
        self.assertIsNone(overflow["perplexity"])
        self.assertTrue(overflow["perplexity_overflow"])
        json.dumps(overflow, allow_nan=False)


class GenerationTests(SingleThreadTests):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; generation precision check skipped")
    def test_cuda_generation_in_supported_precisions(self):
        config = ModelConfig(vocab_size=12, sequence_length=12, layers=1,
                             d_model=12, heads=3, d_ff=24)
        model = DecoderOnlyTransformer(config).to("cuda")
        for precision in ("fp32", "fp16", "bf16"):
            if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
                continue
            with self.subTest(precision=precision):
                before = torch.cuda.get_rng_state().clone()
                sample = generate(model, fixture_tokenizer(), "a b", max_new_tokens=3,
                                  precision=precision)
                self.assertGreater(sample["generated_tokens"], 0)
                torch.testing.assert_close(before, torch.cuda.get_rng_state(), rtol=0, atol=0)

    def test_greedy_context_limit_and_length_limit(self):
        model = FixedLogits([0, -10, 0, 5] + [0] * 8, context=4)
        result = generate(model, fixture_tokenizer(), "a", temperature=0, max_new_tokens=10)
        self.assertEqual(result["generated_tokens"], 3)
        self.assertEqual(result["continuation"], "b b b")
        self.assertEqual(result["text"], "a b b b")
        self.assertEqual(result["stop_reason"], "context_capacity")
        self.assertTrue(model.training)
        model.eval()
        result = generate(model, fixture_tokenizer(), "a", temperature=0, max_new_tokens=1)
        self.assertEqual(result["stop_reason"], "max_new_tokens")
        self.assertFalse(model.training)

    def test_eos_and_no_special_tokens_in_decoded_text(self):
        tokenizer = fixture_tokenizer()
        tokenizer.add_special_tokens(["<|endoftext|>"])
        model = FixedLogits([0, 5] + [0] * 10)
        result = generate(model, tokenizer, "a", temperature=0)
        self.assertEqual(result["stop_reason"], "eos")
        self.assertEqual(result["generated_tokens"], 1)
        self.assertEqual(result["continuation"], "")
        self.assertEqual(result["text"], "a")

    def test_sampling_is_reproducible_and_does_not_consume_global_rng(self):
        model = FixedLogits([0, -100] + [1] * 10, context=16)
        for top_k, temperature in ((0, 0.8), (999, 0.8), (3, 0.8), (1, 1e-300)):
            with self.subTest(top_k=top_k, temperature=temperature):
                before = torch.get_rng_state().clone()
                a = generate(model, fixture_tokenizer(), "a", top_k=top_k,
                             temperature=temperature, seed=123, max_new_tokens=8)
                torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
                b = generate(model, fixture_tokenizer(), "a", top_k=top_k,
                             temperature=temperature, seed=123, max_new_tokens=8)
                self.assertEqual(a, b)

    def test_invalid_prompts_options_and_nonfinite_logits(self):
        model = FixedLogits([0] * 12, context=4)
        for prompt in ("", "  ", "a b c d", "a b c d e"):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                generate(model, fixture_tokenizer(), prompt)
        for options in ({"temperature": -1}, {"temperature": float("nan")},
                        {"temperature": float("inf")}, {"top_k": -1},
                        {"max_new_tokens": 0}, {"seed": -1}, {"seed": 2**63},
                        {"precision": "bad"}, {"precision": "bf16"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                generate(model, fixture_tokenizer(), "a", **options)
        for value in (float("inf"), float("nan")):
            broken = FixedLogits([value] * 12)
            with self.assertRaisesRegex(FloatingPointError, "Nonfinite generation"):
                generate(broken, fixture_tokenizer(), "a")
            self.assertTrue(broken.training)
        with self.assertRaisesRegex(ValueError, "vocabulary/EOS"):
            generate(FixedLogits([0] * 10), fixture_tokenizer(), "a")
        tokenizer = fixture_tokenizer()
        tokenizer.enable_truncation(max_length=1)
        with self.assertRaisesRegex(ValueError, "truncation disabled"):
            generate(model, tokenizer, "a b c d e")


class CheckpointEvaluationTests(SingleThreadTests):
    def test_round_trip_inherits_selection_and_writes_default_report(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            payload, model, _ = checkpoint_fixture(checkpoint)
            before = checkpoint.read_bytes()
            with patch("transformer_training.data.load_tinystories_validation",
                       side_effect=local_validation) as load, \
                    patch("transformer_training.training.make_optimizer",
                          side_effect=AssertionError("Must not create an optimizer")), \
                    patch("transformer_training.checkpoint.restore_rng",
                          side_effect=AssertionError("Must not restore training RNG")):
                report = evaluate_checkpoint(checkpoint, max_new_tokens=2)
            self.assertEqual(load.call_args.kwargs["num_workers"], 0)
            self.assertEqual(load.call_args.kwargs["max_stories"], 3)
            self.assertEqual(load.call_args.kwargs["selection_seed"], 43)
            self.assertEqual(report["checkpoint_updates"], 7)
            self.assertEqual(report["generation"]["prompts"], list(DEFAULT_PROMPTS))
            self.assertEqual([s["seed"] for s in report["samples"]], [42, 43, 44])
            self.assertEqual(report["data"]["splits"]["validation"],
                             payload["data_signature"]["splits"]["validation"])
            expected = evaluate(model, local_validation(**load.call_args.kwargs).validation)
            self.assertAlmostEqual(report["metrics"]["loss"], expected["loss"], places=7)
            output = Path(report["output_dir"])
            self.assertEqual(output.parent, checkpoint.parent / "evaluations")
            self.assertEqual(json.loads((output / "report.json").read_text(encoding="utf-8")), report)
            self.assertIn(DEFAULT_PROMPTS[0], (output / "samples.txt").read_text(encoding="utf-8"))
            self.assertEqual(before, checkpoint.read_bytes())

    def test_cli_overrides_full_split_and_custom_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_fixture(root / "latest.pt", training_device="cuda", precision="bf16")
            arguments = ["--checkpoint", str(root / "latest.pt"), "--device", "cpu",
                         "--precision", "fp32", "--batch-size", "1", "--num-workers", "0",
                         "--full-validation", "--prompt", "a b", "--prompt", "c",
                         "--temperature", "0", "--max-new-tokens", "2",
                         "--seed", "100", "--output-dir", str(root / "quality")]
            with patch("transformer_training.data.load_tinystories_validation",
                       side_effect=local_validation) as load, redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(main(arguments), 0)
            report = json.loads((root / "quality" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["evaluation"]["selection"], "full_validation")
            self.assertEqual(report["evaluation"]["precision"], "fp32")
            self.assertEqual(report["evaluation"]["batch_size"], 1)
            self.assertEqual(report["generation"]["prompts"], ["a b", "c"])
            self.assertEqual([sample["seed"] for sample in report["samples"]], [100, 101])
            self.assertIsNone(load.call_args.kwargs["max_stories"])
            self.assertEqual(report["metrics"]["evaluated_windows"], 4)
            self.assertIn('"perplexity"', stdout.getvalue())
            self.assertIn("Complete text:", stdout.getvalue())

    def test_validation_metadata_rejects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            payload, _, _ = checkpoint_fixture(Path(directory) / "latest.pt")
            saved = payload["data_signature"]
            actual = local_validation(revision=REVISION, tokenizer_revision=TOKENIZER_REVISION,
                                      sequence_length=12, max_stories=3).metadata
            verify_validation_metadata(saved, actual)
            mutations = [lambda x: x.update(tokenizer_revision="c" * 40),
                         lambda x: x.update(eos_id=2),
                         lambda x: x["splits"]["validation"].update(windows_fingerprint="changed"),
                         lambda x: x["splits"]["validation"].update(selected_row_indices=[0, 1, 2])]
            for change in mutations:
                changed = deepcopy(actual)
                change(changed)
                with self.assertRaisesRegex(ValueError, "mismatch|changed"):
                    verify_validation_metadata(saved, changed)
            full = local_validation(revision=REVISION, tokenizer_revision=TOKENIZER_REVISION,
                                    sequence_length=12).metadata
            verify_validation_metadata(saved, full, full_validation=True)

    def test_invalid_checkpoint_and_options_fail_before_data_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload, _, _ = checkpoint_fixture(root / "latest.pt")
            variants = []
            invalid = deepcopy(payload)
            invalid["version"] = -1
            variants.append(invalid)
            invalid = deepcopy(payload)
            invalid["model"].pop("final_norm.bias")
            variants.append(invalid)
            invalid = deepcopy(payload)
            invalid["data_signature"]["revision"] = "main"
            variants.append(invalid)
            invalid = deepcopy(payload)
            invalid["data_signature"]["vocab_size"] = 99
            variants.append(invalid)
            with patch("transformer_training.data.load_tinystories_validation") as load:
                for invalid in variants:
                    save_checkpoint(root / "invalid.pt", invalid)
                    with self.assertRaises((ValueError, RuntimeError)):
                        evaluate_checkpoint(root / "invalid.pt")
                for options in ({"device": "bad"}, {"precision": "bf16"},
                                {"batch_size": 0}, {"num_workers": -1},
                                {"temperature": float("nan")}, {"prompts": [" "]},
                                {"seed": 2**63 - 1}):
                    with self.subTest(options=options), self.assertRaises(ValueError):
                        evaluate_checkpoint(root / "latest.pt", **options)
                load.assert_not_called()
            self.assertFalse((root / "evaluations").exists())

    def test_skip_generation_output_protection_and_provenance_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_fixture(root / "latest.pt")
            with patch("transformer_training.data.load_tinystories_validation",
                       side_effect=local_validation), \
                    patch("transformer_training.quality.generate",
                          side_effect=AssertionError("Generation must be skipped")):
                report = evaluate_checkpoint(root / "latest.pt", output_dir=root / "quality",
                                             skip_generation=True)
            self.assertEqual(report["samples"], [])
            self.assertEqual((root / "quality" / "samples.txt").read_text(), "Generation skipped.\n")
            before = (root / "quality" / "report.json").read_bytes()
            with patch("transformer_training.data.load_tinystories_validation") as load:
                with self.assertRaisesRegex(ValueError, "must be empty"):
                    evaluate_checkpoint(root / "latest.pt", output_dir=root / "quality")
                with self.assertRaisesRegex(ValueError, "must be empty"):
                    evaluate_checkpoint(root / "latest.pt", output_dir=root / "latest.pt")
                load.assert_not_called()
            self.assertEqual(before, (root / "quality" / "report.json").read_bytes())
            bad_data = local_validation(revision=REVISION, tokenizer_revision=TOKENIZER_REVISION,
                                       sequence_length=12, max_stories=3)
            bad_data.metadata["splits"]["validation"]["windows_fingerprint"] = "changed"
            with patch("transformer_training.data.load_tinystories_validation", return_value=bad_data):
                with self.assertRaisesRegex(ValueError, "fingerprints changed"):
                    evaluate_checkpoint(root / "latest.pt", output_dir=root / "failure")
            self.assertFalse((root / "failure").exists())

    def test_offline_subprocess_cli_with_local_hub_stubs(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            checkpoint = temporary / "best.pt"
            _, _, tokenizer = checkpoint_fixture(checkpoint)
            tokenizer.save(str(temporary / "tokenizer.json"))
            before = checkpoint.read_bytes()
            # Assert offline configuration at the FIRST Hugging Face import.
            # Only network boundaries are replaced; real loading/preprocessing,
            # checkpoint reconstruction, evaluation, decoding and reports run.
            code = r'''
import builtins, os, sys
from pathlib import Path
from unittest.mock import patch
root = Path(sys.argv[1])
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'datasets', 'huggingface_hub'}:
        assert os.environ.get('HF_HUB_OFFLINE') == '1'
        assert os.environ.get('HF_DATASETS_OFFLINE') == '1'
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from transformer_training.evaluation_cli import main
def local_data(revision):
    from datasets import Dataset
    data = Dataset.from_dict({'text': ['a b c d e', 'f', '', 'h i', 'j']})
    return data
# Patch only after main has configured offline mode, immediately before its
# deferred validation import. Cache-file identity is tested separately.
from transformer_training import quality
original_evaluate = quality.evaluate_checkpoint
def offline_evaluate(**options):
    assert os.environ['HF_HUB_OFFLINE'] == '1'
    from transformer_training import data
    with patch.object(data, '_load_validation_split', side_effect=local_data), \
         patch.object(data, 'hf_hub_download', return_value=str(root / 'tokenizer.json')), \
         patch.object(data, '_check_dataset_revision'):
        return original_evaluate(**options)
with patch.object(quality, 'evaluate_checkpoint', side_effect=offline_evaluate):
    raise SystemExit(main(['--checkpoint', str(root / 'best.pt'), '--offline',
                          '--prompt', 'a b', '--max-new-tokens', '2',
                          '--output-dir', str(root / 'quality')]))
'''
            env = os.environ.copy()
            env.pop("HF_HUB_OFFLINE", None)
            env.pop("HF_DATASETS_OFFLINE", None)
            result = subprocess.run([sys.executable, "-c", code, directory], cwd=root,
                                    capture_output=True, text=True, timeout=60, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((temporary / "quality" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["generation"]["prompts"], ["a b"])
            self.assertGreater(report["metrics"]["supervised_tokens"], 0)
            self.assertEqual(before, checkpoint.read_bytes())


if __name__ == "__main__":
    unittest.main()
