"""Offline execution correctness and controlled suite recovery; no Hub access."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from transformer_training.checkpoint import capture_rng, load_checkpoint, restore_rng, save_checkpoint
from transformer_training.config import ImplementationConfig, ModelConfig, TrainingConfig
from transformer_training.diagnostics import aggregate_trials, compare_results, run_trial
from transformer_training.experiment_plan import resolve_plan, load_plan
from transformer_training.experiment_report import paired_quality
from transformer_training.experiments import run_suite, worker_environment
from transformer_training.implementations import compiler_snapshot, file_sha256
from transformer_training.model import DecoderOnlyTransformer
from transformer_training.training import Trainer, WarmupCosine, accumulate_gradients, make_optimizer, optimizer_attempt
from test_training import synthetic_loaders


class ImplementationTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.model = ModelConfig(vocab_size=12, sequence_length=4, layers=2,
                                 d_model=12, heads=3, d_ff=24, dropout=0)
        self.training = TrainingConfig(device="cpu", batch_size=2, gradient_accumulation_steps=2,
            epochs=3, num_workers=0, timing_warmup_steps=1, log_every=10)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_sdpa_matches_valid_logits_loss_and_gradients(self):
        reference = DecoderOnlyTransformer(self.model)
        sdpa = DecoderOnlyTransformer(self.model, ImplementationConfig(attention="sdpa", sdpa_backend="math"))
        sdpa.load_state_dict(reference.state_dict())
        for ids, mask in ((torch.tensor([[2]]), torch.tensor([[True]])),
            (torch.tensor([[2, 3, 1, 1], [4, 5, 6, 1]]),
             torch.tensor([[True, True, False, False], [True, True, True, True]]))):
            a, b = reference(ids, mask), sdpa(ids, mask)
            self.assertTrue(torch.isfinite(b).all())
            torch.testing.assert_close(a[mask], b[mask], atol=1e-6, rtol=1e-5)
            changed = ids.clone()
            changed[:, -1] = 11
            torch.testing.assert_close(b[:, :-1], sdpa(changed, mask)[:, :-1], atol=1e-6, rtol=1e-5)
        batches = list(synthetic_loaders().train)
        a = accumulate_gradients(reference, batches)
        b = accumulate_gradients(sdpa, batches)
        self.assertAlmostEqual(a["loss_sum"], b["loss_sum"], places=5)
        for p, q in zip(reference.parameters(), sdpa.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)
        with self.assertRaisesRegex(ValueError, "right-padded"):
            sdpa(torch.tensor([[2, 3]]), torch.tensor([[False, True]]))

    def test_sdpa_eval_disables_attention_and_residual_dropout(self):
        model = DecoderOnlyTransformer(replace(self.model, dropout=.5),
            ImplementationConfig(attention="sdpa", sdpa_backend="math"))
        ids = torch.tensor([[2, 3, 4]])
        model.eval()
        torch.testing.assert_close(model(ids), model(ids), rtol=0, atol=0)
        model.train()
        self.assertFalse(torch.equal(model(ids), model(ids)))

    def test_unsupported_requested_backend_is_not_substituted(self):
        model = DecoderOnlyTransformer(self.model, ImplementationConfig(attention="sdpa", sdpa_backend="cudnn"))
        with self.assertRaises(RuntimeError):
            model(torch.tensor([[2, 3]]))

    def test_checkpointing_preserves_dropout_gradients_and_rng(self):
        for attention in ("reference", "sdpa"):
            with self.subTest(attention=attention):
                impl = ImplementationConfig(attention=attention)
                plain = DecoderOnlyTransformer(replace(self.model, dropout=.25), impl)
                checked = DecoderOnlyTransformer(plain.config, replace(impl, activation_checkpointing=True))
                checked.load_state_dict(plain.state_dict())
                rng = capture_rng()
                a = accumulate_gradients(plain, list(synthetic_loaders().train))
                after = capture_rng()
                restore_rng(rng)
                b = accumulate_gradients(checked, list(synthetic_loaders().train))
                self.assertEqual(a, b)
                torch.testing.assert_close(after["torch"], capture_rng()["torch"], rtol=0, atol=0)
                for p, q in zip(plain.parameters(), checked.parameters()):
                    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
                checked.eval()
                with patch("transformer_training.model.checkpoint", side_effect=AssertionError("eval checkpoint")), torch.no_grad():
                    checked(torch.tensor([[2, 3]]))

    def test_optimizer_variants_preserve_groups_and_updates(self):
        reference = DecoderOnlyTransformer(self.model)
        state = deepcopy(reference.state_dict())
        outcomes = []
        for choice in ("standard", "foreach", "fused"):
            model = DecoderOnlyTransformer(self.model)
            model.load_state_dict(state)
            optimizer = make_optimizer(model, self.training, ImplementationConfig(optimizer_implementation=choice))
            self.assertEqual([g["weight_decay"] for g in optimizer.param_groups], [.1, 0.0])
            params = [p for group in optimizer.param_groups for p in group["params"]]
            self.assertEqual(len(params), len({id(p) for p in model.parameters()}))
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                accumulate_gradients(model, list(synthetic_loaders().train))
                optimizer.step()
            outcomes.append((deepcopy(model.state_dict()), optimizer.state_dict()))
        for state, optimizer in outcomes[1:]:
            for key, value in state.items():
                torch.testing.assert_close(value, outcomes[0][0][key], atol=1e-7, rtol=1e-5)
            for key, entry in optimizer["state"].items():
                for moment in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(entry[moment], outcomes[0][1]["state"][key][moment], atol=1e-8, rtol=1e-5)

    def test_checkpoint_resume_and_diagnostic_overrides(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            impl = ImplementationConfig(attention="sdpa", sdpa_backend="math", optimizer_implementation="foreach")
            trainer = Trainer(self.model, self.training, synthetic_loaders(), root / "run", implementation=impl)
            trainer.fit(stop_after_updates=1)
            payload = load_checkpoint(root / "run" / "latest.pt")
            self.assertEqual(payload["implementation_config"], asdict(impl))
            with self.assertRaisesRegex(ValueError, "implementation"):
                Trainer(self.model, self.training, synthetic_loaders(), root / "bad",
                    resume=root / "run" / "latest.pt", implementation=ImplementationConfig())
            restored = Trainer(self.model, self.training, synthetic_loaders(), root / "diagnostic",
                resume=root / "run" / "latest.pt", diagnostic=True,
                implementation=replace(impl, optimizer_implementation="standard"))
            self.assertTrue(all(not g["foreach"] and not g["fused"] for g in restored.optimizer.param_groups))
            self.assertEqual(restored.state, trainer.state)
            self.assertEqual(restored.checkpoint_sha256, file_sha256(root / "run" / "latest.pt"))

    def test_diagnostic_fused_restore_casts_step_and_preserves_moments(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            trainer = Trainer(self.model, self.training, synthetic_loaders(), root / "run")
            trainer.fit(stop_after_updates=1)
            payload = load_checkpoint(root / "run" / "latest.pt")
            for state in payload["optimizer"]["state"].values():
                state["step"] = state["step"].double()
            save_checkpoint(root / "double-step.pt", payload)
            fused = Trainer(self.model, self.training, synthetic_loaders(), root / "fused",
                resume=root / "double-step.pt", diagnostic=True,
                implementation=ImplementationConfig(optimizer_implementation="fused"))
            self.assertTrue(fused.optimizer._step_supports_amp_scaling)
            for original, restored in zip(payload["optimizer"]["state"].values(), fused.optimizer.state.values()):
                self.assertEqual(restored["step"].dtype, torch.float32)
                self.assertEqual(restored["step"].item(), original["step"].item())
                torch.testing.assert_close(restored["exp_avg"], original["exp_avg"], rtol=0, atol=0)
                torch.testing.assert_close(restored["exp_avg_sq"], original["exp_avg_sq"], rtol=0, atol=0)
            run_trial(fused, warmup=0, measured=1)
            self.assertEqual(fused.state["updates"], 2)
            for group in payload["optimizer"]["param_groups"]:
                group.update(fused=True, foreach=False)
            save_checkpoint(root / "saved-fused.pt", payload)
            standard = Trainer(self.model, self.training, synthetic_loaders(), root / "standard",
                resume=root / "saved-fused.pt", diagnostic=True, implementation=ImplementationConfig())
            self.assertFalse(standard.optimizer._step_supports_amp_scaling)
            run_trial(standard, warmup=0, measured=1)
            self.assertEqual(standard.state["updates"], 2)

    def test_sdpa_evidence_and_declared_comparison(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            def trial(name, impl):
                trainer = Trainer(self.model, self.training, synthetic_loaders(), root / name, implementation=impl)
                return run_trial(trainer, warmup=1, measured=2)
            a = aggregate_trials([trial("ref", ImplementationConfig())])
            b = aggregate_trials([trial("sdpa", ImplementationConfig(attention="sdpa", sdpa_backend="math"))])
            evidence = b["trials"][0]["execution"]["attention_evidence"]
            self.assertEqual(evidence["status"], "observed")
            self.assertIn("math", evidence["observed"])
            self.assertEqual(compare_results(a, b)["status"], "conditions_differ")
            changes = ["implementation.attention", "implementation.sdpa_backend"]
            self.assertEqual(compare_results(a, b, allowed_changes=changes)["status"], "matched")
            b["comparison"]["training"]["batch_size"] = 3
            self.assertIn("training.batch_size", compare_results(a, b, allowed_changes=changes)["unexpected_differences"])
            with self.assertRaises(ValueError):
                compare_results(a, b, allowed_changes=["training.batch_size"])

    def test_compile_boundary_and_state_dict_with_aot_eager(self):
        # AOT eager verifies Dynamo/AOT capture and backward without requiring a
        # local C++/CUDA codegen toolchain. This is not an Inductor speed test.
        self.addCleanup(torch.compiler.reset)
        actual_compile = torch.compile
        def compile_for_test(function, **options):
            self.assertEqual(options["backend"], "inductor")
            self.assertTrue(options["fullgraph"])
            return actual_compile(function, backend="aot_eager", fullgraph=True)
        for attention in ("reference", "sdpa"):
            with self.subTest(attention=attention), patch("torch.compile", side_effect=compile_for_test):
                # Independent models/backends share this method's Dynamo code
                # cache, including earlier CUDA tests. Keep the shape and mode
                # transitions below together, but isolate each attention case.
                torch.compiler.reset()
                impl = ImplementationConfig(attention=attention)
                plain = DecoderOnlyTransformer(self.model, impl)
                model = DecoderOnlyTransformer(self.model, replace(impl, compile=True, activation_checkpointing=True))
                model.load_state_dict(plain.state_dict())
                self.assertEqual(set(model.state_dict()), set(plain.state_dict()))
                for ids in (torch.tensor([[2, 3, 4, 1], [3, 1, 1, 1]]), torch.tensor([[2]])):
                    plain.zero_grad(set_to_none=True)
                    model.zero_grad(set_to_none=True)
                    a, b = plain(ids), model(ids)
                    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
                    a.sum().backward()
                    b.sum().backward()
                    for p, q in zip(plain.parameters(), model.parameters()):
                        torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)
                with self.assertRaisesRegex(ValueError, "right-padded"):
                    model(torch.tensor([[2, 3]]), torch.tensor([[False, True]]))
                model.eval()
                with torch.no_grad():
                    model(torch.tensor([[2, 3]]))
                self.assertEqual(compiler_snapshot(True)["status"], "available")

    def test_compilation_in_measured_interval_is_ineligible(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            trainer = Trainer(self.model, self.training, synthetic_loaders(), Path(directory) / "trial")
            snapshots = iter({"status": "available", "events": list(range(n))} for n in (0, 1, 1, 2))
            with patch("transformer_training.diagnostics.compiler_snapshot", side_effect=lambda _: next(snapshots)):
                result = run_trial(trainer, warmup=1, measured=1)
            self.assertTrue(result["compilation_contaminated"])
            self.assertFalse(result["timings_eligible_for_comparison"])
            report = aggregate_trials([result])
            self.assertEqual(compare_results(report, report)["status"], "timings_ineligible")

    def test_compile_modes_errors_and_validation_remain_explicit(self):
        for mode in ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
            with self.subTest(mode=mode), patch("torch.compile") as compile:
                compile.return_value.side_effect = RuntimeError("compiler failed")
                model = DecoderOnlyTransformer(self.model, ImplementationConfig(compile=True, compile_mode=mode))
                self.assertEqual(compile.call_args.kwargs, {"backend": "inductor", "mode": mode, "fullgraph": True})
                with self.assertRaisesRegex(ValueError, "right-padded"):
                    model(torch.tensor([[2, 3]]), torch.tensor([[False, True]]))
                compile.return_value.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "compiler failed"):
                    model(torch.tensor([[2, 3]]))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; GPU implementation acceptance not verified")
class CudaImplementationTests(unittest.TestCase):
    def test_amp_sdpa_checkpointing_and_fused_overflow_recovery(self):
        model_config = ModelConfig(vocab_size=12, sequence_length=4, layers=1,
                                   d_model=32, heads=4, d_ff=64, dropout=.1)
        batches = list(synthetic_loaders().train)[:2]
        for choice in ("standard", "foreach", "fused"):
            config = TrainingConfig(device="cuda", precision="fp16", batch_size=2,
                                    gradient_accumulation_steps=2, num_workers=0)
            model = DecoderOnlyTransformer(model_config, ImplementationConfig(attention="sdpa",
                optimizer_implementation=choice, activation_checkpointing=True)).cuda()
            optimizer = make_optimizer(model, config)
            scheduler = WarmupCosine(optimizer, config, 3)
            scaler = torch.amp.GradScaler("cuda", init_scale=128)
            first = next(model.parameters())
            original = first.detach().clone()
            hook = first.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
            result = optimizer_attempt(model, optimizer, scaler, scheduler, batches, config, torch.device("cuda"))
            hook.remove()
            self.assertTrue(result["skipped"])
            self.assertEqual(scheduler.completed_updates, 0)
            torch.testing.assert_close(first, original, rtol=0, atol=0)
            result = optimizer_attempt(model, optimizer, scaler, scheduler, batches, config, torch.device("cuda"))
            self.assertFalse(result["skipped"])
            self.assertEqual(scheduler.completed_updates, 1)
        if torch.cuda.is_bf16_supported(including_emulation=False):
            config = replace(config, precision="bf16")
            model = DecoderOnlyTransformer(model_config, ImplementationConfig(attention="sdpa", sdpa_backend="math")).cuda()
            result = accumulate_gradients(model, batches, device="cuda", precision="bf16")
            self.assertTrue(result["loss_sum"] > 0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_inductor_modes_forward_backward_and_partial_batch(self):
        self.addCleanup(torch.compiler.reset)
        config = ModelConfig(vocab_size=12, sequence_length=4, layers=1,
                             d_model=32, heads=4, d_ff=64, dropout=0)
        reference = DecoderOnlyTransformer(config).cuda()
        for mode in ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
            # Each compiler mode is an independent case. Retain both batch
            # shapes in its cache so partial-batch recompilation stays tested.
            torch.compiler.reset()
            model = DecoderOnlyTransformer(config, ImplementationConfig(attention="sdpa", compile=True,
                compile_mode=mode, activation_checkpointing=True)).cuda()
            model.load_state_dict(reference.state_dict())
            for shape in ((2, 4), (1, 4)):
                ids = torch.full(shape, 2, dtype=torch.long, device="cuda")
                reference.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)
                a, b = reference(ids), model(ids)
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
                a.sum().backward()
                b.sum().backward()
                for p, q in zip(reference.parameters(), model.parameters()):
                    torch.testing.assert_close(p.grad, q.grad, atol=1e-4, rtol=1e-3)
                del a, b


class PlanTests(unittest.TestCase):
    def plan(self):
        return {"version": 1, "shared": {"training": {"device": "cpu"}},
                "benchmark": {"repeats": 1, "warmup": 1, "measured": 1},
                "variants": [{"id": "reference"}, {"id": "foreach", "reference": "reference",
                    "implementation": {"optimizer_implementation": "foreach"},
                    "changes": ["implementation.optimizer_implementation"]}]}

    def test_plan_rejects_unsafe_ambiguous_or_uncontrolled_settings(self):
        plan = self.plan()
        resolved = resolve_plan(plan)
        self.assertEqual(resolved["shared"]["training"]["precision"], "fp32")
        self.assertEqual(resolved["shared"]["model"]["d_model"], 160)
        invalid = []
        for mutate in (lambda p: p.update(extra=1),
            lambda p: p["variants"][1].update(id="../bad"),
            lambda p: p["variants"][1].update(id="reference"),
            lambda p: p["variants"][1].update(reference="missing"),
            lambda p: p["variants"][1].update(changes=[]),
            lambda p: p["variants"][1].update(changes=["training.batch_size"]),
            lambda p: p.update(quality={"supervised_token_budget": 10}),
            lambda p: p["shared"].update(data={"revision": "main"}),
            lambda p: p["benchmark"].update(warmup=-1)):
            altered = deepcopy(plan)
            mutate(altered)
            invalid.append(altered)
        for altered in invalid:
            with self.subTest(plan=altered), self.assertRaises((ValueError, TypeError)):
                resolve_plan(altered)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"version": 1, "version": 1}')
            with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
                load_plan(path)

    def test_suite_and_diagnostic_dry_runs_have_no_side_effects(self):
        for command in (["scripts/run_experiments.py", "--plan", "experiments/implementations.json"],
                        ["scripts/benchmark_training.py", "--compile", "--attention", "sdpa"]):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "absent"
                code = ("import runpy,sys,torch; sys.argv=" + repr(command + ["--dry-run", "--output-dir", str(output)]) +
                    ";\ntry: runpy.run_path(sys.argv[0],run_name='__main__')\nexcept SystemExit as e: assert not e.code\n"
                    "assert 'datasets' not in sys.modules; assert 'transformer_training.data' not in sys.modules; "
                    "assert not torch.cuda.is_initialized()")
                result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(output.exists())

    def test_quality_pairs_flag_mismatched_work_and_require_margin(self):
        quality = {"seeds": [42], "supervised_token_budget": 100}
        result = {"processed_tokens": 110, "applied_tokens": 110, "skipped_updates": 0,
                  "validation_identity": "same", "loss": 2.0, "budget_overshoot": 10}
        a = {"42": result}
        b = {"42": dict(result, loss=2.01)}
        pair = paired_quality(b, a, quality)["pairs"][0]
        self.assertEqual(pair["status"], "matched")
        self.assertIsNone(pair["within_declared_margin"])
        self.assertTrue(paired_quality(b, a, dict(quality, degradation_margin=.02))["pairs"][0]["within_declared_margin"])
        b["42"]["applied_tokens"] = 100
        self.assertEqual(paired_quality(b, a, quality)["pairs"][0]["status"], "conditions_differ")

    def test_real_suite_processes_failures_resume_and_manifest(self):
        plan = self.plan()
        plan["shared"].update(model={"vocab_size": 12, "sequence_length": 4, "layers": 1,
            "d_model": 12, "heads": 3, "d_ff": 24, "dropout": 0}, offline=True)
        plan["shared"]["training"].update(batch_size=2, gradient_accumulation_steps=1, epochs=3, num_workers=0)
        plan["variants"].insert(1, {"id": "unsupported", "precision": "bf16", "reference": "reference",
                                  "changes": ["training.precision"]})
        resolved = resolve_plan(plan)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            (root / "sitecustomize.py").write_text("from transformer_training import data\n"
                "from test_training import synthetic_loaders\n"
                "data.load_tinystories = lambda **kwargs: synthetic_loaders()\n")
            pythonpath = os.pathsep.join([str(root), str(Path(__file__).parent.resolve()), os.environ.get("PYTHONPATH", "")])
            with patch.dict(os.environ, {"PYTHONPATH": pythonpath}):
                output = root / "suite"
                report = run_suite(resolved, output)
                self.assertEqual(report["status"], "incomplete")
                self.assertEqual([r["status"] for r in report["rows"]], ["complete", "incomplete", "complete"])
                self.assertIsNotNone(report["rows"][2]["speedup"])
                for suffix in ("json", "csv", "md"):
                    self.assertTrue((output / f"report.{suffix}").exists())
                state = json.loads((output / "suite-state.json").read_text())
                before = state["jobs"]["reference/benchmark/0"]["attempt"]
                report = run_suite(resolved, output, resume=True)
                self.assertEqual(report["jobs"]["reference/benchmark/0"]["attempt"], before)
                self.assertEqual(report["jobs"]["unsupported/benchmark/0"]["attempt"], 2)
                altered = deepcopy(resolved)
                altered["benchmark"]["measured"] = 2
                with self.assertRaisesRegex(ValueError, "changed"):
                    run_suite(altered, output, resume=True)
                artifact = output / state["jobs"]["reference/benchmark/0"]["result"]
                artifact.write_text(artifact.read_text() + " ")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    run_suite(resolved, output, resume=True)

    def test_quality_suite_subprocess_and_interrupted_job_resume(self):
        plan = self.plan()
        plan["shared"].update(model={"vocab_size": 12, "sequence_length": 4, "layers": 1,
            "d_model": 12, "heads": 3, "d_ff": 24, "dropout": 0}, offline=True)
        plan["shared"]["training"].update(batch_size=2, gradient_accumulation_steps=1, epochs=5, num_workers=0)
        plan["quality"] = {"supervised_token_budget": 6, "seeds": [19], "degradation_margin": .02}
        resolved = resolve_plan(plan)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            (root / "sitecustomize.py").write_text(
                "import torch\ntorch.set_num_threads(1)\n"
                "from transformer_training import data\n"
                "from test_evaluation import quality_splits,local_validation\n"
                "from test_data import fixture_tokenizer\n"
                "def load(**options):\n"
                " revision=options.pop('revision'); tokenizer_revision=options.pop('tokenizer_revision')\n"
                " result=data.create_dataloaders(quality_splits(),fixture_tokenizer(),**options)\n"
                " result.metadata.update(dataset=data.DATASET,tokenizer=data.TOKENIZER,revision=revision,tokenizer_revision=tokenizer_revision)\n"
                " return result\n"
                "data.load_tinystories=load\ndata.load_tinystories_validation=local_validation\n")
            pythonpath = os.pathsep.join([str(root), str(Path(__file__).parent.resolve()), os.environ.get("PYTHONPATH", "")])
            with patch.dict(os.environ, {"PYTHONPATH": pythonpath}):
                output = root / "quality-suite"
                report = run_suite(resolved, output)
                self.assertEqual(report["status"], "complete", report["jobs"])
                pair = report["rows"][1]["paired_quality"]["pairs"][0]
                self.assertEqual(pair["status"], "matched")
                self.assertTrue(pair["within_declared_margin"])
                original = report["rows"][1]["quality"]["19"]
                path = output / "suite-state.json"
                state = json.loads(path.read_text())
                job = state["jobs"]["foreach/quality/19"]
                job["status"] = "running"  # Simulate interruption after a saved training checkpoint.
                training_dir = job["training_dir"]
                path.write_text(json.dumps(state))
                report = run_suite(resolved, output, resume=True)
                self.assertEqual(report["status"], "complete", report["jobs"])
                restored = report["rows"][1]["quality"]["19"]
                self.assertEqual(restored["processed_tokens"], original["processed_tokens"])
                self.assertEqual(restored["loss"], original["loss"])
                self.assertEqual(report["jobs"]["foreach/quality/19"]["training_dir"], training_dir)
                self.assertEqual(report["jobs"]["foreach/quality/19"]["attempt"], 2)


if __name__ == "__main__":
    unittest.main()
