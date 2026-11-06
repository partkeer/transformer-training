"""Offline backend contracts and CUDA-only Triton numerical/integration acceptance."""

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from transformer_training.cli import main as train_main
from transformer_training.config import ImplementationConfig, ModelConfig, TrainingConfig
from transformer_training.diagnostics_cli import main as diagnostic_main
from transformer_training.evaluation import evaluate, next_token_loss
from transformer_training.evaluation_cli import build_parser as evaluation_parser
from transformer_training.experiment_plan import load_plan
from transformer_training.implementations import resolve_implementation
from transformer_training.loss import (_validate_triton_inputs, cross_entropy_sum, model_loss_backend,
                                       triton_launch_settings, validate_loss_backend)
from transformer_training.loss_benchmark import (check_loss_and_gradient, main as benchmark_main,
                                                 summarize_cases)
from transformer_training.model import DecoderOnlyTransformer
from transformer_training.training import (CHECKPOINT_VERSION, WarmupCosine, accumulate_gradients,
                                           make_optimizer, optimizer_attempt, validate_resume_settings)
from test_training import synthetic_loaders


def triton_gpu_ready():
    """Select Linux SM75+ CUDA devices; installed-version errors must fail acceptance."""
    return (platform.system() == "Linux" and torch.version.hip is None and torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (7, 5)
            and importlib.util.find_spec("triton") is not None)


class LossContracts(unittest.TestCase):
    def test_reference_sum_and_alignment(self):
        """Preserve the exact default reference, EOS target, padding, and logits gradients."""
        logits = torch.randn(2, 3, 7, requires_grad=True)
        labels = torch.tensor([[1, 6, -100], [3, -100, 6]])
        actual = next_token_loss(logits, labels)
        expected = F.cross_entropy(logits.reshape(-1, 7), labels.reshape(-1), ignore_index=-100, reduction="sum")
        self.assertTrue(torch.equal(actual, expected))
        torch.testing.assert_close(torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0])

    def test_reference_all_ignored_and_empty(self):
        """Summed objectives return differentiable zero for empty/all-ignored inputs."""
        for rows in (0, 3):
            logits = torch.randn(rows, 7, requires_grad=True)
            loss = cross_entropy_sum(logits, torch.full((rows,), -100, dtype=torch.int64))
            loss.backward()
            self.assertEqual(loss.item(), 0)
            self.assertEqual(torch.count_nonzero(logits.grad).item(), 0)

    def test_backend_settings_and_old_checkpoints(self):
        """Old checkpoints default to PyTorch; evaluation can explicitly override the saved backend."""
        self.assertEqual(ImplementationConfig().loss_backend, "pytorch")
        self.assertEqual(resolve_implementation(argparse.Namespace(), {"implementation_config": {}}).loss_backend, "pytorch")
        args = evaluation_parser().parse_args(["--checkpoint", "saved.pt", "--loss-backend", "pytorch"])
        self.assertEqual(resolve_implementation(args, {"implementation_config": {"loss_backend": "triton"}}).loss_backend, "pytorch")
        with self.assertRaises(ValueError):
            ImplementationConfig(loss_backend="automatic")
        self.assertEqual(model_loss_backend(torch.nn.Linear(2, 2)), "pytorch")

    def test_resume_backend_restrictions(self):
        """Normal resume rejects backend changes; diagnostic comparisons allow explicit changes."""
        model, training = ModelConfig(), TrainingConfig()
        payload = {"version": CHECKPOINT_VERSION, "model_config": asdict(model), "training_config": asdict(training)}
        triton = ImplementationConfig(loss_backend="triton")
        validate_resume_settings(payload, model, training)
        with self.assertRaisesRegex(ValueError, "implementation"):
            validate_resume_settings(payload, model, training, triton)
        validate_resume_settings(payload, model, training, triton, diagnostic=True)

    def test_explicit_failure_without_triton_import(self):
        """Unsupported CPU requests explain the explicit fallback and never import the kernel module."""
        with patch("transformer_training.loss.platform.system", return_value="Linux"):
            with self.assertRaisesRegex(ValueError, "loss-backend pytorch"):
                cross_entropy_sum(torch.randn(3, 7), torch.zeros(3, dtype=torch.int64), backend="triton")
        with self.assertRaisesRegex(ValueError, "Unknown loss backend"):
            cross_entropy_sum(torch.randn(3, 7), torch.zeros(3, dtype=torch.int64), backend="typo")

    def test_optional_import_failure_is_actionable(self):
        """A broken optional kernel import must report its cause and the explicit fallback."""
        with patch("transformer_training.loss._validate_triton_inputs"), \
                patch.dict(sys.modules, {"transformer_training.triton_cross_entropy": None}):
            with self.assertRaisesRegex(RuntimeError, "Cannot load Triton.*loss-backend pytorch"):
                cross_entropy_sum(torch.randn(3, 7), torch.zeros(3, dtype=torch.int64), backend="triton")

    def test_runtime_prerequisite_errors(self):
        """Missing CUDA, SM75, and Triton each fail before loading optional kernels."""
        with patch("transformer_training.loss.platform.system", return_value="Linux"), \
                patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "available CUDA"):
                validate_loss_backend("triton", "cuda")
        with patch("transformer_training.loss.platform.system", return_value="Linux"), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_capability", return_value=(7, 0)):
            with self.assertRaisesRegex(ValueError, "capability"):
                validate_loss_backend("triton", "cuda")
        with patch("transformer_training.loss.platform.system", return_value="Linux"), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_capability", return_value=(8, 9)), \
                patch("transformer_training.loss.importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(ValueError, "Triton is unavailable"):
                validate_loss_backend("triton", "cuda")

    def test_turing_requires_triton_32(self):
        """Accept SM75 with Triton 3.2 metadata, rejecting newer or broken installs lazily."""
        with patch("transformer_training.loss.platform.system", return_value="Linux"), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.get_device_capability", return_value=(7, 5)), \
                patch("transformer_training.loss.importlib.util.find_spec", return_value=object()), \
                patch("transformer_training.loss.importlib.metadata.version") as version:
            for installed in ("3.2.0", "3.2.0.post21"):
                version.return_value = installed
                validate_loss_backend("triton", "cuda", 50257)
            for installed in ("3.1.0", "3.3.0", "3.20.0"):
                version.return_value = installed
                with self.subTest(version=installed), self.assertRaisesRegex(ValueError, "Triton 3.2.x"):
                    validate_loss_backend("triton", "cuda")
            version.side_effect = importlib.metadata.PackageNotFoundError("triton")
            with self.assertRaisesRegex(ValueError, "metadata.*reinstall"):
                validate_loss_backend("triton", "cuda")

    def test_turing_rejects_bf16_logits(self):
        """Reject BF16 at the public input boundary even when runtime preflight succeeds."""
        with patch("transformer_training.loss.validate_loss_backend"), \
                patch("torch.cuda.get_device_capability", return_value=(7, 5)):
            with self.assertRaisesRegex(ValueError, "BF16.*FP32 or FP16"):
                _validate_triton_inputs(torch.zeros(3, 7, dtype=torch.bfloat16), torch.zeros(3, dtype=torch.int64))

    def test_input_validation(self):
        """Reject shapes, dtype, layout and label bounds independently of optional GPU prerequisites."""
        logits, targets = torch.randn(3, 7), torch.tensor([0, 6, -100])
        invalid = [(logits[None], targets), (logits, targets[:, None]), (logits, targets[:2]),
                   (logits.double(), targets), (logits, targets.int()),
                   (logits.t(), torch.zeros(7, dtype=torch.int64)),
                   (logits, torch.zeros(6, dtype=torch.int64)[::2]),
                   (logits, torch.tensor([-1, 0, 0])), (logits, torch.tensor([7, 0, 0]))]
        with patch("transformer_training.loss.validate_loss_backend"):
            _validate_triton_inputs(logits, targets)
            for x, y in invalid:
                with self.subTest(shape=x.shape, dtype=x.dtype, target=y), self.assertRaises(ValueError):
                    _validate_triton_inputs(x, y)
            with self.assertRaisesRegex(ValueError, "same device"):
                _validate_triton_inputs(logits, torch.empty(3, dtype=torch.int64, device="meta"))
        for vocab in (0, 65537):
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                triton_launch_settings(vocab)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            next_token_loss(logits.t(), torch.zeros(7, dtype=torch.int64), backend="triton")
        with self.assertRaisesRegex(ValueError, "leading"):
            next_token_loss(logits, targets[None], backend="triton")
        self.assertEqual(triton_launch_settings(50257), {"forward_block": 65536, "forward_warps": 32,
                                                        "backward_block": 1024, "backward_warps": 4})

    def test_shared_backend_and_token_normalisation(self):
        """Shared helpers inherit the backend and preserve uneven-batch token weighting."""
        config = ModelConfig(vocab_size=12, sequence_length=4, layers=1, d_model=8, heads=2, d_ff=16, dropout=0)
        reference = DecoderOnlyTransformer(config)
        selected = DecoderOnlyTransformer(config, ImplementationConfig(loss_backend="triton"))
        selected.load_state_dict(reference.state_dict())
        # Includes unequal token counts and a one-example partial batch.
        batches = list(synthetic_loaders().train)[-2:]
        expected = accumulate_gradients(reference, batches)
        with patch("transformer_training.training.next_token_loss", wraps=next_token_loss) as dispatch:
            # Emulate a valid backend at the dispatcher boundary; this is CPU wiring evidence only.
            with patch("transformer_training.evaluation.cross_entropy_sum", side_effect=lambda x, y, **_: cross_entropy_sum(x, y)):
                actual = accumulate_gradients(selected, batches)
        self.assertTrue(all(c.kwargs["backend"] == "triton" for c in dispatch.call_args_list))
        self.assertAlmostEqual(expected["loss_sum"], actual["loss_sum"], places=5)
        for a, b in zip(reference.parameters(), selected.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        with patch("transformer_training.evaluation.next_token_loss", wraps=next_token_loss) as dispatch:
            with patch("transformer_training.evaluation.cross_entropy_sum", side_effect=lambda x, y, **_: cross_entropy_sum(x, y)):
                evaluate(selected, batches)
        self.assertTrue(all(c.kwargs["backend"] == "triton" for c in dispatch.call_args_list))

    def test_dry_runs_have_no_gpu_or_data_side_effects(self):
        """Triton dry-runs resolve settings without runtime checks, data access, or output creation."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            with patch("torch.cuda.is_available", side_effect=AssertionError("CUDA check")), \
                    patch("transformer_training.loss.validate_loss_backend", side_effect=AssertionError("loss preflight")), \
                    patch("transformer_training.data.load_tinystories", side_effect=AssertionError("data access")), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(train_main(["--loss-backend", "triton", "--dry-run", "--output-dir", str(output)]), 0)
                for kind in ("benchmark", "profile"):
                    self.assertEqual(diagnostic_main(kind, ["--loss-backend", "triton", "--dry-run", "--output-dir", str(output)]), 0)
                self.assertEqual(benchmark_main(["--dry-run", "--output-dir", str(output)]), 0)
            self.assertFalse(output.exists())

    def test_default_import_is_lazy(self):
        """A fresh process can import/default/dry-run without importing Triton or its kernel module."""
        code = ("import sys, torch; from transformer_training.evaluation import next_token_loss; "
                "next_token_loss(torch.zeros(2,3),torch.tensor([0,-100])); "
                "from transformer_training.cli import main; main(['--loss-backend','triton','--dry-run']); "
                "assert 'triton' not in sys.modules; "
                "assert 'transformer_training.triton_cross_entropy' not in sys.modules; "
                "assert not torch.cuda.is_initialized()")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_preflight_precedes_data_and_outputs(self):
        """A real unsupported training request fails before loading data or creating its output."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            with patch("transformer_training.data.load_tinystories", side_effect=AssertionError("data access")), \
                    patch("transformer_training.loss.platform.system", return_value="Windows"):
                with self.assertRaises(SystemExit):
                    train_main(["--loss-backend", "triton", "--device", "cpu", "--output-dir", str(output)])
            self.assertFalse(output.exists())

    def test_standalone_evaluation_override_and_preflight(self):
        """Saved Triton settings fail early on CPU; explicit PyTorch evaluation restores reference behaviour."""
        from test_evaluation import checkpoint_fixture, local_validation
        from transformer_training.checkpoint import save_checkpoint
        from transformer_training.quality import evaluate_checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "saved.pt"
            payload, _, _ = checkpoint_fixture(checkpoint)
            payload["implementation_config"] = asdict(ImplementationConfig(loss_backend="triton"))
            save_checkpoint(checkpoint, payload)
            original = checkpoint.read_bytes()
            output = Path(temporary) / "evaluation"
            with patch("transformer_training.data.load_tinystories_validation", side_effect=AssertionError("data access")):
                with self.assertRaisesRegex(ValueError, "loss-backend pytorch"):
                    evaluate_checkpoint(checkpoint, output_dir=output, skip_generation=True)
            self.assertFalse(output.exists())
            with patch("transformer_training.data.load_tinystories_validation", side_effect=local_validation):
                report = evaluate_checkpoint(checkpoint, output_dir=output, skip_generation=True,
                                             implementation_overrides={"loss_backend": "pytorch"})
            self.assertEqual(report["implementation"]["loss_backend"], "pytorch")
            self.assertEqual(report["execution"]["loss"]["backend"], "pytorch")
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_experiment_pairs(self):
        """All complete-update comparisons change only the loss backend within a precision."""
        plan = load_plan("experiments/fused-cross-entropy.json")
        for variant in plan["variants"]:
            if variant["reference"]:
                self.assertEqual(variant["changes"], ["implementation.loss_backend"])

    def test_unavailable_benchmark_report(self):
        """Unavailable GPU measurements are explicit null-free statuses, never invented zeros."""
        with tempfile.TemporaryDirectory() as temporary, patch("torch.cuda.is_available", return_value=False):
            output = Path(temporary) / "report"
            self.assertEqual(benchmark_main(["--output-dir", str(output)]), 2)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "unavailable")
            self.assertEqual(report["cases"], [])
            self.assertIn("not run", (output / "report.md").read_text())

    def test_aggregation_rejects_partial_repeats(self):
        """A failed repeat prevents a misleading aggregate or speedup comparison."""
        request = {"rows": 7, "vocab": 17, "precision": "fp32", "ignored_fraction": .5, "backend": "triton"}
        summary = summarize_cases([{"status": "failed", "request": request}])
        self.assertEqual(summary["groups"][0]["status"], "incomplete")
        self.assertEqual(summary["groups"][0]["operations"], {})
        self.assertEqual(summary["comparisons"], [])

    def test_aggregation_speed_and_memory_are_separate(self):
        """Matched complete repeats produce the expected operation speed and allocator ratios."""
        cases = []
        for backend, seconds, allocated in (("pytorch", .004, 1000), ("triton", .002, 700)):
            for repeat in range(2):
                request = {"rows": 7, "vocab": 17, "precision": "fp32", "ignored_fraction": .5,
                           "backend": backend, "repeat": repeat}
                names = ("forward", "backward", "forward_backward")
                cases.append({"status": "complete", "request": request,
                              "timings": {n: {"latency_seconds": {"median": seconds},
                                               "supervised_tokens_per_second": 3 / seconds} for n in names},
                              "memory": {n: {"peak_allocated_bytes": allocated,
                                             "peak_reserved_bytes": 2000,
                                             "incremental_peak_allocated_bytes": allocated - 100} for n in names}})
        summary = summarize_cases(cases)
        comparison = summary["comparisons"][0]["operations"]["forward_backward"]
        self.assertEqual(comparison["speedup"], 2)
        self.assertEqual(comparison["peak_allocated_bytes_saved"], 300)
        self.assertAlmostEqual(comparison["peak_allocated_reduction_fraction"], .3)


@unittest.skipUnless(triton_gpu_ready(), "Linux SM75+ CUDA and Triton unavailable; GPU correctness not verified")
class TritonAcceptance(unittest.TestCase):
    def test_shapes_dtypes_targets_and_scales(self):
        """Compare irregular/boundary shapes, ignored targets, and non-unit upstream gradients."""
        for precision, dtype in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
                continue
            for rows, vocab in ((0, 7), (1, 1), (1, 7), (7, 31), (13, 32), (5, 33),
                                (3, 1023), (3, 1024), (3, 1025), (3, 2047), (3, 2048), (3, 2049),
                                (3, 8191), (3, 8192), (3, 8193), (3, 32768), (3, 32769),
                                (7, 50257), (3, 65535), (3, 65536), (257, 17)):
                logits = torch.randn(rows, vocab, device="cuda", dtype=dtype)
                targets = torch.randint(vocab, (rows,), device="cuda")
                if rows:
                    targets[0] = 0
                if rows > 1:
                    targets[-1] = vocab - 1  # GPT-2 EOS for V=50257 is a genuine target.
                    targets[1:-1:2] = -100
                saved = logits.clone()
                for upstream in (0., 1., -2.5, 128.):
                    with self.subTest(precision=precision, rows=rows, vocab=vocab, upstream=upstream):
                        check_loss_and_gradient(logits, targets, "triton", precision, upstream=upstream)
                self.assertTrue(torch.equal(saved, logits))

    def test_extreme_finite_and_all_ignored(self):
        """Check extreme values and ignored zeros in each precision supported by the GPU."""
        for precision, dtype in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation=False):
                continue
            cases = [torch.zeros(5, 50257, device="cuda", dtype=dtype),
                     torch.full((5, 33), 10000., device="cuda", dtype=dtype),
                     torch.linspace(-10000, 10000, 33, device="cuda", dtype=dtype).expand(5, -1).contiguous(),
                     (torch.randn(5, 33, device="cuda") + 1e4).to(dtype)]
            if dtype == torch.float32:
                cases += [torch.full((5, 33), 1e20, device="cuda"),
                          torch.tensor([[1e30, -1e30, 0]], device="cuda").expand(5, -1).contiguous()]
            for logits in cases:
                for ignored in (False, True):
                    targets = torch.full((5,), -100 if ignored else 0, device="cuda", dtype=torch.int64)
                    if not ignored:
                        targets[-1] = logits.shape[1] - 1
                    with self.subTest(precision=precision, vocab=logits.shape[1], ignored=ignored):
                        check_loss_and_gradient(logits, targets, "triton", precision)

    def test_fp32_scalar_and_retained_graph(self):
        """Backward reuses immutable statistics and rejects higher-order autograd explicitly."""
        logits = torch.randn(7, 50257, device="cuda", requires_grad=True)
        targets = torch.tensor([0, 50256, -100, 3, 4, 5, 6], device="cuda")
        original = logits.detach().clone()
        loss = cross_entropy_sum(logits, targets, backend="triton")
        self.assertEqual(loss.dtype, torch.float32)
        first, = torch.autograd.grad(loss, logits, retain_graph=True)
        second, = torch.autograd.grad(loss, logits, grad_outputs=loss.new_tensor(-2.5), retain_graph=True)
        torch.testing.assert_close(second, first * -2.5, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(logits, original))
        with self.assertRaisesRegex(RuntimeError, "first-order"):
            torch.autograd.grad(loss, logits, create_graph=True)

    def test_invalid_cuda_targets(self):
        """Reject bad indices before launching kernels; a subsequent valid call still works."""
        logits = torch.zeros(3, 17, device="cuda")
        for target in (-1, 17):
            with self.assertRaisesRegex(ValueError, "Targets"):
                cross_entropy_sum(logits, torch.tensor([0, target, -100], device="cuda"), backend="triton")
        self.assertTrue(torch.isfinite(cross_entropy_sum(logits, torch.tensor([0, 16, -100], device="cuda"), backend="triton")))

    def test_model_accumulation_optimizer_and_validation(self):
        """Compare eager/compiled gradients, updates and validation using native GPU precisions."""
        model_config = ModelConfig(vocab_size=12, sequence_length=4, layers=1, d_model=16, heads=2, d_ff=32, dropout=0)
        batches = list(synthetic_loaders().train)[-2:]
        # An all-ignored microbatch is allowed when the accumulation group has supervision.
        ignored = {k: v.clone() for k, v in batches[0].items()}
        ignored["labels"].fill_(-100)
        batches = [ignored, *batches]
        for compiled in (False, True):
            for precision in ("fp32", "fp16", "bf16"):
                if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
                    continue
                with self.subTest(compiled=compiled, precision=precision):
                    config = TrainingConfig(device="cuda", precision=precision, batch_size=2, gradient_accumulation_steps=3)
                    implementation = ImplementationConfig(compile=compiled)
                    reference = DecoderOnlyTransformer(model_config, implementation).cuda()
                    candidate = DecoderOnlyTransformer(model_config, replace(implementation, loss_backend="triton")).cuda()
                    candidate.load_state_dict(reference.state_dict())
                    accumulate_gradients(reference, batches, device="cuda", precision=precision)
                    accumulate_gradients(candidate, batches, device="cuda", precision=precision)
                    rtol, atol = {"fp32": (2e-4, 2e-6), "fp16": (1e-2, 2e-4), "bf16": (5e-2, 2e-3)}[precision]
                    for a, b in zip(reference.parameters(), candidate.parameters()):
                        torch.testing.assert_close(a.grad, b.grad, rtol=rtol, atol=atol)
                    optimizers = [make_optimizer(m, config, m.implementation) for m in (reference, candidate)]
                    scalers = [torch.amp.GradScaler("cuda", enabled=precision == "fp16", init_scale=128) for _ in range(2)]
                    schedulers = [WarmupCosine(o, config, 10) for o in optimizers]
                    for _ in range(3):
                        outcomes = [optimizer_attempt(m, o, s, schedule, batches, config, torch.device("cuda"))
                                    for m, o, s, schedule in zip((reference, candidate), optimizers, scalers, schedulers)]
                        self.assertEqual(outcomes[0]["skipped"], outcomes[1]["skipped"])
                        self.assertFalse(outcomes[1]["skipped"])
                        for a, b in zip(reference.parameters(), candidate.parameters()):
                            torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
                    a = evaluate(reference, batches, device="cuda", precision=precision)
                    b = evaluate(candidate, batches, device="cuda", precision=precision)
                    self.assertEqual(a["supervised_tokens"], b["supervised_tokens"])
                    self.assertAlmostEqual(a["loss"], b["loss"], delta=max(1e-4, 2 * atol))
                if compiled:
                    torch.compiler.reset()


if __name__ == "__main__":
    unittest.main()
