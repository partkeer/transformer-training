"""Offline measurement correctness, synthetic updates and profiler/import fixtures."""

from contextlib import redirect_stdout
from dataclasses import asdict, replace
import csv
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

from test_training import synthetic_loaders
from transformer_training.checkpoint import load_checkpoint, save_checkpoint
from transformer_training.config import ModelConfig, TrainingConfig
from transformer_training.diagnostics import aggregate_trials, compare_results, run_trial
from transformer_training.measurements import Invocation, DeviceMemorySampler
from transformer_training.memory_profile import SavedActivations, static_memory, unique_storage_bytes, replay_allocator_peak
from transformer_training.nsight import METRICS, normalize, parse_ncu_csv, run_tool
from transformer_training.reporting import report_training
from transformer_training.traces import analyse_trace, load_nsys_sqlite, union, intersection, subtract
from transformer_training.training import Trainer


class AccountingTests(unittest.TestCase):
    def test_gradient_failures_distinguish_elements_from_norm_reduction(self):
        """Verify nonfinite gradient elements and finite-element norm overflow produce
        distinct failure kinds for measurement reporting.
        """
        from transformer_training.training import clip_gradients, NumericalFailure
        parameter = torch.nn.Parameter(torch.ones(2))
        for value, kind in ((float("inf"), "nonfinite_gradients"), (1e20, "gradient_norm_overflow")):
            parameter.grad = torch.full((2,), value)
            with self.assertRaises(NumericalFailure) as caught:
                clip_gradients([parameter], 1.)
            self.assertEqual(caught.exception.kind, kind)

    def test_bounded_process_timeout(self):
        """Run a short-lived timeout check against a sleeping child and verify the
        bounded process helper raises TimeoutExpired.
        """
        from transformer_training.subprocesses import run_bounded
        with self.assertRaises(subprocess.TimeoutExpired):
            run_bounded([sys.executable, "-c", "import time; time.sleep(30)"], timeout=.2, capture_output=True)

    def test_clocks_warmup_skips_and_all_skipped(self):
        """Use a fake clock to check warm-up exclusion, skipped-token accounting, zero
        useful throughput for all-skipped measured work, and separate
        lifecycle/training rates.
        """
        now = [0.0]
        m = Invocation(1, clock=lambda: now[0])
        m.record(10, 2, False)
        self.assertIsNone(m.snapshot()["warmed_useful_tokens_per_second"])
        m.record(20, 4, True)
        now[0] = 12
        result = m.snapshot()
        self.assertEqual(result["warmed_useful_tokens_per_second"], 0)
        self.assertEqual(result["warmed_processed_tokens_per_second"], 5)
        self.assertEqual(result["whole_run_useful_tokens_per_second"], 10 / 12)
        self.assertEqual(result["skipped_tokens"], 20)
        m.record(30, 2, False)
        self.assertEqual(m.snapshot()["warmed_useful_tokens_per_second"], 5)

    def test_config_targets_and_budget(self):
        """Check validation targets become sorted unique values and reject
        nonfinite/negative targets or nonpositive processed-token budgets.
        """
        self.assertEqual(TrainingConfig(validation_loss_targets=[2, 1, 2]).validation_loss_targets, (1, 2))
        for kwargs in ({"validation_loss_targets": [float("nan")]}, {"validation_loss_targets": [-1]},
                       {"supervised_token_budget": 0}):
            with self.assertRaises(ValueError):
                TrainingConfig(**kwargs)

    def test_sampler_without_device_and_missing_binary(self):
        """Verify missing GPU identity or nvidia-smi yields unavailable device-memory
        evidence without substituting a measured zero.
        """
        sampler = DeviceMemorySampler()
        self.assertEqual(sampler.start().stop()["status"], "unavailable")
        with patch("transformer_training.measurements.subprocess.Popen", side_effect=FileNotFoundError("missing")):
            self.assertEqual(DeviceMemorySampler("GPU-test").start().stop()["status"], "unavailable")


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        """Save the thread count and define a small CPU model and training
        configuration with quality targets for measurement integration tests.
        """
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.model = ModelConfig(vocab_size=12, sequence_length=4, layers=1,
                                 d_model=12, heads=3, d_ff=24, dropout=.2)
        self.config = TrainingConfig(device="cpu", batch_size=2, gradient_accumulation_steps=2,
            epochs=4, num_workers=0, log_every=100, eval_every=100, checkpoint_every=100,
            timing_warmup_steps=1, validation_loss_targets=(100, 0))

    def tearDown(self):
        """Restore the original PyTorch thread count after each measurement integration
        test.
        """
        torch.set_num_threads(self.old_threads)

    def trainer(self, path, config=None, resume=None):
        """Construct a synthetic trainer at the supplied path, optionally overriding
        the configuration or resuming a fixture checkpoint.
        """
        return Trainer(self.model, config or self.config, synthetic_loaders(), path, resume=resume)

    def test_quality_targets_budget_and_resumed_warmup(self):
        """Check token-budget overshoot at accumulation boundaries, quality-target
        status, fresh warm-up on resume, and separate invocation records/plots in
        training reports.
        """
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory)
            trainer = self.trainer(path / "run", replace(self.config, supervised_token_budget=1))
            result = trainer.fit()
            m = result["measurements"]
            self.assertEqual(m["attempts"], 1)
            self.assertIsNone(m["warmed_useful_tokens_per_second"])
            self.assertGreater(result["token_budget_overshoot"], 0)
            self.assertEqual(trainer.scheduler.total_updates, 12)
            self.assertEqual(m["quality_target_status"], {"0.0": "unreached", "100.0": "reached"})
            self.assertEqual(m["processed_tokens"], m["applied_tokens"])
            checkpoint = path / "run" / "latest.pt"
            resumed = self.trainer(checkpoint.parent, resume=checkpoint)
            resumed.fit(stop_after_updates=2)
            self.assertEqual(resumed.invocation.attempts, 1)
            self.assertIsNone(resumed.invocation.snapshot()["warmed_useful_tokens_per_second"])
            report = report_training(checkpoint.parent, path / "report")
            self.assertEqual(len(report["invocations"]), 2)
            self.assertEqual(len(report["plots"]), 2)
            validation = report["validation"][0]
            self.assertEqual(validation["training_processed_tokens"], result["state"]["supervised_tokens"])
            self.assertEqual(validation["validation_tokens"], validation["supervised_tokens"])
            self.assertLess(validation["invocation_lifecycle_seconds"], m["lifecycle_seconds"])

    def test_old_checkpoint_unknown_history_and_exact_continuation(self):
        """Remove newer measurement fields from a checkpoint and verify resumed
        parameters still match while missing applied-token and historical timing
        evidence stays explicitly unknown.
        """
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory)
            original = self.trainer(path / "original")
            original.fit(stop_after_updates=1)
            payload = load_checkpoint(path / "original" / "latest.pt")
            payload.pop("measurements")
            payload["state"].pop("applied_supervised_tokens")
            payload["training_config"].pop("validation_loss_targets")
            payload["training_config"].pop("supervised_token_budget")
            legacy = path / "legacy.pt"
            save_checkpoint(legacy, payload)
            normal = self.trainer(path / "normal", resume=path / "original" / "latest.pt")
            old = self.trainer(path / "old", resume=legacy)
            normal.fit(stop_after_updates=2)
            result = old.fit(stop_after_updates=2)
            for a, b in zip(normal.model.parameters(), old.model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertIsNone(old.state["applied_supervised_tokens"])
            self.assertFalse(result["measurements"]["historical_measurements_known"])

    def test_failure_record_and_no_invalid_checkpoint(self):
        """Inject nonfinite training loss and check failed measurement/numerical-
        failure records are written without saving an invalid latest checkpoint.
        """
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            trainer = self.trainer(Path(directory) / "run")
            with patch("transformer_training.training.accumulate_gradients", side_effect=FloatingPointError("Nonfinite training loss")):
                with self.assertRaises(FloatingPointError):
                    trainer.fit()
            rows = [json.loads(line) for line in (trainer.output_dir / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual([r for r in rows if r["event"] == "numerical_failure"][0]["kind"], "training_loss")
            self.assertEqual(rows[-1]["measurement"]["status"], "failed")
            self.assertFalse((trainer.output_dir / "latest.pt").exists())

    def test_lifecycle_includes_validation_and_checkpoint_with_fake_clock(self):
        """Use controlled update, validation, and checkpoint delays to verify lifecycle
        throughput includes all three while training-attempt throughput includes
        updates only.
        """
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            trainer = self.trainer(Path(directory) / "run", replace(self.config, max_steps=1, timing_warmup_steps=0))
            now = [0.0]
            trainer.invocation.clock = lambda: now[0]
            trainer.invocation.started = 0
            def update(*args, **kwargs):
                """Advance the fake clock by two seconds and return a successful
                synthetic attempt, isolating time accounting from actual
                optimisation.
                """
                now[0] += 2
                return {"loss_sum": 8., "supervised_tokens": 8, "slots": 16, "examples": 4,
                        "skipped": False, "learning_rate": .001, "scaler_before": 1.,
                        "scaler_after": 1., "nonfinite_gradients": False}
            def validate():
                """Advance the fake clock by seven seconds and return fixed validation
                metrics for the lifecycle timing test.
                """
                now[0] += 7
                return {"loss": 1, "supervised_tokens": 10}
            def checkpoint(name):
                """Advance the fake clock by three seconds instead of writing a
                checkpoint, providing a known serialisation cost for lifecycle
                accounting.
                """
                now[0] += 3
            with patch("transformer_training.training.time.perf_counter", side_effect=lambda: now[0]), \
                 patch("transformer_training.training.optimizer_attempt", side_effect=update), \
                 patch.object(trainer, "_validate", side_effect=validate), \
                 patch.object(trainer, "_checkpoint", side_effect=checkpoint):
                result = trainer.fit()["measurements"]
            self.assertEqual(result["lifecycle_seconds"], 12)
            self.assertEqual(result["training_seconds"], 2)
            self.assertEqual(result["whole_run_useful_tokens_per_second"], 8 / 12)
            self.assertEqual(result["warmed_useful_tokens_per_second"], 4)

    def test_multiple_targets_keep_first_crossing_with_nonmonotonic_loss(self):
        """Feed rising and falling validation losses and verify each quality target
        retains its first observed crossing.
        """
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config = replace(self.config, validation_loss_targets=(1., 1.6))
            trainer = self.trainer(Path(directory) / "run", config)
            with patch("transformer_training.training.evaluate", side_effect=[
                {"loss": loss, "supervised_tokens": 10} for loss in (3., 1.5, 2., .9)]), \
                patch.object(trainer, "_checkpoint"):
                for index in range(4):
                    trainer.state["updates"] = index
                    trainer._validate()
            self.assertEqual(trainer.target_observations["1.6"]["updates"], 1)
            self.assertEqual(trainer.target_observations["1.0"]["updates"], 3)

    def test_occupied_diagnostic_destination_is_not_changed(self):
        """Check diagnostic output validation rejects an occupied directory and
        preserves its existing file without adding artifacts.
        """
        from transformer_training.diagnostics_cli import main
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "keep.txt").write_text("preserve")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main("benchmark", ["--output-dir", str(path), "--device", "cpu"])
            self.assertEqual([p.name for p in path.iterdir()], ["keep.txt"])

    def test_benchmark_trials_repeat_and_reject_insufficient_data(self):
        """Compare deterministic synthetic trial losses, aggregation, and comparison
        identity; verify trials save no training checkpoint and reject insufficient
        remaining work.
        """
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory)
            a = run_trial(self.trainer(path / "a"), warmup=1, measured=3)
            b = run_trial(self.trainer(path / "b"), warmup=1, measured=3)
            self.assertEqual([r["loss_sum"] for r in a["measured_attempts"]], [r["loss_sum"] for r in b["measured_attempts"]])
            self.assertEqual(a["measurement"]["measured_attempts"], 3)
            report = aggregate_trials([a, b])
            self.assertEqual(report["throughput"]["count"], 2)
            self.assertEqual(compare_results(report, report)["speedup"], 1)
            changed = dict(report, comparison=dict(report["comparison"], input_mode="fixed"))
            self.assertEqual(compare_results(report, changed)["status"], "conditions_differ")
            self.assertFalse((path / "a" / "latest.pt").exists())
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                run_trial(self.trainer(path / "short", replace(self.config, epochs=1)), warmup=1, measured=10)

    def test_fixed_mode_is_separately_identified(self):
        """Check fixed-input trials label their input mode and reuse an accumulation
        group with a constant supervised-token count.
        """
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            result = run_trial(self.trainer(Path(directory) / "fixed"), warmup=1, measured=2, mode="fixed")
            self.assertEqual(result["input_mode"], "fixed")
            self.assertEqual(len({r["supervised_tokens"] for r in result["measured_attempts"]}), 1)

    def test_memory_profile_does_not_change_updates(self):
        """Compare profiled and unprofiled synthetic parameter updates, verify captured
        memory categories, and reject profiled timing from benchmark aggregation.
        """
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            plain = self.trainer(root / "plain")
            a = run_trial(plain, warmup=1, measured=2)
            memory = self.trainer(root / "memory")
            b = run_trial(memory, warmup=1, measured=2, backend="memory")
            for p, q in zip(plain.model.parameters(), memory.model.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            self.assertEqual(b["memory"]["tensor_peak"]["status"], "available")
            peak = b["memory"]["tensor_peak"]
            self.assertEqual(peak["bytes"], sum(peak["categories"].values()))
            self.assertGreater(b["memory"]["saved_for_backward"]["peak_saved_nonparameter_storage_bytes"], 0)
            self.assertFalse(b["timings_eligible_for_comparison"])
            with self.assertRaisesRegex(ValueError, "Profiled"):
                aggregate_trials([b])

    def test_new_dry_runs_do_not_load_data_or_initialize_cuda(self):
        """Run benchmark/profile dry-runs in fresh processes, checking no data imports,
        CUDA initialisation, or output-directory creation.
        """
        for kind in ("benchmark", "profile"):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "missing"
                code = (
                    "import sys, torch; from transformer_training.diagnostics_cli import main; "
                    f"main({kind!r}, ['--dry-run', '--output-dir', {str(output)!r}]); "
                    "assert 'datasets' not in sys.modules; assert 'transformer_training.data' not in sys.modules; "
                    "assert not torch.cuda.is_initialized()")
                result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(output.exists())

    def test_fresh_process_workers_use_identical_synthetic_workload(self):
        """Launch two bounded workers with a local loader stub and verify distinct
        process IDs but matching workload identities and synthetic loss sequences.
        """
        from transformer_training.diagnostics_cli import launch_worker
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Child-only loader stub; no Hub requests, corpus preprocessing, or real-data updates.
            (root / "sitecustomize.py").write_text(
                "from transformer_training import data\n"
                "from test_training import synthetic_loaders\n"
                "data.load_tinystories = lambda **kwargs: synthetic_loaders()\n")
            request = {"model": asdict(self.model), "training": asdict(self.config),
                "data_options": {}, "checkpoint": None, "offline": True,
                "warmup": 1, "measured": 2, "input_mode": "loader", "backend": "benchmark"}
            pythonpath = os.pathsep.join([str(root), str(Path(__file__).parent.resolve()), os.environ.get("PYTHONPATH", "")])
            with patch.dict(os.environ, {"PYTHONPATH": pythonpath}):
                a, _ = launch_worker(request, root / "a", timeout=60)
                b, _ = launch_worker(request, root / "b", timeout=60)
            self.assertNotEqual(a["process_id"], b["process_id"])
            self.assertEqual([r["loss_sum"] for r in a["measured_attempts"]], [r["loss_sum"] for r in b["measured_attempts"]])
            self.assertEqual(a["comparison"], b["comparison"])


class MemoryTests(unittest.TestCase):
    def test_unindexed_cuda_device_is_resolved_without_allocating(self):
        """Mock the active CUDA index and verify the memory adapter resolves cuda to
        that explicit device without allocating tensors.
        """
        from transformer_training.memory_profile import MemoryAdapter
        with patch("torch.cuda.current_device", return_value=2):
            adapter = MemoryAdapter("cuda")
        self.assertEqual(adapter.device, torch.device("cuda:2"))

    def test_shared_storage_and_lazy_optimizer_state(self):
        """Check alias deduplication, initially absent AdamW state, saved-activation
        release after backward, and materialised optimiser storage after a step.
        """
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters())
        self.assertEqual(unique_storage_bytes([model.weight, model.weight.view(-1)]), model.weight.untyped_storage().nbytes())
        self.assertEqual(static_memory(model, optimizer)["optimizer_state_bytes"], 0)
        tracker = SavedActivations(model)
        with tracker:
            loss = model(torch.ones(2, 4)).square().sum()
            loss.backward()
            del loss
        self.assertEqual(tracker.live, {})
        optimizer.step()
        self.assertGreater(static_memory(model, optimizer)["optimizer_state_bytes"], 0)

    def test_saved_graph_releases_without_backward(self):
        """Verify saved-tensor ownership remains while a graph is alive and disappears
        when its output is discarded without backward.
        """
        model = torch.nn.Linear(4, 4)
        tracker = SavedActivations(model)
        with tracker:
            output = model(torch.ones(2, 4)).square()
        self.assertTrue(tracker.live)
        del output
        self.assertEqual(tracker.live, {})

    def test_allocator_replay_keeps_pending_frees(self):
        """Replay synthetic allocation events and verify a requested free remains live
        until free completion when reconstructing simultaneous peak memory.
        """
        initial = {"segments": [{"device": 0, "address": 100, "blocks": [
            {"state": "active_allocated", "size": 20, "frames": ["parameter"]}]}]}
        final = {"device_traces": [[{"action": "alloc", "addr": 200, "size": 30, "frames": ["activation"]},
            {"action": "free_requested", "addr": 200},
            {"action": "alloc", "addr": 300, "size": 10},
            {"action": "free_completed", "addr": 200}]]}
        peak = replay_allocator_peak(initial, final, 0)
        self.assertEqual(peak["active_bytes"], 60)
        self.assertEqual(len(peak["allocations"]), 3)


class TraceTests(unittest.TestCase):
    def test_interval_operations(self):
        """Check overlap/touch merging, intersection, and busy-interval subtraction
        against manually computed interval results.
        """
        self.assertEqual(union([(0, 4), (2, 6), (6, 8)]), [(0, 8)])
        self.assertEqual(intersection([(0, 5)], [(3, 10)]), [(3, 5)])
        self.assertEqual(subtract([(0, 10)], [(2, 4), (3, 5), (7, 8)]), [(0, 2), (5, 7), (8, 10)])

    def test_overlap_correlation_and_gaps(self):
        """Analyse synthetic overlapping kernels/copies and correlated host launches,
        checking busy/idle durations, exposed transfers, phase attribution, and
        missing-trace handling.
        """
        def event(name, cat, start, length, **args):
            """Build a complete Chrome-style event with microsecond start/duration and
            optional correlation arguments for a synthetic timeline.
            """
            return {"name": name, "cat": cat, "ph": "X", "ts": start, "dur": length, "tid": 1, "args": args}
        events = [event("training/update", "user_annotation", 0, 100),
            event("training/data_acquisition", "user_annotation", 0, 10),
            event("training/forward_loss", "user_annotation", 10, 20),
            event("cudaLaunchKernel", "cuda_runtime", 11, 1, correlation=7),
            event("a", "kernel", 20, 40, correlation=7), event("b", "kernel", 40, 30),
            event("copy", "gpu_memcpy", 10, 20)]
        result = analyse_trace(events)
        self.assertEqual(result["kernel_busy_us"], 50)
        self.assertEqual(result["gpu_busy_us"], 60)
        self.assertEqual(result["copy_compute_overlap_us"], 10)
        self.assertEqual(result["exposed_copy_us"], 10)
        self.assertEqual(result["gpu_idle_us"], 40)
        self.assertEqual(result["correlated_gpu_phase_us"]["training/forward_loss"], 40)
        self.assertEqual(result["idle_gaps"][-1]["attribution"], "unclassified")
        self.assertEqual(analyse_trace([])["status"], "unavailable")

    def test_nsys_export_schema(self):
        """Build a local SQLite fixture and verify Nsight table/string conversion
        produces correctly named kernels and microsecond durations.
        """
        import sqlite3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            connection = sqlite3.connect(path)
            connection.executescript('''CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,globalTid INTEGER);
                INSERT INTO NVTX_EVENTS VALUES(0,100000,'training/update',1);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER,streamId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(10000,50000,1,7);
                CREATE TABLE StringIds(id INTEGER,value TEXT);
                INSERT INTO StringIds VALUES(1,'matrix_multiply');''')
            connection.close()
            result = analyse_trace(load_nsys_sqlite(path))
            self.assertEqual(result["kernel_busy_us"], 40)
            self.assertEqual(result["kernels"][0]["name"], "matrix_multiply")


class NsightTests(unittest.TestCase):
    def test_csv_units_bandwidth_and_flops(self):
        """Parse synthetic counter CSV and verify unit conversion, bandwidth, and
        scalar FLOP rates while rejecting malformed reports and unknown units.
        """
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
        metrics = [(METRICS[0], "usecond", "2"), (METRICS[1], "Kbyte", "1"),
                   (METRICS[2], "byte", "1000"), (METRICS[5], "inst", "20"),
                   (METRICS[6], "inst", "5"), (METRICS[7], "inst", "5")]
        for metric, unit, value in metrics:
            writer.writerow([0, "gemm", metric, unit, value])
        kernel = parse_ncu_csv(stream.getvalue())["kernels"][0]
        self.assertAlmostEqual(kernel["bandwidth_bytes_per_second"], 1e9)
        self.assertAlmostEqual(kernel["fp32_scalar_flops_per_second"], 25e6)
        self.assertEqual(kernel["evidence"]["classification"], "inconclusive")
        with self.assertRaises(ValueError):
            parse_ncu_csv("not a report")
        with self.assertRaises(ValueError):
            normalize("2", "unknown")

    def test_missing_tool_permission_and_failed_process(self):
        """Mock absent tools and denied hardware-counter access, checking each returns
        its distinct unavailable status.
        """
        with patch("transformer_training.nsight.shutil.which", return_value=None):
            self.assertEqual(run_tool(["ncu"])["status"], "missing_tool")
        with patch("transformer_training.nsight.shutil.which", return_value="ncu"), \
             patch("transformer_training.nsight.run_bounded", return_value=subprocess.CompletedProcess([], 1, "", "ERR_NVGPUCTRPERM")):
            self.assertEqual(run_tool(["ncu"])["status"], "permission_denied")

    def test_counter_capture_fresh_output_and_missing_metrics(self):
        """Mock counter discovery and failed captures, checking per-kernel workload
        requests use different output paths and return partial evidence.
        """
        from transformer_training.nsight import capture_compute
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "request.json"
            request.write_text(json.dumps({"output": "unused"}))
            query = {"status": "available", "stdout": "\n".join(METRICS), "stderr": ""}
            sections = {"status": "available", "stdout": "", "stderr": ""}
            failure = {"status": "permission_denied", "value": None, "reason": "ERR_NVGPUCTRPERM"}
            with patch("transformer_training.nsight.run_tool", side_effect=[query, sections, failure, failure]):
                result = capture_compute(["python", "--worker", str(request)], root, [{"name": "a"}, {"name": "b"}])
            self.assertEqual(result["status"], "partial")
            outputs = [json.loads((root / f"request-{i}.json").read_text())["output"] for i in (0, 1)]
            self.assertNotEqual(*outputs)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; hardware measurements not verified")
class CudaMeasurementTests(unittest.TestCase):
    def test_cuda_benchmark_and_memory_capture(self):
        """When CUDA is available, run a short synthetic memory capture and verify
        nonzero allocated memory and available tensor-peak attribution.
        """
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config = TrainingConfig(device="cuda", batch_size=2, gradient_accumulation_steps=2, epochs=2, num_workers=0)
            model = ModelConfig(vocab_size=12, sequence_length=4, layers=1, d_model=12, heads=3, d_ff=24)
            trainer = Trainer(model, config, synthetic_loaders(), Path(directory) / "cuda")
            result = run_trial(trainer, warmup=1, measured=2, backend="memory")
            self.assertGreater(result["peak_allocated_bytes"], 0)
            self.assertEqual(result["memory"]["tensor_peak"]["status"], "available")


if __name__ == "__main__":
    unittest.main()
