"""Synthetic DDP correctness, accounting and CLI checks; no TinyStories access."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from datetime import timedelta
import copy
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from transformer_training.config import ImplementationConfig, ModelConfig, TrainingConfig
from transformer_training.data import NextTokenCollator, TinyStoriesLoaders
from transformer_training.distributed import (DistributedContext, ScalingBatchSampler,
    scaling_loader, workload_fingerprint)
from transformer_training.gpu_scaling import (build_parser, build_report, launch_trial, main,
    resolve_request, run_rank, schedule_horizon, summarise_trial, worker_command, write_report)
from transformer_training.loss import model_loss_backend, validate_loss_backend
from transformer_training.measurements import source_identity
from transformer_training.model import DecoderOnlyTransformer
from transformer_training.subprocesses import run_bounded
from transformer_training.training import (accumulate_gradients, data_signature, make_optimizer,
    optimizer_attempt, seed_everything, WarmupCosine)


def fixture_loaders():
    """Return nine unequal-length EOS windows, including a one-row final update."""
    examples = [{'tokens': row} for row in (
        [2, 3, 4, 1], [3, 1], [4, 5, 1], [6, 7, 8, 1], [8, 1],
        [9, 10, 1], [2, 1], [3, 4, 1], [5, 6, 7, 1])]
    loader = DataLoader(examples, batch_size=2, collate_fn=NextTokenCollator(4, 1),
                        generator=torch.Generator().manual_seed(123))
    return TinyStoriesLoaders(loader, loader, {'sequence_length': 4, 'vocab_size': 12,
        'seed': 42, 'splits': {'train': {'windows_fingerprint': 'fixture-v1'},
                             'validation': {'windows_fingerprint': 'fixture-v1'}}})


def fixture_config(precision='fp32'):
    """Keep global batch eight and two epochs to exercise four updates and tails."""
    return TrainingConfig(device='cpu' if precision == 'fp32' else 'cuda',
                          batch_size=2, gradient_accumulation_steps=4, epochs=2,
                          num_workers=0, precision=precision)


def fixture_model(implementation=None):
    """Create a tiny dropout-free reference whose gradients can be compared directly."""
    return DecoderOnlyTransformer(ModelConfig(vocab_size=12, sequence_length=4, layers=1,
        d_model=12, heads=3, d_ff=24, dropout=0), implementation)


def update_records(device, precision='fp32', loss_backend='pytorch', distributed=False):
    """Record unclipped gradients and multiple AdamW updates on identical global groups."""
    seed_everything(42)
    config = fixture_config(precision)
    context = DistributedContext(device) if distributed else None
    raw = fixture_model(ImplementationConfig(loss_backend=loss_backend)).to(device)
    model = (DistributedDataParallel(raw, device_ids=[device.index] if device.type == 'cuda'
                                    else None, broadcast_buffers=False)
             if distributed else raw)
    optimizer = make_optimizer(raw, config)
    scheduler = WarmupCosine(optimizer, config, 4)
    scaler = torch.amp.GradScaler('cuda', enabled=precision == 'fp16')
    results = []
    # A local context supplies the same batch protocol without process-group collectives.
    loader_context = context if context is not None else type('Local', (), {
        'rank': 0, 'world_size': 1, 'device': device})()
    for epoch in range(2):
        loader, sizes = scaling_loader(fixture_loaders().train, config, epoch, loader_context)
        iterator = iter(loader)
        for size in sizes:
            batches = [next(iterator) for _ in range(size)]
            optimizer.zero_grad(set_to_none=True)
            # FP32 gradient checks are independent of AMP scale and clipping.
            if precision == 'fp32':
                accumulate_gradients(model, batches, device=device, distributed=context)
                gradients = [p.grad.detach().cpu().clone() for p in raw.parameters()]
            else:
                gradients = []
            result = optimizer_attempt(model, optimizer, scaler, scheduler, batches,
                                       config, device, distributed=context)
            results.append({'result': result, 'gradients': gradients,
                            'weights': {k: v.detach().cpu().clone() for k, v in raw.state_dict().items()},
                            'scheduler': scheduler.state_dict(), 'scale': scaler.get_scale()})
    if distributed:
        # DDP configuration attributes must survive wrapping even without Triton execution.
        raw.implementation = replace(raw.implementation, loss_backend='triton')
        if model_loss_backend(model) != 'triton':
            raise AssertionError('DDP wrapper lost explicitly selected Triton loss')
        raw.implementation = replace(raw.implementation, loss_backend=loss_backend)
    return results, (raw, model, optimizer, scaler, scheduler, config, batches, context)


def distributed_fixture(rank, directory, backend, precision, loss_backend):
    """Run a bounded subprocess rank and save evidence for the parent unittest."""
    torch.set_num_threads(1)
    device = torch.device(f'cuda:{rank}' if backend == 'nccl' else 'cpu')
    if backend == 'nccl':
        torch.cuda.set_device(device)
    dist.init_process_group(backend, rank=rank, world_size=2,
        init_method=(Path(directory) / 'rendezvous').resolve().as_uri(), timeout=timedelta(seconds=60))
    try:
        records, objects = update_records(device, precision, loss_backend, distributed=True)
        raw, model, optimizer, scaler, scheduler, config, batches, context = objects
        overflow = None
        if precision == 'fp16':
            def force_overflow(gradient):
                """Inject one rank's overflow; DDP must deliver it to every replica."""
                return torch.full_like(gradient, float('inf')) if rank == 0 else gradient
            hook = next(raw.parameters()).register_hook(force_overflow)
            before = {k: v.detach().clone() for k, v in raw.state_dict().items()}
            completed = scheduler.completed_updates
            overflow = optimizer_attempt(model, optimizer, scaler, scheduler, batches,
                                         config, device, distributed=context)
            hook.remove()
            assert overflow['skipped'] and scheduler.completed_updates == completed
            assert all(torch.equal(v, before[k]) for k, v in raw.state_dict().items())
        torch.save({'records': records, 'overflow': overflow}, Path(directory) / f'rank-{rank}.pt')
        # Exercise the full benchmark runner, including untimed validation and report gathering.
        if backend == 'gloo':
            request = resolve_request(build_parser().parse_args([
                '--batch-size', '2', '--gradient-accumulation-steps', '4', '--epochs', '2',
                '--num-workers', '0', '--warmup-attempts', '1', '--measured-attempts', '3',
                '--validation-batches', '2', '--repeats', '1']))
            request['model'] = asdict(fixture_model().config)
            request['training'] = asdict(fixture_config())
            request['prepared_data'] = data_signature(fixture_loaders().metadata)
            request['output'] = str(Path(directory) / 'runner')
            trial = run_rank(request, fixture_loaders(), device, env={'source': 'synthetic'})
            if rank == 0:
                assert trial['world_size'] == 2 and trial['validation']['actual_batches'] == 2
    finally:
        dist.destroy_process_group()


def trial_fixture(world=1, repeat=1):
    """Create compact rank evidence for report eligibility and clock accounting tests."""
    record = {'seconds': 2., 'loss_sum': 12., 'supervised_tokens': 6, 'examples': 3,
              'slots': 12, 'skipped': False, 'scaler_after': 1., 'learning_rate': .01,
              'nonfinite_gradients': False}
    ranks = [{'rank': rank, 'identity': {'workload_fingerprint': 'same',
              'environment': {'source': {'content_sha256': 'same-source'}}},
              'work': [{'examples': 3, 'supervised_tokens': 6, 'slots': 12}],
              'warmup_records': [], 'records': [dict(record, seconds=2. + rank)],
              'measured_seconds': 2. + rank, 'peak_allocated_bytes': 100 + rank,
              'peak_reserved_bytes': 200 + rank, 'timings_eligible': True,
              'environment': {'gpu': 'test', 'gpu_memory_bytes': 1000,
                              'compute_capability': [8, 0], 'gpu_uuid': f'GPU-{rank}'},
              'validation': {'status': 'complete', 'loss': 2.}} for rank in range(world)]
    return dict(summarise_trial(ranks), repeat=repeat)


class ScalingAccountingTests(unittest.TestCase):
    def test_source_identity_excludes_only_generated_outputs(self):
        """Changing a custom log leaves source identity stable; changing source does not."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(['git', 'init', '--quiet', str(root)], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Test',
                            '-c', 'user.email=test@example.invalid', 'commit', '--allow-empty',
                            '--quiet', '-m', 'Fixture'], check=True, capture_output=True)
            (root / 'source.py').write_text('original')
            output = root / 'custom outputs'
            output.mkdir()
            log = output / 'worker.log'
            log.write_text('first')
            # Point the helper at an isolated Git root without touching this checkout.
            with patch('transformer_training.measurements.__file__',
                       str(root / 'src' / 'transformer_training' / 'measurements.py')):
                before = source_identity(exclude_paths=[output])
                log.write_text('changed log')
                self.assertEqual(before, source_identity(exclude_paths=[output]))
                self.assertNotEqual(before, source_identity())
                (root / 'source.py').write_text('changed source')
                self.assertNotEqual(before, source_identity(exclude_paths=[output]))

    def test_global_groups_keep_all_real_indices_and_mark_only_dummies(self):
        """Cover full groups, partial batches, and a rank with no real tail work."""
        for size in (1, 3, 6, 9, 17):
            baseline = ScalingBatchSampler(size, 2, 4, 42, 0)
            split = [ScalingBatchSampler(size, 2, 4, 42, 0, rank, 2) for rank in (0, 1)]
            self.assertEqual(split[0].group_sizes(), split[1].group_sizes())
            batches = [list(sampler) for sampler in split]
            real = [row for rank in batches for batch in rank for row, dummy in batch if not dummy]
            self.assertEqual(sorted(real), list(range(size)))
            base_groups = list(baseline.groups())
            offset = 0
            for group, local_size in zip(base_groups, split[0].group_sizes()):
                reunited = [row for step in range(local_size) for rank in range(2)
                            for row, dummy in batches[rank][offset + step] if not dummy]
                self.assertEqual(group, reunited)
                offset += local_size
        sampler = ScalingBatchSampler(1, 2, 4, 42, 0, 1, 2)
        self.assertTrue(all(dummy for batch in sampler for _, dummy in batch))

    def test_schedule_workload_identity_and_invalid_geometry(self):
        """Global schedule counts tails, caps max_steps, and rejects insufficient data."""
        config = fixture_config()
        self.assertEqual(schedule_horizon(9, config, 4), 4)
        self.assertEqual(schedule_horizon(9, replace(config, max_steps=3), 3), 3)
        self.assertEqual(workload_fingerprint(9, config, 4), workload_fingerprint(9, config, 4))
        self.assertNotEqual(workload_fingerprint(9, config, 4),
                            workload_fingerprint(9, replace(config, seed=43), 4))
        for function in (lambda: schedule_horizon(9, config, 5),
                         lambda: workload_fingerprint(9, config, 5),
                         lambda: ScalingBatchSampler(9, 2, 3, 42, 0, 0, 2)):
            with self.assertRaises(ValueError):
                function()

    def test_global_throughput_uses_slowest_rank_and_counts_tokens_once(self):
        """Unequal rank durations use maximum window and maximum per-update latency."""
        trial = trial_fixture(2)
        self.assertEqual(trial['useful_tokens_per_second'], 6 / 3)
        self.assertEqual(trial['processed_tokens'], 6)
        self.assertEqual(trial['latency_seconds']['median'], 3)
        self.assertEqual(trial['memory']['sum_allocated_bytes'], 201)
        ranks = copy.deepcopy(trial['rank_results'])
        ranks[1]['records'][0]['scaler_after'] = 2.
        with self.assertRaisesRegex(ValueError, 'scaler'):
            summarise_trial(ranks)

    def test_report_speedup_and_ineligibility(self):
        """Eligible reports compute ratios; changed work, hardware and failures suppress them."""
        request = resolve_request(build_parser().parse_args(['--repeats', '1']))
        one, two = trial_fixture(), trial_fixture(2)
        two['useful_tokens_per_second'] = 5.
        report = build_report(request, [one, two])
        self.assertAlmostEqual(report['comparison']['speedup'], 5 / 3)
        self.assertAlmostEqual(report['comparison']['two_gpu_efficiency'], 5 / 6)
        self.assertEqual(report['groups']['2']['memory_per_gpu'][1]['allocated']['median'], 101)
        for mutation in ('work', 'conditions', 'compile', 'hardware', 'skips', 'device', 'source'):
            candidate = copy.deepcopy(two)
            if mutation == 'work':
                candidate['work'][0]['examples'] += 1
            elif mutation == 'conditions':
                candidate['identity']['changed'] = True
            elif mutation == 'compile':
                candidate['timings_eligible'] = False
            elif mutation == 'hardware':
                candidate['rank_results'][1]['environment']['gpu'] = 'different'
            elif mutation == 'skips':
                candidate['skipped_updates'] = 1
            elif mutation == 'device':
                candidate['rank_results'][0]['environment']['gpu_uuid'] = 'different'
            else:
                candidate['identity']['environment']['source'] = {'status': 'unavailable'}
            self.assertIsNone(build_report(request, [one, candidate])['comparison']['speedup'])
        self.assertIsNone(build_report(request, [one], status='failed')['comparison']['speedup'])
        with tempfile.TemporaryDirectory() as directory:
            write_report(report, Path(directory))
            self.assertEqual(set(p.name for p in Path(directory).iterdir()),
                             {'report.json', 'report.md', 'report.csv'})

    def test_dry_run_has_no_data_cuda_process_or_output_side_effects(self):
        """Resolve Triton/compile settings without importing the backend or creating outputs."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'absent'
            with patch('transformer_training.gpu_scaling.prepare_data', side_effect=AssertionError), \
                 patch('torch.cuda.is_available', side_effect=AssertionError), \
                 patch('torch.cuda.device_count', side_effect=AssertionError), \
                 patch('subprocess.Popen', side_effect=AssertionError), \
                 patch('transformer_training.gpu_scaling.preflight', side_effect=AssertionError), \
                 patch('transformer_training.model.DecoderOnlyTransformer.__init__', side_effect=AssertionError), \
                 redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(main(['--dry-run', '--loss-backend', 'triton', '--compile',
                                       '--output-dir', str(output)]), 0)
            result = json.loads(stream.getvalue())
            self.assertEqual(result['geometry'][1]['accumulation_per_gpu'], 4)
            self.assertEqual(result['geometry'][1]['global_batch_size'], 64)
            self.assertFalse(output.exists())

    def test_cli_rejects_invalid_counts_and_occupied_output(self):
        """Fail before hardware/data work for invalid geometry or occupied destinations."""
        for args in (['--gradient-accumulation-steps', '3'], ['--gradient-accumulation-steps', '1'],
                     ['--warmup-attempts', '-1'], ['--repeats', '0'], ['--validation-batches', '-1'],
                     ['--max-steps', '2'], ['--revision', 'main']):
            with self.assertRaises(ValueError):
                resolve_request(build_parser().parse_args(args))
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'keep.txt'
            marker.write_text('original')
            with patch('transformer_training.gpu_scaling.preflight', side_effect=AssertionError), \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--output-dir', directory])
            self.assertEqual(marker.read_text(), 'original')
            self.assertFalse((Path(directory) / 'report.json').exists())
        command = worker_command(2, 'request.json', 'logs')
        self.assertIn('--nproc-per-node=2', command)
        self.assertIn('--max-restarts=0', command)

    def test_launcher_records_timeout_and_worker_failure(self):
        """Failed/timeout child jobs retain status and cannot become successful trials."""
        request = resolve_request(build_parser().parse_args([]))
        for error in (subprocess.TimeoutExpired('torchrun', 1),
                      subprocess.CompletedProcess(['torchrun'], 3)):
            with tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / 'trial'
                kwargs = {'side_effect': error} if isinstance(error, Exception) else {'return_value': error}
                with patch('transformer_training.gpu_scaling.run_bounded', **kwargs), \
                     self.assertRaises(RuntimeError):
                    launch_trial(request, 2, destination)
                status = json.loads((destination / 'status.json').read_text())
                self.assertIn(status['status'], {'failed', 'timeout'})
                self.assertTrue((destination / 'worker.log').exists())

    def test_validation_time_and_selection_are_outside_the_measured_window(self):
        """An artificially slow evaluator changes validation time, never training timing."""
        request = resolve_request(build_parser().parse_args([
            '--batch-size', '2', '--gradient-accumulation-steps', '4', '--epochs', '2',
            '--num-workers', '0', '--warmup-attempts', '1', '--measured-attempts', '1',
            '--validation-batches', '2']))
        request['model'] = asdict(fixture_model().config)
        request['training'] = asdict(fixture_config())
        request['prepared_data'] = data_signature(fixture_loaders().metadata)
        now = [0.]

        def clock():
            """Advance one unit per observation, independent of actual execution time."""
            now[0] += 1
            return now[0]

        def slow_evaluate(model, loader, **kwargs):
            """Add a large validation-only delay while inspecting the actual subset."""
            now[0] += 1000
            batches = list(loader)
            self.assertEqual(len(batches), 2)
            return {'loss': 2., 'supervised_tokens': 9}

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                request['output'] = directory
                with patch('transformer_training.gpu_scaling.evaluate', side_effect=slow_evaluate), \
                     patch('transformer_training.gpu_scaling.time.perf_counter', side_effect=clock):
                    trial = run_rank(request, fixture_loaders(), torch.device('cpu'),
                                     env={'source': {'content_sha256': 'synthetic'}})
                self.assertLess(trial['measured_seconds'], 1000)
                self.assertGreater(trial['validation']['seconds'], 1000)
                self.assertEqual(trial['validation']['actual_batches'], 2)
                self.assertIsNone(trial['memory']['sum_reserved_bytes'])
        finally:
            torch.set_num_threads(old_threads)
        skipped = resolve_request(build_parser().parse_args(['--validation-batches', '0']))
        full = resolve_request(build_parser().parse_args(['--full-validation']))
        self.assertEqual(skipped['protocol']['validation_batches'], 0)
        self.assertIsNone(full['protocol']['validation_batches'])


class DistributedCorrectnessTests(unittest.TestCase):
    def run_distributed(self, backend='gloo', precision='fp32', loss_backend='pytorch'):
        """Use a bounded process tree so a collective regression cannot hang the suite."""
        with tempfile.TemporaryDirectory() as directory:
            process = run_bounded([sys.executable, str(Path(__file__).resolve()), '--fixture',
                directory, backend, precision, loss_backend], timeout=150, capture_output=True)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            ranks = [torch.load(Path(directory) / f'rank-{rank}.pt', weights_only=True)
                     for rank in (0, 1)]
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                reference, _ = update_records(torch.device('cpu'))
            finally:
                torch.set_num_threads(previous_threads)
            for rank in ranks:
                for actual, expected in zip(rank['records'], reference):
                    self.assertEqual(actual['result']['examples'], expected['result']['examples'])
                    self.assertEqual(actual['result']['supervised_tokens'], expected['result']['supervised_tokens'])
                    self.assertEqual(actual['result']['slots'], expected['result']['slots'])
                    if precision == 'fp32':
                        for a, b in zip(actual['gradients'], expected['gradients']):
                            torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-6)
                        for key, value in actual['weights'].items():
                            torch.testing.assert_close(value, expected['weights'][key], rtol=3e-4, atol=3e-6)
                    self.assertEqual(actual['scheduler']['total_updates'], 4)
                self.assertEqual(rank['records'][-1]['result']['examples'], 1)
            for a, b in zip(ranks[0]['records'], ranks[1]['records']):
                self.assertEqual(a['result'], b['result'])
                for key in a['weights']:
                    torch.testing.assert_close(a['weights'][key], b['weights'][key], rtol=0, atol=0)
            if backend == 'gloo':
                trial = json.loads((Path(directory) / 'runner' / 'trial.json').read_text())
                self.assertEqual(trial['validation']['actual_batches'], 2)
                self.assertEqual(trial['validation']['supervised_tokens'], 9)
                self.assertEqual(len(trial['rank_results'][0]['records']), 3)
                self.assertTrue(trial['timings_eligible'])
                self.assertIsNone(trial['memory']['max_allocated_bytes'])
            if precision == 'fp16':
                self.assertTrue(all(rank['overflow']['skipped'] for rank in ranks))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo unavailable')
    def test_two_rank_gloo_gradients_updates_and_benchmark_runner(self):
        """Verify token weighting, EOS, padding and dummy-only tails against a CPU reference."""
        self.run_distributed()

    @unittest.skipUnless(platform.system() == 'Linux' and torch.cuda.device_count() >= 2
                         and dist.is_nccl_available(), 'Two Linux CUDA GPUs and NCCL required')
    def test_two_gpu_nccl_fp32(self):
        """Run the same gradient/update equivalence check on two real CUDA devices."""
        self.run_distributed('nccl')

    @unittest.skipUnless(platform.system() == 'Linux' and torch.cuda.device_count() >= 2
                         and dist.is_nccl_available(), 'Two Linux CUDA GPUs and NCCL required')
    def test_two_gpu_nccl_mixed_precision_and_overflow(self):
        """Check rank agreement for FP16 overflow and natively supported BF16 updates."""
        self.run_distributed('nccl', 'fp16')
        if all(torch.cuda.get_device_capability(index) >= (8, 0) for index in (0, 1)):
            self.run_distributed('nccl', 'bf16')

    @unittest.skipUnless(platform.system() == 'Linux' and torch.cuda.device_count() >= 2
                         and dist.is_nccl_available(), 'Two Linux CUDA GPUs and NCCL required')
    def test_two_gpu_triton_loss(self):
        """Execute the selected Triton loss through DDP when both devices support it."""
        try:
            for index in (0, 1):
                validate_loss_backend('triton', f'cuda:{index}', 12)
        except ValueError as error:
            self.skipTest(str(error))
        self.run_distributed('nccl', loss_backend='triton')


if __name__ == '__main__':
    if len(sys.argv) == 6 and sys.argv[1] == '--fixture':
        mp.spawn(distributed_fixture, args=tuple(sys.argv[2:]), nprocs=2, join=True)
    else:
        unittest.main()
