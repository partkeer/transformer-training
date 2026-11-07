"""Fresh-process fixed-global-batch comparisons of one GPU and two-GPU DDP."""

import argparse
from dataclasses import asdict, fields, replace
from datetime import timedelta
import csv
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import atomic_json, atomic_text
from .config import ImplementationConfig, ModelConfig, TrainingConfig
from .data_identity import DATASET_REVISION, TOKENIZER_REVISION
from .diagnostics import distribution, field_differences
from .distributed import DistributedContext, scaling_loader, workload_fingerprint
from .evaluation import evaluate
from .implementations import (add_implementation_arguments, compiler_snapshot,
                              compilation_changed, execution_report, resolve_implementation)
from .loss import validate_loss_backend
from .measurements import digest
from .model import DecoderOnlyTransformer
from .subprocesses import run_bounded
from .training import (WarmupCosine, data_signature, environment, make_optimizer,
                       optimizer_attempt, seed_everything, validate_device,
                       validate_output_directory)


def build_parser():
    """Expose relevant training options plus the bounded GPU-scaling protocol."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    omitted = {'vocab_size', 'device', 'eval_every', 'checkpoint_every', 'log_every',
               'timing_warmup_steps', 'supervised_token_budget', 'validation_loss_targets'}
    for cls in (ModelConfig, TrainingConfig):
        for field in fields(cls):
            if field.name in omitted:
                continue
            default = getattr(cls(), field.name)
            parser.add_argument('--' + field.name.replace('_', '-'), default=default,
                                type=int if default is None else type(default))
    add_implementation_arguments(parser)
    parser.add_argument('--revision', default=DATASET_REVISION)
    parser.add_argument('--tokenizer-revision', default=TOKENIZER_REVISION)
    parser.add_argument('--max-stories', type=int)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup-attempts', type=int, default=10)
    parser.add_argument('--measured-attempts', type=int, default=50)
    validation = parser.add_mutually_exclusive_group()
    validation.add_argument('--validation-batches', type=int, default=32,
                            help='Untimed validation batches; zero skips the sanity check')
    validation.add_argument('--full-validation', action='store_true')
    parser.add_argument('--timeout-seconds', type=int, default=1800,
                        help='Time limit for each entire worker job, including validation')
    parser.add_argument('--torch-threads', type=int, default=1,
                        help='CPU compute threads per rank, equal in both configurations')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    return parser


def resolve_request(args):
    """Validate configuration without device/data access or filesystem mutations."""
    model = ModelConfig(**{f.name: getattr(args, f.name) for f in fields(ModelConfig)
                           if f.name != 'vocab_size'})
    training = TrainingConfig(**{f.name: getattr(args, f.name) for f in fields(TrainingConfig)
                                 if hasattr(args, f.name)})
    accumulation = training.gradient_accumulation_steps
    if accumulation < 2 or accumulation % 2:
        raise ValueError('One-GPU gradient accumulation must be even and at least 2')
    for name in ('repeats', 'measured_attempts', 'timeout_seconds', 'torch_threads'):
        if getattr(args, name) < 1:
            raise ValueError(name + ' must be positive')
    if args.warmup_attempts < 0 or args.validation_batches < 0:
        raise ValueError('Warm-up and validation batch counts must be nonnegative')
    if args.max_stories is not None and args.max_stories < 1:
        raise ValueError('max_stories must be positive')
    for revision in (args.revision, args.tokenizer_revision):
        if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
            raise ValueError('Dataset/tokenizer revisions must be pinned 40-character commit IDs')
    if training.max_steps is not None and training.max_steps < args.warmup_attempts + args.measured_attempts:
        raise ValueError('max_steps is smaller than warm-up plus measured attempts')
    return {'version': 1, 'model': asdict(model), 'training': asdict(training),
            'implementation': asdict(resolve_implementation(args)), 'offline': args.offline,
            'data_options': {'revision': args.revision, 'tokenizer_revision': args.tokenizer_revision,
                'max_stories': args.max_stories, 'sequence_length': model.sequence_length,
                'batch_size': training.batch_size, 'seed': training.seed,
                'num_workers': training.num_workers, 'pin_memory': True,
                'exclude_validation_overlap': True},
            'protocol': {'repeats': args.repeats, 'warmup': args.warmup_attempts,
                'measured': args.measured_attempts,
                'validation_batches': None if args.full_validation else args.validation_batches,
                'timeout_seconds': args.timeout_seconds, 'torch_threads': args.torch_threads},
            'geometry': [{'world_size': world, 'batch_size_per_gpu': training.batch_size,
                'accumulation_per_gpu': accumulation // world,
                'global_batch_size': training.effective_batch_size,
                'loader_workers_total': training.num_workers * world} for world in (1, 2)]}


def worker_command(world_size, request_path, log_dir):
    """Build a shell-free torchrun command with standalone rendezvous and rank logs."""
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
            f'--nproc-per-node={world_size}', '--max-restarts=0', '--log-dir', str(log_dir),
            '--redirects=3', '--tee=3', '--module', 'transformer_training.gpu_scaling',
            '--worker', str(request_path)]


def preflight(request):
    """Check Linux/CUDA/NCCL and both visible GPUs before any corpus preparation."""
    if platform.system() != 'Linux':
        raise ValueError('GPU scaling requires Linux or WSL2 with NVIDIA CUDA and NCCL')
    if not dist.is_available() or not dist.is_nccl_available():
        raise ValueError('GPU scaling requires a PyTorch build with NCCL support')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise ValueError('GPU scaling requires at least two visible CUDA GPUs')
    config = TrainingConfig(**request['training'])
    for index in (0, 1):
        device = validate_device(replace(config, device=f'cuda:{index}'))
        validate_loss_backend(request['implementation']['loss_backend'], device,
                              request['model']['vocab_size'])
    try:
        result = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True,
                                text=True, timeout=10, check=False)
        topology = {'status': 'available' if result.returncode == 0 else 'unavailable',
                    'output': result.stdout if result.returncode == 0 else result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        topology = {'status': 'unavailable', 'reason': str(error)}
    return {'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'topology': topology}


def prepare_data(request):
    """Populate the pinned, audited cache outside trials; retain existing offline rules."""
    if request['offline']:
        os.environ['HF_HUB_OFFLINE'] = os.environ['HF_DATASETS_OFFLINE'] = '1'
    from .data import load_tinystories
    return load_tinystories(**request['data_options'])


def schedule_horizon(size, config, attempts):
    """Return the global-batch schedule horizon or reject an insufficient workload."""
    horizon = config.epochs * math.ceil(size / config.effective_batch_size)
    if config.max_steps is not None:
        horizon = min(horizon, config.max_steps)
    if horizon < attempts:
        raise ValueError('Insufficient data/schedule for all warm-up and measured attempts; '
                         'increase --epochs/--max-stories or reduce attempt counts')
    return horizon


def initial_weights_hash(model):
    """Hash initial tensor values on CPU before timing to verify identical replicas."""
    checksum = hashlib.sha256()
    for name, value in model.state_dict().items():
        checksum.update(name.encode())
        checksum.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def summarise_trial(ranks):
    """Combine rank durations conservatively, counting global tokens only once."""
    if not ranks or [r['rank'] for r in ranks] != list(range(len(ranks))):
        raise ValueError('Missing or unordered rank results')
    first = ranks[0]
    for other in ranks[1:]:
        if other['identity'] != first['identity'] or other['work'] != first['work']:
            raise ValueError('Rank workload/configuration identities disagree')
        for phase in ('warmup_records', 'records'):
            if [{k: r[k] for k in ('skipped', 'scaler_after', 'learning_rate')} for r in other[phase]] != [
                    {k: r[k] for k in ('skipped', 'scaler_after', 'learning_rate')} for r in first[phase]]:
                raise ValueError('Rank update/scaler decisions disagree')
    seconds = max(r['measured_seconds'] for r in ranks)
    if seconds <= 0:
        raise ValueError('Measured window must have positive duration')
    records = first['records']
    tokens = sum(r['supervised_tokens'] for r in records)
    applied = sum(r['supervised_tokens'] for r in records if not r['skipped'])
    skipped = sum(r['skipped'] for r in records)
    allocated = [r['peak_allocated_bytes'] for r in ranks]
    reserved = [r['peak_reserved_bytes'] for r in ranks]
    return {'status': 'complete', 'world_size': len(ranks), 'identity': first['identity'],
            'work': first['work'], 'rank_results': ranks, 'measured_seconds': seconds,
            'useful_tokens_per_second': applied / seconds,
            'processed_tokens_per_second': tokens / seconds,
            'processed_tokens': tokens, 'applied_tokens': applied, 'skipped_updates': skipped,
            'numerical_failures': sum(r['nonfinite_gradients'] for r in records),
            'warmup_skipped_updates': sum(r['skipped'] for r in first['warmup_records']),
            'training_loss': sum(r['loss_sum'] for r in records) / tokens,
            'validation': first['validation'],
            'latency_seconds': distribution([max(rank['records'][i]['seconds'] for rank in ranks)
                                               for i in range(len(records))]),
            'memory': {'per_gpu_allocated_bytes': allocated,
                       'max_allocated_bytes': max(allocated) if None not in allocated else None,
                       'sum_allocated_bytes': sum(allocated) if None not in allocated else None,
                       'per_gpu_reserved_bytes': reserved,
                       'max_reserved_bytes': max(reserved) if None not in reserved else None,
                       'sum_reserved_bytes': sum(reserved) if None not in reserved else None},
            'timings_eligible': all(r['timings_eligible'] for r in ranks)}


def run_rank(request, loaders, device, *, env=None):
    """Run real bounded updates on one rank; CPU/Gloo is available only to tests.

    All workers receive the one-GPU configuration: the batch sampler partitions
    its global groups. Setup, warm-up, validation and evidence capture are outside
    the measured window. No training checkpoints or resumable state are written.
    """
    config = TrainingConfig(**request['training'])
    implementation = ImplementationConfig(**request['implementation'])
    context = DistributedContext(device)
    protocol = request['protocol']
    attempts = protocol['warmup'] + protocol['measured']
    horizon = schedule_horizon(len(loaders.train.dataset), config, attempts)
    if data_signature(loaders.metadata) != request['prepared_data']:
        raise ValueError('Prepared data identity changed between preparation and worker')
    seed_everything(config.seed)
    raw = DecoderOnlyTransformer(ModelConfig(**request['model']), implementation).to(device)
    identity = {'model': request['model'], 'training': request['training'],
                'implementation': request['implementation'], 'data': request['prepared_data'],
                'protocol': protocol, 'schedule_updates': horizon,
                'workload_fingerprint': workload_fingerprint(len(loaders.train.dataset), config, attempts),
                'initial_weights_sha256': initial_weights_hash(raw)}
    env = env if env is not None else environment(torch.device(device),
        source_exclude=(Path(request['output']).resolve().parent,))
    identity['environment'] = {key: value for key, value in env.items() if key not in {
        'device', 'gpu', 'gpu_memory_bytes', 'compute_capability', 'gpu_uuid'}}
    identities = context.gather(identity)
    if any(other != identity for other in identities):
        raise ValueError('Rank model, data, source, or software identities disagree')
    model = (DistributedDataParallel(raw, device_ids=[device.index] if device.type == 'cuda' else None,
                                     broadcast_buffers=False)
             if context.world_size > 1 else raw)
    optimizer = make_optimizer(raw, config, implementation)
    scheduler = WarmupCosine(optimizer, config, horizon)
    scaler = torch.amp.GradScaler('cuda', enabled=config.precision == 'fp16')
    seed_everything((config.seed + context.rank) % 2**63)
    model.train()
    representative = {}
    epoch, group_index = 0, 0
    loader, group_sizes = scaling_loader(loaders.train, config, epoch, context)
    iterator = iter(loader)

    def sync():
        """Finish local CUDA work without adding a rank barrier to each attempt."""
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def attempt():
        """Time delivery, transfers, global loss weighting and one complete update."""
        nonlocal epoch, group_index, iterator, group_sizes, loader
        if group_index == len(group_sizes):
            epoch += 1
            loader, group_sizes = scaling_loader(loaders.train, config, epoch, context)
            iterator, group_index = iter(loader), 0
        sync()
        before = compiler_snapshot(implementation.compile)
        started = time.perf_counter()
        batches = list(itertools.islice(iterator, group_sizes[group_index]))
        if len(batches) != group_sizes[group_index]:
            raise ValueError('Data loader ended before its declared accumulation group')
        result = optimizer_attempt(model, optimizer, scaler, scheduler, batches, config,
                                   device, distributed=context)
        sync()
        seconds = time.perf_counter() - started
        after = compiler_snapshot(implementation.compile)
        for batch in batches:
            if not batch['_scaling_dummy'].all():
                representative.setdefault(tuple(batch['input_ids'].shape), batch)
        group_index += 1
        return dict(result, seconds=seconds,
                    compilation_detected=compilation_changed(before, after),
                    compiler_telemetry_available=after['status'] in {'disabled', 'available'})

    warmup = [attempt() for _ in range(protocol['warmup'])]
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    sync()
    context.barrier()
    started = time.perf_counter()
    records = [attempt() for _ in range(protocol['measured'])]
    sync()
    measured_seconds = time.perf_counter() - started
    memory = {'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)
              if device.type == 'cuda' else None,
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)
              if device.type == 'cuda' else None}
    context.barrier()
    validation = {'status': 'skipped'}
    raw.zero_grad(set_to_none=True)
    if context.rank == 0 and protocol['validation_batches'] != 0:
        limit = protocol['validation_batches']
        validation_loader = (loaders.validation if limit is None else
                             itertools.islice(loaders.validation, limit))
        started_validation = time.perf_counter()
        validation = dict(evaluate(raw, validation_loader, device=device, precision=config.precision),
                          status='complete', scope='full selected validation' if limit is None
                          else 'bounded sanity check', requested_batches=limit,
                          actual_batches=len(loaders.validation) if limit is None else
                          min(limit, len(loaders.validation)),
                          selection_sha256=digest({'data': request['prepared_data'],
                              'batch_size': config.batch_size, 'batch_limit': limit}),
                          seconds=time.perf_counter() - started_validation)
    from .attention_evidence import capture_attention
    raw.attention_evidence = capture_attention(raw, list(representative.values()),
        device=device, precision=config.precision, environment=env)
    result = {'rank': context.rank, 'environment': env, 'identity': identity,
              'work': [{k: r[k] for k in ('examples', 'supervised_tokens', 'slots')}
                       for r in warmup + records],
              'records': records, 'warmup_records': warmup, 'measured_seconds': measured_seconds,
              **memory, 'allocator_measurement': 'CUDA allocator' if device.type == 'cuda'
              else 'unavailable on CPU',
              'validation': validation, 'execution': execution_report(raw),
              'timings_eligible': all(not r['compilation_detected'] and
                  r['compiler_telemetry_available'] for r in records)}
    rank_dir = Path(request['output']) / f'rank-{context.rank}'
    atomic_json(rank_dir / 'result.json', result)
    ranks = context.gather(result)
    if context.rank == 0:
        trial = summarise_trial(ranks)
        atomic_json(Path(request['output']) / 'trial.json', trial)
        return trial
    return None


def worker(request):
    """Bind the local GPU, load prepared caches and run a rank with failure evidence."""
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    output = Path(request['output']) / f'rank-{rank}'
    output.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(request['protocol']['torch_threads'])
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
        validate_device(replace(TrainingConfig(**request['training']), device=str(device)))
        validate_loss_backend(request['implementation']['loss_backend'], device,
                              request['model']['vocab_size'])
        # The parent has already populated pinned caches, so workers need no network.
        loaders = prepare_data(dict(request, offline=True))
        if int(os.environ.get('WORLD_SIZE', '1')) > 1:
            dist.init_process_group('nccl', timeout=timedelta(
                seconds=request['protocol']['timeout_seconds']))
        trial = run_rank(request, loaders, device)
        atomic_json(output / 'status.json', {'status': 'complete'})
        return trial
    except BaseException as error:
        atomic_json(output / 'failure.json', {'status': 'failed', 'reason': str(error),
                                             'traceback': traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def launch_trial(request, world, destination):
    """Run an isolated torchrun job and reject missing, partial or failed results."""
    from .experiments import worker_environment
    destination.mkdir(parents=True, exist_ok=False)
    worker_request = dict(request, output=str(destination.resolve()))
    path = destination / 'request.json'
    atomic_json(path, worker_request)
    command = worker_command(world, path.resolve(), (destination / 'logs').resolve())
    env = worker_environment(destination.resolve() / 'compiler-cache')
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = str(request['protocol']['torch_threads'])
    try:
        with (destination / 'worker.log').open('w', encoding='utf-8') as log:
            process = run_bounded(command, timeout=request['protocol']['timeout_seconds'],
                                  stdout=log, stderr=subprocess.STDOUT, env=env)
        if process.returncode:
            raise RuntimeError(f'Worker exited {process.returncode}; see {destination / "worker.log"}')
        trial = json.loads((destination / 'trial.json').read_text(encoding='utf-8'))
        if trial['status'] != 'complete' or trial['world_size'] != world:
            raise ValueError('Worker did not produce a complete trial for the requested GPU count')
        atomic_json(destination / 'status.json', {'status': 'complete', 'command': command})
        return trial
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        atomic_json(destination / 'status.json', {'status': 'timeout' if isinstance(error,
                    subprocess.TimeoutExpired) else 'failed', 'reason': str(error), 'command': command})
        raise RuntimeError(f'GPU scaling trial failed: {error}; logs: {destination}') from error


def build_report(request, trials, *, status='complete', failure=None):
    """Aggregate repeats and expose speedup only when all comparison checks pass."""
    groups = {}
    issues = []
    for world in (1, 2):
        selected = [t for t in trials if t['world_size'] == world]
        if len(selected) != request['protocol']['repeats']:
            issues.append(f'Incomplete {world}-GPU repeats')
        groups[str(world)] = {'repeats': len(selected),
            'throughput': distribution([t['useful_tokens_per_second'] for t in selected]),
            'processed_throughput': distribution([t['processed_tokens_per_second'] for t in selected]),
            'update_seconds': distribution([t['latency_seconds']['median'] for t in selected]),
            'training_loss': distribution([t['training_loss'] for t in selected]),
            'validation_loss': distribution([t['validation']['loss'] for t in selected
                                             if t['validation']['status'] == 'complete']),
            'skipped_updates': sum(t['skipped_updates'] for t in selected),
            'numerical_failures': sum(t['numerical_failures'] for t in selected),
            'memory': {key: distribution([t['memory'][key] for t in selected
                                         if t['memory'][key] is not None]) for key in (
                'max_allocated_bytes', 'sum_allocated_bytes', 'max_reserved_bytes', 'sum_reserved_bytes')},
            'memory_per_gpu': [{'rank': rank, **{key: distribution([
                t['memory'][f'per_gpu_{key}_bytes'][rank] for t in selected
                if t['memory'][f'per_gpu_{key}_bytes'][rank] is not None])
                for key in ('allocated', 'reserved')}} for rank in range(world)]}
    reference = trials[0] if trials else None
    for trial in trials:
        differences = field_differences(reference['identity'], trial['identity'])
        if differences:
            issues.append('Changed conditions: ' + ', '.join(differences))
        if trial['work'] != reference['work']:
            issues.append('Consumed real-example/token/slot counts differ')
        if not trial['timings_eligible']:
            issues.append('Compilation occurred during measurement or telemetry is unavailable')
        if trial['skipped_updates'] or trial['warmup_skipped_updates']:
            issues.append('Skipped optimiser updates occurred; training trajectories differ')
        if trial.get('numerical_failures'):
            issues.append('Nonfinite gradients occurred')
        source = trial['identity'].get('environment', {}).get('source', {})
        if not isinstance(source, dict) or not source.get('content_sha256'):
            issues.append('Source content identity could not be verified')
    devices = [r['environment'] for t in trials for r in t['rank_results']]
    signatures = {json.dumps({k: e.get(k) for k in ('gpu', 'gpu_memory_bytes', 'compute_capability')},
                             sort_keys=True) for e in devices}
    if len(signatures) > 1:
        issues.append('Heterogeneous GPU models, capacities, or compute capabilities')
    baselines = {t['repeat']: t for t in trials if t['world_size'] == 1}
    for trial in (t for t in trials if t['world_size'] == 2):
        baseline = baselines.get(trial['repeat'])
        if baseline and baseline['rank_results'][0]['environment'].get('gpu_uuid') != (
                trial['rank_results'][0]['environment'].get('gpu_uuid')):
            issues.append('The baseline GPU differs from rank zero of the two-GPU trial')
        uuids = [r['environment'].get('gpu_uuid') for r in trial['rank_results']]
        if any(value is None for value in uuids) or len(set(uuids)) != 2:
            issues.append('Two distinct GPU identities could not be verified')
    if status != 'complete':
        issues.append('Comparison did not complete')
    a, b = (groups[str(world)]['throughput']['median'] for world in (1, 2))
    speedup = b / a if not issues and a and b is not None else None
    return {'version': 1, 'kind': 'gpu_scaling', 'status': status, 'failure': failure,
            'request': request, 'groups': groups, 'trials': trials,
            'comparison': {'status': 'eligible' if not issues else 'ineligible',
                           'issues': sorted(set(issues)), 'speedup': speedup,
                           'two_gpu_efficiency': speedup / 2 if speedup is not None else None},
            'timing_scope': 'loader delivery, transfers, global weighting, forward/backward, DDP '
                            'communication, clipping, optimizer and scheduler; includes between-attempt '
                            'bookkeeping and epoch transitions; excludes trial setup, '
                            'warm-up, validation, evidence capture and output writing',
            'memory_scope': 'per-rank CUDA allocator peaks during measured attempts after warm-up; '
                            'sum is the sum of per-device peaks, not a simultaneous device-wide peak'}


def write_report(report, output):
    """Write JSON, a compact Markdown comparison and per-repeat CSV measurements."""
    def cell(value):
        """Format missing values explicitly and keep measured numbers readable."""
        return 'unavailable' if value is None else f'{value:.6g}' if isinstance(value, float) else str(value)

    atomic_json(output / 'report.json', report)
    lines = ['# One versus two GPU training', '',
             f"Status: {report['status']}; comparison: {report['comparison']['status']}.", '',
             '| GPUs | Repeats | Useful tokens/s | Median update (s) | Max GPU allocated (bytes) | Validation loss |',
             '|---|---:|---:|---:|---:|---:|']
    for world, group in report['groups'].items():
        values = [world, group['repeats'], group['throughput']['median'],
                  group['update_seconds']['median'], group['memory']['max_allocated_bytes']['median'],
                  group['validation_loss']['median']]
        lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
    comparison = report['comparison']
    if comparison['speedup'] is not None:
        lines.extend(['', f"Speedup: {comparison['speedup']:.3f}x. "
                      f"Two-GPU efficiency: {comparison['two_gpu_efficiency']:.1%}."])
    else:
        lines.extend(['', 'Speedup withheld:', *['- ' + issue for issue in comparison['issues']]])
    if report['failure']:
        lines.extend(['', 'Failure: ' + report['failure']])
    lines.extend(['', '| GPUs | Processed tokens/s | Training loss | Skipped measured updates | Numerical failures |',
                  '|---|---:|---:|---:|---:|'])
    for world, group in report['groups'].items():
        lines.append('| ' + ' | '.join(map(cell, [world, group['processed_throughput']['median'],
            group['training_loss']['median'], group['skipped_updates'], group['numerical_failures']])) + ' |')
    lines.extend(['', '| GPUs | Rank | Allocated peak (bytes) | Reserved peak (bytes) |',
                  '|---|---:|---:|---:|'])
    for world, group in report['groups'].items():
        for gpu in group['memory_per_gpu']:
            lines.append('| ' + ' | '.join(map(cell, [world, gpu['rank'], gpu['allocated']['median'],
                                                      gpu['reserved']['median']])) + ' |')
    lines.extend(['', '| GPUs | Max reserved (bytes) | Sum allocated peaks (bytes) | Sum reserved peaks (bytes) |',
                  '|---|---:|---:|---:|'])
    for world, group in report['groups'].items():
        lines.append('| ' + ' | '.join(map(cell, [world,
            *[group['memory'][key]['median'] for key in ('max_reserved_bytes', 'sum_allocated_bytes',
                                                         'sum_reserved_bytes')]])) + ' |')
    lines.extend(['', report['timing_scope'] + '.', '', report['memory_scope'] + '.', '',
                  'Validation is untimed. A bounded validation check is a sanity check, '
                  'not evidence of equal final model quality. Raw per-GPU memory, losses, skips, '
                  'environments and rank records are retained in report.json.'])
    atomic_text(output / 'report.md', '\n'.join(lines) + '\n')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=['repeat', 'gpus', 'useful_tokens_per_second',
        'processed_tokens_per_second', 'median_update_seconds', 'max_allocated_bytes',
        'sum_allocated_bytes', 'max_reserved_bytes', 'sum_reserved_bytes', 'training_loss',
        'validation_loss', 'skipped_updates', 'numerical_failures', 'timings_eligible'])
    writer.writeheader()
    for trial in report['trials']:
        writer.writerow({'repeat': trial['repeat'], 'gpus': trial['world_size'],
            **{k: trial[k] for k in ('useful_tokens_per_second', 'processed_tokens_per_second',
                'training_loss', 'skipped_updates', 'numerical_failures', 'timings_eligible')},
            'median_update_seconds': trial['latency_seconds']['median'],
            **{k: trial['memory'][k] for k in ('max_allocated_bytes', 'sum_allocated_bytes',
                                             'max_reserved_bytes', 'sum_reserved_bytes')},
            'validation_loss': trial['validation'].get('loss')})
    atomic_text(output / 'report.csv', stream.getvalue())


def main(argv=None):
    """Resolve, prepare and compare one/two GPUs, or print a side-effect-free dry-run."""
    parser = build_parser()
    args = parser.parse_args(argv)
    created, trials = False, []
    output = args.output_dir or Path('runs') / f'gpu-scaling-{time.time_ns()}'
    try:
        request = resolve_request(args)
        validate_output_directory(output)
        if args.dry_run:
            print(json.dumps({'dry_run': True, **request,
                'commands': [worker_command(world, output / f'gpu-{world}' / 'request.json',
                                             output / f'gpu-{world}' / 'logs') for world in (1, 2)]}, indent=2))
            return 0
        hardware = preflight(request)
        output.mkdir(parents=True, exist_ok=True)
        created = True
        atomic_json(output / 'hardware.json', hardware)
        prepared_at = time.perf_counter()
        loaders = prepare_data(request)
        request['model']['vocab_size'] = loaders.metadata['vocab_size']
        request['prepared_data'] = data_signature(loaders.metadata)
        size = len(loaders.train.dataset)
        schedule_horizon(size, TrainingConfig(**request['training']),
                         request['protocol']['warmup'] + request['protocol']['measured'])
        atomic_json(output / 'data.json', loaders.metadata)
        atomic_json(output / 'preparation.json', {'seconds': time.perf_counter() - prepared_at})
        atomic_json(output / 'request.json', request)
        del loaders
        for repeat in range(1, request['protocol']['repeats'] + 1):
            for world in ((1, 2) if repeat % 2 else (2, 1)):
                print(f'Repeat {repeat}/{request["protocol"]["repeats"]}: {world} GPU(s)', flush=True)
                trial = launch_trial(request, world, output / f'repeat-{repeat}-gpu-{world}')
                trial['repeat'] = repeat
                trials.append(trial)
                write_report(build_report(request, trials, status='running'), output)
        write_report(build_report(request, trials), output)
        print(f'Report: {output / "report.md"}', flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        if created:
            write_report(build_report(request, trials, status='failed', failure=str(error)), output)
            atomic_json(output / 'failure.json', {'status': 'failed', 'reason': str(error)})
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        worker(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')))
    else:
        raise SystemExit(main())
