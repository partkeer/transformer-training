"""Token-correct DDP updates and deterministic, nonduplicating benchmark batches."""

from contextlib import nullcontext
import hashlib
import itertools
import math

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler

from .training import EpochSampler


class DistributedContext:
    """Collectives for an explicitly opted-in update; a single rank is a no-op."""

    def __init__(self, device, group=None):
        """Use an initialised process group, or local execution when none exists."""
        self.device, self.group = torch.device(device), group
        self.world_size = dist.get_world_size(group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(group) if dist.is_initialized() else 0

    def sum(self, values):
        """Return globally summed numeric values on the backend-compatible device."""
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(tensor, group=self.group)
        return tensor.tolist()

    def gather(self, value):
        """Gather JSON-safe setup/results outside timing, retaining rank order."""
        if self.world_size == 1:
            return [value]
        results = [None] * self.world_size
        dist.all_gather_object(results, value, group=self.group)
        return results

    def barrier(self):
        """Align ranks before the measured window, without CUDA work on CPU."""
        if self.world_size > 1:
            dist.barrier(group=self.group)

    def accumulation_context(self, model, final):
        """Synchronise only the last forward/backward pair in an update group."""
        return model.no_sync() if self.world_size > 1 and not final else nullcontext()

    def check_update(self, skipped, scale, completed_updates):
        """Reject divergent AMP/scaler/scheduler states before the next attempt."""
        state = torch.tensor([int(skipped), scale, completed_updates],
                             dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            states = [torch.empty_like(state) for _ in range(self.world_size)]
            dist.all_gather(states, state, group=self.group)
            if any(not torch.equal(state, other) for other in states):
                raise RuntimeError("Ranks disagree on skipped update, scaler, or scheduler state")


class ScalingDataset(Dataset):
    """Read real windows or explicitly marked zero-supervision dummy windows."""

    def __init__(self, dataset):
        """Wrap an existing prepared dataset without copying or retokenising it."""
        self.dataset = dataset

    def __len__(self):
        """Return the number of real windows, excluding dummy execution slots."""
        return len(self.dataset)

    def __getitem__(self, key):
        """Return a window plus its dummy marker from a sampler's (index, flag)."""
        index, dummy = key
        return dict(self.dataset[index], _scaling_dummy=dummy)


class ScalingCollator:
    """Reuse next-token collation while making dummy forwards gradient-neutral."""

    def __init__(self, collator):
        """Retain the existing collator, including EOS and shifted-label policy."""
        self.collator = collator

    def __call__(self, examples):
        """Mask dummy targets, keeping valid input/mask shapes for DDP forward."""
        batch = self.collator(examples)
        dummy = torch.tensor([row['_scaling_dummy'] for row in examples], dtype=torch.bool)
        batch['labels'][dummy] = -100
        batch['_scaling_dummy'] = dummy
        return batch


class ScalingBatchSampler(Sampler):
    """Partition identical global updates into whole rank-local microbatches."""

    def __init__(self, size, batch_size, accumulation, seed, epoch, rank=0, world_size=1):
        """Validate rank geometry and record the one-GPU accumulation setting.

        Partial groups are kept; dummy slots equalise forward/backward counts.
        Real data is never padded, dropped, or supervised more than once.
        """
        if (size < 1 or batch_size < 1 or accumulation < 1 or world_size < 1
                or accumulation % world_size or not 0 <= rank < world_size):
            raise ValueError('Invalid scaling batch geometry')
        self.size, self.batch_size, self.accumulation = size, batch_size, accumulation
        self.seed, self.epoch, self.rank, self.world_size = seed, epoch, rank, world_size
        self.global_batch = batch_size * accumulation

    def group_sizes(self):
        """Return rank-local microbatch counts for each global update in this epoch."""
        return [math.ceil(math.ceil(min(self.global_batch, self.size - start) /
                                    self.batch_size) / self.world_size)
                for start in range(0, self.size, self.global_batch)]

    def groups(self):
        """Yield each global index group from the existing deterministic permutation."""
        order = iter(EpochSampler(self.size, self.seed, self.epoch))
        while group := list(itertools.islice(order, self.global_batch)):
            yield group

    def __iter__(self):
        """Yield rank-local batches and explicitly ignored slots at uneven tails."""
        for group in self.groups():
            batches = [group[start:start + self.batch_size]
                       for start in range(0, len(group), self.batch_size)]
            for step in range(math.ceil(len(batches) / self.world_size)):
                index = step * self.world_size + self.rank
                dummy = index >= len(batches)
                yield [(row, dummy) for row in (batches[-1] if dummy else batches[index])]

    def __len__(self):
        """Return the number of rank-local batches, including dummy execution slots."""
        return sum(self.group_sizes())


def scaling_loader(loader, config, epoch, context):
    """Build one epoch's sharded loader and group lengths, preserving worker isolation."""
    sampler = ScalingBatchSampler(len(loader.dataset), config.batch_size,
                                  config.gradient_accumulation_steps, config.seed,
                                  epoch, context.rank, context.world_size)
    result = DataLoader(ScalingDataset(loader.dataset), batch_sampler=sampler,
                        collate_fn=ScalingCollator(loader.collate_fn),
                        num_workers=config.num_workers, pin_memory=context.device.type == 'cuda',
                        generator=torch.Generator().manual_seed(
                            (config.seed + epoch + 100000 + context.rank) % 2**63))
    return result, sampler.group_sizes()


def workload_fingerprint(size, config, attempts):
    """Hash real indices and update boundaries, independent of GPU count and dummies."""
    checksum = hashlib.sha256()
    consumed = 0
    for epoch in range(config.epochs):
        sampler = ScalingBatchSampler(size, config.batch_size,
                                      config.gradient_accumulation_steps, config.seed, epoch)
        for group in sampler.groups():
            checksum.update((str(epoch) + ':' + ','.join(map(str, group)) + ';').encode())
            consumed += 1
            if consumed == attempts:
                return checksum.hexdigest()
    raise ValueError('Insufficient data for requested warm-up and measured attempts')
