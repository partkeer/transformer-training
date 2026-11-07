"""Summed next-token objectives and explicit, lazy loss-backend selection."""

import importlib.metadata
import importlib.util
import platform

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100
FALLBACK = "Select --loss-backend pytorch to use the reference implementation."


def model_loss_backend(model):
    """Read the configured loss through a DDP wrapper; plain modules use PyTorch.

    DDP does not forward arbitrary configuration attributes from its module.
    Unwrapping explicitly prevents an opted-in Triton loss from silently changing.
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    return getattr(getattr(model, "implementation", None), "loss_backend", "pytorch")


def triton_launch_settings(vocab_size):
    """Return deterministic forward/block and backward/tile sizes without importing Triton.

    The supported vocabulary range is 1..65536. These initial launch choices are
    explicit rather than autotuned; measurements must establish their performance.
    """
    if not 1 <= vocab_size <= 65536:
        raise ValueError("Triton cross-entropy requires 1 <= vocabulary size <= 65536. " + FALLBACK)
    block = 1 << (vocab_size - 1).bit_length()
    warps = 4 if block <= 2048 else 8 if block <= 8192 else 16 if block <= 32768 else 32
    return {"forward_block": block, "forward_warps": warps,
            "backward_block": 1024, "backward_warps": 4}


def validate_loss_backend(backend, device, vocab_size=None):
    """Check runtime prerequisites before real work; never call this during dry-run.

    PyTorch needs no extra dependency. Triton requires Linux, NVIDIA CUDA and
    SM75 or newer; Turing (SM75) additionally requires the Triton 3.2 series.
    Read package metadata without importing Triton or compiling a kernel.
    Checking CUDA can initialise it, so dry runs must bypass this function.
    """
    if backend == "pytorch":
        return
    if backend != "triton":
        raise ValueError(f"Unknown loss backend: {backend!r}")
    if vocab_size is not None:
        triton_launch_settings(vocab_size)
    device = torch.device(device)
    if platform.system() != "Linux" or device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Triton cross-entropy requires Linux with NVIDIA CUDA. " + FALLBACK)
    if not torch.cuda.is_available():
        raise ValueError("Triton cross-entropy requires an available CUDA device. " + FALLBACK)
    capability = torch.cuda.get_device_capability(device)
    if capability < (7, 5):
        raise ValueError("Triton cross-entropy requires compute capability >= 7.5. " + FALLBACK)
    if importlib.util.find_spec("triton") is None:
        raise ValueError("Triton is unavailable; install the version compatible with your CUDA PyTorch. " + FALLBACK)
    if capability < (8, 0):
        try:
            version = importlib.metadata.version("triton")
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError("Triton package metadata is unavailable; reinstall triton==3.2.0. " + FALLBACK) from error
        if not version.startswith("3.2."):
            raise ValueError("Turing GPUs (compute capability 7.5) require Triton 3.2.x; "
                             "use PyTorch 2.6.0 with triton==3.2.0. " + FALLBACK)


def _validate_triton_inputs(logits, targets):
    """Reject unsupported tensors and out-of-range labels before any kernel access.

    Target validation synchronises CUDA by reading one boolean; public-operation
    benchmarks deliberately include that overhead. Finite logits are a documented
    precondition, rather than adding a vocabulary-wide finite-value scan.
    BF16 logits require SM80 or newer; Turing supports FP32 and FP16 only.
    """
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("Triton loss expects logits [N, V] and targets [N]. " + FALLBACK)
    if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Triton logits must be FP32, FP16 or BF16. " + FALLBACK)
    if targets.dtype != torch.int64:
        raise ValueError("Triton targets must be int64. " + FALLBACK)
    if not logits.is_contiguous() or not targets.is_contiguous():
        raise ValueError("Triton logits and targets must be contiguous. " + FALLBACK)
    if logits.device != targets.device:
        raise ValueError("Triton logits and targets must be on the same device. " + FALLBACK)
    validate_loss_backend("triton", logits.device, logits.shape[1])
    if logits.dtype == torch.bfloat16 and torch.cuda.get_device_capability(logits.device) < (8, 0):
        raise ValueError("Triton BF16 logits require compute capability >= 8.0; use FP32 or FP16 on Turing. " + FALLBACK)
    invalid = (targets != IGNORE_INDEX) & ((targets < 0) | (targets >= logits.shape[1]))
    if invalid.any().item():
        raise ValueError("Targets must be -100 or in [0, vocabulary size).")


def cross_entropy_sum(logits, targets, *, backend="pytorch"):
    """Return summed cross-entropy for aligned [N,V] logits and [N] class indices.

    Ignore only -100; never normalise or shift targets. PyTorch preserves its
    native dtype/autocast behaviour. Triton returns an FP32 scalar and supports
    first-order gradients in the logits dtype, including arbitrary scalar upstream
    scaling. Its finite, contiguous input contract is checked before lazy import.
    Unsupported Triton requests fail explicitly; callers can select PyTorch.
    """
    if backend == "pytorch":
        return F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX, reduction="sum")
    if backend != "triton":
        raise ValueError(f"Unknown loss backend: {backend!r}")
    _validate_triton_inputs(logits, targets)
    try:
        from .triton_cross_entropy import triton_cross_entropy_sum
    except Exception as error:
        raise RuntimeError(f"Cannot load Triton cross-entropy: {error}. {FALLBACK}") from error
    return triton_cross_entropy_sum(logits, targets)
