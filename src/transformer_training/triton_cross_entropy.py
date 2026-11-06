"""Lazy NVIDIA Triton cross-entropy kernels; import only after runtime validation.

No full-vocabulary FP32 copy or log-softmax is stored. The forward retains input
logits plus two FP32 statistics per row; backward writes a fresh logits gradient.
Only first-order autograd for finite, contiguous supported inputs is provided.
"""

import torch
import triton
import triton.language as tl

from .loss import FALLBACK, triton_launch_settings


@triton.jit
def _forward(Logits, Targets, Maxima, LogSums, Losses, V: tl.constexpr, BLOCK: tl.constexpr):
    """Fuse stable row reduction and target selection, storing only row statistics/loss."""
    row = tl.program_id(0).to(tl.int64)
    target = tl.load(Targets + row)
    if target == -100:
        tl.store(Maxima + row, 0.0)
        tl.store(LogSums + row, 0.0)
        tl.store(Losses + row, 0.0)
    else:
        columns = tl.arange(0, BLOCK)
        values = tl.load(Logits + row * V + columns, columns < V, other=-float("inf")).to(tl.float32)
        maximum = tl.max(values, 0)
        log_sum = tl.log(tl.sum(tl.exp(values - maximum), 0))
        selected = tl.load(Logits + row * V + target).to(tl.float32)
        tl.store(Maxima + row, maximum)
        tl.store(LogSums + row, log_sum)
        # Subtract before adding log_sum: large common offsets must not cancel it.
        tl.store(Losses + row, (maximum - selected) + log_sum)


@triton.jit
def _backward(Logits, Targets, Maxima, LogSums, Upstream, Gradient,
              V: tl.constexpr, BLOCK: tl.constexpr):
    """Write scaled softmax-minus-target gradients by row/vocabulary tile, including zeros."""
    row = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = columns < V
    target = tl.load(Targets + row)
    if target == -100:
        tl.store(Gradient + row * V + columns, 0.0, mask)
    else:
        values = tl.load(Logits + row * V + columns, mask, other=0.0).to(tl.float32)
        maximum = tl.load(Maxima + row)
        log_sum = tl.load(LogSums + row)
        scale = tl.load(Upstream).to(tl.float32)
        probability = tl.exp((values - maximum) - log_sum)
        gradient = (probability - (columns == target).to(tl.float32)) * scale
        tl.store(Gradient + row * V + columns, gradient, mask)


class _CrossEntropy(torch.autograd.Function):
    """Connect the row kernels to first-order autograd without mutating saved tensors."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, logits, targets):
        """Return FP32 summed loss, saving original inputs and separate stable statistics."""
        rows, vocab = logits.shape
        settings = triton_launch_settings(vocab)
        maxima = torch.empty(rows, dtype=torch.float32, device=logits.device)
        log_sums = torch.empty_like(maxima)
        losses = torch.empty_like(maxima)
        if rows:
            try:
                with torch.cuda.device(logits.device):
                    _forward[(rows,)](logits, targets, maxima, log_sums, losses, vocab,
                                      settings["forward_block"], num_warps=settings["forward_warps"])
            except Exception as error:
                raise RuntimeError(f"Triton cross-entropy forward failed: {error}. {FALLBACK}") from error
        ctx.save_for_backward(logits, targets, maxima, log_sums)
        return losses.sum()

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        """Apply any scalar upstream gradient and return fresh logits gradients only.

        Reject create_graph=True rather than silently returning incorrect higher
        derivatives. Saved row statistics remain reusable for retained graphs.
        """
        if torch.is_grad_enabled():
            raise RuntimeError("Triton cross-entropy supports first-order gradients only. " + FALLBACK)
        logits, targets, maxima, log_sums = ctx.saved_tensors
        gradient = torch.empty_like(logits)
        rows, vocab = logits.shape
        settings = triton_launch_settings(vocab)
        if rows:
            try:
                with torch.cuda.device(logits.device):
                    _backward[(rows, triton.cdiv(vocab, settings["backward_block"]))](
                        logits, targets, maxima, log_sums, grad_output, gradient, vocab,
                        settings["backward_block"], num_warps=settings["backward_warps"])
            except Exception as error:
                raise RuntimeError(f"Triton cross-entropy backward failed: {error}. {FALLBACK}") from error
        return gradient, None


def triton_cross_entropy_sum(logits, targets):
    """Apply the custom autograd operation to tensors validated by the public dispatcher."""
    return _CrossEntropy.apply(logits, targets)
