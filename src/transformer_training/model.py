"""Small pre-LayerNorm decoder with switchable execution implementations."""

import math
import time
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

from .config import ImplementationConfig, ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config, implementation=None):
        super().__init__()
        self.implementation = implementation or ImplementationConfig()
        self.heads = config.heads
        self.head_dim = config.d_model // config.heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x, allowed):
        b, t, c = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        if self.implementation.attention == "sdpa":
            # Valid queries precede every padded key under our right-prefix
            # contract. Causality alone therefore isolates all valid outputs.
            backends = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
                        "efficient": SDPBackend.EFFICIENT_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}
            requested = self.implementation.sdpa_backend
            # Re-enter during checkpoint recomputation as well as the first pass.
            context = sdpa_kernel(backends[requested]) if requested != "auto" else nullcontext()
            with context:
                x = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                    dropout_p=self.dropout.p if self.training else 0.0)
        else:
            scores = (q / math.sqrt(self.head_dim)) @ k.transpose(-2, -1)
            scores = scores.masked_fill(~allowed, float("-inf"))
            # Accumulate softmax in FP32, including in mixed-precision runs.
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
            x = self.dropout(weights) @ v
        x = x.transpose(1, 2).contiguous().reshape(b, t, c)
        return self.residual_dropout(self.output(x))


class DecoderBlock(nn.Module):
    def __init__(self, config, implementation=None):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config, implementation)
        self.ff_norm = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff), nn.GELU(),
            nn.Linear(config.d_ff, config.d_model), nn.Dropout(config.dropout),
        )

    def forward(self, x, allowed):
        x = x + self.attention(self.attention_norm(x), allowed)
        return x + self.ff(self.ff_norm(x))


class DecoderOnlyTransformer(nn.Module):
    """Return unshifted logits; the data loader already shifts next-token labels.

    Masks must contain a nonempty True prefix followed by right padding. Padded
    query logits are finite but unspecified and must be ignored by the loss.
    """

    def __init__(self, config: ModelConfig, implementation=None):
        super().__init__()
        self.config = config
        self.implementation = implementation or ImplementationConfig()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.sequence_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(DecoderBlock(config, self.implementation) for _ in range(config.layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        # Initialise tied embeddings once, rather than visiting the shared head twice.
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        for block in self.blocks:
            block.apply(self._init_module)
            for projection in (block.attention.output, block.ff[2]):
                nn.init.normal_(projection.weight, std=0.02 / math.sqrt(2 * config.layers))
        self._init_module(self.final_norm)
        self.register_buffer("causal_mask", torch.ones(
            config.sequence_length, config.sequence_length, dtype=torch.bool
        ).tril(), persistent=False)
        self.compile_wrapper_seconds = 0.0
        self._compiled_forward = None
        if self.implementation.compile:
            started = time.perf_counter()
            self._compiled_forward = torch.compile(self._forward_tensors, backend="inductor",
                mode=self.implementation.compile_mode, fullgraph=True)
            self.compile_wrapper_seconds = time.perf_counter() - started

    @staticmethod
    def _init_module(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None):
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be int64 [batch, sequence]")
        b, t = input_ids.shape
        if b < 1 or not 1 <= t <= self.config.sequence_length:
            raise ValueError("Empty batch or sequence, or sequence exceeds model context")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if (attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool
                or attention_mask.device != input_ids.device):
            raise ValueError("attention_mask must be bool, on the same device and shape")
        if (not attention_mask[:, 0].all()
                or (attention_mask[:, 1:] & ~attention_mask[:, :-1]).any()):
            raise ValueError("attention_mask must contain nonempty right-padded prefixes")
        # Backend selection and value-dependent validation stay outside the graph.
        backends = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
                    "efficient": SDPBackend.EFFICIENT_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION}
        requested = self.implementation.sdpa_backend
        context = sdpa_kernel(backends[requested]) if requested != "auto" else nullcontext()
        with context:
            return (self._compiled_forward or self._forward_tensors)(input_ids, attention_mask)

    def _forward_tensors(self, input_ids, attention_mask):
        t = input_ids.shape[1]
        allowed = (self.causal_mask[None, None, :t, :t] & attention_mask[:, None, None, :]
                   if self.implementation.attention == "reference" else None)
        positions = torch.arange(t, device=input_ids.device)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for block in self.blocks:
            if self.implementation.activation_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, allowed, use_reentrant=False, preserve_rng_state=True)
            else:
                x = block(x, allowed)
        return self.lm_head(self.final_norm(x))
