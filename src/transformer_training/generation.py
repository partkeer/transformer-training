"""Small, uncached autoregressive story generation for qualitative review."""

import math

import torch

from .evaluation import autocast_context

DEFAULT_PROMPTS = (
    "Once upon a time, a little girl",
    "Tom found a small box in the garden.",
    "The puppy was afraid of the rain.",
)


def validate_generation_options(*, max_new_tokens, temperature, top_k, seed):
    """Reject invalid token limits, nonfinite or negative temperature, negative top-k,
    and seeds outside [0, 2**63). Temperature zero and top-k zero remain valid
    greedy and full-vocabulary options.
    """
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    if not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("generation seed must be in [0, 2**63)")


def encode_prompt(tokenizer, prompt, context_length):
    """Return prompt token IDs without adding EOS or truncating the opening.

    Reject blank prompts, tokenizers with padding/truncation enabled, and inputs
    that leave no context space for at least one generated token.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompts must be nonblank strings")
    if tokenizer.padding is not None or tokenizer.truncation is not None:
        raise ValueError("Generation tokenizer must have padding and truncation disabled")
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    if not ids or len(ids) >= context_length:
        raise ValueError("Prompt must encode to 1 to context_length - 1 tokens; "
                         "shorten the prompt to leave room for generation")
    return ids


@torch.no_grad()
def generate(model, tokenizer, prompt, *, max_new_tokens=128, temperature=0.8,
             top_k=50, seed=42, precision="fp32"):
    """Generate one continuation, preserving prompt, model mode and global RNG.

    Temperature zero is greedy. Positive top_k keeps at most that many tokens;
    zero keeps the whole vocabulary. Stop before exceeding the learned context.
    The token count includes a generated EOS; decoded text omits special tokens.
    Reproducibility is within the same hardware/software/precision environment.
    """
    validate_generation_options(max_new_tokens=max_new_tokens, temperature=temperature,
                                top_k=top_k, seed=seed)
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16 or fp16")
    ids = encode_prompt(tokenizer, prompt, model.config.sequence_length)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None or tokenizer.get_vocab_size() != model.config.vocab_size:
        raise ValueError("Generation tokenizer vocabulary/EOS does not match model")
    device = next(model.parameters()).device
    if device.type == "cpu" and precision != "fp32":
        raise ValueError("This reference supports CPU only with fp32 precision")
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    new_ids = []
    was_training = model.training
    model.eval()
    stop_reason = "max_new_tokens"
    try:
        for _ in range(max_new_tokens):
            with autocast_context(device, precision):
                logits = model(tokens)[:, -1, :].float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite generation logits")
            if temperature == 0:
                next_id = int(logits.argmax(dim=-1).item())
            else:
                # Subtract first so tiny positive temperatures cannot create +inf.
                scale = max(temperature, torch.finfo(logits.dtype).tiny)
                logits = (logits - logits.max(dim=-1, keepdim=True).values) / scale
                if top_k:
                    values, indices = logits.topk(min(top_k, logits.size(-1)), dim=-1)
                    selected = torch.multinomial(values.softmax(dim=-1), 1,
                                                 generator=generator)
                    next_id = int(indices.gather(-1, selected).item())
                else:
                    next_id = int(torch.multinomial(logits.softmax(dim=-1), 1,
                                                   generator=generator).item())
            new_ids.append(next_id)
            # Release compiled outputs before the next CUDA-graph invocation.
            del logits
            if next_id == eos_id:
                stop_reason = "eos"
                break
            tokens = torch.cat((tokens, tokens.new_tensor([[next_id]])), dim=1)
            if tokens.size(1) == model.config.sequence_length:
                stop_reason = "context_capacity"
                break
    finally:
        model.train(was_training)
    return {
        "prompt": prompt, "seed": seed, "prompt_tokens": len(ids),
        "generated_tokens": len(new_ids), "stop_reason": stop_reason,
        "continuation": tokenizer.decode(new_ids, skip_special_tokens=True),
        "text": tokenizer.decode(ids + new_ids, skip_special_tokens=True),
    }
