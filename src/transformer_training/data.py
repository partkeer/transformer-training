"""Load official TinyStories splits as padded, next-token PyTorch batches.

Each story is tokenised independently and ends with EOS. Long stories are split
into windows with one overlapping token so every next-token pair is retained.
No window crosses a story or split boundary. The model must apply causal masking.
"""

from dataclasses import dataclass
import importlib.metadata
from pathlib import Path
import platform
import random
import re

from datasets import DatasetDict, Features, List, Value, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer
import torch
from torch.utils.data import DataLoader

DATASET = "roneneldan/TinyStories"
# Match the existing EDA; the tokenizer is also pinned to an immutable Hub commit.
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TOKENIZER = "openai-community/gpt2"
TOKENIZER_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
IGNORE_INDEX = -100


def _resolve_revision(repo, revision, *, dataset=False):
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    api = HfApi()
    info = api.dataset_info if dataset else api.model_info
    return info(repo, revision=revision).sha


def _check_dataset_revision(splits, revision):
    """Reject datasets' offline fallback if it selected another cached commit.

    With the pinned datasets version, Hub Arrow files live directly under the
    resolved commit directory. Check before sampling or mapping changes paths.
    """
    for name in ("train", "validation"):
        files = splits[name].cache_files
        if not files or any(Path(file["filename"]).parent.name != revision for file in files):
            raise RuntimeError(
                f"Cannot verify {name} cache matches dataset revision {revision}. "
                "Load the requested revision online to populate the matching cache."
            )


def tokenise_stories(batch, indices, *, tokenizer, sequence_length, eos_id,
                     source_indices=None):
    """A batched Arrow map: skip blank text, reject invalid schema, keep tails."""
    texts, source_rows = [], []
    for text, index in zip(batch["text"], indices):
        if not isinstance(text, str):
            raise ValueError(f"Expected string text at source row {index}")
        if text.strip():
            texts.append(text)
            source_rows.append(index if source_indices is None else source_indices[index])
    windows, rows = [], []
    for encoding, index in zip(
        tokenizer.encode_batch(texts, add_special_tokens=False), source_rows
    ):
        tokens = encoding.ids + [eos_id]
        for start in range(0, len(tokens) - 1, sequence_length):
            windows.append(tokens[start:start + sequence_length + 1])
            rows.append(index)
    return {"tokens": windows, "source_row": rows}


@dataclass(frozen=True)
class NextTokenCollator:
    """Right-pad inputs; mask only padded targets, never genuine EOS targets."""

    sequence_length: int
    pad_id: int

    def __call__(self, examples):
        shape = (len(examples), self.sequence_length)
        inputs = torch.full(shape, self.pad_id, dtype=torch.long)
        labels = torch.full(shape, IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros(shape, dtype=torch.bool)
        for row, example in enumerate(examples):
            tokens = torch.tensor(example["tokens"], dtype=torch.long)
            length = len(tokens) - 1
            if not 1 <= length <= self.sequence_length:
                raise ValueError("Windows must contain 2 to sequence_length + 1 tokens")
            inputs[row, :length] = tokens[:-1]
            labels[row, :length] = tokens[1:]
            attention_mask[row, :length] = True
        return {"input_ids": inputs, "labels": labels,
                "attention_mask": attention_mask}


@dataclass
class TinyStoriesLoaders:
    train: DataLoader
    validation: DataLoader
    metadata: dict


def create_dataloaders(
    splits: DatasetDict,
    tokenizer: Tokenizer,
    *,
    sequence_length: int = 256,
    batch_size: int = 8,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_stories: int | None = None,
) -> TinyStoriesLoaders:
    """Prepare Arrow-cached windows and loaders; max_stories samples each split.

    Training windows shuffle each epoch using a dedicated seeded generator.
    Validation order is fixed. Partial batches are kept in both splits. Data and
    collation stay on CPU, including when pin_memory=True. No deduplication is
    performed: preserving official split membership is not a leakage audit.
    """
    if sequence_length < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("sequence_length/batch_size must be positive; num_workers >= 0")
    if max_stories is not None and max_stories < 1:
        raise ValueError("max_stories must be positive or None for the full splits")
    if not {"train", "validation"}.issubset(splits):
        raise ValueError("Expected official train and validation splits")
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("Tokenizer must define <|endoftext|> as EOS")
    # Work on a copy; caller settings cannot silently truncate or pad stories.
    tokenizer = Tokenizer.from_str(tokenizer.to_str())
    tokenizer.no_padding()
    tokenizer.no_truncation()
    loaders, split_metadata = {}, {}
    for offset, name in enumerate(("train", "validation")):
        split = splits[name]
        if "text" not in split.column_names:
            raise ValueError(f"{name} has no text column")
        original_rows = len(split)
        # Sort sampled indices to preserve original story order within the subset.
        indices = None
        if max_stories is not None and max_stories < len(split):
            indices = sorted(random.Random(seed + offset).sample(
                range(len(split)), max_stories))
            split = split.select(indices)
        windows = split.map(
            tokenise_stories, batched=True, with_indices=True, batch_size=1000,
            fn_kwargs={"tokenizer": tokenizer, "sequence_length": sequence_length,
                       "eos_id": eos_id, "source_indices": indices},
            remove_columns=split.column_names,
            features=Features({"tokens": List(Value("int32")),
                               "source_row": Value("int64")}),
            desc=f"Tokenise {name}",
        )
        if not len(windows):
            raise ValueError(f"{name} has no usable stories after filtering blank text")
        loaders[name] = DataLoader(
            windows, batch_size=batch_size, shuffle=name == "train",
            generator=torch.Generator().manual_seed(seed + offset),
            collate_fn=NextTokenCollator(sequence_length, eos_id),
            num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        )
        split_metadata[name] = {
            "original_stories": original_rows, "selected_stories": len(split),
            "windows": len(windows), "source_fingerprint": split._fingerprint,
            "windows_fingerprint": windows._fingerprint,
            "selection_seed": seed + offset, "selected_row_indices": indices,
        }
    metadata = {
        "sequence_length": sequence_length, "batch_size": batch_size,
        "seed": seed, "num_workers": num_workers, "pin_memory": pin_memory,
        "max_stories": max_stories, "drop_last": False,
        "vocab_size": tokenizer.get_vocab_size(), "eos_id": eos_id, "pad_id": eos_id,
        "ignore_index": IGNORE_INDEX,
        "policy": "Per-story windows; append EOS; retain tails; skip blank text; "
                  "reject non-string text; no deduplication; train-only shuffle",
        "splits": split_metadata,
    }
    return TinyStoriesLoaders(loaders["train"], loaders["validation"], metadata)


def load_tinystories(
    *,
    revision: str = DATASET_REVISION,
    tokenizer_revision: str = TOKENIZER_REVISION,
    **loader_options,
) -> TinyStoriesLoaders:
    """Download/cache TinyStories and GPT-2 tokenisation, then create CPU loaders.

    HF_HOME controls cache location. Set HF_HUB_OFFLINE=1 before importing this
    module for a cached-only run (use immutable revisions in offline mode).
    The pretrained tokenizer is fixed; nothing is fitted on either split here.
    """
    revision = _resolve_revision(DATASET, revision, dataset=True)
    tokenizer_revision = _resolve_revision(TOKENIZER, tokenizer_revision)
    tokenizer_path = hf_hub_download(TOKENIZER, "tokenizer.json",
                                     revision=tokenizer_revision)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    splits = load_dataset(DATASET, revision=revision)
    _check_dataset_revision(splits, revision)
    result = create_dataloaders(splits, tokenizer, **loader_options)
    result.metadata.update(
        dataset=DATASET, revision=revision, tokenizer=TOKENIZER,
        tokenizer_revision=tokenizer_revision,
        environment={"python": platform.python_version(),
                     "platform": platform.platform(),
                     **{name: importlib.metadata.version(name) for name in
                        ("torch", "datasets", "tokenizers", "huggingface-hub")}},
    )
    return result
