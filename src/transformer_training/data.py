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

from datasets import DatasetDict, Features, List, Value, load_dataset, load_dataset_builder
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer
import torch
from torch.utils.data import DataLoader

from .audit import audit_overlap

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
    for name in splits:
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
    for position, (text, index) in enumerate(zip(batch["text"], indices)):
        if not isinstance(text, str):
            raise ValueError(f"Expected string text at source row {index}")
        if text.strip():
            texts.append(text)
            if "_source_row" in batch:
                source_rows.append(batch["_source_row"][position])
            else:
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


@dataclass
class TinyStoriesValidation:
    validation: DataLoader
    tokenizer: Tokenizer
    metadata: dict


def _prepare_tokenizer(tokenizer):
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        raise ValueError("Tokenizer must define <|endoftext|> as EOS")
    tokenizer = Tokenizer.from_str(tokenizer.to_str())
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer, eos_id


def _prepare_split(split, tokenizer, *, name, sequence_length, batch_size,
                   selection_seed, num_workers, pin_memory, max_stories,
                   original_rows):
    """Shared selection/map arguments preserve existing Arrow fingerprints."""
    if "text" not in split.column_names:
        raise ValueError(f"{name} has no text column")
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    eligible_rows = len(split)
    indices = None
    if max_stories is not None and max_stories < len(split):
        indices = sorted(random.Random(selection_seed).sample(
            range(len(split)), max_stories))
        split = split.select(indices)
        if "_source_row" in split.column_names:
            indices = list(split["_source_row"])
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
    loader = DataLoader(
        windows, batch_size=batch_size, shuffle=name == "train",
        generator=torch.Generator().manual_seed(selection_seed),
        collate_fn=NextTokenCollator(sequence_length, eos_id),
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    return loader, {
        "original_stories": original_rows, "eligible_stories": eligible_rows,
        "selected_stories": len(split),
        "windows": len(windows), "source_fingerprint": split._fingerprint,
        "windows_fingerprint": windows._fingerprint,
        "selection_seed": selection_seed, "selected_row_indices": indices,
    }


def _exclude_audited_rows(batch, indices, *, excluded):
    keep = [position for position, row in enumerate(indices) if row not in excluded]
    result = {name: [values[position] for position in keep]
              for name, values in batch.items()}
    result["_source_row"] = [indices[position] for position in keep]
    return result


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
    exclude_validation_overlap: bool = False,
    audit_cache_dir: str | Path | None = None,
) -> TinyStoriesLoaders:
    """Prepare Arrow-cached windows and loaders; max_stories samples each split.

    Training windows shuffle each epoch using a dedicated seeded generator.
    Validation order is fixed. Partial batches are kept in both splits. Data and
    collation stay on CPU, including when pin_memory=True. By default no audit or
    deduplication is performed. Opt-in exclusion audits full splits before sampling
    and removes exact validation matches from training, preserving source IDs.
    """
    if sequence_length < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("sequence_length/batch_size must be positive; num_workers >= 0")
    if max_stories is not None and max_stories < 1:
        raise ValueError("max_stories must be positive or None for the full splits")
    if not {"train", "validation"}.issubset(splits):
        raise ValueError("Expected official train and validation splits")
    tokenizer, eos_id = _prepare_tokenizer(tokenizer)
    audit = audit_overlap(splits, audit_cache_dir) if exclude_validation_overlap else None
    loaders, split_metadata = {}, {}
    for offset, name in enumerate(("train", "validation")):
        split = splits[name]
        if "text" not in split.column_names:
            raise ValueError(f"{name} has no text column")
        original_rows = len(split)
        if name == "train" and audit is not None and audit["removed_train_rows"]:
            split = split.map(
                _exclude_audited_rows, batched=True, with_indices=True,
                fn_kwargs={"excluded": frozenset(audit["removed_train_rows"])},
                desc="Exclude exact validation matches from training",
            )
        loaders[name], split_metadata[name] = _prepare_split(
            split, tokenizer, name=name, sequence_length=sequence_length,
            batch_size=batch_size, selection_seed=seed + offset,
            num_workers=num_workers, pin_memory=pin_memory,
            max_stories=max_stories, original_rows=original_rows,
        )
    metadata = {
        "sequence_length": sequence_length, "batch_size": batch_size,
        "seed": seed, "num_workers": num_workers, "pin_memory": pin_memory,
        "max_stories": max_stories, "drop_last": False,
        "exclude_validation_overlap": exclude_validation_overlap,
        "vocab_size": tokenizer.get_vocab_size(), "eos_id": eos_id, "pad_id": eos_id,
        "ignore_index": IGNORE_INDEX,
        "policy": "Per-story windows; append EOS; retain tails; skip blank text; "
                  "reject non-string text; no deduplication; train-only shuffle",
        "splits": split_metadata,
    }
    if audit is not None:
        metadata["audit"] = audit
        metadata["policy"] = metadata["policy"].replace(
            "no deduplication", "exclude exact validation text from training only")
    return TinyStoriesLoaders(loaders["train"], loaders["validation"], metadata)


def create_validation_loader(
    split, tokenizer, *, sequence_length=256, batch_size=8,
    selection_seed=43, num_workers=0, pin_memory=False, max_stories=None,
) -> TinyStoriesValidation:
    """Prepare only validation, using the same policy as create_dataloaders.

    selection_seed is the validation seed itself (training seed + 1).
    The input is a single official validation Dataset, never a DatasetDict.
    """
    if sequence_length < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("sequence_length/batch_size must be positive; num_workers >= 0")
    if max_stories is not None and max_stories < 1:
        raise ValueError("max_stories must be positive or None for the full split")
    tokenizer, eos_id = _prepare_tokenizer(tokenizer)
    loader, provenance = _prepare_split(
        split, tokenizer, name="validation", sequence_length=sequence_length,
        batch_size=batch_size, selection_seed=selection_seed,
        num_workers=num_workers, pin_memory=pin_memory,
        max_stories=max_stories, original_rows=len(split),
    )
    return TinyStoriesValidation(loader, tokenizer, {
        "sequence_length": sequence_length, "batch_size": batch_size,
        "num_workers": num_workers, "pin_memory": pin_memory,
        "max_stories": max_stories, "drop_last": False,
        "vocab_size": tokenizer.get_vocab_size(), "eos_id": eos_id, "pad_id": eos_id,
        "ignore_index": IGNORE_INDEX, "splits": {"validation": provenance},
    })


def load_tinystories_validation(
    *, revision=DATASET_REVISION, tokenizer_revision=TOKENIZER_REVISION,
    **loader_options,
) -> TinyStoriesValidation:
    """Load only the official validation split and pinned tokenizer.

    Set offline environment variables before importing this module. Loading
    does not access or audit train; existing checkpoint provenance records the
    overlap exclusion performed before training.
    """
    revision = _resolve_revision(DATASET, revision, dataset=True)
    tokenizer_revision = _resolve_revision(TOKENIZER, tokenizer_revision)
    tokenizer_path = hf_hub_download(TOKENIZER, "tokenizer.json",
                                     revision=tokenizer_revision)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    split = _load_validation_split(revision)
    _check_dataset_revision({"validation": split}, revision)
    result = create_validation_loader(split, tokenizer, **loader_options)
    result.metadata.update(dataset=DATASET, revision=revision, tokenizer=TOKENIZER,
                           tokenizer_revision=tokenizer_revision)
    return result


def _load_validation_split(revision):
    """Read an existing split, or prepare validation in a separate cache.

    datasets 5.1 load_dataset(split='validation') still prepares every split.
    Restrict the builder's data files BEFORE preparation instead. Its original
    config identity is retained, so as_dataset produces the same fingerprint
    as training; a separate cache avoids leaving a partial training cache.
    """
    from datasets.config import HF_DATASETS_CACHE

    try:
        cached = load_dataset_builder(DATASET, revision=revision)
    except (FileNotFoundError, ConnectionError):
        cached = None
    if cached is not None and (Path(cached.cache_dir) / "dataset_info.json").is_file():
        split = cached.as_dataset(split="validation")
        _check_dataset_revision({"validation": split}, revision)
        return split
    builder = load_dataset_builder(
        DATASET, revision=revision,
        cache_dir=str(Path(HF_DATASETS_CACHE) / "validation-only"),
    )
    if builder.config.data_files is not None:
        builder.config.data_files = type(builder.config.data_files)(
            {"validation": builder.config.data_files["validation"]})
        if builder.info.splits is not None:
            builder.info.splits = type(builder.info.splits)(
                {"validation": builder.info.splits["validation"]})
        # Existing Hub metadata may describe the combined size of both splits.
        builder.info.download_size = None
        builder.info.dataset_size = None
        builder.info.size_in_bytes = None
    builder.download_and_prepare()
    return builder.as_dataset(split="validation")


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
    if loader_options.get("exclude_validation_overlap"):
        from huggingface_hub.constants import HF_HOME

        loader_options.setdefault("audit_cache_dir", Path(HF_HOME) / "training_audits")
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
