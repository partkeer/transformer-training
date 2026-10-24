"""Load the official TinyStories splits and write reproducible, sampled EDA."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import re

ROOT = Path(__file__).resolve().parents[1]
# Keep downloaded data and plotting caches inside this repository.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi

DATASET = "roneneldan/TinyStories"
WORD = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*")


def describe(values):
    if not values:
        return None
    return dict(zip(
        ["min", "p25", "median", "p75", "p90", "p95", "p99", "max"],
        map(float, np.percentile(values, [0, 25, 50, 75, 90, 95, 99, 100])),
    ), mean=float(np.mean(values)), std=float(np.std(values)))


def analyse(rows):
    """Word counts are regex words, never subword/model token counts."""
    characters, lengths = [], []
    words, fingerprints = Counter(), Counter()
    missing = non_string = empty = 0
    examples = []
    for row in rows:
        text = row.get("text")
        if text is None:
            missing += 1
            continue
        if not isinstance(text, str):
            non_string += 1
            continue
        empty += not bool(text.strip())
        tokens = WORD.findall(text.lower())
        characters.append(len(text))
        lengths.append(len(tokens))
        words.update(tokens)
        fingerprints[hashlib.sha256(text.encode("utf-8")).hexdigest()] += 1
        if text.strip() and len(examples) < 3:
            examples.append(text[:500])
    summary = {
        "analysed_rows": len(rows), "string_rows": len(lengths),
        "missing_or_null_text": missing, "non_string_text": non_string,
        "empty_or_whitespace_text": empty,
        "duplicate_rows_beyond_first": sum(n - 1 for n in fingerprints.values()),
        "characters": describe(characters), "words": describe(lengths),
        "total_words": sum(lengths), "unique_lowercase_words": len(words),
        "top_words": words.most_common(25),
        "stories_over_word_threshold": {
            str(n): sum(x > n for x in lengths) for n in (128, 256, 512, 1024)
        },
        "example_excerpts": examples,
    }
    return summary, fingerprints, characters, lengths


def write_report(result, output):
    lines = ["# TinyStories exploratory analysis", "",
             f"Dataset: [{DATASET}](https://huggingface.co/datasets/{DATASET}).",
             f"Revision: `{result['revision']}`. Seed: {result['seed']}.", "",
             "The full official splits were loaded; statistics below describe only the "
             "selected rows. Selection is uniform without replacement within each split, "
             "using independent seeded Python random generators. No splits are merged.", "",
             "Words are lowercase English-letter regex matches, with internal apostrophes "
             "retained. They are **not model tokens**. Characters use Python `len(text)`. "
             "Quantiles use NumPy's default linear interpolation; standard deviation is "
             "the population value. Vocabulary and frequencies include stop words.", "",
             "| Split | Full rows | Analysed rows | Mean words | Median | p95 | Max |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, summary in result["splits"].items():
        stats = summary["words"]
        values = [f"{stats[k]:.1f}" if stats else "n/a"
                  for k in ("mean", "median", "p95", "max")]
        lines.append(f"| {name} | {summary['full_rows']:,} | "
                     f"{summary['analysed_rows']:,} | " + " | ".join(values) + " |")
    lines += ["", "![Story length distributions](length_distributions.png)", ""]
    for name, summary in result["splits"].items():
        lines += [f"## {name}", "",
                  f"Schema: `{summary['features']}`.", "",
                  f"Missing/null: {summary['missing_or_null_text']}; non-string: "
                  f"{summary['non_string_text']}; empty/whitespace: "
                  f"{summary['empty_or_whitespace_text']}; exact duplicate rows beyond "
                  f"the first: {summary['duplicate_rows_beyond_first']}.", "",
                  f"Observed lowercase vocabulary: {summary['unique_lowercase_words']:,} words.", "",
                  "Frequent words (occurrences): " + ", ".join(
                      f"{word} ({count:,})" for word, count in summary["top_words"][:15]), "",
                  "Stories longer than each word threshold: " + ", ".join(
                      f"{threshold}: {count:,}" for threshold, count in
                      summary["stories_over_word_threshold"].items()), "",
                  "Example excerpts (up to 500 characters, from sampled rows):", ""]
        for example in summary["example_excerpts"]:
            lines += ["> " + example.replace("\n", "\n> "), ""]
    lines += ["## Split overlap and interpretation", "",
              f"Distinct exact texts shared by the analysed train and validation rows: "
              f"**{result['sample_overlap_unique_texts']}** (SHA-256 of unchanged UTF-8 text). "
              "This is a sample-only exact-match check, not a full-corpus leakage audit; "
              "paraphrases and whitespace differences are not detected.", "",
              "Before training, explicitly filter empty stories and audit exact duplicates "
              "across the full splits. If overlap is confirmed, keep validation fixed and "
              "remove matching training stories; record the cleaning rule and resulting "
              "split sizes. The current EDA does not modify or deduplicate the source data.", "",
              "Use the length distribution to guide a later tokenizer-specific analysis. "
              "For example, a 256-word story may exceed 256 subword tokens. Do not choose "
              "a context length or estimate truncation from word counts alone. Measure "
              "token lengths with the chosen tokenizer and consider packing short stories "
              "with explicit end-of-story boundaries. Keep the official validation split "
              "separate and fit any tokenizer on training data only.", "",
              "No tokenizer, training run, GPU benchmark, or performance claim is included. "
              "Machine-readable statistics, selected original row indices, environment "
              "versions and dataset fingerprints are recorded in `summary.json`.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=10_000,
                        help="Rows per split; 0 analyses every row (more time/memory).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default="main", help="Hub commit, tag or branch.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "tinystories")
    args = parser.parse_args()
    if args.sample_size < 0:
        parser.error("--sample-size must be non-negative")
    revision = HfApi().dataset_info(DATASET, revision=args.revision).sha
    print(f"Loading {DATASET} at {revision}", flush=True)
    dataset = load_dataset(DATASET, revision=revision)
    if not {"train", "validation"}.issubset(dataset):
        raise ValueError("Expected official train and validation splits")
    result = {
        "dataset": DATASET, "revision": revision, "seed": args.seed,
        "requested_sample_size": args.sample_size,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        **{p: importlib.metadata.version(p) for p in
                           ("datasets", "huggingface-hub", "numpy", "pyarrow", "matplotlib")}},
        "splits": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    hashes = {}
    plot_values = {}
    for offset, name in enumerate(("train", "validation")):
        split = dataset[name]
        if "text" not in split.column_names:
            raise ValueError(f"{name} has no text column")
        n = len(split) if args.sample_size == 0 else min(args.sample_size, len(split))
        indices = sorted(random.Random(args.seed + offset).sample(range(len(split)), n))
        summary, hashes[name], chars, lengths = analyse(split.select(indices))
        summary.update(full_rows=len(split), features=str(split.features),
                       dataset_fingerprint=split._fingerprint,
                       selection_seed=args.seed + offset, row_indices=indices)
        result["splits"][name] = summary
        plot_values[name] = (chars, lengths)
        print(f"{name}: loaded {len(split):,}; analysed {n:,}", flush=True)
    for metric, (ax, label) in enumerate(zip(axes, ("Characters", "Regex words"))):
        # Common bin edges make the split proportions directly comparable.
        combined = [v for values in plot_values.values() for v in values[metric]]
        bins = np.histogram_bin_edges(combined, bins=60)
        for name, split_values in plot_values.items():
            values = split_values[metric]
            n = result["splits"][name]["analysed_rows"]
            ax.hist(values, bins=bins, alpha=0.5, weights=np.full(len(values), 1 / max(1, len(values))),
                    label=f"{name} (n={n:,})")
            ax.set(xlabel=f"{label} per story", ylabel="Fraction of sampled string rows")
            ax.legend()
    result["sample_overlap_unique_texts"] = len(hashes["train"].keys() & hashes["validation"].keys())
    fig.suptitle("TinyStories: sampled story lengths (not model tokens)")
    fig.savefig(args.output_dir / "length_distributions.png", dpi=160)
    plt.close(fig)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(result, args.output_dir)
    print(f"Report: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
