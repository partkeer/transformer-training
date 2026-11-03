"""Prepare TinyStories dataloaders and inspect a few batches from each split."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))


def main():
    """Parse loader options, prepare both official splits, and inspect CPU batch shapes
    and supervised-token counts. Write settings and observations to the requested
    JSON report; --batches limits inspection, not preprocessing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-stories", type=int,
                        help="Uniform sample per split; omit to prepare all stories.")
    parser.add_argument("--batches", type=int, default=2,
                        help="Batches to inspect per split, after preparing all selected stories.")
    parser.add_argument("--revision", help="Dataset commit/tag/branch; default is the EDA commit.")
    parser.add_argument("--tokenizer-revision", help="GPT-2 commit/tag/branch; default is pinned.")
    parser.add_argument("--offline", action="store_true", help="Use cached data/tokenizer only.")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports" / "tinystories" / "dataloader.json")
    args = parser.parse_args()
    if args.batches < 1:
        parser.error("--batches must be positive")
    if args.offline:
        # Hugging Face reads these on import, before the package below is loaded.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    from transformer_training.data import load_tinystories

    options = {name: getattr(args, name) for name in
               ("sequence_length", "batch_size", "seed", "num_workers",
                "pin_memory", "max_stories")}
    if args.revision:
        options["revision"] = args.revision
    if args.tokenizer_revision:
        options["tokenizer_revision"] = args.tokenizer_revision
    loaders = load_tinystories(**options)
    observed = {}
    for name in ("train", "validation"):
        loader = getattr(loaders, name)
        iterator = iter(loader)
        batches = []
        for _ in range(min(args.batches, len(loader))):
            batch = next(iterator)
            batches.append({"shape": list(batch["input_ids"].shape),
                            "dtype": str(batch["input_ids"].dtype),
                            "supervised_tokens": int((batch["labels"] != -100).sum())})
        observed[name] = batches
        del iterator
        print(f"{name}: {len(loader.dataset):,} windows, {len(loader):,} batches; "
              f"first batch {batches[0]}", flush=True)
    result = dict(loaders.metadata, inspected_batches=observed,
                  created_utc=datetime.now(timezone.utc).isoformat())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Settings and batch inspection: {args.output}", flush=True)


if __name__ == "__main__":
    main()
