"""Full-split exact-text leakage audit; validation membership is immutable."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from .checkpoint import atomic_json

POLICY_VERSION = "sha256-utf8-nonblank-training-exclusion-v1"


def audit_overlap(splits, cache_dir=None):
    """Return counts and original train rows to exclude, before any sampling.

    Blank rows are counted separately; duplicate/overlap counts refer to usable
    nonblank stories. No case, Unicode or whitespace normalisation is applied.
    Only digests and row numbers are retained, never all full story strings.
    """
    identity = {
        "policy_version": POLICY_VERSION,
        "sources": {name: {"fingerprint": splits[name]._fingerprint,
                           "rows": len(splits[name])}
                    for name in ("train", "validation")},
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir) / f"{key}.json" if cache_dir is not None else None
    if cache is not None and cache.exists():
        result = json.loads(cache.read_text(encoding="utf-8"))
        if result["identity"] != identity:
            raise ValueError("Audit cache identity mismatch")
        return result
    summaries, removed, validation_hashes = {}, [], set()
    for name in ("validation", "train"):
        split = splits[name]
        if "text" not in split.column_names:
            raise ValueError(f"{name} has no text column")
        counts = Counter()
        blanks = 0
        for start in range(0, len(split), 1000):
            for offset, text in enumerate(split[start:start + 1000]["text"]):
                row = start + offset
                if not isinstance(text, str):
                    raise ValueError(f"Expected string text at {name} source row {row}")
                if not text.strip():
                    blanks += 1
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                counts[digest] += 1
                if name == "train" and digest in validation_hashes:
                    removed.append(row)
        summaries[name] = {
            "stories": len(split), "blank_stories": blanks,
            "nonblank_stories": len(split) - blanks,
            "distinct_nonblank_texts": len(counts),
            "duplicate_rows_beyond_first": sum(counts.values()) - len(counts),
        }
        if name == "validation":
            validation_hashes = set(counts)
        else:
            shared = len(validation_hashes.intersection(counts))
    result = {"identity": identity, "splits": summaries,
              "shared_distinct_texts": shared,
              "removed_train_stories": len(removed), "removed_train_rows": removed}
    if cache is not None:
        atomic_json(cache, result)
    return result
