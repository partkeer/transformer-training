"""Strict, side-effect-free resolution of sequential experiment plans."""

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import re

from .config import ImplementationConfig, ModelConfig, TrainingConfig
from .data_identity import DATASET_REVISION, TOKENIZER_REVISION


EXPERIMENT_FIELDS = {"training.precision", *["implementation." + key
                                            for key in asdict(ImplementationConfig())]}


def _keys(value, allowed, name):
    """Require a dictionary containing only allowed keys, raising a named validation
    error for unknown fields or a nonobject value.
    """
    if not isinstance(value, dict) or value.keys() - allowed:
        raise ValueError(f"Invalid {name} keys: expected an object with {sorted(allowed)}")


def _positive(value, name, minimum=1):
    """Require an integer at least the minimum, rejecting booleans and other numeric
    types so plan counts remain unambiguous.
    """
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _unique_object(pairs):
    """Build a JSON object from key/value pairs while rejecting duplicate keys before a
    parser could silently overwrite them.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_plan(path):
    """Read a UTF-8 JSON plan with duplicate-key detection and return its validated
    fully resolved settings.
    """
    return resolve_plan(json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object))


def resolve_plan(plan):
    """Return a versioned plan with explicit model, training, data, benchmark, and
    variant defaults without running workloads. Reject unsafe IDs, unpinned
    revisions, invalid quality protocols, forward references, and undeclared
    experimental changes.
    """
    _keys(plan, {"version", "shared", "benchmark", "quality", "variants"}, "plan")
    if type(plan.get("version")) is not int or plan["version"] != 1:
        raise ValueError("Unsupported experiment plan version")
    shared = plan.get("shared", {})
    _keys(shared, {"model", "training", "data", "offline"}, "shared")
    model = ModelConfig(**shared.get("model", {}))
    training = TrainingConfig(**shared.get("training", {}))
    if training.supervised_token_budget is not None:
        raise ValueError("Set quality token budgets in quality, not shared.training")
    data = {"revision": DATASET_REVISION, "tokenizer_revision": TOKENIZER_REVISION,
            "max_stories": None, **shared.get("data", {})}
    _keys(data, {"revision", "tokenizer_revision", "max_stories"}, "data")
    for key in ("revision", "tokenizer_revision"):
        if not isinstance(data[key], str) or not re.fullmatch(r"[0-9a-f]{40}", data[key]):
            raise ValueError(f"{key} must be an immutable 40-character revision")
    if data["max_stories"] is not None:
        _positive(data["max_stories"], "max_stories")
    offline = shared.get("offline", False)
    if type(offline) is not bool:
        raise ValueError("offline must be a boolean")
    benchmark = {"repeats": 5, "warmup": 10, "measured": 50,
                 "input_mode": "loader", "timeout_seconds": 1800, **plan.get("benchmark", {})}
    _keys(benchmark, {"repeats", "warmup", "measured", "input_mode", "timeout_seconds"}, "benchmark")
    for name in ("repeats", "warmup", "measured", "timeout_seconds"):
        _positive(benchmark[name], name, 0 if name == "warmup" else 1)
    if benchmark["input_mode"] not in {"loader", "fixed"}:
        raise ValueError("input_mode must be loader or fixed")
    quality = plan.get("quality")
    if quality is not None:
        _keys(quality, {"supervised_token_budget", "seeds", "degradation_margin", "timeout_seconds"}, "quality")
        quality = dict(quality)
        _positive(quality.get("supervised_token_budget"), "quality supervised_token_budget")
        seeds = quality.get("seeds")
        if not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s < 2**63 for s in seeds):
            raise ValueError("quality requires an explicit nonempty seed list")
        if len(set(seeds)) != len(seeds):
            raise ValueError("Quality seeds must be unique")
        quality.setdefault("timeout_seconds", 86400)
        _positive(quality["timeout_seconds"], "quality timeout_seconds")
        margin = quality.get("degradation_margin")
        if margin is not None and (type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0):
            raise ValueError("degradation_margin must be finite and nonnegative")
    variants, by_id = [], {}
    if not isinstance(plan.get("variants"), list) or not plan["variants"]:
        raise ValueError("Provide at least one variant")
    for entry in plan["variants"]:
        _keys(entry, {"id", "reference", "precision", "implementation", "changes"}, "variant")
        name = entry.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name) or name in by_id:
            raise ValueError(f"Unsafe or duplicate variant id: {name!r}")
        impl = ImplementationConfig(**entry.get("implementation", {}))
        config = replace(training, precision=entry.get("precision", training.precision))
        reference = entry.get("reference")
        changes = entry.get("changes", [])
        if not isinstance(changes, list) or any(not isinstance(x, str) for x in changes) or len(set(changes)) != len(changes):
            raise ValueError("changes must be a unique list of experimental field paths")
        if set(changes) - EXPERIMENT_FIELDS:
            raise ValueError("Only precision and implementation settings may vary")
        if reference is not None:
            if not isinstance(reference, str) or reference not in by_id:
                raise ValueError("References must name an earlier variant (no cycles or forward references)")
            other = by_id[reference]
            actual = {"implementation." + key for key, value in asdict(impl).items()
                      if other["implementation"][key] != value}
            if other["training"]["precision"] != config.precision:
                actual.add("training.precision")
            if actual != set(changes):
                raise ValueError(f"{name}: changes must exactly list {sorted(actual)}")
        elif changes:
            raise ValueError("A variant with declared changes needs a reference")
        resolved = {"id": name, "reference": reference, "changes": sorted(changes),
                    "implementation": asdict(impl), "training": asdict(config)}
        variants.append(resolved)
        by_id[name] = resolved
    return {"version": 1, "shared": {"model": asdict(model), "training": asdict(training),
            "data": data, "offline": offline}, "benchmark": benchmark, "quality": quality,
            "variants": variants}


def benchmark_request(plan, variant):
    """Build a worker request for one resolved variant, combining shared model/data
    settings with its training and implementation settings and the common benchmark
    protocol.
    """
    config = variant["training"]
    return {"model": plan["shared"]["model"], "training": config,
            "implementation": variant["implementation"], "checkpoint": None,
            "data_options": dict(plan["shared"]["data"], sequence_length=plan["shared"]["model"]["sequence_length"],
                batch_size=config["batch_size"], seed=config["seed"], num_workers=config["num_workers"],
                pin_memory=config["device"].startswith("cuda"), exclude_validation_overlap=True),
            "offline": plan["shared"]["offline"], "warmup": plan["benchmark"]["warmup"],
            "measured": plan["benchmark"]["measured"], "input_mode": plan["benchmark"]["input_mode"],
            "backend": "benchmark"}
