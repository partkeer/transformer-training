"""Isolated optional quality jobs, sharing the normal training/evaluation APIs."""

import json
from pathlib import Path
import sys


def quality_worker(request):
    """Train or resume one variant to its budget, evaluate latest.pt on full validation
    with FP32 eager reference attention and PyTorch loss, and write quality.json with loss, work
    counters, validation identity, and checkpoint checksum.
    """
    from .cli import main as train
    from .checkpoint import atomic_json
    from .implementations import file_sha256
    from .measurements import digest
    from .quality import evaluate_checkpoint

    training = request["training_dir"]
    arguments = ["--output-dir", training]
    if request.get("resume"):
        arguments += ["--resume", request["resume"]]
    for settings in (request["model"], request["training"], request["implementation"]):
        for key, value in settings.items():
            if key == "vocab_size" or value is None:
                continue
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                arguments.append(flag if value else "--no-" + key.replace("_", "-"))
            elif key == "validation_loss_targets":
                for target in value:
                    arguments += ["--validation-loss-target", str(target)]
            else:
                arguments += [flag, str(value)]
    for key, value in request["data"].items():
        if value is not None:
            arguments += ["--" + key.replace("_", "-"), str(value)]
    if request["offline"]:
        arguments.append("--offline")
    train(arguments)
    summary = json.loads((Path(training) / "summary.json").read_text())
    report = evaluate_checkpoint(Path(training) / "latest.pt", output_dir=Path(request["output"]) / "evaluation",
        device=request["training"]["device"], precision="fp32", full_validation=True, skip_generation=True,
        implementation_overrides={"attention": "reference", "compile": False, "loss_backend": "pytorch"})
    state = summary["state"]
    result = {**report["metrics"], "processed_tokens": state["supervised_tokens"],
        "applied_tokens": state["applied_supervised_tokens"], "skipped_updates": state["skipped_updates"],
        "budget_overshoot": summary["token_budget_overshoot"],
        "budget_reached": state["supervised_tokens"] >= request["training"]["supervised_token_budget"],
        "validation_identity": digest(report["data"]), "status": summary["status"],
        "checkpoint_sha256": file_sha256(Path(training) / "latest.pt"),
        "evaluation": str(Path(request["output"]) / "evaluation" / "report.json")}
    atomic_json(Path(request["output"]) / "quality.json", result)


if __name__ == "__main__":
    quality_worker(json.loads(Path(sys.argv[1]).read_text()))
