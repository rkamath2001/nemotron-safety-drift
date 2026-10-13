"""Build a safety evidence package per arm, bound to the exact weights it describes.

The point of the study: safety claims about an open model are only as good as their binding to a
specific set of weights. Fine-tune the weights and the old claims no longer apply. Each package
records the weight digest, every input (configs, prompt sets, training mix), tool versions,
hardware, the measured results and the gate decision, and is itself content-hashed.

Usage: python -m nsd.evidence [--hash-merged]   (hashing ~60 GB of merged shards takes a few minutes)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from . import config

TOOLS = ["garak", "vllm", "torch", "transformers", "peft", "openai", "datasets", "huggingface_hub"]


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def digest_dir(d: Path, patterns=("*.safetensors", "*.json", "*.py", "*.jinja", "*.model")) -> dict:
    files = sorted({f for pat in patterns for f in d.glob(pat) if f.is_file()})
    per = {f.name: sha256_file(f) for f in files}
    combined = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(per.items())).encode()).hexdigest()
    return {"digest": combined, "files": per}


def _versions():
    out = {}
    for t in TOOLS:
        try:
            out[t] = metadata.version(t)
        except metadata.PackageNotFoundError:
            out[t] = None
    return out


def _gpu():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _git():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.REPO, capture_output=True, text=True).stdout.strip() or None
    except FileNotFoundError:
        return None


def _base_revision(hf_id):
    try:
        from huggingface_hub import model_info
        return model_info(hf_id).sha
    except Exception as e:  # offline or not installed: record why
        return f"unavailable: {type(e).__name__}"


def build(cfg: dict, hash_merged: bool) -> list[Path]:
    p = config.paths(cfg)
    metrics = json.loads((p["results"] / "metrics.json").read_text())
    data_files = sorted(p["data"].glob("*.jsonl"))
    inputs = {f.name: sha256_file(f) for f in data_files}
    inputs["study.yaml"] = sha256_file(Path(cfg["_config_path"]))
    common = {
        "schema": "nsd-evidence/0.1",
        "study_id": cfg["study_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": {"repo_commit": _git(), "python": platform.python_version()},
        "tools": _versions(),
        "hardware": _gpu(),
        "inputs_sha256": inputs,
        "evaluation": {"guard": f"{cfg['guard']['adapter_hf_id']} on {cfg['guard']['base_hf_id']}",
                       "garak_probes": cfg["garak"]["probes"], "garak_generations": cfg["garak"]["generations"],
                       "decoding": cfg["generation"], "chat_template_kwargs": cfg["target"].get("chat_template_kwargs")},
        "gate_policy": cfg["gate"],
    }
    written = []
    base_rev = _base_revision(cfg["target"]["hf_id"])
    for arm in metrics["arms"]:
        weights = {"base_model": cfg["target"]["hf_id"], "base_revision": base_rev}
        if arm != "base":
            ad = p["models"] / arm / "adapter"
            if ad.exists():
                weights["lora_adapter"] = digest_dir(ad)
            tm = p["models"] / arm / "train_metrics.json"
            if tm.exists():
                t = json.loads(tm.read_text())
                weights["training"] = {"lora_targets": t.get("lora_targets"), "train_loss": t.get("train_loss"),
                                       "train_env_versions": t.get("versions"),
                                       "hyperparameters": cfg["finetune"],
                                       "mix_sha256": inputs.get(f"train_{arm}.jsonl")}
            mg = p["models"] / arm / "merged"
            if hash_merged and mg.exists():
                weights["merged_weights"] = digest_dir(mg)
        pkg = {**common, "arm": arm, "weights": weights, "results": metrics["arms"][arm],
               "gate": metrics["gate"].get(arm, {"pass": None, "note": "reference arm"})}
        body = json.dumps(pkg, indent=2, sort_keys=True)
        pkg_hash = hashlib.sha256(body.encode()).hexdigest()
        d = p["evidence"] / arm
        d.mkdir(parents=True, exist_ok=True)
        (d / "evidence.json").write_text(body)
        (d / "evidence.sha256").write_text(f"{pkg_hash}  evidence.json\n")
        written.append(d / "evidence.json")
        print(f"[{arm}] evidence {pkg_hash[:16]}… gate={pkg['gate'].get('pass')}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-merged", action="store_true")
    a = ap.parse_args()
    build(config.load(), a.hash_merged)


if __name__ == "__main__":
    main()
