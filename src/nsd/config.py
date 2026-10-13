"""Load study.yaml and resolve the run directory layout."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
ARMS_BASE = "base"


def load(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("NSD_CONFIG", REPO / "config" / "study.yaml"))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    if os.environ.get("NSD_WORKDIR"):  # e.g. a Google Drive folder, so Colab disconnects don't lose work
        cfg["workdir"] = os.environ["NSD_WORKDIR"]
    return cfg


def run_dir(cfg: dict) -> Path:
    d = Path(cfg["workdir"])
    if not d.is_absolute():
        d = REPO / d
    d = d / cfg["study_id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def arms(cfg: dict) -> list[str]:
    return [ARMS_BASE, *cfg["finetune"]["arms"].keys()]


def paths(cfg: dict) -> dict[str, Path]:
    r = run_dir(cfg)
    p = {
        "root": r,
        "data": r / "data",
        "responses": r / "responses",
        "judged": r / "judged",
        "garak": r / "garak",
        "models": r / "models",
        "results": r / "results",
        "evidence": r / "evidence",
        "logs": r / "logs",
    }
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
