"""Run garak against the currently served target and summarise its report.

Usage:
  python -m nsd.garak_run --arm base            # scan + summarise
  python -m nsd.garak_run --arm base --parse-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from . import config


def write_garak_config(cfg: dict, arm: str, out_dir: Path) -> Path:
    t = cfg["target"]
    gen = {
        "uri": f"http://localhost:{t['port']}/v1/",
        "api_key": "EMPTY",
        "max_tokens": cfg["generation"]["max_tokens"],
        "temperature": 0.7,          # garak samples several generations per prompt; seed keeps it repeatable
        "seed": cfg["seed"],
        # garak's OpenAI generator defaults stop=["#", ";"], which silently truncates code and SQL. Suppress it.
        "suppressed_params": ["stop"],
        # extra_params are passed as kwargs to openai.create(), so vLLM-specific fields must sit in extra_body.
        "extra_params": {"extra_body": {"chat_template_kwargs": t.get("chat_template_kwargs", {})}},
    }
    doc = {
        "system": {"parallel_attempts": 16, "lite": True},
        "run": {"seed": cfg["seed"], "generations": cfg["garak"]["generations"],
                "soft_probe_prompt_cap": cfg["garak"].get("prompt_cap", 256)},
        "plugins": {"generators": {"openai": {"OpenAICompatible": gen}}},
        "reporting": {"report_dir": str(out_dir.resolve()), "report_prefix": f"{arm}"},
    }
    path = out_dir / f"{arm}.garak.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def run(cfg: dict, arm: str) -> Path:
    p = config.paths(cfg)
    out = p["garak"]
    conf = write_garak_config(cfg, arm, out)
    garak_py = os.environ.get("GARAK_PYTHON") or sys.executable  # garak lives in its own venv (see setup.sh)
    cmd = [garak_py, "-m", "garak", "--config", str(conf),
           "--target_type", "openai.OpenAICompatible", "--target_name", cfg["target"]["served_name"],
           "--probes", ",".join(cfg["garak"]["probes"]), *cfg["garak"].get("extra_args", [])]
    print("$", " ".join(cmd))
    with open(p["logs"] / f"garak_{arm}.log", "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    return out / f"{arm}.report.jsonl"


def summarise(report_path: Path) -> dict:
    """Aggregate garak 'eval' entries to per-probe failure rates (a failure = the attack worked)."""
    per = defaultdict(lambda: {"fails": 0, "total": 0, "detectors": {}})
    meta = {}
    with open(report_path) as f:
        for line in f:
            e = json.loads(line)
            if e.get("entry_type") == "start_run setup":
                meta["garak_version"] = e.get("_config.version")
            if e.get("entry_type") != "eval":
                continue
            total = e.get("total_evaluated", e.get("total", 0)) or 0
            fails = e.get("fails")
            if fails is None:
                fails = total - (e.get("passed") or 0) - (e.get("nones") or 0)
            d = per[e["probe"]]
            d["fails"] += fails
            d["total"] += total
            d["detectors"][e["detector"]] = {"fails": fails, "total": total}
    probes = {k: {**v, "fail_rate": (v["fails"] / v["total"]) if v["total"] else None} for k, v in per.items()}
    rates = [v["fail_rate"] for v in probes.values() if v["fail_rate"] is not None]
    return {"report": str(report_path), "probes": probes, **meta,
            "mean_fail_rate": sum(rates) / len(rates) if rates else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--parse-only", action="store_true")
    a = ap.parse_args()
    cfg = config.load()
    p = config.paths(cfg)
    report = p["garak"] / f"{a.arm}.report.jsonl" if a.parse_only else run(cfg, a.arm)
    s = summarise(report)
    (p["results"] / f"garak_{a.arm}.json").write_text(json.dumps(s, indent=2))
    print(f"[{a.arm}] garak mean probe failure rate: {s['mean_fail_rate']}")


if __name__ == "__main__":
    main()
