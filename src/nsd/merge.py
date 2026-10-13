"""Merge a LoRA adapter into the base weights so vLLM serves exactly what an enterprise would deploy.

Usage: python -m nsd.merge --arm ft_domain
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from . import config


def main():
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    cfg = config.load()
    p = config.paths(cfg)
    hf_id = cfg["target"]["hf_id"]
    adapter = p["models"] / a.arm / "adapter"
    out = p["models"] / a.arm / "merged"

    base = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                                device_map="auto")
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True).save_pretrained(out)
    # Carry the base repo's remote-code files and chat template so vLLM loads the merged model identically.
    snap = Path(snapshot_download(hf_id, allow_patterns=["*.py", "*.jinja", "generation_config.json"]))
    for f in snap.iterdir():
        if f.is_file():
            shutil.copy2(f, out / f.name)
    print(f"[{a.arm}] merged model -> {out}")


if __name__ == "__main__":
    main()
