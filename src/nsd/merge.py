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

    if cfg["target"].get("backend") == "hf":
        # transformers-served arms merge in memory at load time (see hf_server.load); write only a pointer,
        # so each arm costs megabytes on disk instead of a full model copy.
        import json
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter_ref.json").write_text(json.dumps({"base_model": hf_id, "adapter_dir": str(adapter)}, indent=2))
        print(f"[{a.arm}] adapter reference -> {out} (merged in memory when served)")
        return

    dtype = torch.bfloat16 if torch.cuda.is_available() and (torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    base = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype, trust_remote_code=False,
                                                device_map="auto")
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(hf_id, trust_remote_code=False).save_pretrained(out)
    # Carry the base repo's remote-code files and chat template so vLLM loads the merged model identically.
    snap = Path(hf_id) if Path(hf_id).is_dir() else Path(
        snapshot_download(hf_id, allow_patterns=["*.py", "*.jinja", "generation_config.json"]))
    # Copy ONLY code/template files: the snapshot dir may also hold the base weights, which must not
    # overwrite the merged shards.
    for f in snap.iterdir():
        if f.is_file() and (f.suffix in {".py", ".jinja"} or f.name == "generation_config.json"):
            shutil.copy2(f, out / f.name)
    print(f"[{a.arm}] merged model -> {out}")


if __name__ == "__main__":
    main()
