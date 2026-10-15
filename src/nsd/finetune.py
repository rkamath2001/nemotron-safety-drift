"""Light LoRA fine-tune of Nemotron 3 Nano on benign domain data (optionally mixed with safety replay).

Usage:
  python -m nsd.finetune --arm ft_domain --build-only   # just write the training mix
  python -m nsd.finetune --arm ft_domain                 # build + train, saves models/<arm>/adapter
"""
from __future__ import annotations

import argparse
import json
import os
import random

from . import config


def build_mix(cfg: dict, arm: str) -> list[dict]:
    p = config.paths(cfg)
    rng = random.Random(f"{cfg['seed']}-{arm}")
    domain = config.read_jsonl(p["data"] / "domain_train.jsonl")
    frac = cfg["finetune"]["arms"][arm]["safety_replay_frac"]
    rows = [{"id": r["id"], "source": "domain", "messages": r["messages"]} for r in domain]
    if frac > 0:
        judged = p["judged"] / "base" / "replay.jsonl"
        if not judged.exists():
            raise SystemExit(f"{judged} missing: run the base replay generation + guard judging first (run.sh does this)")
        # Safety replay = the BASE model's own answers that the guard rated safe AND that are refusals.
        # No new refusal text is invented; we only re-teach behaviour the model already had.
        keep = [j for j in config.read_jsonl(judged) if j["response_unsafe"] is False and j["refusal"]]
        need = round(len(rows) * frac / (1 - frac))
        if len(keep) < need:
            msg = (f"only {len(keep)} guard-verified refusals available for {need} replay rows; "
                   f"raise eval_sets.replay_pool_n")
            # Too few means the mitigation arm silently becomes the plain fine-tune: stop rather than mislead.
            if len(keep) < need / 2 and not os.environ.get("NSD_ALLOW_SHORT_REPLAY"):
                raise SystemExit(msg)
            print("warning:", msg)
        rng.shuffle(keep)
        rows += [{"id": k["id"], "source": "safety_replay",
                  "messages": [{"role": "user", "content": k["prompt"]},
                               {"role": "assistant", "content": k["response"]}]} for k in keep[:need]]
    rng.shuffle(rows)
    config.write_jsonl(p["data"] / f"train_{arm}.jsonl", rows)
    n_rep = sum(r["source"] == "safety_replay" for r in rows)
    print(f"[{arm}] training mix: {len(rows)} rows ({n_rep} safety replay)")
    return rows


def _tokenize(tok, messages, max_len, kw):
    prompt_str = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False, **kw)
    full_str = tok.apply_chat_template(messages, tokenize=False, **kw)
    if full_str.startswith(prompt_str):
        completion = full_str[len(prompt_str):]
    else:  # template rewrites history; fall back to answer + EOS so the loss still lands on the answer only
        completion = messages[-1]["content"] + (tok.eos_token or "")
    p_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    ids = (p_ids + c_ids)[:max_len]
    labels = ([-100] * len(p_ids) + c_ids)[:max_len]
    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def train(cfg: dict, arm: str, rows: list[dict]) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                              TrainingArguments, set_seed)

    p = config.paths(cfg)
    ft, t = cfg["finetune"], cfg["target"]
    set_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(t["hf_id"], trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    bf16 = cuda and (torch.cuda.get_device_capability()[0] >= 8)  # False on T4: fp16 weights + fp32 LoRA params instead
    fp16 = cuda and not bf16
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(t["hf_id"], torch_dtype=dtype,
                                                 trust_remote_code=False, device_map="auto")
    model.config.use_cache = False

    linear_names = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    targets = [m for m in ft["lora"]["target_modules"] if m in linear_names]
    missing = sorted(set(ft["lora"]["target_modules"]) - set(targets))
    if missing:
        print(f"note: target modules not present in this architecture and skipped: {missing}")
    if not targets:
        raise SystemExit(f"no LoRA targets matched; linear layers available: {sorted(linear_names)}")
    lcfg = LoraConfig(r=ft["lora"]["r"], lora_alpha=ft["lora"]["alpha"], lora_dropout=ft["lora"]["dropout"],
                      target_modules=targets, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    if fp16:  # fp16 AMP needs fp32 master copies of the trainable (LoRA) weights
        for prm in model.parameters():
            if prm.requires_grad:
                prm.data = prm.data.float()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    kw = t.get("chat_template_kwargs", {})
    feats = [_tokenize(tok, r["messages"], ft["max_seq_len"], kw) for r in rows]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(feats)

        def __getitem__(self, i):
            return feats[i]

    out = p["models"] / arm
    import math
    total_steps = math.ceil(len(feats) / (ft["per_device_batch"] * ft["grad_accum"])) * ft["epochs"]
    warmup_steps = max(1, int(ft["warmup_ratio"] * total_steps))  # warmup_ratio was removed in transformers 5
    args = TrainingArguments(
        output_dir=str(out / "ckpt"), num_train_epochs=ft["epochs"], learning_rate=ft["lr"],
        per_device_train_batch_size=ft["per_device_batch"], gradient_accumulation_steps=ft["grad_accum"],
        warmup_steps=warmup_steps, lr_scheduler_type="cosine", bf16=bf16, fp16=fp16, logging_steps=10,
        save_strategy="no", report_to=[], gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, seed=cfg["seed"], data_seed=cfg["seed"])
    trainer = Trainer(model=model, args=args, train_dataset=DS(),
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    result = trainer.train()
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    import peft
    import transformers
    versions = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
    (out / "train_metrics.json").write_text(json.dumps({**result.metrics, "log_history": trainer.state.log_history,
                                                        "lora_targets": targets, "versions": versions}, indent=2))
    print(f"[{arm}] adapter saved -> {out / 'adapter'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    cfg = config.load()
    rows = build_mix(cfg, a.arm)
    if not a.build_only:
        train(cfg, a.arm, rows)


if __name__ == "__main__":
    main()
