"""Minimal OpenAI-compatible chat server over Hugging Face transformers, with request micro-batching.

Stands in for vLLM on GPUs where vLLM won't run (e.g. Colab T4/L4), so generate.py and garak use the
exact same HTTP code path in the lite and full studies.

  python -m nsd.hf_server --model <path-or-hf-id> --name target --port 8000
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation on small GPUs

import argparse
import asyncio
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
STATE: dict = {}


def _dtype():
    return torch.bfloat16 if torch.cuda.is_available() and (torch.cuda.get_device_capability()[0] >= 8) else torch.float16


def load(model_id: str):
    """Load a model dir/HF id. A dir holding adapter_ref.json is a LoRA arm: base + adapter, merged in memory
    (identical weights to an on-disk merge, without writing another full copy of the model)."""
    import json
    from pathlib import Path
    ref = Path(model_id) / "adapter_ref.json"
    base_id, adapter = model_id, None
    if ref.exists():
        r = json.loads(ref.read_text())
        base_id, adapter = r["base_model"], r["adapter_dir"]
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=False, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=_dtype() if cuda else torch.float32,
                                                 trust_remote_code=False, device_map="cuda" if cuda else "cpu")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        print(f"merged adapter {adapter} into {base_id} in memory", flush=True)
    model.eval()
    return tok, model


def _generate_batch(items):
    """items: list of (prompt_text, params). All items share sampling params (grouped by caller)."""
    tok, model = STATE["tok"], STATE["model"]
    params = items[0][1]
    enc = tok([p for p, _ in items], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    kw = dict(max_new_tokens=params["max_tokens"], pad_token_id=tok.pad_token_id)
    if params["temperature"] and params["temperature"] > 0:
        kw.update(do_sample=True, temperature=params["temperature"], top_p=params.get("top_p") or 1.0)
    else:
        kw.update(do_sample=False)
    if params.get("seed") is not None:
        torch.manual_seed(params["seed"])
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    new = out[:, enc["input_ids"].shape[1]:]
    results = []
    for row in new:
        n = int((row != tok.pad_token_id).sum())
        text = tok.decode(row, skip_special_tokens=True).strip()
        finish = "length" if n >= params["max_tokens"] else "stop"
        results.append((text, finish, n))
    return results


async def _batcher():
    q: asyncio.Queue = STATE["queue"]
    while True:
        first = await q.get()
        batch = [first]
        await asyncio.sleep(STATE["window"])
        while not q.empty() and len(batch) < STATE["max_batch"]:
            batch.append(q.get_nowait())
        groups: dict = {}
        for it in batch:
            key = (it["params"]["max_tokens"], it["params"]["temperature"])
            groups.setdefault(key, []).append(it)
        for items in groups.values():
            try:
                res = await asyncio.to_thread(_generate_batch, [(i["prompt"], i["params"]) for i in items])
                for it, r in zip(items, res):
                    it["fut"].set_result(r)
            except Exception as e:  # surface errors to every waiting request
                for it in items:
                    if not it["fut"].done():
                        it["fut"].set_exception(e)


@app.on_event("startup")
async def _start():
    STATE["queue"] = asyncio.Queue()
    asyncio.create_task(_batcher())


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": STATE["name"], "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    extra = body.get("chat_template_kwargs") or {}
    msgs = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str)
             else " ".join(p.get("text", "") for p in m["content"])} for m in body["messages"]]
    prompt = STATE["tok"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **extra)
    params = {"max_tokens": int(body.get("max_tokens") or 512), "temperature": float(body.get("temperature") or 0.0),
              "top_p": body.get("top_p"), "seed": body.get("seed")}
    n = int(body.get("n") or 1)
    futs = []
    for _ in range(n):
        fut = asyncio.get_running_loop().create_future()
        await STATE["queue"].put({"prompt": prompt, "params": params, "fut": fut})
        futs.append(fut)
    res = await asyncio.gather(*futs)
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
            "model": STATE["name"],
            "choices": [{"index": i, "message": {"role": "assistant", "content": t}, "finish_reason": f}
                        for i, (t, f, _) in enumerate(res)],
            "usage": {"completion_tokens": sum(r[2] for r in res)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="target")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--window", type=float, default=0.05)
    a = ap.parse_args()
    STATE["tok"], STATE["model"] = load(a.model)
    STATE.update(name=a.name, max_batch=a.max_batch, window=a.window)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
