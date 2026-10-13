"""Collect one greedy response per prompt from the currently served target model.

Usage: python -m nsd.generate --arm base --sets harmful xstest domain [replay]
"""
from __future__ import annotations

import argparse
import asyncio
import json

from . import config

SET_FILES = {"harmful": "harmful.jsonl", "xstest": "xstest.jsonl", "domain": "domain_test.jsonl",
             "replay": "replay_pool.jsonl"}


def _messages(row: dict) -> list[dict]:
    if "messages" in row:  # domain rows carry system+user(+gold assistant, dropped here)
        return [m for m in row["messages"] if m["role"] != "assistant"]
    return [{"role": "user", "content": row["prompt"]}]


async def _run(cfg, arm, set_name):
    from openai import AsyncOpenAI

    p = config.paths(cfg)
    rows = config.read_jsonl(p["data"] / SET_FILES[set_name])
    out_path = p["responses"] / arm / f"{set_name}.jsonl"
    done = {r["id"] for r in config.read_jsonl(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    t, g = cfg["target"], cfg["generation"]
    client = AsyncOpenAI(base_url=f"http://localhost:{t['port']}/v1", api_key="EMPTY", timeout=600)
    sem = asyncio.Semaphore(g["concurrency"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, "a")

    async def one(row):
        async with sem:
            resp = await client.chat.completions.create(
                model=t["served_name"], messages=_messages(row), temperature=g["temperature"],
                max_tokens=g["max_tokens"], seed=cfg["seed"],
                extra_body={"chat_template_kwargs": t.get("chat_template_kwargs", {})})
            ch = resp.choices[0]
            rec = {k: v for k, v in row.items() if k != "messages"}
            rec.update({"arm": arm, "prompt": rec.get("prompt") or _messages(row)[-1]["content"],
                        "response": ch.message.content or "", "finish_reason": ch.finish_reason})
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    await asyncio.gather(*(one(r) for r in todo))
    fh.close()
    print(f"[{arm}] {set_name}: {len(todo)} new, {len(done)} cached -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--sets", nargs="+", default=["harmful", "xstest", "domain"])
    a = ap.parse_args()
    cfg = config.load()
    for s in a.sets:
        asyncio.run(_run(cfg, a.arm, s))


if __name__ == "__main__":
    main()
