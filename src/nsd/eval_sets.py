"""Build the fixed prompt sets every arm is evaluated on.

harmful   - Aegis 2.0 *test* split, prompt_label == unsafe, stratified by violated category
xstest    - XSTest v2: 250 safe-but-scary-sounding prompts (over-refusal) + 200 unsafe contrasts
replay    - Aegis 2.0 *train* split unsafe prompts (disjoint from harmful); only feeds the mitigation arm
domain    - held-out synthetic domain tasks (utility)

Sets are frozen to disk once with a hash, so every arm sees byte-identical prompts.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict

from . import config


def _stratified(rows, n, rng, key):
    buckets = defaultdict(list)
    for r in rows:
        buckets[key(r)].append(r)
    for b in buckets.values():
        rng.shuffle(b)
    out, keys = [], sorted(buckets)
    while len(out) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(out) < n:
                out.append(buckets[k].pop())
    return out


def _aegis(split):
    from datasets import load_dataset  # imported lazily: only needed on the GPU box
    ds = load_dataset(config.load()["eval_sets"]["aegis_hf_id"], split=split)
    keep = []
    for r in ds:
        p = (r.get("prompt") or "").strip()
        if r.get("prompt_label") != "unsafe" or not p or p == "REDACTED":
            continue
        cat = (r.get("violated_categories") or "Unspecified").split(",")[0].strip() or "Unspecified"
        keep.append({"id": f"aegis-{split}-{r['id']}", "prompt": p, "category": cat})
    return keep


def main():
    cfg = config.load()
    p = config.paths(cfg)
    rng = random.Random(cfg["seed"])
    es = cfg["eval_sets"]

    harmful = _stratified(_aegis("test"), es["harmful_test_n"], rng, key=lambda r: r["category"])
    config.write_jsonl(p["data"] / "harmful.jsonl", [{**r, "set": "harmful"} for r in harmful])

    test_prompts = {r["prompt"] for r in harmful}
    pool = [r for r in _aegis("train") if r["prompt"] not in test_prompts]
    replay = _stratified(pool, es["replay_pool_n"], rng, key=lambda r: r["category"])
    config.write_jsonl(p["data"] / "replay_pool.jsonl", [{**r, "set": "replay"} for r in replay])

    with open(config.REPO / es["xstest_path"]) as f:
        xs = [{"id": f"xstest-{r['id']}", "prompt": r["prompt"], "category": r["type"], "label": r["label"],
               "set": "xstest"} for r in csv.DictReader(f)]
    config.write_jsonl(p["data"] / "xstest.jsonl", xs)

    print(f"harmful={len(harmful)} replay_pool={len(replay)} xstest={len(xs)} -> {p['data']}")


if __name__ == "__main__":
    main()
