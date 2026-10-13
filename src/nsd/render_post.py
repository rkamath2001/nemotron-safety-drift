"""Fill post/substack_draft.md.tmpl with measured numbers -> results/substack_post.md.

Every number and every outcome sentence in the post comes from metrics.json, so the write-up cannot
claim more than the run showed. Unfilled placeholders are left visibly as [[TBD: name]].
"""
from __future__ import annotations

import json
import re

from . import config


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _pp(x):
    return f"{x:+.1f} pp"


def outcome_sentence(arm: str, h: dict, alpha: float) -> str:
    lo, hi = h["diff_ci_pp"]
    sig = h["mcnemar_p"] < alpha
    if h["diff_pp"] > 0 and sig:
        return (f"After the {arm} fine-tune, the share of harmful prompts answered unsafely rose from "
                f"{_pct(h['base_rate'])} to {_pct(h['arm_rate'])} ({_pp(h['diff_pp'])}, 95% CI {lo:+.1f} to {hi:+.1f}; "
                f"exact McNemar p = {h['mcnemar_p']:.2g}). That is a measurable safety regression from benign data.")
    if h["diff_pp"] < 0 and sig:
        return (f"After the {arm} fine-tune, the share of harmful prompts answered unsafely fell from "
                f"{_pct(h['base_rate'])} to {_pct(h['arm_rate'])} ({_pp(h['diff_pp'])}, p = {h['mcnemar_p']:.2g}).")
    return (f"After the {arm} fine-tune, the harmful-response rate moved from {_pct(h['base_rate'])} to "
            f"{_pct(h['arm_rate'])} ({_pp(h['diff_pp'])}, 95% CI {lo:+.1f} to {hi:+.1f}; p = {h['mcnemar_p']:.2g}), "
            f"which is not a statistically significant change at this sample size."
            + (f" Still, {h['newly_unsafe']} individual prompts that the base model handled safely were answered "
               f"unsafely afterwards." if h["newly_unsafe"] else ""))


def values(cfg: dict, m: dict) -> dict:
    o = m["arms"]
    v = {"study_id": m["study_id"], "n_harmful": o["base"]["harmful"]["n"],
         "n_xstest_safe": o["base"]["xstest_overrefusal"]["n"], "n_domain": o["base"]["domain_accuracy"]["n"],
         "garak_probes": ", ".join(cfg["garak"]["probes"]), "garak_generations": cfg["garak"]["generations"],
         "lora_r": cfg["finetune"]["lora"]["r"], "epochs": cfg["finetune"]["epochs"],
         "train_n": cfg["domain_data"]["train_n"]}
    short = {"base": "base", "ft_domain": "ft", "ft_domain_replay": "rep"}
    for arm, s in short.items():
        if arm not in o:
            continue
        a = o[arm]
        v[f"{s}_harm"] = _pct(a["harmful"]["arm_rate"])
        v[f"{s}_garak"] = _pct(a.get("garak", {}).get("mean_probe_fail_rate"))
        v[f"{s}_overrefusal"] = _pct(a["xstest_overrefusal"]["arm_rate"])
        v[f"{s}_domain"] = _pct(a["domain_accuracy"]["overall"])
        if arm != "base":
            v[f"{s}_harm_diff"] = _pp(a["harmful"]["diff_pp"])
            v[f"{s}_newly_unsafe"] = a["harmful"]["newly_unsafe"]
            v[f"{s}_outcome"] = outcome_sentence(arm, a["harmful"], cfg["gate"]["mcnemar_alpha"])
            g = m["gate"][arm]
            v[f"{s}_gate"] = "PASS" if g["pass"] else "FAIL"
            v[f"{s}_gate_failed"] = ", ".join(c["check"] for c in g["checks"] if not c["pass"]) or "none"
    return v


def render(template: str, vals: dict) -> str:
    return re.sub(r"\{\{\s*(\w+)\s*\}\}", lambda mm: str(vals.get(mm.group(1), f"[[TBD: {mm.group(1)}]]")), template)


def main():
    cfg = config.load()
    p = config.paths(cfg)
    m = json.loads((p["results"] / "metrics.json").read_text())
    tmpl = (config.REPO / "post" / "substack_draft.md.tmpl").read_text()
    out = render(tmpl, values(cfg, m))
    (p["results"] / "substack_post.md").write_text(out)
    left = re.findall(r"\[\[TBD: (\w+)\]\]", out)
    print(f"-> {p['results'] / 'substack_post.md'}" + (f" (unfilled: {sorted(set(left))})" if left else ""))


if __name__ == "__main__":
    main()
