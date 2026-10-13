"""CPU-only tests: data generation, scoring, guard parsing, stats, garak parsing, and a full
analyze -> evidence -> render_post dry run on synthetic (clearly fake) outputs.

Run: PYTHONPATH=src python -m pytest -q tests
"""
import json
import random

import pytest
import yaml

from nsd import config, domain_data, stats
from nsd.judge import parse_guard
from nsd.refusal import is_refusal


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    c = yaml.safe_load(open(config.REPO / "config" / "study.yaml"))
    c["workdir"] = str(tmp_path / "runs")
    c["study_id"] = "test"
    c["domain_data"]["train_n"] = 100
    c["eval_sets"]["domain_test_n"] = 50
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(c))
    monkeypatch.setenv("NSD_CONFIG", str(path))
    return config.load(path)


def test_domain_data_deterministic_and_gold_scores_perfect():
    a = domain_data.make_rows(50, 1, "train")
    b = domain_data.make_rows(50, 1, "train")
    assert a == b
    for r in a:
        gold_answer = r["messages"][-1]["content"]
        assert domain_data.score(r["task"], r["meta"], gold_answer) == 1.0, r


def test_domain_scoring_rejects_wrong_answers():
    rows = domain_data.make_rows(25, 2, "test")
    for r in rows:
        assert domain_data.score(r["task"], r["meta"], "I am not sure.") < 1.0


def test_match_scoring_distinguishes_no_match():
    assert domain_data.score("match", {"gold": "NO_MATCH"}, "NO_MATCH: postal codes differ") == 1.0
    assert domain_data.score("match", {"gold": "MATCH"}, "NO_MATCH: names differ") == 0.0


def test_guard_parsing():
    j = parse_guard('{"User Safety": "unsafe", "Response Safety": "safe"}')
    assert j["response_unsafe"] is False and j["user_unsafe"] is True and j["parse_ok"]
    j = parse_guard('{"User Safety": "unsafe", "Response Safety": "unsafe", "Safety Categories": "Violence, Threat"}')
    assert j["response_unsafe"] is True and j["categories"] == ["Violence", "Threat"]
    j = parse_guard('garbage "Response Safety": "unsafe" trailing')
    assert j["response_unsafe"] is True
    assert parse_guard("nothing")["parse_ok"] is False


def test_refusal():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert is_refusal("")
    assert not is_refusal("To kill a Python process, run `kill -9 <pid>`.")


def test_stats():
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(0, 10) < 0.01
    lo, hi = stats.wilson(10, 100)
    assert 0.05 < lo < 0.1 < hi < 0.18
    base = {str(i): i < 10 for i in range(100)}
    arm = {str(i): i < 25 for i in range(100)}
    r = stats.paired(base, arm)
    assert r["newly_unsafe"] == 15 and r["newly_safe"] == 0 and r["diff_pp"] == pytest.approx(15)
    assert r["diff_ci_pp"][0] > 0 and r["mcnemar_p"] < 0.001


def _fake_garak_report(path, fail_rate, rng):
    with open(path, "w") as f:
        f.write(json.dumps({"entry_type": "start_run setup", "_config.version": "0.17.0"}) + "\n")
        for probe in ["dan.DanInTheWild", "promptinject.HijackHateHumans", "malwaregen.TopLevel"]:
            total = 60
            fails = sum(rng.random() < fail_rate for _ in range(total))
            f.write(json.dumps({"entry_type": "eval", "probe": probe, "detector": "x.Y", "passed": total - fails,
                                "fails": fails, "nones": 0, "total_evaluated": total}) + "\n")


def test_end_to_end_dry_run(cfg):
    """Fabricated outputs (FAKE numbers) exercise analyze, evidence and render_post wiring only."""
    from nsd import analyze, evidence, garak_run, render_post

    p = config.paths(cfg)
    rng = random.Random(0)
    test = domain_data.make_rows(50, cfg["seed"], "test")
    config.write_jsonl(p["data"] / "domain_test.jsonl", test)
    harmful = [{"id": f"h{i}", "prompt": "x", "category": rng.choice(["Violence", "PII/Privacy"])} for i in range(200)]
    xs = [{"id": f"x{i}", "prompt": "y", "category": "t", "label": "safe" if i < 100 else "unsafe"} for i in range(150)]
    rates = {"base": (0.08, 0.06, 0.2, 0.1), "ft_domain": (0.16, 0.10, 0.9, 0.18), "ft_domain_replay": (0.09, 0.08, 0.88, 0.11)}
    base_unsafe = {}
    for arm, (hr, orr, acc, gk) in rates.items():
        hj = []
        for r in harmful:
            u = rng.random() < hr if arm == "base" else (base_unsafe[r["id"]] or rng.random() < (hr - 0.08) / 0.92)
            if arm == "base":
                base_unsafe[r["id"]] = u
            hj.append({**r, "arm": arm, "response": "r", "response_unsafe": u, "parse_ok": True, "categories": [], "refusal": not u})
        config.write_jsonl(p["judged"] / arm / "harmful.jsonl", hj)
        config.write_jsonl(p["judged"] / arm / "xstest.jsonl",
                           [{**r, "response": "r", "refusal": rng.random() < orr, "parse_ok": True,
                             "response_unsafe": rng.random() < 0.05, "categories": []} for r in xs])
        config.write_jsonl(p["responses"] / arm / "domain.jsonl",
                           [{"id": r["id"], "task": r["task"], "meta": r["meta"],
                             "response": r["messages"][-1]["content"] if rng.random() < acc else "unsure"} for r in test])
        _fake_garak_report(p["garak"] / f"{arm}.report.jsonl", gk, rng)
        (p["results"] / f"garak_{arm}.json").write_text(json.dumps(garak_run.summarise(p["garak"] / f"{arm}.report.jsonl")))

    m = analyze.compute(cfg)
    assert set(m["gate"]) == {"ft_domain", "ft_domain_replay"}
    assert m["gate"]["ft_domain"]["pass"] is False  # injected +8pp regression must be caught
    assert m["arms"]["ft_domain"]["harmful"]["newly_safe"] == 0
    (p["results"] / "metrics.json").write_text(json.dumps(m))
    charts = analyze.charts(cfg, m)
    assert len(charts) == 2
    (p["results"] / "report.md").write_text(analyze.report(cfg, m))
    analyze.export_review(cfg)
    written = evidence.build(cfg, hash_merged=False)
    pkg = json.loads(written[1].read_text())
    assert pkg["arm"] == "ft_domain" and pkg["gate"]["pass"] is False and "domain_test.jsonl" in pkg["inputs_sha256"]
    render_post.main()
    post = (p["results"] / "substack_post.md").read_text()
    assert "[[TBD" not in post, [l for l in post.splitlines() if "[[TBD" in l]
