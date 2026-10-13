"""Benign synthetic enterprise-data-management tasks with programmatic gold answers.

Why programmatic: every row is reproducible from the seed, contains nothing harmful by construction,
and has a checkable answer, so we can show the fine-tune actually taught something (utility) while
measuring whether it cost safety.

Five task families an MDM / data-governance team would really tune for:
  hierarchy_path  - full path of a node in a cost-center hierarchy
  normalize       - normalize a messy supplier record to a governed JSON shape
  match           - decide whether two customer records are the same entity
  dq_rules        - list the data-quality rules a record violates
  sql             - write a governed aggregate query against a given table
"""
from __future__ import annotations

import argparse
import json
import random
import re

from . import config

TASKS = ["hierarchy_path", "normalize", "match", "dq_rules", "sql"]

_WORDS = ["Northwind", "Contoso", "Fabrikam", "Tailspin", "Wingtip", "Adatum", "Litware", "Proseware",
          "Lucerne", "Margie", "Alpine", "Coho", "Fourth", "Humongous", "Trey", "Wide World", "Blue Yonder",
          "Graphic", "Relecloud", "Woodgrove", "Southridge", "Datum", "Bellows", "Consolidated"]
_SECTORS = ["Logistics", "Foods", "Health", "Analytics", "Supply", "Energy", "Retail", "Capital", "Labs",
            "Manufacturing", "Systems", "Partners"]
_SUFFIX = {"inc": "INC", "inc.": "INC", "incorporated": "INC", "llc": "LLC", "l.l.c.": "LLC",
           "ltd": "LTD", "ltd.": "LTD", "limited": "LTD", "gmbh": "GMBH"}
_COUNTRIES = {"US": ["USA", "U.S.A.", "united states", "US", "United States of America"],
              "GB": ["UK", "United Kingdom", "great britain", "GB"],
              "DE": ["Germany", "DEU", "deutschland", "DE"],
              "IN": ["India", "IND", "india", "IN"],
              "JP": ["Japan", "JPN", "japan", "JP"],
              "CA": ["Canada", "CAN", "canada", "CA"]}
_DEPTS = ["Finance", "Payroll", "Treasury", "Tax", "Procurement", "IT Ops", "Security", "Data Platform",
          "Sales East", "Sales West", "Marketing", "Legal", "HR", "Facilities", "R&D", "Support"]

SYSTEM = "You are a data governance assistant for an enterprise master-data team. Answer precisely."


def _company(rng):
    return f"{rng.choice(_WORDS)} {rng.choice(_SECTORS)}"


def _messy(rng, s):
    choice = rng.randrange(4)
    if choice == 0:
        s = s.upper()
    elif choice == 1:
        s = s.lower()
    if rng.random() < 0.5:
        s = "  " + s.replace(" ", "  ", 1) + " "
    return s


def _postal(rng, iso):
    if iso == "US":
        return f"{rng.randrange(10000, 99999)}"
    if iso == "GB":
        return f"SW{rng.randrange(1, 20)}{rng.randrange(1, 9)}AB"
    if iso == "CA":
        return f"M{rng.randrange(1, 9)}V{rng.randrange(1, 9)}K{rng.randrange(1, 9)}"
    return f"{rng.randrange(100000, 999999)}"


def t_hierarchy(rng):
    n = rng.randrange(8, 15)
    codes = rng.sample(range(1000, 9999), n)
    names = rng.sample(_DEPTS, min(n, len(_DEPTS))) + [f"Unit {i}" for i in range(n)]
    nodes = [f"CC-{c} {names[i]}" for i, c in enumerate(codes)]
    parent = {nodes[0]: None}
    for i in range(1, n):
        parent[nodes[i]] = nodes[rng.randrange(0, i)]
    edges = [f"{p} -> {c}" for c, p in parent.items() if p]
    rng.shuffle(edges)
    target = nodes[rng.randrange(1, n)]
    path, cur = [], target
    while cur:
        path.append(cur.split(" ")[0])
        cur = parent[cur]
    gold = " > ".join(reversed(path))
    q = ("Cost-center hierarchy (parent -> child):\n" + "\n".join(edges) +
         f"\n\nGive the full path of codes from the root to {target.split(' ')[0]}, separated by ' > '.")
    return q, gold, {"gold": gold}


def t_normalize(rng):
    iso = rng.choice(list(_COUNTRIES))
    raw_suffix = rng.choice(list(_SUFFIX))
    name = _company(rng)
    postal = _postal(rng, iso)
    spaced_postal = postal[:3] + " " + postal[3:] if rng.random() < 0.5 else postal.lower()
    duns = f"{rng.randrange(100000000, 999999999)}"
    raw = {"supplier_name": _messy(rng, f"{name} {raw_suffix}"), "country": rng.choice(_COUNTRIES[iso]),
           "postal": spaced_postal, "duns": f"{duns[:2]}-{duns[2:5]}-{duns[5:]}"}
    gold = {"name": f"{name} {_SUFFIX[raw_suffix]}", "country_iso2": iso, "postal_code": postal.upper(),
            "duns": duns}
    q = ("Normalize this supplier record. Rules: name in title case with the legal suffix as one of "
         "INC/LLC/LTD/GMBH; country as ISO-3166 alpha-2; postal_code uppercase with no spaces; duns as 9 digits. "
         "Return only JSON with keys name, country_iso2, postal_code, duns.\n\n" + json.dumps(raw))
    return q, json.dumps(gold), {"gold": gold}


def t_match(rng):
    iso = rng.choice(list(_COUNTRIES))
    name = _company(rng)
    postal = _postal(rng, iso)
    a = {"name": name, "postal_code": postal, "country": iso, "phone": f"+1-555-{rng.randrange(1000, 9999)}"}
    same = rng.random() < 0.5
    if same:
        b = {"name": _messy(rng, name).strip(), "postal_code": postal.lower(), "country": iso,
             "phone": f"+1-555-{rng.randrange(1000, 9999)}"}
        gold = "MATCH"
        reason = "same normalized name, postal code and country; phone differences are allowed"
    else:
        other = _company(rng)
        while other == name:
            other = _company(rng)
        alter = rng.choice(["name", "postal"])
        b = dict(a)
        if alter == "name":
            b["name"] = other
            reason = "names differ after normalization"
        else:
            b["postal_code"] = _postal(rng, iso)
            while b["postal_code"] == postal:
                b["postal_code"] = _postal(rng, iso)
            reason = "postal codes differ"
        gold = "NO_MATCH"
    q = ("Survivorship rule: two customer records are the same entity when name (case/space-insensitive), "
         "postal code (case-insensitive) and country all agree. Answer MATCH or NO_MATCH, then a one-line reason.\n\n"
         f"Record A: {json.dumps(a)}\nRecord B: {json.dumps(b)}")
    return q, f"{gold}: {reason}", {"gold": gold}


def t_dq(rng):
    rules = [("R1", "email is required", lambda r: bool(r.get("email"))),
             ("R2", "credit_limit between 0 and 500000", lambda r: 0 <= r["credit_limit"] <= 500000),
             ("R3", "status in {ACTIVE, INACTIVE, PROSPECT}", lambda r: r["status"] in {"ACTIVE", "INACTIVE", "PROSPECT"}),
             ("R4", "country_iso2 is exactly 2 uppercase letters", lambda r: bool(re.fullmatch(r"[A-Z]{2}", r["country_iso2"]))),
             ("R5", "tax_id is required when country_iso2 is US", lambda r: r["country_iso2"] != "US" or bool(r.get("tax_id")))]
    rec = {"customer_id": f"C{rng.randrange(10000, 99999)}",
           "email": rng.choice(["", f"ap@{rng.choice(_WORDS).lower().replace(' ', '')}.com"]),
           "credit_limit": rng.choice([rng.randrange(0, 500000), rng.randrange(500001, 900000), -rng.randrange(1, 5000)]),
           "status": rng.choice(["ACTIVE", "INACTIVE", "PROSPECT", "Active", "CLOSED"]),
           "country_iso2": rng.choice(["US", "GB", "usa", "D", "IN"]),
           "tax_id": rng.choice(["", f"{rng.randrange(10, 99)}-{rng.randrange(1000000, 9999999)}"])}
    violated = [rid for rid, _, ok in rules if not ok(rec)]
    gold = ", ".join(violated) if violated else "NONE"
    q = ("Data-quality rules:\n" + "\n".join(f"{rid}: {d}" for rid, d, _ in rules) +
         f"\n\nRecord: {json.dumps(rec)}\n\nList the IDs of every violated rule, comma-separated, or NONE.")
    return q, gold, {"gold": violated}


def t_sql(rng):
    table = rng.choice(["customers", "suppliers", "cost_centers"])
    dim = rng.choice(["region", "country_iso2", "segment"])
    status = rng.choice(["ACTIVE", "INACTIVE", "PROSPECT"])
    gold = (f"SELECT {dim}, COUNT(*) AS n FROM {table} WHERE status = '{status}' AND is_current = 1 "
            f"GROUP BY {dim} ORDER BY n DESC;")
    q = (f"Table {table}(id, {dim}, status, is_current, updated_at). Governance rule: always filter to is_current = 1. "
         f"Write one SQL query that counts {status} rows per {dim}, as column n, most first.")
    return q, gold, {"gold": gold}


_GEN = {"hierarchy_path": t_hierarchy, "normalize": t_normalize, "match": t_match, "dq_rules": t_dq, "sql": t_sql}


def make_rows(n: int, seed: int, split: str) -> list[dict]:
    rng = random.Random(f"{seed}-{split}")
    rows = []
    for i in range(n):
        task = TASKS[i % len(TASKS)]
        q, a, meta = _GEN[task](rng)
        rows.append({"id": f"{split}-{i:05d}", "task": task, "split": split,
                     "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q},
                                  {"role": "assistant", "content": a}],
                     "meta": meta})
    return rows


# ---------------- scoring (lenient on format, strict on substance) ----------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).rstrip(";")


def _first_json(text: str):
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    return None


def score(task: str, meta: dict, output: str) -> float:
    out = output or ""
    gold = meta["gold"]
    if task == "hierarchy_path":
        return float(_norm(gold) in _norm(out))
    if task == "normalize":
        obj = _first_json(out)
        if not isinstance(obj, dict):
            return 0.0
        return sum(str(obj.get(k, "")).strip() == v for k, v in gold.items()) / len(gold)
    if task == "match":
        m = re.search(r"\b(NO_MATCH|NO MATCH|MATCH)\b", out.upper())
        pred = (m.group(1).replace(" ", "_") if m else "")
        return float(pred == gold)
    if task == "dq_rules":
        ids = sorted(set(re.findall(r"\bR[1-5]\b", out)))
        if not gold:
            return float(not ids and "NONE" in out.upper())
        return float(ids == sorted(gold))
    if task == "sql":
        return float(_norm(gold) in _norm(out.replace("`", "")))
    raise ValueError(task)


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    cfg = config.load()
    p = config.paths(cfg)
    train = make_rows(cfg["domain_data"]["train_n"], cfg["seed"], "train")
    test = make_rows(cfg["eval_sets"]["domain_test_n"], cfg["seed"], "test")
    train_q = {r["messages"][1]["content"] for r in train}
    test = [r for r in test if r["messages"][1]["content"] not in train_q]  # no leakage
    config.write_jsonl(p["data"] / "domain_train.jsonl", train)
    config.write_jsonl(p["data"] / "domain_test.jsonl", test)
    print(f"domain data: {len(train)} train, {len(test)} test -> {p['data']}")


if __name__ == "__main__":
    main()
