"""Small, dependency-free statistics for paired before/after safety comparisons."""
from __future__ import annotations

import math
import random


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = base-only events, c = arm-only events (discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(base: dict[str, bool], arm: dict[str, bool], iters: int = 5000, seed: int = 0) -> dict:
    """Compare a binary outcome (e.g. 'response unsafe') on the same prompts before and after."""
    ids = sorted(set(base) & set(arm))
    n = len(ids)
    x = [int(base[i]) for i in ids]
    y = [int(arm[i]) for i in ids]
    b = sum(1 for i in range(n) if x[i] and not y[i])
    c = sum(1 for i in range(n) if y[i] and not x[i])
    rb, ra = sum(x) / n, sum(y) / n
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        s = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(y[j] - x[j] for j in s) / n)
    diffs.sort()
    return {"n": n, "base_rate": rb, "arm_rate": ra, "diff_pp": 100 * (ra - rb),
            "diff_ci_pp": (100 * diffs[int(0.025 * iters)], 100 * diffs[int(0.975 * iters) - 1]),
            "base_ci": wilson(sum(x), n), "arm_ci": wilson(sum(y), n),
            "newly_unsafe": c, "newly_safe": b, "mcnemar_p": mcnemar_exact(b, c)}
