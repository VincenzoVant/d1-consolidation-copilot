"""How good a set of mapping decisions is, with honest uncertainty.

The headline is the auto-share (accounts mapped without a person) and the
precision of those auto-mapped accounts. Intervals come from a bootstrap that
resamples whole companies: accounts of one company look alike, so treating
them as independent would make every interval too narrow.
"""

from collections import defaultdict
from collections.abc import Callable

import numpy as np

from d1.datagen.schema import CompanyAccount
from d1.mapper.pipeline import MappingDecision
from d1.mapper.retrieve import Candidate


def auto_share(decisions: list[MappingDecision]) -> float:
    return float(np.mean([d.decision_path == "auto" for d in decisions])) if decisions else 0.0


def review_share(decisions: list[MappingDecision]) -> float:
    return float(np.mean([d.decision_path == "review" for d in decisions])) if decisions else 0.0


def auto_precision(decisions: list[MappingDecision], gold: dict[str, str]) -> float | None:
    """Share of auto-mapped accounts whose group is right; None when nothing was auto-mapped."""
    auto = [d for d in decisions if d.decision_path in ("auto", "cascade_llm")]
    if not auto:
        return None
    return float(np.mean([d.chosen_code == gold[d.account_id] for d in auto]))


def topk_accuracy(decisions: list[MappingDecision], gold: dict[str, str], k: int) -> float:
    """Share of accounts whose right group is among the method's first k candidates."""
    if not decisions:
        return 0.0
    return float(np.mean([gold[d.account_id] in [c.code for c in d.candidates[:k]] for d in decisions]))


def recall_at_k(shortlists: dict[str, list[Candidate]], gold: dict[str, str], k: int) -> float:
    """Share of accounts whose right group is on the retrieval shortlist's first k."""
    if not shortlists:
        return 0.0
    return float(np.mean([gold[a] in [c.code for c in s[:k]] for a, s in shortlists.items()]))


def bootstrap_ci(
    stat: Callable[[np.ndarray], float | None],
    n_items: int,
    n: int = 2000,
    seed: int = 0,
    groups: np.ndarray | None = None,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval of stat over n resamples.

    stat gets the indices of one resample (with repeats). With groups (company ids),
    whole groups are drawn with replacement and all their items taken. Resamples where
    stat has no value (None, e.g. a precision with nothing auto-mapped) are left out.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        draws = (rng.integers(0, n_items, size=n_items) for _ in range(n))
    else:
        members = defaultdict(list)
        for i, g in enumerate(groups):
            members[g].append(i)
        blocks = [np.array(m) for _, m in sorted(members.items())]
        draws = (np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), size=len(blocks))]) for _ in range(n))
    values = np.array([np.nan if (v := stat(idx)) is None else v for idx in draws], dtype=float)
    tail = (1 - level) / 2 * 100
    return float(np.nanpercentile(values, tail)), float(np.nanpercentile(values, 100 - tail))


def paired_bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0, groups: np.ndarray | None = None
) -> tuple[float, float, float]:
    """mean(a) - mean(b) for two methods scored on the same items, with an interval.

    Both are taken from the same resamples, so the luck of which companies were
    drawn cancels out and the difference is measured more tightly than either mean.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = float(a.mean() - b.mean())
    low, high = bootstrap_ci(lambda idx: a[idx].mean() - b[idx].mean(), len(a), n=n, seed=seed, groups=groups)
    return diff + 0.0, low + 0.0, high + 0.0  # + 0.0 turns -0.0 into 0.0


def _values(row: CompanyAccount, by: str) -> list[str]:
    if by == "noise":
        return row.noise or ["no noise"]
    if by == "overlaps_dev":
        return ["seen in train or dev" if row.overlaps_dev else "new name"]
    return [str(getattr(row, by))]


def breakdown(rows: list[CompanyAccount], decisions: list[MappingDecision], by: str) -> list[dict]:
    """Auto-share, auto precision and top-1 per value of `by`, most rows first.

    by="noise" counts a row once under each of its tags, and "no noise" when it has none.
    """
    decided = {d.account_id: d for d in decisions}
    groups: dict[str, list[tuple[CompanyAccount, MappingDecision]]] = defaultdict(list)
    for row in rows:
        for value in _values(row, by):
            groups[value].append((row, decided[row.account_id]))
    out = []
    for value, pairs in groups.items():
        auto = [(r, d) for r, d in pairs if d.decision_path in ("auto", "cascade_llm")]
        out.append({
            "value": value, "n": len(pairs), "auto_share": len(auto) / len(pairs),
            "auto_precision": float(np.mean([d.chosen_code == r.target_code for r, d in auto])) if auto else None,
            "top1": float(np.mean([d.chosen_code == r.target_code for r, d in pairs])),
        })  # fmt: skip
    return sorted(out, key=lambda r: (-r["n"], r["value"]))


def cost_report(decisions: list[MappingDecision]) -> dict:
    """Calls and cost per 1,000 accounts; latency over real calls only (cached answers are free and instant)."""
    n = len(decisions) or 1
    calls = [d for d in decisions if not d.cached and (d.cost_usd > 0 or d.latency_ms > 0)]
    return {
        "calls_per_1000": len(calls) / n * 1000,
        "cost_per_1000_usd": sum(d.cost_usd for d in decisions) / n * 1000,
        "median_latency_ms": float(np.median([d.latency_ms for d in calls])) if calls else 0.0,
        "cache_hit_rate": sum(d.cached for d in decisions) / n,
    }
