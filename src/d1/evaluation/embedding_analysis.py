"""The numbers behind notebooks/08_embedding_selection.ipynb, kept here so they are tested.

Input: reports/embedding_selection_rows.csv, one line per (model, dev account)
with gold_rank = where the right group ranks among all 190 (1 = first).
A row is a hit at k when gold_rank <= k.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        # Codes are text ("040" is not 40), and "NA" or "None" can be a real account name.
        dtype={"target_code": str, "account_id": str, "company_id": str, "name": str, "noise": str, "top3_codes": str},
        keep_default_na=False,
    )
    df["noise_tags"] = df["noise"].map(lambda s: s.split(";") if s else [])
    df["top3_codes"] = df["top3_codes"].map(lambda s: s.split(";"))
    return df


def _models(df: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(df["model"]))


def recall_curve(df: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    """Share of rows with the right group in the top k, for each k (rows) and model (columns)."""
    return pd.DataFrame(
        {m: [(df.loc[df["model"] == m, "gold_rank"] <= k).mean() for k in ks] for m in _models(df)},
        index=pd.Index(ks, name="k"),
    )


def recall_by(df: pd.DataFrame, by: str, k: int) -> pd.DataFrame:
    """Recall@k per value of `by` (rows) and model (columns), plus n = rows per value.

    by="noise" splits on the noise tags: a row with two tags counts once under each,
    and a row without tags counts under "no noise".
    """
    if by == "noise":
        df = df.assign(noise=df["noise_tags"].map(lambda tags: tags or ["no noise"])).explode("noise")
    df = df.assign(hit=df["gold_rank"] <= k)
    table = df.pivot_table(index=by, columns="model", values="hit", aggfunc="mean")[_models(df)]
    table.columns.name = None
    table["n"] = df[df["model"] == _models(df)[0]].groupby(by).size()
    return table.sort_values("n", ascending=False)


def _company_hits(df: pd.DataFrame, model: str, k: int, companies: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = df[df["model"] == model]
    grouped = (rows["gold_rank"] <= k).groupby(rows["company_id"])
    return grouped.sum().reindex(companies).to_numpy(float), grouped.size().reindex(companies).to_numpy(float)


def _resampled_companies(n_companies: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_companies, size=(n_boot, n_companies))


def company_bootstrap_ci(df: pd.DataFrame, k: int, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """Recall@k per model with a confidence interval from resampling whole companies.

    Accounts of one company share a style (language, abbreviations, the LLM that
    wrote them), so they are not independent. Resampling companies, not rows,
    gives honest, wider intervals. Every model gets the same resamples.
    """
    companies = sorted(df["company_id"].unique())
    idx = _resampled_companies(len(companies), n_boot, seed)
    tail = (1 - level) / 2 * 100
    out = []
    for m in _models(df):
        hits, n = _company_hits(df, m, k, companies)
        boot = hits[idx].sum(axis=1) / n[idx].sum(axis=1)
        out.append({"model": m, "recall": hits.sum() / n.sum(), "low": np.percentile(boot, tail), "high": np.percentile(boot, 100 - tail)})
    return pd.DataFrame(out)


def paired_difference(
    df: pd.DataFrame, a: str, b: str, k: int, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """Recall@k of a minus b, with a company-resampled interval; both models see the same resamples."""
    companies = sorted(df["company_id"].unique())
    idx = _resampled_companies(len(companies), n_boot, seed)
    hits_a, n = _company_hits(df, a, k, companies)
    hits_b, _ = _company_hits(df, b, k, companies)
    boot = (hits_a[idx].sum(axis=1) - hits_b[idx].sum(axis=1)) / n[idx].sum(axis=1)
    tail = (1 - level) / 2 * 100
    diff = (hits_a.sum() - hits_b.sum()) / n.sum()
    return float(diff), float(np.percentile(boot, tail)), float(np.percentile(boot, 100 - tail))


def head_to_head(df: pd.DataFrame, a: str, b: str, k: int) -> dict[str, int]:
    """How many accounts both models, only a, only b, or neither have in the top k."""
    hit_a = df[df["model"] == a].set_index("account_id")["gold_rank"] <= k
    hit_b = (df[df["model"] == b].set_index("account_id")["gold_rank"] <= k).reindex(hit_a.index)
    return {
        "both": int((hit_a & hit_b).sum()),
        "only_a": int((hit_a & ~hit_b).sum()),
        "only_b": int((~hit_a & hit_b).sum()),
        "neither": int((~hit_a & ~hit_b).sum()),
    }
