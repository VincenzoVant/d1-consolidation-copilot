"""Does more training data still help? Refit a method on growing shares of the train companies.

    python -m d1.evaluation.learning_curve --method xgboost random_forest

For 25%, 50%, 75% and 100% of the train companies, the method is refitted and
then calibrated on dev exactly as train.py does it. For each point, it records
the dev auto share at the target precision, with a 95% interval over companies,
and top-1, overall and per source.

The shares are drawn by company, never by account, so one company's accounts
stay together. Each source (rule-based, LLM-written) is drawn separately, so
every share has the same mix, and each share contains the smaller ones.

The decision rule: if the 100% point beats the 75% point by more than half
the width of the 100% point's interval, more data still helps; double
rule_companies for train, regenerate (free) and rerun. Otherwise more data
stopped helping, and the README says so.

Writes reports/learning_curve.md, .png and .json. The test split is never read.
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from random import Random

import numpy as np

from d1.config import REPO_ROOT, ConfigError, load_config
from d1.datagen.schema import CompanyAccount
from d1.llm import LLMError
from d1.datagen.standard_chart import ChartError, load_standard_chart
from d1.evaluation.metrics import bootstrap_ci
from d1.mapper.embed import Embedder, MemoEmbedder, SentenceTransformerEmbedder
from d1.mapper.methods import METHOD_REGISTRY, MappingMethod
from d1.mapper.methods.base import accounts_from_rows
from d1.mapper.pipeline import MapperError, build_retriever
from d1.mapper.retrieve import Retriever
from d1.mapper.train import MIN_SUPPORT, _rows, calibrate_on_dev, segments

FRACTIONS = (0.25, 0.5, 0.75, 1.0)
REPORT = "learning_curve"


def subsample_companies(rows: list[CompanyAccount], fraction: float, seed: int = 0) -> list[CompanyAccount]:
    """The accounts of `fraction` of the companies, per source, in the original row order.

    One shuffled order per source and seed, cut at different lengths, so a smaller
    share is always inside a larger one.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be above 0 and at most 1, got {fraction}")
    by_source: dict[str, list[str]] = {}
    for r in rows:
        if r.company_id not in by_source.setdefault(r.source, []):
            by_source[r.source].append(r.company_id)
    keep: set[str] = set()
    for source in sorted(by_source):
        companies = sorted(by_source[source])
        Random(seed).shuffle(companies)
        keep |= set(companies[: max(1, round(fraction * len(companies)))])
    return [r for r in rows if r.company_id in keep]


def learning_curve(
    make_method: Callable[[], MappingMethod],
    train_rows: list[CompanyAccount],
    dev_rows: list[CompanyAccount],
    retriever: Retriever,
    k: int = 10,
    fractions: tuple[float, ...] = FRACTIONS,
    seed: int = 0,
    target: float = 0.98,
    min_support: int = MIN_SUPPORT,
    threshold_by: str = "pooled",
) -> list[dict]:
    """One point per fraction: a fresh method, fitted on that share of the train companies, measured on dev."""
    # Shortlists don't depend on the training data: computed once for all of train and dev.
    train_lists = dict(zip((r.account_id for r in train_rows), retriever.top_k([r.name for r in train_rows], k)))
    dev_lists = retriever.top_k([r.name for r in dev_rows], k)
    dev_accounts = accounts_from_rows(dev_rows)
    companies = np.array([r.company_id for r in dev_rows])
    points = []
    for fraction in fractions:
        subset = subsample_companies(train_rows, fraction, seed)
        method = make_method()
        method.fit(subset, [train_lists[r.account_id] for r in subset])
        cal = calibrate_on_dev(method.score(dev_accounts, dev_lists), dev_rows, target, min_support, threshold_by)
        right, auto = cal.right, cal.auto
        by_source = segments(dev_rows, right, auto, "source")
        points.append({
            "fraction": fraction, "n_train_companies": len({r.company_id for r in subset}), "n_train_accounts": len(subset),
            "top1": float(right.mean()), "auto_share": float(auto.mean()),
            "auto_share_ci": list(bootstrap_ci(lambda idx: auto[idx].mean(), len(dev_rows), groups=companies)),
            "auto_precision": float(right[auto].mean()) if auto.any() else None, "t_auto": cal.t_auto,
            "by_source": {s: {"n": v["n"], "top1": v["top1"], "auto_share": v["auto_share"]} for s, v in by_source.items()},
        })  # fmt: skip
    return points


def more_data_helps(points: list[dict]) -> bool:
    """True if the 100% point beats the 75% point by more than half the 100% point's interval."""
    by_fraction = {p["fraction"]: p for p in points}
    if 0.75 not in by_fraction or 1.0 not in by_fraction:
        raise ValueError("the decision rule compares the 0.75 and 1.0 points, and one of them is missing")
    last = by_fraction[1.0]
    low, high = last["auto_share_ci"]
    return last["auto_share"] - by_fraction[0.75]["auto_share"] > (high - low) / 2


# --- the report --------------------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.1%}"


def _verdict(name: str, points: list[dict]) -> str:
    by_fraction = {p["fraction"]: p for p in points}
    gain = by_fraction[1.0]["auto_share"] - by_fraction[0.75]["auto_share"]
    low, high = by_fraction[1.0]["auto_share_ci"]
    numbers = f"75% → 100% of the companies changes the auto share by {gain * 100:+.1f} points; the interval's half-width is {(high - low) / 2 * 100:.1f}"
    if more_data_helps(points):
        return f"- **{name}:** {numbers}. More data still helps: double `rule_companies` for train, regenerate and rerun."
    return f"- **{name}:** {numbers}, so more data stopped helping."


def render_report(curves: dict[str, list[dict]]) -> str:
    lines = ["# Learning curve", "",
             "Each method refitted on a growing share of the train companies (whole companies, the same mix of rule-based and "
             "LLM-written), calibrated on dev the same way every time. Auto share at the target precision, with a 95% interval "
             "over dev companies.", ""]  # fmt: skip
    for name, points in curves.items():
        lines += [f"## {name}", "",
                  "| share of train | companies | accounts | top-1 | auto-mapped | 95% interval | precision | auto (rules) | auto (LLM) |",
                  "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for p in points:
            low, high = p["auto_share_ci"]
            src = p["by_source"]
            lines.append(f"| {p['fraction']:.0%} | {p['n_train_companies']} | {p['n_train_accounts']:,} | {_pct(p['top1'])} | {_pct(p['auto_share'])} | "
                         f"{low:.1%} – {high:.1%} | {_pct(p['auto_precision'])} | {_pct(src.get('rules', {}).get('auto_share'))} | "
                         f"{_pct(src.get('llm', {}).get('auto_share'))} |")  # fmt: skip
        lines.append("")
    lines += ["## Decision", ""] + [_verdict(name, points) for name, points in curves.items()] + [""]
    return "\n".join(lines)


def plot(curves: dict[str, list[dict]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # files only: no window
    import matplotlib.pyplot as plt

    from d1.evaluation.plotting import INK_2, METHOD_COLORS, apply_style, pct

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, points in curves.items():
        x = [p["n_train_accounts"] for p in points]
        color = METHOD_COLORS.get(name, INK_2)
        ax.fill_between(x, [p["auto_share_ci"][0] for p in points], [p["auto_share_ci"][1] for p in points], color=color, alpha=0.15, linewidth=0)
        ax.plot(x, [p["auto_share"] for p in points], color=color, marker="o", markersize=5, label=name)
    ax.set_xlabel("train accounts the model learned from")
    ax.set_ylabel("dev auto-mapped at the target precision")
    ax.yaxis.set_major_formatter(pct())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title("More training data: does the auto share still grow?")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(curves: dict[str, list[dict]], folder: Path) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    md, png, data = (folder / f"{REPORT}.{ext}" for ext in ("md", "png", "json"))
    md.write_text(render_report(curves), encoding="utf-8")
    plot(curves, png)
    decisions = {name: more_data_helps(points) for name, points in curves.items()}
    data.write_text(json.dumps({"curves": curves, "more_data_helps": decisions}, indent=2) + "\n", encoding="utf-8")
    return [md, png, data]


def main(argv: list[str] | None = None, root: Path = REPO_ROOT, embedder_factory: Callable[[str], Embedder] = SentenceTransformerEmbedder) -> int:
    parser = argparse.ArgumentParser(prog="python -m d1.evaluation.learning_curve", description="Refit methods on growing shares of train.")
    parser.add_argument("--method", nargs="+", required=True, help="methods that learn from train, such as xgboost random_forest")
    args = parser.parse_args(argv)
    try:
        config = load_config(root)
        unknown = [m for m in args.method if m not in METHOD_REGISTRY]
        if unknown:
            raise MapperError(f"{', '.join(unknown)}: not built yet")
        chart = load_standard_chart(config.data.chart_path, expected=config.data.expected_groups)
        dev, train = _rows(config, "dev"), _rows(config, "train")
        retriever = build_retriever(config, MemoEmbedder(embedder_factory(config.models.embedding_model)), chart)
        mapping = config.models.mapping
        curves = {}
        for name in args.method:
            if not METHOD_REGISTRY[name](config, retriever).needs_training:
                raise MapperError(f"{name} learns nothing from train, so it has no learning curve")
        for name in args.method:
            curves[name] = learning_curve(lambda: METHOD_REGISTRY[name](config, retriever), train, dev, retriever,  # noqa: B023
                                          k=mapping.candidates_k, target=mapping.target_precision, threshold_by=mapping.threshold_by)  # fmt: skip
            print(_verdict(name, curves[name]).replace("**", ""), flush=True)
        for path in write_report(curves, config.root / "reports"):
            print(f"wrote {path.relative_to(config.root)}")
    except (ConfigError, ChartError, MapperError, LLMError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
