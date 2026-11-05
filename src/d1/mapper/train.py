"""Train and calibrate a mapping method, and save it.

    python -m d1.mapper.train --method fuzzy|embeddings|...|all

For one method:
  1. fit it on the train split, if it learns (XGBoost, random forest);
  2. let it score every dev account's shortlist;
  3. calibrate out of fold on dev (companies kept together), refit the calibrator
     on all of dev, and choose t_auto: the lowest confidence at which >= target
     precision of auto-mapped dev accounts are right, both on the out-of-fold
     probabilities and on the saved calibrator's (the stricter of the two);
  4. write artifacts/methods/<method>/:
     method.json (+ model files), calibrator.json, thresholds.json, dev_summary.json.

The test split is never read here.
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from d1.config import REPO_ROOT, Config, ConfigError, Method, load_config
from d1.datagen.schema import CompanyAccount, read_jsonl
from d1.datagen.standard_chart import ChartError, load_standard_chart
from d1.mapper.calibrate import Calibrator, choose_threshold, cross_fit_probs, feature_matrix, stricter_threshold
from d1.mapper.embed import Embedder, SentenceTransformerEmbedder
from d1.mapper.methods import METHOD_REGISTRY
from d1.mapper.methods.base import accounts_from_rows
from d1.mapper.pipeline import MapperError, Thresholds, artifact_dir, build_retriever

MIN_SUPPORT = 50  # a threshold must be backed by at least this many auto-mapped dev accounts


def _rows(config: Config, split: str) -> list[CompanyAccount]:
    path = config.data.generated_dir / f"{split}.jsonl"
    if not path.exists():
        raise MapperError(f"{path} not found: run `python -m d1.datagen build` first")
    return read_jsonl(path)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def train_method(config: Config, name: Method, embedder_factory: Callable[[str], Embedder] = SentenceTransformerEmbedder) -> dict:
    """Fit (if needed), calibrate on dev and save one method; returns its dev summary."""
    if name not in METHOD_REGISTRY:
        raise MapperError(f"{name}: not built yet")
    k = config.models.mapping.candidates_k
    target = config.models.mapping.target_precision
    chart = load_standard_chart(config.data.chart_path, expected=config.data.expected_groups)
    method = METHOD_REGISTRY[name](config)
    # Read the data first: a missing file should fail in a second, not after loading a 2 GB model.
    train = _rows(config, "train") if method.needs_training else []
    dev = _rows(config, "dev")
    embedder = embedder_factory(config.models.embedding_model)
    retriever = build_retriever(config, embedder, chart)

    if method.needs_training:
        method.fit(train, retriever.top_k([r.name for r in train], k))

    outputs = method.score(accounts_from_rows(dev), retriever.top_k([r.name for r in dev], k))
    right = np.array([o.chosen_code == r.target_code for o, r in zip(outputs, dev)])
    picked = np.array([o.chosen_code is not None for o in outputs])
    X = feature_matrix([list(o.scores.values()) for o in outputs])
    companies = np.array([r.company_id for r in dev])

    # Accounts without a pick are never auto-mapped: probability 0, and left out of the fit.
    oof, p = np.zeros(len(dev)), np.zeros(len(dev))
    folds = min(5, len(set(companies[picked])))
    oof[picked] = cross_fit_probs(X[picked], right[picked], folds=folds, seed=0, groups=companies[picked])
    calibrator = Calibrator().fit(X[picked], right[picked])
    p[picked] = calibrator.predict(X[picked])  # exactly what the Mapper will compute for these accounts
    t_oof = choose_threshold(oof, right, target, min_support=MIN_SUPPORT)
    t_auto = stricter_threshold(t_oof, choose_threshold(p, right, target, min_support=MIN_SUPPORT))

    folder = artifact_dir(config, name)
    folder.mkdir(parents=True, exist_ok=True)
    method.save(folder)
    _write_json(folder / "calibrator.json", calibrator.to_json())
    thresholds = Thresholds(method=name, t_auto=t_auto, target_precision=target, fitted_on="dev", embedding_model=embedder.model_id,
                            group_text=retriever.group_text)
    _write_json(folder / "thresholds.json", thresholds.model_dump())

    # The dev numbers use the saved calibrator, so an evaluation run on dev reproduces them exactly.
    auto = (p >= t_auto) & picked if t_auto is not None else np.zeros(len(dev), bool)
    auto_oof = (oof >= t_auto) & picked if t_auto is not None else np.zeros(len(dev), bool)
    by_source = {}
    for source in ("rules", "llm"):
        s = np.array([r.source == source for r in dev])
        if s.any():
            by_source[source] = {"n": int(s.sum()), "top1": float(right[s].mean()), "auto_share": float(auto[s].mean()),
                                 "auto_precision": float(right[auto & s].mean()) if (auto & s).any() else None}  # fmt: skip
    summary = {
        "method": name, "version": method.version(), "n_dev": len(dev), "top1": float(right.mean()),
        "no_pick": int((~picked).sum()), "t_auto": t_auto, "auto_share": float(auto.mean()),
        "auto_precision": float(right[auto].mean()) if auto.any() else None, "by_source": by_source,
        "out_of_fold": {"t_auto": t_oof, "auto_share": float(auto_oof.mean()),
                        "auto_precision": float(right[auto_oof].mean()) if auto_oof.any() else None},
    }  # fmt: skip
    _write_json(folder / "dev_summary.json", summary)
    return summary


def _describe(s: dict) -> str:
    if s["t_auto"] is None:
        return f"{s['method']}: top-1 {s['top1']:.1%}; no threshold reaches the target precision, so every account goes to review"
    return (f"{s['method']}: top-1 {s['top1']:.1%}; auto-map {s['auto_share']:.1%} of dev at {s['auto_precision']:.1%} precision "
            f"(threshold {s['t_auto']:.3f})")  # fmt: skip


def main(argv: list[str] | None = None, root: Path = REPO_ROOT, embedder_factory: Callable[[str], Embedder] = SentenceTransformerEmbedder) -> int:
    parser = argparse.ArgumentParser(prog="python -m d1.mapper.train", description="Train, calibrate on dev and save a mapping method.")
    parser.add_argument("--method", required=True, help="a method from config/models.yaml, or all")
    args = parser.parse_args(argv)
    try:
        config = load_config(root)
        configured = config.models.mapping.methods
        if args.method == "all":
            names = configured
        elif args.method in configured:
            names = [args.method]
        else:
            raise MapperError(f"unknown method {args.method!r}: choose from {', '.join(configured)} or all")
        if len(names) == 1 and names[0] not in METHOD_REGISTRY:
            raise MapperError(f"{names[0]}: not built yet")
        for name in names:
            if name not in METHOD_REGISTRY:
                print(f"{name}: not built yet, skipped")
                continue
            print(_describe(train_method(config, name, embedder_factory)), flush=True)
    except (ConfigError, ChartError, MapperError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
