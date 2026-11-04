"""Run a trained method on a split and write its metrics.

    python -m d1.evaluation.run_mapping --split dev --method fuzzy
    python -m d1.evaluation.run_mapping --split test --method <name> --reported

Writes reports/results/<run_id>/metrics.json, tables.md and decisions.csv (one line per account).

The guard on test: the test split is the final exam, so a run on it needs
--reported, reads the hand-checked labels (test_checked.jsonl, never test.jsonl),
and is appended to reports/test_runs.jsonl with the git commit and a fingerprint
of the config. Train is refused: the methods learned from it.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from d1.config import REPO_ROOT, Config, ConfigError, load_config
from d1.datagen.schema import read_jsonl
from d1.datagen.standard_chart import ChartError
from d1.evaluation.metrics import (
    auto_precision, auto_share, bootstrap_ci, breakdown, cost_report, recall_at_k, review_share, topk_accuracy,
)  # fmt: skip
from d1.mapper.embed import Embedder, SentenceTransformerEmbedder
from d1.mapper.methods.base import accounts_from_rows
from d1.mapper.pipeline import Mapper, MapperError

RESULTS_DIR = Path("reports") / "results"
TEST_LOG = Path("reports") / "test_runs.jsonl"
LABEL_FILES = {"dev": "dev.jsonl", "test": "test_checked.jsonl"}
BREAKDOWNS = ["source", "language", "noise", "industry", "overlaps_dev"]
DECISION_COLUMNS = [
    "account_id", "company_id", "source", "language", "noise", "overlaps_dev", "name", "target_code",
    "chosen_code", "right", "confidence", "decision_path", "status", "top3",
]  # fmt: skip


class RunError(Exception):
    """The run is not allowed or can't start; the message says what to do."""


def config_sha256(config: Config) -> str:
    h = hashlib.sha256()
    for name in ("models.yaml", "data.yaml"):
        h.update((config.root / "config" / name).read_bytes())
    return h.hexdigest()


def git_state(root: Path) -> tuple[str, bool]:
    """The current commit and whether there are uncommitted changes; ("unknown", False) outside git."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def _run_dir(config: Config, split: str, method: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = config.root / RESULTS_DIR / f"{stamp}-{split}-{method}"
    path, i = base, 2
    while path.exists():  # two runs in the same second never overwrite each other
        path, i = base.with_name(f"{base.name}-{i}"), i + 1
    return path


def evaluate(config: Config, split: str, method: str, embedder_factory: Callable[[str], Embedder] = SentenceTransformerEmbedder) -> dict:
    """Map every account of the split with the trained method and measure the decisions."""
    path = config.data.generated_dir / LABEL_FILES[split]
    if not path.exists():
        hint = "python -m d1.datagen apply-review" if split == "test" else "python -m d1.datagen build"
        raise RunError(f"{path} not found: run `{hint}` first")
    rows = read_jsonl(path)
    mapper = Mapper.from_artifacts(config, method, embedder_factory=embedder_factory)
    k = mapper.k
    accounts = accounts_from_rows(rows)
    shortlists = mapper.retriever.top_k([a.name for a in accounts], k)
    decisions = mapper.map(accounts, shortlists=shortlists)

    gold = {r.account_id: r.target_code for r in rows}
    companies = np.array([r.company_id for r in rows])
    auto = np.array([d.decision_path in ("auto", "cascade_llm") for d in decisions])
    right = np.array([d.chosen_code == gold[d.account_id] for d in decisions])

    def precision(idx):
        return right[idx][auto[idx]].mean() if auto[idx].any() else None

    overall = {
        "auto_share": auto_share(decisions),
        "auto_share_ci": bootstrap_ci(lambda idx: auto[idx].mean(), len(rows), groups=companies),
        "auto_precision": auto_precision(decisions, gold),
        "auto_precision_ci": bootstrap_ci(precision, len(rows), groups=companies) if auto.any() else None,
        "review_share": review_share(decisions),
        "top1": topk_accuracy(decisions, gold, 1),
        "top1_ci": bootstrap_ci(lambda idx: right[idx].mean(), len(rows), groups=companies),
        "top3": topk_accuracy(decisions, gold, 3),
        "recall_at_k": recall_at_k({a.account_id: s for a, s in zip(accounts, shortlists)}, gold, k),
        "k": k,
    }
    per_account = [
        {
            "account_id": r.account_id, "company_id": r.company_id, "source": r.source, "language": r.language,
            "noise": ";".join(r.noise), "overlaps_dev": int(r.overlaps_dev), "name": r.name, "target_code": r.target_code,
            "chosen_code": d.chosen_code or "", "right": int(d.chosen_code == r.target_code), "confidence": f"{d.confidence:.6f}",
            "decision_path": d.decision_path, "status": d.status, "top3": ";".join(c.code for c in d.candidates[:3]),
        }
        for r, d in zip(rows, decisions)
    ]  # fmt: skip
    return {
        "per_account": per_account,
        "split": split, "method": method, "labels_file": path.name, "n": len(rows),
        "label_status": dict(Counter(r.label_status for r in rows)),
        "model_versions": decisions[0].model_versions if decisions else {},
        "thresholds": mapper.thresholds.model_dump(), "overall": overall,
        "breakdowns": {by: breakdown(rows, decisions, by) for by in BREAKDOWNS},
        "cost": cost_report(decisions),
    }  # fmt: skip


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{x:.1%}"


def render_tables(m: dict) -> str:
    o = m["overall"]
    ci = lambda pair: f"{pair[0]:.1%} – {pair[1]:.1%}" if pair else "–"  # noqa: E731
    lines = [
        f"# {m['method']} on {m['split']}", "",
        f"{m['n']} accounts from `{m['labels_file']}`; threshold {m['thresholds']['t_auto']}; target precision {m['thresholds']['target_precision']:.0%}.", "",
        "| measure | value | 95% interval (companies resampled) |", "|---|---|---|",
        f"| auto-mapped | {_pct(o['auto_share'])} | {ci(o['auto_share_ci'])} |",
        f"| precision of auto-mapped | {_pct(o['auto_precision'])} | {ci(o['auto_precision_ci'])} |",
        f"| sent to review | {_pct(o['review_share'])} | |",
        f"| top-1 | {_pct(o['top1'])} | {ci(o['top1_ci'])} |",
        f"| top-3 | {_pct(o['top3'])} | |",
        f"| recall@{o['k']} (shortlist) | {_pct(o['recall_at_k'])} | |", "",
    ]  # fmt: skip
    for by, table in m["breakdowns"].items():
        lines += [f"## By {by}", "", f"| {by} | accounts | auto-mapped | precision of auto-mapped | top-1 |", "|---|---|---|---|---|"]
        lines += [f"| {r['value']} | {r['n']} | {_pct(r['auto_share'])} | {_pct(r['auto_precision'])} | {_pct(r['top1'])} |" for r in table]
        lines.append("")
    c = m["cost"]
    lines += ["## Cost", "", f"{c['calls_per_1000']:.0f} calls and ${c['cost_per_1000_usd']:.2f} per 1,000 accounts; "
              f"median latency {c['median_latency_ms']:.0f} ms; cache hits {c['cache_hit_rate']:.0%}.", ""]  # fmt: skip
    return "\n".join(lines)


def main(argv: list[str] | None = None, root: Path = REPO_ROOT, embedder_factory: Callable[[str], Embedder] = SentenceTransformerEmbedder) -> int:
    parser = argparse.ArgumentParser(prog="python -m d1.evaluation.run_mapping", description="Run a trained method on a split and write its metrics.")
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument("--method", required=True)
    parser.add_argument("--reported", action="store_true", help="required for the test split: this run is reported and logged")
    args = parser.parse_args(argv)
    try:
        if args.split == "train":
            raise RunError("the train split is refused: the methods were trained on it, so it would measure memory, not mapping")
        if args.split == "test" and not args.reported:
            raise RunError("the test split is the final exam: add --reported to run it, and the run will be logged in reports/test_runs.jsonl")
        config = load_config(root)
        metrics = evaluate(config, args.split, args.method, embedder_factory)
        run = _run_dir(config, args.split, args.method)
        metrics = {"run_id": run.name, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **metrics}
        run.mkdir(parents=True)
        per_account = metrics.pop("per_account")
        with open(run / "decisions.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DECISION_COLUMNS)
            writer.writeheader()
            writer.writerows(per_account)
        (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        (run / "tables.md").write_text(render_tables(metrics), encoding="utf-8")
        if args.split == "test":
            sha, dirty = git_state(config.root)
            entry = {"run_id": run.name, "git_sha": sha, "uncommitted_changes": dirty, "config_sha256": config_sha256(config),
                     "method": args.method, "timestamp": metrics["created_at"]}  # fmt: skip
            log = config.root / TEST_LOG
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a", encoding="utf-8") as f:  # append only: the log is never rewritten
                f.write(json.dumps(entry) + "\n")
        o = metrics["overall"]
        print(f"{args.method} on {args.split}: auto-mapped {_pct(o['auto_share'])} at {_pct(o['auto_precision'])} precision; "
              f"top-1 {_pct(o['top1'])} -> {run.relative_to(config.root)}")  # fmt: skip
    except (ConfigError, ChartError, MapperError, RunError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
