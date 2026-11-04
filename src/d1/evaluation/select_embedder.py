"""Choose the embedding model on dev.

    python -m d1.evaluation.select_embedder     # -> reports/embedding_selection.md (+ _rows.csv)

Every candidate model builds the top-k shortlist for every dev account. The
winner is the model whose shortlist most often contains the right group
(recall@k), because every mapping method chooses from that shortlist and so
can never beat it. Ties go to the better top-1, then to the faster model.

Next to the report, reports/embedding_selection_rows.csv keeps one line per
(model, dev account) with the rank of the right group among all 190, so the
notebook in notebooks/ can draw every comparison without loading a model.
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import date
from pathlib import Path

from d1.config import REPO_ROOT, ConfigError, load_config
from d1.datagen.schema import CompanyAccount, StandardAccount, read_jsonl
from d1.datagen.standard_chart import ChartError, load_standard_chart
from d1.mapper.embed import Embedder, SentenceTransformerEmbedder
from d1.mapper.retrieve import Candidate, Retriever

REPORT_PATH = Path("reports") / "embedding_selection.md"
ROWS_PATH = Path("reports") / "embedding_selection_rows.csv"
SUMMARY_PATH = Path("reports") / "embedding_selection.json"
ROWS_COLUMNS = [
    "model", "account_id", "company_id", "language", "source", "code_scheme", "noise", "name",
    "target_code", "gold_rank", "top1_score", "top2_score", "top3_codes",
]  # fmt: skip


class ModelLoadError(Exception):
    """A candidate model could not be downloaded or loaded."""


def shortlist_scores(shortlists: list[list[Candidate]], gold: list[str], k: int) -> dict[str, float]:
    """Share of rows whose right group is first, in the top 3, and in the top k."""
    if len(shortlists) != len(gold):
        raise ValueError(f"{len(shortlists)} shortlists but {len(gold)} gold codes")
    n = len(gold) or 1

    def hit_rate(depth: int) -> float:
        return sum(g in [c.code for c in s[:depth]] for s, g in zip(shortlists, gold)) / n

    return {"top1": hit_rate(1), "top3": hit_rate(3), "recall_at_k": hit_rate(k)}


def _recall_by(rows: list[CompanyAccount], shortlists: list[list[Candidate]], k: int, field: str) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[getattr(row, field)].append(i)
    return {
        value: shortlist_scores([shortlists[i] for i in idx], [rows[i].target_code for i in idx], k)["recall_at_k"]
        for value, idx in sorted(groups.items())
    }


def compare_embedders(
    model_names: list[str],
    dev_rows: list[CompanyAccount],
    chart: list[StandardAccount],
    k: int,
    embedder_factory: Callable[[str], Embedder] = SentenceTransformerEmbedder,
    clock: Callable[[], float] = time.perf_counter,
) -> list[dict]:
    """Score every model's shortlists on the dev rows, in the order given."""
    if any(r.split != "dev" for r in dev_rows):
        # Choosing a model is tuning: doing it on test would leak into the final result.
        raise ValueError("compare_embedders takes dev rows only")
    names = [r.name for r in dev_rows]
    gold = [r.target_code for r in dev_rows]
    results = []
    for model_name in model_names:
        try:
            embedder = embedder_factory(model_name)
            embedder.encode(["warm-up"], "query")  # loads (and the first time downloads) the model, outside the timing
        except Exception as e:  # download and load errors come in many types (HTTP, OS, torch)
            raise ModelLoadError(f"could not load {model_name}: {_first_line(e)}") from e
        start = clock()
        # No cache: every model pays for embedding the 190 groups, so the times compare fairly.
        # The full ranking (k = all groups) gives the rank of the right group even when it is not in the top k.
        retriever = Retriever(embedder, chart)
        shortlists = retriever.top_k(names, len(chart))
        seconds = clock() - start
        per_row = [
            {
                "account_id": row.account_id,
                "gold_rank": next(i for i, c in enumerate(s, start=1) if c.code == row.target_code),
                "top1_score": s[0].score,
                "top2_score": s[1].score if len(s) > 1 else 0.0,
                "top3_codes": [c.code for c in s[:3]],
            }
            for row, s in zip(dev_rows, shortlists)
        ]
        results.append({
            "model": model_name,
            **shortlist_scores(shortlists, gold, k),
            "seconds": seconds,
            "n_rows": len(dev_rows),
            "dim": int(retriever.standard_embeddings.shape[1]),
            "recall_by_language": _recall_by(dev_rows, shortlists, k, "language"),
            "recall_by_source": _recall_by(dev_rows, shortlists, k, "source"),
            "per_row": per_row,
        })  # fmt: skip
        del embedder, retriever  # free the model's memory before loading the next one
    return results


def _first_line(e: Exception) -> str:
    text = str(e).strip().splitlines()
    return (text[0] if text else type(e).__name__)[:300]


def choose_best(results: list[dict]) -> dict:
    """Highest recall@k; ties to the higher top-1, then to the fewer seconds."""
    return max(results, key=lambda r: (r["recall_at_k"], r["top1"], -r["seconds"]))


def _pct(x: float) -> str:
    return f"{x:.1%}"


def render_report(results: list[dict], best: dict, k: int) -> str:
    n_rows = results[0]["n_rows"]
    languages = list(results[0]["recall_by_language"])
    sources = list(results[0]["recall_by_source"])
    lines = [
        "# Choosing the embedding model",
        "",
        f"Run on {date.today().isoformat()} on {n_rows} dev rows, with shortlists of k = {k}. "
        f"Seconds = embedding the 190 standard groups and every dev name, on the machine that ran it (model already loaded).",
        "",
        f"**Rule:** the highest recall@{k} wins, because every mapping method chooses from this shortlist "
        "and can never beat it. Ties go to the higher top-1, then to the faster model.",
        "",
        f"| model | top-1 | top-3 | recall@{k} | seconds |",
        "|---|---|---|---|---|",
        *[f"| {r['model']} | {_pct(r['top1'])} | {_pct(r['top3'])} | {_pct(r['recall_at_k'])} | {r['seconds']:.1f} |" for r in results],
        "",
        f"**Chosen: `{best['model']}`** (recall@{k} {_pct(best['recall_at_k'])}, top-1 {_pct(best['top1'])}).",
        "",
        f"## Recall@{k} by language and by source",
        "",
        "| model | " + " | ".join(languages + sources) + " |",
        "|---|" + "---|" * (len(languages) + len(sources)),
        *[
            f"| {r['model']} | "
            + " | ".join(_pct(r["recall_by_language"][x]) for x in languages)
            + " | "
            + " | ".join(_pct(r["recall_by_source"][x]) for x in sources)
            + " |"
            for r in results
        ],
        "",
        "Note: the LLM-written dev rows carry the generating LLM's own labels, which are not hand-checked. "
        "On the test split the hand check found 13% of those labels wrong, so the `llm` column understates "
        "every model by a similar margin. That affects all models alike, so the comparison holds.",
        "",
    ]
    return "\n".join(lines)


def write_rows(results: list[dict], dev_rows: list[CompanyAccount], path: Path) -> None:
    """One CSV line per (model, dev row): the input of the notebook."""
    by_id = {r.account_id: r for r in dev_rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROWS_COLUMNS)
        writer.writeheader()
        for result in results:
            for d in result["per_row"]:
                row = by_id[d["account_id"]]
                writer.writerow({
                    "model": result["model"], "account_id": row.account_id, "company_id": row.company_id,
                    "language": row.language, "source": row.source, "code_scheme": row.code_scheme,
                    "noise": ";".join(row.noise), "name": row.name, "target_code": row.target_code,
                    "gold_rank": d["gold_rank"], "top1_score": f"{d['top1_score']:.6f}",
                    "top2_score": f"{d['top2_score']:.6f}", "top3_codes": ";".join(d["top3_codes"]),
                })  # fmt: skip


def main(
    argv: list[str] | None = None,
    root: Path = REPO_ROOT,
    embedder_factory: Callable[[str], Embedder] = SentenceTransformerEmbedder,
) -> int:
    argparse.ArgumentParser(prog="python -m d1.evaluation.select_embedder", description=__doc__.split("\n")[0]).parse_args(argv)
    try:
        config = load_config(root)
        dev_path = config.data.generated_dir / "dev.jsonl"
        if not dev_path.exists():
            raise ConfigError(f"{dev_path} not found: run `python -m d1.datagen build` first")
        chart = load_standard_chart(config.data.chart_path, expected=config.data.expected_groups)
        dev_rows = read_jsonl(dev_path)
    except (ConfigError, ChartError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    k = config.models.mapping.candidates_k
    results = []
    for model_name in config.models.embedding_candidates:
        print(f"{model_name} ...", flush=True)
        try:
            (result,) = compare_embedders([model_name], dev_rows, chart, k, embedder_factory)
        except ModelLoadError as e:
            # No report with a model missing: the choice would not be among all candidates.
            print(f"error: {e} (models already downloaded are kept, so run it again to retry)", file=sys.stderr)
            return 1
        print(f"  top-1 {_pct(result['top1'])}, recall@{k} {_pct(result['recall_at_k'])}, {result['seconds']:.1f} s")
        results.append(result)
    best = choose_best(results)
    path = config.root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(results, best, k), encoding="utf-8")
    write_rows(results, dev_rows, config.root / ROWS_PATH)
    summary = {"k": k, "chosen": best["model"], "models": [{x: v for x, v in r.items() if x != "per_row"} for r in results]}
    (config.root / SUMMARY_PATH).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"report -> {path}, per-row results -> {config.root / ROWS_PATH}")
    print(f"best: {best['model']}; set embedding_model to it in config/models.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
