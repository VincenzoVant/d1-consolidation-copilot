"""Command line for the data generator.

    python -m d1.datagen fetch-llm --split train|dev|test|all   # LLM-written companies, once
    python -m d1.datagen build                                  # all three splits, offline
    python -m d1.datagen export-review                          # test rows to check -> CSV
    python -m d1.datagen apply-review                           # your verdicts -> test_checked.jsonl

Known errors print to stderr (one line; a list for review problems) and exit with code 1.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from d1.config import REPO_ROOT, Config, ConfigError, load_config
from d1.datagen.handcheck import ReviewError, apply_review, export_review
from d1.datagen.llm_charts import FrozenOutputMissing, LLMOutputError
from d1.datagen.schema import CompanyAccount, read_jsonl, write_jsonl
from d1.datagen.splits import build_all, fetch_missing_llm
from d1.datagen.standard_chart import ChartError, load_standard_chart
from d1.llm import LLMBackend, LLMError, make_backend

KNOWN_ERRORS = (ConfigError, ChartError, FrozenOutputMissing, LLMOutputError, LLMError, ReviewError)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m d1.datagen", description="Build D1's synthetic data.")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-llm", help="have the generator model write the LLM companies that are not frozen yet")
    fetch.add_argument("--split", required=True, choices=["train", "dev", "test", "all"])
    fetch.add_argument("--workers", type=_positive_int, default=4, help="calls at the same time (default 4)")
    commands.add_parser("build", help="write train/dev/test JSONL from the rules and the frozen LLM files")
    commands.add_parser("export-review", help="write the test rows to check by hand to the review CSV")
    commands.add_parser("apply-review", help="apply the verdicts in the review CSV to the test split")
    return parser


def _test_rows(config: Config) -> list[CompanyAccount]:
    path = config.data.generated_dir / "test.jsonl"
    if not path.exists():
        raise ReviewError([f"{path} not found: run `python -m d1.datagen build` first"])
    return read_jsonl(path)


def _review(config: Config, command: str) -> None:
    rows = _test_rows(config)
    chart = load_standard_chart(config.data.chart_path, expected=config.data.expected_groups)
    path, sample, seed = config.data.review_path, config.data.review_rule_sample, config.data.splits["test"].seed
    if command == "export-review":
        n = export_review(rows, chart, path, sample, seed)
        print(f"wrote {n} rows to {path}")
        print("fill in verdict for every row: ok, fix (and put the right group in corrected_target) or drop")
        return
    result = apply_review(rows, chart, path, sample, seed)
    out = config.data.generated_dir / "test_checked.jsonl"
    write_jsonl(result.rows, out)
    for source, counts in result.counts.items():
        print(f"{source}: " + ", ".join(f"{verdict} {n}" for verdict, n in counts.items()))
    changed = result.counts["rules"]["fix"] + result.counts["rules"]["drop"]
    if result.rule_fix_rate is not None:
        print(f"rule-based spot check fix rate: {result.rule_fix_rate:.1%} ({changed} of {sample})")
    print(f"{len(result.rows)} test rows -> {out}")


def main(
    argv: list[str] | None = None,
    backend_factory: Callable[[Config], LLMBackend] = make_backend,
    root: Path = REPO_ROOT,
) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(root)
        if args.command == "fetch-llm":
            paths = fetch_missing_llm(config, args.split, backend_factory(config), max_workers=args.workers, log=print)
            print(f"{len(paths)} new frozen files in {config.data.frozen_llm_dir}")
        elif args.command in ("export-review", "apply-review"):
            _review(config, args.command)
        else:
            for split, path in build_all(config).items():
                with open(path, encoding="utf-8") as f:
                    print(f"{split}: {sum(1 for _ in f)} accounts -> {path}")
    except KNOWN_ERRORS as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
