"""Command line for the data generator.

    python -m d1.datagen fetch-llm --split train|dev|test|all   # LLM-written companies, once
    python -m d1.datagen build                                  # all three splits, offline

Known errors print one line to stderr and exit with code 1.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from d1.config import REPO_ROOT, Config, ConfigError, load_config
from d1.datagen.llm_charts import FrozenOutputMissing, LLMOutputError
from d1.datagen.splits import build_all, fetch_missing_llm
from d1.datagen.standard_chart import ChartError
from d1.llm import LLMBackend, LLMError, make_backend

KNOWN_ERRORS = (ConfigError, ChartError, FrozenOutputMissing, LLMOutputError, LLMError)


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
    return parser


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
