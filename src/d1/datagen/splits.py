"""Train, dev and test splits: which companies each one has, and building them.

Every company is fixed by its split and its index i:
  - company_plan gives its industry, language and source (rules or llm);
  - its random generator is Random(split seed * 1000 + i).
So building twice gives byte-identical files, and a deleted LLM company is
fetched again with exactly the same groups.

Two steps, because only the first needs an LLM:
  fetch_missing_llm  LLM-written companies -> data/frozen/llm_charts/ (once)
  build_all          rule-based + frozen LLM companies -> data/generated/*.jsonl
"""

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from random import Random
from typing import Literal

from pydantic import BaseModel, ConfigDict

from d1.config import Config, SplitConfig
from d1.datagen.llm_charts import (
    FrozenOutputMissing,
    LLMOutputError,
    fetch_llm_company,
    frozen_path,
    load_llm_company,
    select_llm_groups,
)
from d1.datagen.noise import strip_accents
from d1.datagen.rule_charts import generate_rule_company
from d1.datagen.schema import CompanyAccount, Language, Split, StandardAccount, write_jsonl
from d1.datagen.standard_chart import load_standard_chart
from d1.llm import LOGIN_HINT, LLMBackend

SPLITS: tuple[Split, ...] = ("train", "dev", "test")

# Failures that will not fix themselves by asking again: stop the whole run.
FATAL_HINTS = (LOGIN_HINT, "install Claude Code", "ANTHROPIC_API_KEY")


class CompanySpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    company_id: str
    split: Split
    industry: str
    language: Language
    source: Literal["rules", "llm"]
    index: int
    n_accounts: int


def company_plan(split: Split, cfg: SplitConfig, industries: list[str], languages: list[Language]) -> list[CompanySpec]:
    """The companies of one split: rule-based first, then LLM-written.

    Industry cycles fastest; the language shifts by one every full round of
    industries, so every industry meets every language and both sources.
    """
    plan = []
    for i in range(cfg.n_companies):
        source = "rules" if i < cfg.rule_companies else "llm"
        plan.append(
            CompanySpec(
                company_id=f"{split}-c{i:03d}",
                split=split,
                industry=industries[i % len(industries)],
                language=languages[(i + i // len(industries)) % len(languages)],
                source=source,
                index=i,
                n_accounts=cfg.rule_accounts_per_company if source == "rules" else cfg.llm_accounts_per_company,
            )
        )
    return plan


def _company_rng(cfg: SplitConfig, spec: CompanySpec) -> Random:
    return Random(cfg.seed * 1000 + spec.index)


def _plans(config: Config, splits: tuple[Split, ...]) -> list[tuple[SplitConfig, CompanySpec]]:
    d = config.data
    return [(d.splits[s], spec) for s in splits for spec in company_plan(s, d.splits[s], d.industries, d.languages)]


# --- names seen before ------------------------------------------------------------


_PUNCTUATION = re.compile(r"[.,/\-]")


def normalize_name(name: str) -> str:
    """Compare names the way a reader would: no case, accents, . , / - or extra spaces."""
    return " ".join(_PUNCTUATION.sub(" ", strip_accents(name.casefold())).split())


def mark_overlaps(rows: list[CompanyAccount], seen_rows: list[CompanyAccount]) -> list[CompanyAccount]:
    """Copies of rows, with overlaps_dev set where the normalized name occurs in seen_rows."""
    seen = {normalize_name(r.name) for r in seen_rows}
    return [r.model_copy(update={"overlaps_dev": normalize_name(r.name) in seen}) for r in rows]


# --- fetching the LLM-written companies ------------------------------------------------


def fetch_missing_llm(
    config: Config,
    split: Split | Literal["all"],
    backend: LLMBackend,
    max_workers: int = 4,
    log: Callable[[str], None] = lambda message: None,
) -> list[Path]:
    """Fetch every LLM company of the split(s) that has no frozen file yet.

    Calls run max_workers at a time. One failed company does not stop the
    others: they are saved, and the same command retries only what is missing.
    A failure that will repeat for every company (not logged in, CLI missing)
    stops the run. Returns the new frozen files.
    """
    chart = load_standard_chart(config.data.chart_path, expected=config.data.expected_groups)
    frozen_dir = config.data.frozen_llm_dir
    todo = []
    for cfg, spec in _plans(config, SPLITS if split == "all" else (split,)):
        if spec.source != "llm":
            continue
        rng = _company_rng(cfg, spec)
        groups = select_llm_groups(chart, spec.n_accounts, rng)
        code_scheme = rng.choice(["llm", "legacy"])
        if not frozen_path(frozen_dir, spec.company_id).exists():
            todo.append((spec, groups, code_scheme))

    log(f"{len(todo)} to fetch with {config.models.generator_model}, {max_workers} at a time")

    def fetch(spec: CompanySpec, groups: list[StandardAccount], code_scheme: str) -> Path:
        return fetch_llm_company(
            backend,
            generator_model=config.models.generator_model,
            mapping_model=config.models.mapping_model,
            company_id=spec.company_id,
            split=spec.split,
            industry=spec.industry,
            language=spec.language,
            code_scheme=code_scheme,
            groups=groups,
            n_accounts=spec.n_accounts,
            frozen_dir=frozen_dir,
        )

    fetched: list[Path] = []
    failures: list[str] = []
    total_cost = 0.0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch, *item): item[0] for item in todo}
        for done, future in enumerate(as_completed(futures), start=1):
            spec = futures[future]
            try:
                path = future.result()
            except LLMOutputError as e:
                failures.append(str(e))
                log(f"[{done}/{len(todo)}] {spec.company_id} FAILED: {e}")
                if any(hint in str(e) for hint in FATAL_HINTS):
                    for pending in futures:
                        pending.cancel()
                    raise LLMOutputError(f"stopped: {e}") from None
                continue
            usage = json.loads(path.read_text(encoding="utf-8"))["usage"]
            total_cost += usage["cost_usd"] or 0.0
            fetched.append(path)
            log(f"[{done}/{len(todo)}] {spec.company_id} ({spec.split}, {spec.industry}, {spec.language}) "
                f"{usage['latency_ms'] / 1000:.0f}s")  # fmt: skip

    log(f"fetched {len(fetched)}, failed {len(failures)}, cost ${total_cost:.2f}")
    if failures:
        raise LLMOutputError(
            f"{len(failures)} of {len(todo)} companies failed (the others are saved); run the same command again to retry them"
        )
    return sorted(fetched)


# --- building the splits ----------------------------------------------------------------


def build_split(split: Split, config: Config, chart: list[StandardAccount]) -> list[CompanyAccount]:
    """Every account of one split, company by company in plan order."""
    rows: list[CompanyAccount] = []
    for cfg, spec in _plans(config, (split,)):
        if spec.source == "rules":
            rows += generate_rule_company(
                spec.company_id, split, spec.industry, spec.language, chart, spec.n_accounts, _company_rng(cfg, spec)
            )
            continue
        company = load_llm_company(config.data.frozen_llm_dir, spec.company_id, split, spec.n_accounts)
        # A frozen file made under a different plan would quietly unbalance the split.
        for field in ("industry", "language"):
            if getattr(company[0], field) != getattr(spec, field):
                raise LLMOutputError(
                    f"{spec.company_id}.json has {field} {getattr(company[0], field)}, but the plan says {getattr(spec, field)}"
                )
        rows += company
    return rows


def build_all(config: Config) -> dict[Split, Path]:
    """Build train, dev and test and write them to generated_dir as JSONL.

    Nothing is written unless all three splits build, and every LLM company must
    be fetched first; build never calls an LLM.
    """
    llm_ids = [spec.company_id for _, spec in _plans(config, SPLITS) if spec.source == "llm"]
    missing = [c for c in llm_ids if not frozen_path(config.data.frozen_llm_dir, c).exists()]
    if missing:
        raise FrozenOutputMissing(
            f"{len(missing)} of {len(llm_ids)} LLM companies are not fetched yet (first: {missing[0]}): "
            "run `python -m d1.datagen fetch-llm --split all`"
        )

    chart = load_standard_chart(config.data.chart_path, expected=config.data.expected_groups)
    rows = {split: build_split(split, config, chart) for split in SPLITS}
    # Test names that also occur in train or dev measure memory, not generalisation.
    rows["test"] = mark_overlaps(rows["test"], rows["train"] + rows["dev"])

    paths = {}
    for split, split_rows in rows.items():
        paths[split] = config.data.generated_dir / f"{split}.jsonl"
        write_jsonl(split_rows, paths[split])
    return paths
