"""Group descriptions: the everyday words the official group names lack.

    python -m d1.mapper.descriptions     # asks the description model, once; answers frozen in data/frozen/group_descriptions/

The shortlist embeds each standard group by its official names. Bookkeepers
write "Gross wages shop staff", not "Remuneration and Direct Social Benefits",
so the right group can end up far down the list. Each group therefore gets a
one-line description plus common ledger words in nl/fr/en, embedded with the names.

Leakage rules:
  - written by description_model (Sonnet), never by generator_model (Opus), which
    wrote the LLM company accounts: its own wording would match its own accounts;
  - the prompt holds the standard chart only, never an account from any split.

The 190 groups are asked in batches of 19 neighbouring groups (one call each),
and every answer is frozen like the LLM company charts: written once, then only
read, so the vectors and every result built on them rebuild identically.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from d1.config import REPO_ROOT, ConfigError, load_config
from d1.datagen.schema import StandardAccount
from d1.datagen.standard_chart import ChartError, load_standard_chart
from d1.llm import LLMBackend, LLMError, make_backend

PROMPT_PATH = REPO_ROOT / "prompts" / "describe_groups.md"
PROMPT_VERSION = "v1"
SYSTEM = "You describe the groups of the Belgian standard chart of accounts. Return JSON only."
BATCH_SIZE = 19  # 10 calls for 190 groups; neighbouring groups in one call can be told apart
TIMEOUT_S = 300
COMMAND = "python -m d1.mapper.descriptions"

DESCRIPTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "description": {"type": "string"},
                    "words_nl": {"type": "array", "items": {"type": "string"}},
                    "words_fr": {"type": "array", "items": {"type": "string"}},
                    "words_en": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["code", "description", "words_nl", "words_fr", "words_en"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["groups"],
    "additionalProperties": False,
}


class DescriptionError(Exception):
    """The descriptions are missing, incomplete or came from the wrong model."""


class GroupDescription(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    description: str = Field(min_length=1)
    words_nl: list[str]
    words_fr: list[str]
    words_en: list[str]

    def text(self) -> str:
        """What is added to the group's embedded text: the sentence, then the words per language."""
        per_language = [", ".join(w.strip() for w in ws if w.strip()) for ws in (self.words_nl, self.words_fr, self.words_en)]
        words = "; ".join(p for p in per_language if p)
        return f"{self.description.strip()} {words}".strip()


def batches(chart: list[StandardAccount], size: int = BATCH_SIZE) -> list[tuple[int, list[StandardAccount]]]:
    """The chart in code order, cut into numbered batches (1, 2, ...)."""
    ordered = sorted(chart, key=lambda a: a.code)
    return [(i // size + 1, ordered[i : i + size]) for i in range(0, len(ordered), size)]


def build_prompt(groups: list[StandardAccount]) -> str:
    rows = [f"| {g.code} | {g.name_en} | {g.name_nl} | {g.name_fr} | {g.parent_name_en} |" for g in groups]
    return PROMPT_PATH.read_text(encoding="utf-8").format(n_groups=len(groups), groups_table="\n".join(rows))


def parse_descriptions(response: dict, codes: list[str]) -> list[GroupDescription]:
    """One entry per expected code, in the order of codes; anything else is an error."""
    try:
        entries = [GroupDescription.model_validate(e) for e in response.get("groups", [])]
    except ValidationError as e:
        raise DescriptionError(f"malformed entry: {e.errors()[0]['msg']}") from None
    seen: dict[str, GroupDescription] = {}
    for e in entries:
        if e.code in seen:
            raise DescriptionError(f"group {e.code} described twice")
        if not e.description.strip():
            raise DescriptionError(f"group {e.code} has an empty description")
        seen[e.code] = e
    unknown = sorted(set(seen) - set(codes))
    missing = [c for c in codes if c not in seen]
    if unknown:
        raise DescriptionError(f"answer has groups that were not asked: {', '.join(unknown)}")
    if missing:
        raise DescriptionError(f"answer is missing groups: {', '.join(missing)}")
    return [seen[c] for c in codes]


def _batch_path(frozen_dir: Path, number: int) -> Path:
    return Path(frozen_dir) / f"batch-{number:02d}.json"


def fetch_batch(
    backend: LLMBackend,
    *,
    writer_model: str,
    generator_model: str,
    number: int,
    groups: list[StandardAccount],
    frozen_dir: Path,
) -> Path:
    """Ask for one batch and freeze the answer; does nothing if it is frozen already."""
    path = _batch_path(frozen_dir, number)
    if path.exists():
        return path
    codes = [g.code for g in groups]
    try:
        response = backend.complete(build_prompt(groups), schema=DESCRIPTION_SCHEMA, model=writer_model, system=SYSTEM, timeout_s=TIMEOUT_S)
    except LLMError as e:
        raise DescriptionError(f"batch {number}: {e}") from None
    if response.model == generator_model:
        raise DescriptionError(f"batch {number}: answered by {response.model}, the generator model; the descriptions must come from another model")
    try:
        parse_descriptions(response.structured, codes)
    except DescriptionError as e:
        raise DescriptionError(f"batch {number}: {e}") from None
    record = {
        "batch": number,
        "requested_model": writer_model,
        "served_model": response.model,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest(),
        "group_codes": codes,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "usage": {"input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                  "cost_usd": response.cost_usd, "latency_ms": response.latency_ms},  # fmt: skip
        "response": response.structured,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)  # never a half-written file that the next run would skip
    return path


def load_entries(frozen_dir: Path, chart: list[StandardAccount]) -> dict[str, GroupDescription]:
    """Every group's frozen description; fails unless the chart is covered exactly once."""
    folder = Path(frozen_dir)
    files = sorted(folder.glob("batch-*.json")) if folder.is_dir() else []
    if not files:
        raise DescriptionError(f"no descriptions in {folder}: run `{COMMAND}` first")
    entries: dict[str, GroupDescription] = {}
    for f in files:
        for raw in json.loads(f.read_text(encoding="utf-8"))["response"]["groups"]:
            e = GroupDescription.model_validate(raw)
            if e.code in entries:
                raise DescriptionError(f"group {e.code} is described twice ({f.name})")
            entries[e.code] = e
    codes = {a.code for a in chart}
    unknown = sorted(set(entries) - codes)
    if unknown:
        raise DescriptionError(f"described groups not in the chart: {', '.join(unknown)}")
    missing = sorted(codes - set(entries))
    if missing:
        raise DescriptionError(f"no description for {len(missing)} groups ({', '.join(missing[:5])}...): run `{COMMAND}`")
    return entries


def load_descriptions(frozen_dir: Path, chart: list[StandardAccount]) -> dict[str, str]:
    """group code -> the text added to that group in retrieval."""
    return {code: e.text() for code, e in load_entries(frozen_dir, chart).items()}


def main(argv: list[str] | None = None, root: Path = REPO_ROOT, backend: LLMBackend | None = None) -> int:
    argparse.ArgumentParser(prog=COMMAND, description=__doc__.split("\n")[0]).parse_args(argv)
    try:
        config = load_config(root)
        chart = load_standard_chart(config.data.chart_path, expected=config.data.expected_groups)
        backend = backend or make_backend(config)
        folder = config.data.frozen_descriptions_dir
        for number, groups in batches(chart):
            done = _batch_path(folder, number).exists()
            fetch_batch(backend, writer_model=config.models.description_model, generator_model=config.models.generator_model,
                        number=number, groups=groups, frozen_dir=folder)  # fmt: skip
            print(f"batch {number:02d} ({groups[0].code}-{groups[-1].code}): {'already frozen' if done else 'written'}", flush=True)
        load_entries(folder, chart)  # all 190, exactly once
    except (ConfigError, ChartError, LLMError, DescriptionError) as e:
        print(f"error: {e} (frozen batches are kept, so run it again to continue)", file=sys.stderr)
        return 1
    print(f"190 group descriptions in {folder.relative_to(config.root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
