"""Data shapes shared across D1: standard accounts, company accounts, JSONL I/O."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from d1.config import Language, Split

__all__ = ["Language", "Split", "StandardAccount", "CompanyAccount", "write_jsonl", "read_jsonl"]


class StandardAccount(BaseModel):
    """One 3-digit group of the Belgian standard chart: a mapping target."""

    model_config = ConfigDict(frozen=True)
    
    code: str = Field(pattern=r"^\d{3}$") 
    code_end: str | None = None  
    parent_code: str | None  
    class_code: str = Field(pattern=r"^\d$")
    name_en: str = Field(min_length=1)
    name_fr: str = Field(min_length=1)
    name_nl: str = Field(min_length=1)
    parent_name_en: str  
    class_name_en: str

    def name(self, lang: Literal["nl", "fr", "en"]) -> str:
        return {"en": self.name_en, "fr": self.name_fr, "nl": self.name_nl}[lang]


class CompanyAccount(BaseModel):
    """One account in a synthetic company chart, with its correct target group."""

    account_id: str
    company_id: str
    industry: str
    split: Split
    source: Literal["rules", "llm"] # -- either the LLM or the noise rules
    language: Language
    code_scheme: Literal["pcmn", "legacy", "llm"]
    code: str
    name: str = Field(min_length=1)
    noise: list[str]
    target_code: str  
    overlaps_dev: bool = False  
    # unchecked: not reviewed yet; constructed: rule-based row outside the hand-check
    # sample (label correct by construction); checked / corrected: hand-checked.
    label_status: Literal["unchecked", "constructed", "checked", "corrected"] = "unchecked"


def write_jsonl(rows: list[BaseModel], path: Path) -> None:
    """Write one JSON object per line, creating the parent folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row.model_dump_json() + "\n")


def read_jsonl(path: Path) -> list[CompanyAccount]:
    """Read company accounts written by write_jsonl, skipping blank lines."""
    with open(path, encoding="utf-8") as f:
        return [CompanyAccount.model_validate_json(line) for line in f if line.strip()]
