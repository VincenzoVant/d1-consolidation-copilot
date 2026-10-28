"""Load and validate the YAML files in config/.

Every module reads settings through load_config(), so a typo or an
inconsistent value fails once, at start-up, with the file name in the message.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]

Language = Literal["nl", "fr", "en", "mixed"]
Split = Literal["train", "dev", "test"]
Method = Literal["fuzzy", "embeddings", "xgboost", "random_forest", "laya", "llm"]


class ConfigError(Exception):
    """A config file is missing, unreadable or inconsistent."""


class _Strict(BaseModel):
    # Unknown keys are errors, so a misspelt setting cannot be silently ignored.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Price(_Strict):
    input: float = Field(ge=0)
    output: float = Field(ge=0)


class MappingConfig(_Strict):
    candidates_k: int = Field(ge=2, le=20)  # Laya degrades above ~20 options
    target_precision: float = Field(gt=0, le=1)
    methods: list[Method] = Field(min_length=1)
    laya_model: str
    selected_method: Method | None = None

    @model_validator(mode="after")
    def _check(self) -> "MappingConfig":
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods contains a duplicate")
        if self.selected_method is not None and self.selected_method not in self.methods:
            raise ValueError(f"selected_method {self.selected_method} is not in methods")
        return self


class ModelsConfig(_Strict):
    generator_model: str
    mapping_model: str
    llm_backend: Literal["claude_cli", "anthropic_api"]
    embedding_model: str
    embedding_candidates: list[str] = Field(min_length=1)
    llm_prices_per_mtok: dict[str, Price]
    mapping: MappingConfig

    @model_validator(mode="after")
    def _check(self) -> "ModelsConfig":
        if self.generator_model == self.mapping_model:
            raise ValueError("generator_model and mapping_model must differ")
        if self.mapping_model not in self.llm_prices_per_mtok:
            raise ValueError(f"llm_prices_per_mtok has no entry for {self.mapping_model}")
        return self


class SplitConfig(_Strict):
    seed: int
    rule_companies: int = Field(ge=0)
    rule_accounts_per_company: int = Field(ge=1)
    llm_companies: int = Field(ge=0)
    llm_accounts_per_company: int = Field(ge=1)

    @property
    def n_companies(self) -> int:
        return self.rule_companies + self.llm_companies

    @property
    def n_rule_accounts(self) -> int:
        return self.rule_companies * self.rule_accounts_per_company

    @property
    def n_llm_accounts(self) -> int:
        return self.llm_companies * self.llm_accounts_per_company

    @property
    def n_accounts(self) -> int:
        return self.n_rule_accounts + self.n_llm_accounts


class DataConfig(_Strict):
    chart_path: Path
    expected_groups: int = Field(ge=1)
    industries: list[str] = Field(min_length=1)
    languages: list[Language] = Field(min_length=1)
    splits: dict[Split, SplitConfig]
    generated_dir: Path
    frozen_llm_dir: Path
    review_path: Path
    review_rule_sample: int = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> "DataConfig":
        missing = {"train", "dev", "test"} - set(self.splits)
        if missing:
            raise ValueError(f"splits is missing: {', '.join(sorted(missing))}")
        seeds = [split.seed for split in self.splits.values()]
        if len(set(seeds)) != len(seeds):
            raise ValueError("every split must use a different seed")
        for name, split in self.splits.items():
            if split.n_companies == 0:
                raise ValueError(f"split {name} has no companies")
        if self.review_rule_sample > self.splits["test"].n_rule_accounts:
            raise ValueError("review_rule_sample is larger than the number of rule-based test rows")
        return self


class Config(_Strict):
    root: Path
    data: DataConfig
    models: ModelsConfig


_PATH_FIELDS = ("chart_path", "generated_dir", "frozen_llm_dir", "review_path")


def _read_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"{path.name}: file not found at {path}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path.name}: invalid YAML: {e}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: expected a mapping at the top level")
    return data


def _validate(model: type[BaseModel], data: dict, filename: str):
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"{filename}: {e}") from None


def load_config(root: Path = REPO_ROOT) -> Config:
    """Read config/models.yaml and config/data.yaml under root.

    Relative paths in data.yaml are resolved against root.
    """
    root = Path(root).resolve()
    data_raw = _read_yaml(root / "config" / "data.yaml")
    for key in _PATH_FIELDS:
        if key in data_raw:
            data_raw[key] = root / data_raw[key]
    data = _validate(DataConfig, data_raw, "data.yaml")
    models = _validate(ModelsConfig, _read_yaml(root / "config" / "models.yaml"), "models.yaml")
    return Config(root=root, data=data, models=models)
