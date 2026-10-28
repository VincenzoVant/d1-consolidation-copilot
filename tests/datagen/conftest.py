import re
from pathlib import Path

import pytest
import yaml

from d1.config import REPO_ROOT
from d1.llm import LLMResponse

REAL_CHART = REPO_ROOT / "data/source/odoo_l10n_be/account.group-be.csv"

# Small splits: the same structure as the real config, built in well under a second.
SMALL_SPLITS = {
    "train": {"seed": 3003, "rule_companies": 3, "rule_accounts_per_company": 20, "llm_companies": 2, "llm_accounts_per_company": 10},
    "dev": {"seed": 1001, "rule_companies": 2, "rule_accounts_per_company": 20, "llm_companies": 1, "llm_accounts_per_company": 10},
    "test": {"seed": 2002, "rule_companies": 3, "rule_accounts_per_company": 20, "llm_companies": 2, "llm_accounts_per_company": 10},
}


class FakeChartWriter:
    """Stands in for the generator model: reads the groups and count from the prompt
    and answers with a valid chart (one account per group, then second accounts)."""

    def __init__(self, model="claude-opus-5-5"):
        self.model = model
        self.calls = 0

    def complete(self, prompt, schema, model, system, timeout_s=120.0):
        self.calls += 1
        codes = re.findall(r"^\| (\d{3}) \|", prompt, flags=re.MULTILINE)
        n = int(re.search(r"exactly (\d+) accounts", prompt).group(1))
        rows = [{"target_code": c, "code": f"{c}0", "name": f"Rekening {c}"} for c in codes]
        rows += [{"target_code": c, "code": f"{c}1", "name": f"Rekening {c} - Gent"} for c in codes]
        return LLMResponse(structured={"accounts": rows[:n]}, model=self.model, cost_usd=0.03, latency_ms=5)


def _point_at_real_chart(root: Path, **changes) -> Path:
    path = root / "config" / "data.yaml"
    data = yaml.safe_load(path.read_text())
    data.update(chart_path=str(REAL_CHART), **changes)
    path.write_text(yaml.safe_dump(data))
    return root


@pytest.fixture
def fake_writer() -> FakeChartWriter:
    return FakeChartWriter()


@pytest.fixture
def full_root(config_root) -> Path:
    """A tmp repo root with the real split sizes; generated and frozen files go to tmp."""
    return _point_at_real_chart(config_root)


@pytest.fixture
def small_root(config_root) -> Path:
    """A tmp repo root with small splits (5 + 3 + 5 companies, 5 of them LLM-written)."""
    return _point_at_real_chart(config_root, splits=SMALL_SPLITS, review_rule_sample=10)
