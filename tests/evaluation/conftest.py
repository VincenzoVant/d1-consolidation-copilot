import shutil
from random import Random

import pytest

from d1.config import REPO_ROOT, load_config
from d1.datagen.rule_charts import generate_rule_company
from d1.datagen.schema import write_jsonl
from d1.mapper.train import train_method


def _split(split, chart, n_companies, label_status="unchecked"):
    rows = []
    for c, language in enumerate(["nl", "fr", "en", "mixed"][:n_companies]):
        rows += generate_rule_company(f"{split}-c{c:03d}", split, "retail", language, chart, 50, Random({"dev": 100, "test": 200}[split] + c), code_scheme="legacy")
    # Mislabel every fourth row, so methods are sometimes wrong and calibration sees both outcomes.
    rows = [r.model_copy(update={"target_code": "604" if r.target_code != "604" else "600"}) if i % 4 == 0 else r for i, r in enumerate(rows)]
    return [r.model_copy(update={"label_status": label_status}) for r in rows]


@pytest.fixture
def trained_root(config_root, chart, FakeEmbedder):
    """A repo root with the real chart, small dev and test splits, and a trained fuzzy method."""
    shutil.copytree(REPO_ROOT / "data" / "source", config_root / "data" / "source")
    config = load_config(config_root)
    out = config.data.generated_dir
    write_jsonl(_split("dev", chart, 4), out / "dev.jsonl")
    write_jsonl(_split("test", chart, 2), out / "test.jsonl")
    write_jsonl(_split("test", chart, 2, label_status="checked"), out / "test_checked.jsonl")
    train_method(config, "fuzzy", lambda name: FakeEmbedder())
    return config_root
