import shutil
from random import Random

import pytest

from d1.config import REPO_ROOT, load_config
from d1.datagen.rule_charts import generate_rule_company
from d1.datagen.schema import write_jsonl


@pytest.fixture
def mapper_root(config_root, chart):
    """A repo root with the real chart and small train/dev splits (no test split at all)."""
    shutil.copytree(REPO_ROOT / "data" / "source", config_root / "data" / "source")
    config = load_config(config_root)
    for split, n_companies in (("train", 2), ("dev", 4)):
        rows = []
        for c, language in enumerate(["nl", "fr", "en", "mixed"][:n_companies]):
            rows += generate_rule_company(f"{split}-c{c:03d}", split, "retail", language, chart, 50, Random(c), code_scheme="legacy")
        # Mislabel every fourth row, so even an exact-name method is sometimes wrong (calibration needs both).
        rows = [r.model_copy(update={"target_code": "604" if r.target_code != "604" else "600"}) if i % 4 == 0 else r for i, r in enumerate(rows)]
        write_jsonl(rows, config.data.generated_dir / f"{split}.jsonl")
    return config_root


class _FirstPickBackend:
    """Stands in for Claude: picks each account's first candidate, more sure for odd-numbered labels.

    Asked for a ranked list, it ranks the account's candidates in their listed order. asked collects
    the account names of every call, so a test can see which accounts reached the LLM.
    """

    def __init__(self):
        self.asked: list[str] = []

    def complete(self, prompt, schema, model, system, timeout_s=120.0):
        from d1.llm import LLMResponse

        answers = []
        item = schema["properties"]["answers"]["items"]["properties"]
        for label in item["account"]["enum"]:
            block = prompt.split(f"## Account {label}\n")[1].split("## Account")[0]
            self.asked.append(block.split("Name: ")[1].split("\n")[0])
            codes = [line[2:].split(":")[0] for line in block.split("Candidate groups:\n")[1].splitlines() if line.startswith("- ")]
            top = 90 if int(label[1:]) % 2 else 40
            if "choices" in item:
                choices = [{"code": c, "probability": max(top - 10 * k, 1)} for k, c in enumerate(codes[:5])]
                answers.append({"account": label, "choices": choices, "reason": "first"})
            else:
                answers.append({"account": label, "choice": codes[0], "confidence": top, "reason": "first"})
        return LLMResponse(structured={"answers": answers}, model=model, latency_ms=1.0, cost_usd=0.001)


@pytest.fixture
def FirstPickBackend():
    return _FirstPickBackend
