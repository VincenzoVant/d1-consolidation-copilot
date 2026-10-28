from pathlib import Path

import pytest
import yaml

from d1.config import REPO_ROOT, ConfigError, load_config


def _edit(root: Path, name: str, **changes) -> None:
    path = root / "config" / name
    data = yaml.safe_load(path.read_text())
    data.update(changes)
    path.write_text(yaml.safe_dump(data))


def test_load_config_reads_repo_files():
    cfg = load_config()
    assert cfg.data.expected_groups == 190
    test = cfg.data.splits["test"]
    assert test.n_accounts == 600 and test.n_llm_accounts == 200
    assert test.n_llm_accounts + cfg.data.review_rule_sample == 250  # rows to check by hand
    assert cfg.data.splits["dev"].n_accounts == 3500
    assert cfg.data.splits["train"].n_accounts == 12000
    assert cfg.models.mapping_model == "claude-haiku-5-5"
    assert cfg.models.llm_backend == "claude_cli"
    assert cfg.models.mapping.methods == ["fuzzy", "embeddings", "xgboost", "random_forest", "laya", "llm"]
    assert cfg.models.mapping.selected_method is None
    assert cfg.models.embedding_model in cfg.models.embedding_candidates


def test_paths_resolved_against_root(config_root):
    cfg = load_config(config_root)
    assert cfg.data.chart_path == config_root / "data/source/odoo_l10n_be/account.group-be.csv"
    assert cfg.data.generated_dir.is_absolute()
    assert load_config().data.chart_path.is_file()


def test_generator_and_mapping_model_must_differ(config_root):
    _edit(config_root, "models.yaml", generator_model="claude-haiku-5-5")
    with pytest.raises(ConfigError, match="must differ"):
        load_config(config_root)


def test_split_seeds_must_differ(config_root):
    path = config_root / "config" / "data.yaml"
    data = yaml.safe_load(path.read_text())
    data["splits"]["test"]["seed"] = data["splits"]["train"]["seed"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="seed"):
        load_config(config_root)


def test_all_three_splits_required(config_root):
    path = config_root / "config" / "data.yaml"
    data = yaml.safe_load(path.read_text())
    del data["splits"]["train"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="train"):
        load_config(config_root)


def _edit_mapping(root: Path, **changes) -> None:
    path = root / "config" / "models.yaml"
    data = yaml.safe_load(path.read_text())
    data["mapping"].update(changes)
    path.write_text(yaml.safe_dump(data))


def test_too_many_candidates_rejected(config_root):
    _edit_mapping(config_root, candidates_k=25)
    with pytest.raises(ConfigError, match="candidates_k"):
        load_config(config_root)


def test_selected_method_must_be_compared(config_root):
    _edit_mapping(config_root, methods=["fuzzy", "xgboost"], selected_method="laya")
    with pytest.raises(ConfigError, match="selected_method"):
        load_config(config_root)


def test_unknown_method_and_backend_rejected(config_root):
    _edit_mapping(config_root, methods=["svm"])
    with pytest.raises(ConfigError, match="models.yaml"):
        load_config(config_root)
    _edit_mapping(config_root, methods=["fuzzy"])
    _edit(config_root, "models.yaml", llm_backend="openai")
    with pytest.raises(ConfigError, match="llm_backend"):
        load_config(config_root)


def test_unknown_key_rejected(config_root):
    _edit(config_root, "models.yaml", mapping_modle="typo")
    with pytest.raises(ConfigError, match="models.yaml"):
        load_config(config_root)


def test_missing_file_raises_config_error(config_root):
    (config_root / "config" / "data.yaml").unlink()
    with pytest.raises(ConfigError, match="data.yaml"):
        load_config(config_root)


def test_invalid_yaml_raises_config_error(config_root):
    (config_root / "config" / "models.yaml").write_text("generator_model: [unclosed\n")
    with pytest.raises(ConfigError, match="models.yaml"):
        load_config(config_root)


def test_review_sample_cannot_exceed_rule_test_rows(config_root):
    _edit(config_root, "data.yaml", review_rule_sample=10_000)
    with pytest.raises(ConfigError, match="review_rule_sample"):
        load_config(config_root)
