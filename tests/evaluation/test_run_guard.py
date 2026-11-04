import json

import pytest

from d1.config import load_config
from d1.datagen.schema import read_jsonl, write_jsonl
from d1.evaluation.run_mapping import main


def _run(root, FakeEmbedder, *args):
    return main(list(args), root=root, embedder_factory=lambda name: FakeEmbedder())


def _only_run_dir(root):
    (run,) = (root / "reports" / "results").iterdir()
    return run


# --- dev runs -----------------------------------------------------------------------------


def test_dev_run_writes_metrics_and_tables(trained_root, FakeEmbedder, capsys):
    assert _run(trained_root, FakeEmbedder, "--split", "dev", "--method", "fuzzy") == 0
    run = _only_run_dir(trained_root)
    assert run.name.endswith("-dev-fuzzy")
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["split"] == "dev" and metrics["method"] == "fuzzy" and metrics["n"] == 200
    for key in ("auto_share", "auto_precision", "review_share", "top1", "top3", "recall_at_k"):
        assert key in metrics["overall"]
    low, high = metrics["overall"]["auto_share_ci"]
    assert low <= metrics["overall"]["auto_share"] <= high
    assert {"source", "language", "noise", "industry", "overlaps_dev"} <= set(metrics["breakdowns"])
    assert metrics["model_versions"]["method"].startswith("fuzzy")
    tables = (run / "tables.md").read_text()
    assert tables.startswith("# fuzzy on dev") and "| language |" in tables
    assert "auto-mapped" in capsys.readouterr().out


def test_dev_run_is_not_logged_as_a_test_run(trained_root, FakeEmbedder):
    _run(trained_root, FakeEmbedder, "--split", "dev", "--method", "fuzzy")
    assert not (trained_root / "reports" / "test_runs.jsonl").exists()


# --- the guard on test ----------------------------------------------------------------------


def test_test_split_requires_reported_flag(trained_root, FakeEmbedder, capsys):
    assert _run(trained_root, FakeEmbedder, "--split", "test", "--method", "fuzzy") == 1
    assert "--reported" in capsys.readouterr().err
    assert not (trained_root / "reports" / "results").exists()  # nothing was run


def test_reported_run_is_logged(trained_root, FakeEmbedder):
    assert _run(trained_root, FakeEmbedder, "--split", "test", "--method", "fuzzy", "--reported") == 0
    (entry,) = [json.loads(line) for line in (trained_root / "reports" / "test_runs.jsonl").read_text().splitlines()]
    assert entry["method"] == "fuzzy" and entry["run_id"] == _only_run_dir(trained_root).name
    assert len(entry["config_sha256"]) == 64 and "git_sha" in entry and "timestamp" in entry
    # A second reported run adds a line; the log is never rewritten.
    _run(trained_root, FakeEmbedder, "--split", "test", "--method", "fuzzy", "--reported")
    assert len((trained_root / "reports" / "test_runs.jsonl").read_text().splitlines()) == 2


def test_test_split_uses_checked_file(trained_root, FakeEmbedder):
    config = load_config(trained_root)
    checked = config.data.generated_dir / "test_checked.jsonl"
    # Change one label only in the checked file: the run must see that label, not the one in test.jsonl.
    rows = read_jsonl(checked)
    rows = [rows[0].model_copy(update={"target_code": "100", "label_status": "corrected"})] + rows[1:]
    write_jsonl(rows, checked)
    _run(trained_root, FakeEmbedder, "--split", "test", "--method", "fuzzy", "--reported")
    metrics = json.loads((_only_run_dir(trained_root) / "metrics.json").read_text())
    assert metrics["labels_file"] == "test_checked.jsonl" and metrics["n"] == len(rows)
    assert metrics["label_status"] == {"checked": len(rows) - 1, "corrected": 1}


def test_test_split_without_checked_file_names_apply_review(trained_root, FakeEmbedder, capsys):
    (load_config(trained_root).data.generated_dir / "test_checked.jsonl").unlink()
    assert _run(trained_root, FakeEmbedder, "--split", "test", "--method", "fuzzy", "--reported") == 1
    assert "apply-review" in capsys.readouterr().err


def test_train_split_refused(trained_root, FakeEmbedder, capsys):
    assert _run(trained_root, FakeEmbedder, "--split", "train", "--method", "fuzzy") == 1
    assert "trained on" in capsys.readouterr().err


def test_untrained_method_names_the_train_command(trained_root, FakeEmbedder, capsys):
    assert _run(trained_root, FakeEmbedder, "--split", "dev", "--method", "embeddings") == 1
    assert "python -m d1.mapper.train --method embeddings" in capsys.readouterr().err


def test_dev_run_matches_the_training_summary(trained_root, FakeEmbedder):
    # Training reports its dev numbers with the saved calibrator, exactly as the Mapper works,
    # so an evaluation run on dev can never disagree with them.
    _run(trained_root, FakeEmbedder, "--split", "dev", "--method", "fuzzy")
    metrics = json.loads((_only_run_dir(trained_root) / "metrics.json").read_text())
    summary = json.loads((trained_root / "artifacts" / "methods" / "fuzzy" / "dev_summary.json").read_text())
    assert metrics["overall"]["auto_share"] == summary["auto_share"]
    assert metrics["overall"]["auto_precision"] == summary["auto_precision"]
    assert metrics["overall"]["top1"] == summary["top1"]


def test_dev_run_writes_one_decision_line_per_account(trained_root, FakeEmbedder):
    import csv

    from d1.evaluation.run_mapping import DECISION_COLUMNS

    _run(trained_root, FakeEmbedder, "--split", "dev", "--method", "fuzzy")
    with open(_only_run_dir(trained_root) / "decisions.csv", encoding="utf-8", newline="") as f:
        lines = list(csv.DictReader(f))
    assert len(lines) == 200 and list(lines[0]) == DECISION_COLUMNS
    assert {line["decision_path"] for line in lines} <= {"auto", "review"}
    assert all(line["right"] in ("1", "0") for line in lines)
