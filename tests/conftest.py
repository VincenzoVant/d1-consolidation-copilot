import shutil
from pathlib import Path

import pytest

from d1.config import REPO_ROOT

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def config_root(tmp_path: Path) -> Path:
    """A throwaway repo root holding a copy of config/, safe to edit in a test."""
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    return tmp_path
