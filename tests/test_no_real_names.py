"""No real (or invented) company, bank, brand or person names in the data or anywhere in the repo.

The LLM-written charts first held names like the company's banks, software
suppliers and subsidiaries; they were replaced by generic words (see
data/frozen/llm_charts/*.json, "names_scrubbed"). These tests keep it that way.

The banned names are stored as hashes, so this file names nobody either. A text
is cut into words (once at every non-letter, once keeping "." and "-" inside a
word) and every word, word pair and word triple is hashed and looked up.
"""

import csv
import hashlib
import json
import re
from pathlib import Path

import pytest

from d1.config import REPO_ROOT

ORGS = frozenset({
    "ee9021c84f33366b",
    "b026c66d55951f76",
    "1206d210d9d40e72",
    "5344ae4677efc34b",
    "bac8736b4055203b",
    "b369b5e9c27e4bbc",
    "f3daf83f5feeabd2",
    "651b2c67d5361363",
    "b51026e4444f98ec",
    "50f08e64c5a6192c",
    "7d1507284a5757ca",
    "79f43c3b9c23fbdf",
    "7ee073a80eff3bf2",
    "a425cae4bd837e7c",
    "73928fd07ddea2ab",
    "4b7d8a710be7f36d",
    "95aca8501bbd9aaf",
    "50be81be0b536010",
    "6952b2ee90c8382e",
    "e3ea121183668bf6",
    "96680c4456e510ac",
    "f3dd35c3710e1a45",
    "02953a6c6db2975e",
    "f1454f676ceb2558",
    "f945a6a2d4c867b6",
    "b2f9ff6a9b85e196",
    "9e92e57403549bb2",
    "b004e65215cc622b",
    "25c2afa75e27befd",
    "66a8d715f0b8d886",
    "e9a91839b938c9f0",
    "34f0b09b3a971ad8",
    "95cdde78204069f5",
    "ddc1daf94653ea78",
    "9b90f817e44d9be6",
    "2ab1cf5850901c9e",
    "2b8151e682bdc9a7",
    "9c03d5590832c6b7",
    "a5cba03a94205501",
    "65c9fbc27700aaef",
    "5d058152e20a05ec",
    "86783bed199c7854",
    "41215b9b3eb2426d",
    "b4625cf2307b8770",
    "9ebbdd6ae35bbc59",
    "58b31f8e432a330b",
    "74182a1f33627ae0",
    "de4bb8fcff6f529f",
    "b364af0199546fcc",
    "e74a5f91c69f2118",
    "6a106c4ba8cf9eac",
    "50493d190ca7301d",
    "abef54f42989348f",
    "12af27e00f3e57b2",
    "1508762a1758bc78",
    "9d9313777e46caeb",
    "296c41d29aa6e142",
    "2382e4b16b7f5294",
    "30b98ca8dbfd7ae4",
    "33c74b669c6f80a6",
    "027d59970c306e9c",
    "d239a449d3fbd0fe",
    "bf900933c14ab46d",
    "62e298811207a8f9",
    "f84171f2f1bb54c2",
    "381af75ea13c414c",
    "12723d560fbf189f",
    "5c046bec7e12bb50",
    "f3526c3f0020fe82",
    "7c06b7dc46340b8c",
    "699e98ee9ed5a3f0",
    "33aa3535c4995e7e",
    "1296caae771fe519",
    "7070dc8509c487fe",
    "371e8fa7e8c4e224",
    "422b3f0476b564a4",
    "dd3921aecdf70d68",
    "91af7d847f80f22c",
    "85ccf5df294271b4",
    "6c2949a5b9698d22",
    "7e2db47e76b90667",
    "aa7d397c4ccf6faa",
    "935f43036f95cf22",
    "4a5b5a6a8e2d15af",
    "917ebb3396b2ff2e",
    "9fbf261b62c1d7c0",
    "3b03d4b408671f86",
    "edb76738a48f2257",
    "a3df02f9cc7330e9",
    "2a0a94880924c956",
    "3fb498e174078991",
    "280714648a11a776",
    "5a80dd03c4354fef",
    "e95abc84ee5a3e9f",
    "d0cf59f28314aa8e",
    "2795ac99d9275d7b",
    "7679289f40e76fac",
    "f993411275cf8e4e",
    "900d1c1837bc980e",
    "107430fff8393f1b",
    "af886bf2598756cb",
    "573abb3ca06e6abb",
    "c15d426909238eef",
    "862efdc617054597",
    "de4e9e60784006b2",
    "f1a206d9615c86db",
    "2c2375f7c68a842c",
    "e57f6fd1bb31fa04",
    "286389827bb038c0",
    "1ebb7ba14a0d7f9b",
    "ba6240fa3683935c",
    "78944f78d5ae1080",
    "dbbec30d468431a8",
    "55e6b83841f3d0e3",
    "1a036108f2915823",
    "2bed2032ed3d8300",
    "8a261ec240791ac6",
    "381eb9c4a65808d4",
    "1f9d5f1c5a968571",
    "6a43529c5d88bff2",
    "50ebd222d24323f5",
    "fa55b1d9f3d15546",
    "68522c83e0ceb7d3",
    "456121013305d00e",
    "6e1ae5dc1ae867f5",
})
DATA_ONLY = frozenset({
    "bf9866e2ef5a1e1a",
    "ef94814c0bb166eb",
    "f7358372358d1b28",
    "b043864b7ffb308f",
    "7bd4d92935007807",
    "09474e10a624857f",
    "404e91050d105f97",
    "65cac3e92782e3cd",
    "63891ae740856864",
    "77754110e188ce7c",
    "5b381c47cfd78c0e",
    "2ef83b0808111bb9",
})

ALL_NAMES = ORGS | DATA_ONLY  # people and the data source's name may appear in the docs, never in the accounts

FINE = re.compile(r"[0-9a-zà-ÿ]+")
COARSE = re.compile(r"[0-9a-zà-ÿ]+(?:[.\-][0-9a-zà-ÿ]+)*")
SKIP_DIRS = {".git", ".venv", ".cache", "__pycache__", ".pytest_cache", "generated"}
SKIP_FILES = {"uv.lock", "requirements.txt"}
TEXT_SUFFIXES = {".md", ".py", ".json", ".jsonl", ".csv", ".yaml", ".yml", ".txt", ".toml", ".ipynb"}


def _hash(term: str) -> str:
    return hashlib.sha256(term.encode()).hexdigest()[:16]


def found(text: str, banned: frozenset[str]) -> list[str]:
    """The words (and word pairs and triples) of text whose hash is banned."""
    hits = set()
    low = text.lower()
    for pattern in (FINE, COARSE):
        words = pattern.findall(low)
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                term = " ".join(words[i : i + n])
                if _hash(term) in banned:
                    hits.add(term)
    return sorted(hits)


def _notebook_text(path: Path) -> str:
    """Sources and text outputs only: image data would give random short words."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts = []
    for cell in nb["cells"]:
        parts.append("".join(cell.get("source", [])))
        for out in cell.get("outputs", []):
            parts.append("".join(out.get("text", [])))
            for kind, value in out.get("data", {}).items():
                if kind.startswith("text/"):
                    parts.append("".join(value))
    return "\n".join(parts)


def _repo_files():
    for path in sorted(REPO_ROOT.rglob("*")):
        rel = path.relative_to(REPO_ROOT)
        if any(p in SKIP_DIRS for p in rel.parts) or path.name in SKIP_FILES:
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES and path.name != Path(__file__).name:
            yield rel, path


def test_the_lookup_finds_a_name_in_a_word_pair_and_inside_a_hyphenated_word():
    banned = frozenset({_hash("acme"), _hash("acme bank")})
    assert found("Current account Acme", banned) == ["acme"]
    assert found("Loan ACME Bank 2024", banned) == ["acme", "acme bank"]
    assert found("Acme-payouts suspense", banned) == ["acme"]
    assert found("Current account", banned) == []


def test_account_names_hold_no_company_brand_or_person_names():
    problems = []
    for f in sorted((REPO_ROOT / "data" / "frozen" / "llm_charts").glob("*.json")):
        for account in json.loads(f.read_text(encoding="utf-8"))["response"]["accounts"]:
            if hits := found(account["name"], ALL_NAMES):
                problems.append(f"{f.name}: {account['name']!r} {hits}")
    with open(REPO_ROOT / "data" / "labels" / "test_review.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            if hits := found(row["name"], ALL_NAMES) + found(row["note"], ORGS):
                problems.append(f"test_review.csv {row['account_id']}: {hits}")
    assert problems == []


def test_repo_files_hold_no_company_or_brand_names():
    problems = []
    for rel, path in _repo_files():
        text = _notebook_text(path) if path.suffix == ".ipynb" else path.read_text(encoding="utf-8", errors="ignore")
        if hits := found(text, ORGS):
            problems.append(f"{rel}: {hits[:5]}")
    assert problems == []


@pytest.mark.parametrize("folder", ["llm_charts"])
def test_scrubbed_charts_say_so(folder):
    files = sorted((REPO_ROOT / "data" / "frozen" / folder).glob("*.json"))
    scrubbed = [f for f in files if "names_scrubbed" in json.loads(f.read_text(encoding="utf-8"))]
    assert len(scrubbed) == len(files) == 78
