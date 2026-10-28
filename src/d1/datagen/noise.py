"""Rule-based noise: make clean standard account names look like real ledger entries.

Each operation takes a name and returns a changed name. apply_noise() runs them
in a fixed order, each with its own probability, and reports which ones
actually changed the string, so results can later be broken down by noise type.

All randomness comes from the Random object passed in, so the same seed always
produces the same noisy name.
"""

import re
import unicodedata
from random import Random

# Per language: lowercase word or phrase -> abbreviation (as a bookkeeper would type it).
ABBREVIATIONS: dict[str, dict[str, str]] = {
    "nl": {
        "afschrijvingen": "afschr.",
        "voorzieningen": "voorz.",
        "vorderingen": "vord.",
        "leveranciers": "lev.",
        "materiële": "mat.",
        "immateriële": "immat.",
        "waardeverminderingen": "waardeverm.",
        "belastingen": "bel.",
        "handelsgoederen": "handelsg.",
        "grondstoffen": "grondst.",
        "bezoldigingen": "bezold.",
        "financiële": "fin.",
        "rekeningen": "rek.",
    },
    "fr": {
        "réductions de valeur": "RDV",
        "immobilisations": "immob.",
        "amortissements": "amort.",
        "provisions": "prov.",
        "fournisseurs": "fourn.",
        "créances": "créan.",
        "financières": "fin.",
        "corporelles": "corp.",
        "incorporelles": "incorp.",
        "marchandises": "march.",
        "rémunérations": "rémun.",
        "approvisionnements": "approv.",
    },
    "en": {
        "depreciation": "depr.",
        "provisions": "prov.",
        "receivables": "rec.",
        "payables": "pay.",
        "intangible": "intang.",
        "tangible": "tang.",
        "financial": "fin.",
        "amounts": "amts.",
        "remuneration": "remun.",
        "miscellaneous": "misc.",
        "operating": "oper.",
    },
}

# Probability that each operation is attempted, in the order they are applied:
# abbreviate first (typos would stop words matching the dictionary), truncate last.
NOISE_PROBS: dict[str, float] = {
    "abbreviation": 0.4,
    "typo": 0.15,
    "case": 0.25,
    "accents_stripped": 0.2,
    "truncated": 0.1,
}


def _pattern(language: str) -> re.Pattern[str]:
    """One regex matching any dictionary entry as a whole word, longest entries first."""
    if language not in ABBREVIATIONS:
        raise ValueError(f"no abbreviations for language {language!r}; use one of {sorted(ABBREVIATIONS)}")
    entries = sorted(ABBREVIATIONS[language], key=len, reverse=True)
    # \b only matches at a word boundary, so "tangible" is not found inside "intangible".
    return re.compile(r"\b(" + "|".join(map(re.escape, entries)) + r")\b", re.IGNORECASE)


_PATTERNS = {language: _pattern(language) for language in ABBREVIATIONS}


def abbreviate(name: str, language: str) -> str:
    """Replace every known word or phrase with its abbreviation.

    A replaced word keeps the capital of its first letter:
    "Afschrijvingen op materiële" -> "Afschr. op mat.".
    """
    pattern = _PATTERNS.get(language) or _pattern(language)  # unknown language raises

    def replace(match: re.Match[str]) -> str:
        found = match.group(0)
        short = ABBREVIATIONS[language][found.lower()]
        return short[0].upper() + short[1:] if found[0].isupper() else short

    return pattern.sub(replace, name)


def typo(name: str, rng: Random) -> str:
    """Make one typo (swap, drop or double a letter) in one word of 5+ letters."""
    words = name.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not candidates:
        return name
    i = rng.choice(candidates)
    word = words[i]
    kind = rng.choice(["swap", "drop", "double"])
    if kind == "swap":
        # Only swap two different letters, otherwise the "typo" changes nothing.
        spots = [j for j in range(1, len(word) - 1) if word[j] != word[j + 1]]
        if not spots:
            kind = "drop"
        else:
            j = rng.choice(spots)
            word = word[:j] + word[j + 1] + word[j] + word[j + 2 :]
    if kind == "drop":
        j = rng.randrange(1, len(word))
        word = word[:j] + word[j + 1 :]
    elif kind == "double":
        j = rng.randrange(1, len(word))
        word = word[:j] + word[j] + word[j:]
    words[i] = word
    return " ".join(words)


def strip_accents(name: str) -> str:
    """Remove accents, as old systems without é/ë support do: "Créances" -> "Creances"."""
    decomposed = unicodedata.normalize("NFKD", name)  # "é" becomes "e" + a separate accent mark
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def change_case(name: str, rng: Random) -> str:
    """Write the whole name in UPPER or lower case."""
    return name.upper() if rng.random() < 0.5 else name.lower()


def truncate(name: str, max_len: int = 30) -> str:
    """Cut to max_len characters, as legacy systems with short name fields do."""
    return name[:max_len].rstrip()


def apply_noise(name: str, language: str, rng: Random) -> tuple[str, list[str]]:
    """Apply each operation with its NOISE_PROBS probability.

    Returns the noisy name and the tags of the operations that changed it.
    All on/off decisions are drawn first, one per operation, so changing one
    probability never changes which of the other operations fire.
    """
    draws = {op: rng.random() for op in NOISE_PROBS}
    tags = []
    for op, p in NOISE_PROBS.items():
        if draws[op] >= p:
            continue
        if op == "abbreviation":
            new = abbreviate(name, language)
        elif op == "typo":
            new = typo(name, rng)
        elif op == "case":
            new = change_case(name, rng)
        elif op == "accents_stripped":
            new = strip_accents(name)
        else:
            new = truncate(name)
        if new.strip() and new != name:
            name = new
            tags.append(op)
    return name, tags
