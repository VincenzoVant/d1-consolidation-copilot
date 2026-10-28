"""Rule-based company charts: one fictional company's own chart of accounts.

Recipe for a company with n accounts:
  1. pick round(0.8 * n) different standard groups;
  2. give each group a name in the company's language (or, for a "mixed"
     company, a per-group choice of nl/fr/en, sometimes a bilingual "nl / fr"
     label) and run it through the noise rules;
  3. fill the remaining rows with sub-accounts of those groups
     ("Lev. - Brussel"), which reuse their group's noisy name;
  4. number everything with the company's code scheme and sort by code.

Every row keeps the group it came from as target_code: the answer key.
"""

from random import Random
from typing import Literal

from d1.datagen.noise import apply_noise
from d1.datagen.schema import CompanyAccount, Language, Split, StandardAccount

# Suffixes a bookkeeper adds to split one account into several.
SUB_QUALIFIERS: dict[str, list[str]] = {
    "nl": ["Brussel", "Gent", "Antwerpen", "oud", "project A", "BE", "intern"],
    "fr": ["Bruxelles", "Liège", "Namur", "ancien", "projet A", "BE", "interne"],
    "en": ["UK", "EU", "legacy", "project A", "other", "BE", "internal"],
}

BASE_SHARE = 0.8  # share of rows that are a group's main account; the rest are sub-accounts
BILINGUAL_P = 0.3  # in a mixed company, chance a group gets an "nl / fr" label

CodeScheme = Literal["pcmn", "legacy"]


def _group_name(account: StandardAccount, language: Language, rng: Random) -> tuple[str, str, list[str]]:
    """Pick the clean name for one group: (name, language used for noise, tags)."""
    if language != "mixed":
        return account.name(language), language, []
    # Groups whose Dutch and French names are the same word ("Goodwill") never get a
    # bilingual label. The draw still happens, so the random sequence does not depend on it.
    if rng.random() < BILINGUAL_P and account.name_nl != account.name_fr:
        # The noise dictionary is applied in Dutch, the first half of the label.
        return f"{account.name_nl} / {account.name_fr}", "nl", ["bilingual"]
    lang = rng.choice(["nl", "fr", "en"])
    return account.name(lang), lang, []


def _sub_qualifiers(lang: str, count: int, rng: Random) -> list[str]:
    """count different qualifiers; numbered once the list runs out."""
    pool = SUB_QUALIFIERS[lang]
    picked = rng.sample(pool, min(count, len(pool)))
    return picked + [f"{i}" for i in range(2, 2 + count - len(picked))]


def legacy_codes(targets: list[str], rng: Random) -> list[str]:
    """Sequential 4-digit codes, skipping any that would start with the row's target code."""
    code = rng.randint(100, 700) * 10
    codes = []
    for target in targets:
        while str(code).startswith(target):
            code += 10
        codes.append(str(code))
        code += 10
    return codes


def generate_rule_company(
    company_id: str,
    split: Split,
    industry: str,
    language: Language,
    chart: list[StandardAccount],
    n_accounts: int,
    rng: Random,
    code_scheme: CodeScheme | None = None,
) -> list[CompanyAccount]:
    """Generate one company's chart of n_accounts rows from the standard chart.

    code_scheme is drawn at random unless given (tests pass it to force one).
    industry is recorded on every row; it only shapes the LLM-written charts.
    """
    if n_accounts < 1:
        raise ValueError("n_accounts must be at least 1")
    n_base = round(BASE_SHARE * n_accounts)
    if n_base > len(chart):
        raise ValueError(
            f"{n_accounts} accounts need {n_base} different groups, but the chart has only {len(chart)} groups"
        )
    scheme: CodeScheme = code_scheme or rng.choice(["pcmn", "legacy"])

    groups = sorted(rng.sample(chart, n_base), key=lambda a: a.code)

    # One noisy name per group; sub-accounts reuse it, as in a real chart.
    base: dict[str, tuple[str, str, list[str]]] = {}
    for account in groups:
        clean, noise_lang, tags = _group_name(account, language, rng)
        noisy, noise_tags = apply_noise(clean, noise_lang, rng)
        base[account.code] = (noisy, noise_lang, tags + noise_tags)

    sub_parents = [rng.choice(groups).code for _ in range(n_accounts - n_base)]
    subs: dict[str, list[str]] = {}
    for code in sorted(set(sub_parents)):
        subs[code] = _sub_qualifiers(base[code][1], sub_parents.count(code), rng)

    # (target, name, tags) in chart order: each group's main account, then its sub-accounts.
    rows: list[tuple[str, str, list[str]]] = []
    for account in groups:
        name, _, tags = base[account.code]
        rows.append((account.code, name, tags))
        for qualifier in subs.get(account.code, []):
            rows.append((account.code, f"{name} - {qualifier}", tags + ["sub_account"]))

    if scheme == "pcmn":
        codes, seen = [], {}
        for target, _, _ in rows:
            seen[target] = seen.get(target, -1) + 1
            codes.append(f"{target}{seen[target]:03d}")
        extra_tags: list[str] = []
    else:
        codes = legacy_codes([target for target, _, _ in rows], rng)
        extra_tags = ["legacy_code"]

    return [
        CompanyAccount(
            account_id=f"{company_id}-{i:03d}",
            company_id=company_id,
            industry=industry,
            split=split,
            source="rules",
            language=language,
            code_scheme=scheme,
            code=code,
            name=name,
            noise=tags + extra_tags,
            target_code=target,
        )
        for i, (code, (target, name, tags)) in enumerate(zip(codes, rows))
    ]
