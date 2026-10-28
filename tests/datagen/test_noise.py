from random import Random

import pytest

from d1.datagen.noise import (
    ABBREVIATIONS,
    NOISE_PROBS,
    abbreviate,
    apply_noise,
    change_case,
    strip_accents,
    truncate,
    typo,
)

# --- abbreviate -----------------------------------------------------------


def test_abbreviate_nl_keeps_first_letter_capital():
    assert abbreviate("Afschrijvingen op materiële vaste activa", "nl") == "Afschr. op mat. vaste activa"


def test_abbreviate_fr_phrase_and_word():
    assert abbreviate("Réductions de valeur sur créances", "fr") == "RDV sur créan."


def test_abbreviate_whole_words_only():
    # "tangible" must not be found inside "Intangible"
    assert abbreviate("Intangible fixed assets", "en") == "Intang. fixed assets"


def test_abbreviate_leaves_unknown_words_alone():
    assert abbreviate("Huur gebouwen", "nl") == "Huur gebouwen"


def test_abbreviate_unknown_language_raises():
    with pytest.raises(ValueError, match="de"):
        abbreviate("Abschreibungen", "de")


def test_dictionary_keys_are_lowercase():
    for words in ABBREVIATIONS.values():
        assert all(key == key.lower() for key in words)


# --- the other single operations ------------------------------------------


def test_strip_accents():
    assert strip_accents("Créances à un an, matériële") == "Creances a un an, materiele"


def test_truncate_cuts_to_max_len_without_trailing_space():
    out = truncate("Voorzieningen voor pensioenen en soortgelijke verplichtingen")
    assert len(out) <= 30 and out == out.rstrip()
    assert truncate("Huur") == "Huur"


def test_typo_changes_one_long_word_and_is_seeded():
    name = "Herwaarderingsmeerwaarden op vaste activa"
    a, b = typo(name, Random(4)), typo(name, Random(4))
    assert a == b and a != name
    changed = [(x, y) for x, y in zip(name.split(), a.split()) if x != y]
    assert len(changed) == 1 and len(changed[0][0]) >= 5


def test_typo_leaves_short_words_alone():
    assert typo("BTW op 21", Random(0)) == "BTW op 21"


def test_change_case():
    out = change_case("Leveranciers", Random(0))
    assert out in {"LEVERANCIERS", "leveranciers"}


# --- apply_noise ----------------------------------------------------------

# Has an abbreviable word, an accent outside any abbreviable word, and stays
# longer than 30 characters after abbreviation, so every op can change it.
NAME = "Afschrijvingen op terreinen en gebouwen - éénmalige herwaardering"


def test_apply_noise_is_seeded_and_non_empty():
    a = apply_noise(NAME, "nl", Random(7))
    b = apply_noise(NAME, "nl", Random(7))
    assert a == b
    assert a[0].strip()
    assert set(a[1]) <= set(NOISE_PROBS)


def test_tags_only_for_ops_that_changed_the_string():
    for seed in range(200):
        _, tags = apply_noise("ABC", "en", Random(seed))
        assert "abbreviation" not in tags and "accents_stripped" not in tags and "truncated" not in tags


def test_tag_rates_match_noise_probs():
    # Over many draws each op fires about as often as NOISE_PROBS says
    # (this name can be abbreviated and has accents, so both always change it).
    n = 4000
    counts = {op: 0 for op in NOISE_PROBS}
    for seed in range(n):
        for tag in apply_noise(NAME, "nl", Random(seed))[1]:
            counts[tag] += 1
    for op in ("abbreviation", "accents_stripped", "truncated"):
        assert abs(counts[op] / n - NOISE_PROBS[op]) < 0.03, op


def test_noise_never_returns_empty():
    for seed in range(500):
        assert apply_noise("Ab", "en", Random(seed))[0].strip()
