"""The injector: which words get phonemes, and where the stress goes.

`GhanaInjector` is the QA and training-inspection path -- deployments use the
compiled espeak dictionary instead -- but both are generated from the same two
decisions tested here: what the lexicon says, and which syllable to stress.
"""

from __future__ import annotations

import pytest

from poto_tts.inject import FUNCTION_WORDS, GhanaInjector, nuclei, stress_index


@pytest.fixture(scope="module")
def inj():
    return GhanaInjector()


# -- which words are injected ---------------------------------------------


def test_lexicon_words_become_phonemes(inj):
    assert inj("Kwabena") == "[[kwA:b'InA:]]"


def test_unknown_words_are_left_as_text(inj):
    """espeak reads them as English. This is what keeps OOV words from vanishing,
    the failure mode a lexicon-only frontend has."""
    assert inj("zzzqx") == "zzzqx"


def test_numbers_are_left_to_espeak(inj):
    """espeak expands them ('250' -> two hundred fifty), which is exactly why the
    frontend does not try to."""
    assert inj("2026") == "2026"
    assert inj("It cost 250.") == "[[It]] [[k'Ost]] 250."


def test_punctuation_stays_outside_the_brackets(inj):
    """Punctuation inside `[[ ]]` is read as a phoneme and truncates the entry,
    but outside it still does its phrasing job."""
    out = inj("Kwabena, Achimota.")
    assert out.endswith(".")
    assert "," in out
    assert ",]]" not in out and ".]]" not in out


def test_hyphen_splits_words(inj):
    assert inj("Akufo-Addo").count("[[") == 2


def test_case_is_irrelevant(inj):
    assert inj("KWABENA") == inj("kwabena") == inj("Kwabena")


def test_lexicon_override():
    custom = GhanaInjector(lexicon={"zzzqx": "k w a"})
    assert custom("zzzqx") == "[[kw'A:]]"


# -- stress ----------------------------------------------------------------


def test_every_injected_content_word_is_stressed(inj):
    """espeak adds no stress of its own to inline phonemes, so an unmarked word is
    unstressed. Without this the whole utterance is monotone."""
    out = inj("Kwame Nkrumah met Nana Asantewaa")
    assert all("'" in chunk for chunk in out.split() if chunk.startswith("[["))


def test_function_words_are_not_stressed(inj):
    """'at', 'the', 'on' all take primary stress otherwise, and the sentence gets
    the rhythm of a word list."""
    assert "'" not in inj("at")
    assert "'" not in inj("the")
    assert "the" in FUNCTION_WORDS


def test_english_words_keep_english_stress(inj):
    """'convention' is in the Ghana lexicon too, and still belongs on 'ven'."""
    assert inj("convention") == "[[kOnv'EnS@n]]"


def test_ghanaian_names_fall_back_to_penultimate(inj):
    """espeak reads 'Okuapenhene' as /ˈoʊkjuːˌeɪpənhˌiːn/ -- five nuclei like the
    lexicon, so a syllable-count check alone would borrow its first-syllable
    stress. The consonant skeleton disagrees (/j/ against /w/), so it declines."""
    assert inj("Okuapenhene") == "[[okwA:pEnh'EnE]]"


def test_penultimate_is_the_default():
    phones = ["o", "k", "w", "a", "p", "ɛ", "n", "h", "ɛ", "n", "ɛ"]
    assert stress_index(phones) == nuclei(phones)[-2]


def test_single_syllable_words_take_the_only_vowel():
    assert stress_index(["m", "ɛ", "t"]) == 1


def test_no_vowel_means_no_stress():
    assert stress_index(["s", "t"]) is None


def test_diphthong_counts_as_one_syllable():
    assert nuclei(["p", "r", "a", "ɪ", "s"]) == [2]


def test_english_stress_can_be_disabled():
    """'policy' is antepenultimate in English (PO-li-cy) and penultimate under the
    Ghanaian default, so it is a word where the flag actually shows."""
    assert GhanaInjector(english_stress=True)("policy") == "[[p'OlIsi]]"
    assert GhanaInjector(english_stress=False)("policy") == "[[pOl'Isi]]"


# -- introspection ---------------------------------------------------------


def test_coverage(inj):
    assert inj.coverage("Kwabena Achimota Nkrumah") == 1.0
    assert inj.coverage("zzzqx flurbleglonk") == 0.0
