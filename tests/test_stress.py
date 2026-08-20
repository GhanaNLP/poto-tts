"""Where the stress mark goes on a dictionary entry.

The lexicon records segments and not stress, so every entry needs one chosen: the
Ghanaian penultimate rule, or espeak's own stress for a word English also has.
"""

from __future__ import annotations

import pytest

from poto_tts.stress import FUNCTION_WORDS, nuclei, stress_index


def test_penultimate_is_the_default():
    phones = ["o", "k", "w", "a", "p", "ɛ", "n", "h", "ɛ", "n", "ɛ"]
    assert stress_index(phones) == nuclei(phones)[-2]
def test_single_syllable_words_take_the_only_vowel():
    assert stress_index(["m", "ɛ", "t"]) == 1
def test_no_vowel_means_no_stress():
    assert stress_index(["s", "t"]) is None
def test_diphthong_counts_as_one_syllable():
    assert nuclei(["p", "r", "a", "ɪ", "s"]) == [2]
def test_english_word_takes_espeak_stress_despite_the_joiner():
    """'yesterday' is initial-stressed in English; espeak writes /deɪ/ with a
    zero-width joiner, which used to defeat the skeleton comparison."""
    phones = ["j", "ɛ", "s", "t", "a", "d", "e", "i"]
    espeak = "jˈɛstɚdˌe‍ɪ"
    at = stress_index(phones, espeak, known_english=True)
    assert at == nuclei(phones)[0], "English stress should land on the first nucleus"
def test_ghanaian_name_keeps_penultimate_even_when_skeletons_agree():
    """espeak's guess at 'Kwabena' has the same consonants as the lexicon, so a
    skeleton test accepts it. It must still not donate its stress."""
    phones = ["k", "w", "a", "b", "ɪ", "n", "a"]
    espeak = "kwˈe‍ɪbnə"
    at = stress_index(phones, espeak, known_english=False)
    assert at == nuclei(phones)[-2], "a name espeak guessed at keeps penultimate stress"
def test_known_english_does_not_override_a_syllable_count_mismatch():
    """The count check still guards: espeak reading a different number of
    syllables means the ordinal mapping is meaningless, whatever the word is."""
    phones = ["k", "w", "a", "b", "ɪ", "n", "a"]          # three nuclei
    assert stress_index(phones, "kwˈebn", known_english=True) == nuclei(phones)[-2]
