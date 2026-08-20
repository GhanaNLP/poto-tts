"""The IPA -> espeak mnemonic mapping, and what espeak really does with it.

The round-trip tests are the ones that matter. A mapping table can be internally
consistent and still wrong about espeak's behaviour, and espeak's failure mode is
silent: an invalid mnemonic makes it discard the rest of the string and return a
fragment, with no error. So the alphabet is checked symbol by symbol against the
real binary, and the words that motivated each mapping decision are checked too.
"""

from __future__ import annotations

import shutil

import pytest

from poto_tts.mnemonics import (CONSONANTS, DIPHTHONGS, LONG_VOWELS, VOWELS,
                                MnemonicError, injection, phonemise, verify)

espeak = shutil.which("espeak-ng")
needs_espeak = pytest.mark.skipif(espeak is None, reason="needs the espeak-ng binary")


# -- the mapping -----------------------------------------------------------


@pytest.mark.parametrize(
    "phones, expected",
    [
        (["k", "w", "a", "b", "ɪ", "n", "a"], "kwabIna"),        # Kwabena
        (["a", "tʃ", "i", "m", "o", "t", "a"], "atSimota"),      # Achimota
        (["o", "k", "w", "a", "p", "ɛ", "n", "h", "ɛ", "n", "ɛ"], "okwapEnhEnE"),
        (["ɲ", "a", "ŋ", "kp", "a", "n", "i"], "njaNkpani"),
        (["ɡb", "e", "d", "e"], "gbede"),
        (["t", "w", "u", "m", "a", "s", "i"], "twumasi"),         # Twumasi
        (["ʃ", "i", "t", "o"], "Sito"),
        (["ç", "i"], "Ci"),
        ([], ""),
    ],
)
def test_injection(phones, expected):
    assert injection(phones) == expected


def test_a_uses_the_context_stable_mnemonic():
    """/a/ is 'a', not '0'.

    '0' is read as ɔː before /r/, which both mispronounces the word and collides
    with genuine /ɔ/ -- a contrast Akan needs. 'a' is ɑː in every context.
    """
    assert VOWELS["a"] == "a"
    assert injection(["s", "a", "r", "a"]) == "sara"


def test_ash_merges_into_a():
    """Ghanaian English merges TRAP into [a], and the lexicon writes some
    Ghanaian words with æ ('Asantewaa' is stored 'æsæntɪwæ')."""
    assert injection(["b", "æ", "t"]) == "bat"


def test_face_and_goat_are_monophthongs():
    """Ghanaian English has [fes] and [got]; PRICE, MOUTH and CHOICE keep glides."""
    assert injection(["f", "e", "ɪ", "s"]) == "fes"
    assert injection(["ɡ", "o", "ʊ", "t"]) == "got"
    assert injection(["p", "r", "a", "ɪ", "s"]) == "praIs"
    assert injection(["m", "a", "ʊ", "θ"]) == "maUT"
    assert injection(["tʃ", "ɔ", "ɪ", "s"]) == "tSOIs"


def test_length_is_looked_up_not_appended():
    """'a:' would be read as two vowels rather than one long one."""
    assert injection(["m", "aː", "m"]) == "mam"
    assert injection(["b", "iː", "t"]) == "bi:t"
    assert injection(["b", "ɛː", "t"]) == "bE:t"


def test_stress_mark_placement():
    assert injection(["k", "w", "a", "b", "ɪ", "n", "a"], stress_at=4) == "kwab'Ina"
    assert injection(["n", "a", "n", "a"], stress_at=1, secondary_at=3) == "n'an,a"


def test_unmappable_phone_raises():
    """Raised, not dropped: a dropped phone here is wrong in every utterance
    containing the word, not once."""
    with pytest.raises(MnemonicError):
        injection(["k", "ǀ", "a"])


def test_no_spaces_in_output():
    """A space inside `[[ ]]` is a word boundary to espeak, which resets stress."""
    assert " " not in injection(["k", "w", "a", "m", "e"], stress_at=2)


# -- what espeak actually does --------------------------------------------


@needs_espeak
@pytest.mark.parametrize("mnemonic", sorted(
    set(VOWELS.values()) | set(LONG_VOWELS.values()) | set(DIPHTHONGS.values())))
def test_every_vowel_mnemonic_is_valid(mnemonic):
    """An invalid mnemonic truncates silently, so each one is checked alone."""
    assert len(phonemise(f"[[b{mnemonic}t]]").replace("‍", "")) >= 3


@needs_espeak
@pytest.mark.parametrize("mnemonic", sorted(m for m in set(CONSONANTS.values()) if m))
def test_every_consonant_mnemonic_is_valid(mnemonic):
    assert len(phonemise(f"[[a{mnemonic}a]]").replace("‍", "")) >= 3


@needs_espeak
@pytest.mark.parametrize(
    "phones, expected_ipa",
    [
        (["k", "w", "a", "m", "e"], "kwame"),                 # Kwame
        (["ɲ", "a", "m", "ɪ"], "njamɪ"),                      # Nyame, ny
        (["dʒ", "a", "s", "i"], "dʒasi"),                     # Gyasi, gy
        (["tʃ", "e", "i"], "tʃei"),                            # Kyei, ky
        (["t", "w", "u", "m", "a", "s", "i"], "twumasi"),     # Twumasi, tw
        (["d", "w", "a", "m", "ɪ", "n", "a"], "dwamɪna"),   # Dwamena, dw
        (["ɲ", "a", "ŋ", "kp", "a", "n", "i"], "njaŋkpani"),  # Nyankpani, kp
        (["ɡb", "e", "d", "e"], "ɡbede"),                      # gb
    ],
)
def test_espeak_reads_back_what_we_meant(phones, expected_ipa):
    """The Twi digraphs, which are the whole point of controlling the phones."""
    got = phonemise(f"[[{injection(phones)}]]")
    got = "".join(c for c in got.replace("‍", "") if c not in "ˈˌ")
    assert got == expected_ipa


@needs_espeak
def test_verify_accepts_a_good_injection():
    phones = ["k", "w", "a", "b", "ɪ", "n", "a"]
    assert verify(phones, injection(phones))


@needs_espeak
def test_verify_rejects_a_truncated_injection():
    """'ny' is not a valid en-us mnemonic: espeak returns 'n' and drops the rest.
    This is the failure `verify` exists to catch."""
    assert not verify(["ɲ", "a", "n", "i"], "nyani")
