"""The IPA -> lfn mapping, and what espeak actually does with it.

The round-trip tests are the ones that matter: a table can be internally consistent
and still wrong about espeak's behaviour. Both bugs this file guards against were
found that way rather than by reading the table.
"""

from __future__ import annotations

import collections
import shutil

import pytest

from poto_tts.respell import CONSONANTS, VOWELS, read_back, respell

needs_espeak = pytest.mark.skipif(shutil.which("espeak-ng") is None,
                                  reason="needs the espeak-ng binary")


@pytest.mark.parametrize("phones, expected", [
    (["k", "w", "a", "b", "ɪ", "n", "a"], "kwabina"),        # Kwabena
    (["a", "tʃ", "i", "m", "o", "t", "a"], "atximota"),      # ch is /kh/ in lfn
    (["o", "k", "w", "a", "p", "ɛ", "n", "h", "ɛ", "n", "ɛ"], "okwapenhene"),
    (["ʃ", "i", "t", "o"], "xito"),                          # ʃ is x
    (["ð", "ə"], "da"),                                      # th-stopping, no schwa
    (["dʒ", "a", "s", "i"], "djasi"),                        # Gyasi
    ([], ""),
])
def test_respell(phones, expected):
    assert respell(phones) == expected


def test_length_marks_are_dropped():
    """Doubling a vowel in lfn reads as two syllables, worse than a short one."""
    assert respell(["m", "aː", "m", "eː"]) == "mame"


def test_every_lexicon_phone_can_be_mapped():
    """The guard against the bug that dropped ɒ silently: `on` became `n`, read as
    /ˈen/, because the table had no entry and unknown phones are dropped."""
    pytest.importorskip("ghana_english_g2p")
    from ghana_english_g2p.core import _load_lexicon

    inventory = collections.Counter()
    for prons in _load_lexicon().values():
        inventory.update(prons[0])
    unmapped = [p for p in inventory
                if p.rstrip("ːˑ̩̥̯̃ʰʷʲ͡") not in VOWELS
                and p.rstrip("ːˑ̩̥̯̃ʰʷʲ͡") not in CONSONANTS
                and p != "ŋ"]
    assert not unmapped, f"phones with no lfn mapping: {unmapped}"


def test_turned_alpha_is_mapped():
    """ɒ specifically: 548 lexicon entries use it, `on` among them."""
    assert respell(["ɒ", "n"]) == "an"


# -- the velar nasal, which needs three different answers -----------------


@needs_espeak
@pytest.mark.parametrize("phones, expected_ipa, why", [
    (["s", "ɪ", "ŋ"], "siŋ", "word-final ng reads clean"),
    (["ŋ", "k", "r", "u", "m", "a"], "nkɾuma", "word-initial ng gets a vowel prepended"),
    (["ɲ", "a", "ŋ", "k", "p", "a", "n", "i"], "njankpani",
     "medial ng would insert a ɡ; note the lexicon writes this k + p, not kp"),
    (["b", "a", "ŋ", "k"], "bank",
     "the cost: a medial ŋ goes alveolar rather than gaining a spurious ɡ"),
])
def test_velar_nasal_positions(phones, expected_ipa, why):
    got = "".join(c for c in read_back(respell(phones)).replace("‍", "")
                  if c not in "ˈˌ")
    assert got == expected_ipa, why


@needs_espeak
@pytest.mark.parametrize("phones, expected_ipa", [
    (["k", "w", "a", "b", "ɪ", "n", "a"], "kwabina"),
    (["ɲ", "a", "m", "ɪ"], "njami"),                          # Nyame
    (["kp", "a", "kp", "o"], "kpakpo"),
    (["ɡb", "e", "d", "e"], "ɡbede"),
    (["dʒ", "o", "n"], "dʒon"),
])
def test_espeak_reads_back_what_we_meant(phones, expected_ipa):
    got = "".join(c for c in read_back(respell(phones)).replace("‍", "")
                  if c not in "ˈˌ")
    assert got == expected_ipa
