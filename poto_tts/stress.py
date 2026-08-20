"""Where the stress mark goes on a dictionary entry.

The lexicon records segments, not stress, so every entry needs one chosen. Ghanaian
words take the penultimate syllable, which is the Akan pattern. A word English also has
can borrow espeak's stress instead, since espeak reads those from its own dictionary
and gets them right -- that is what `known_english` selects.

`_same_shape` guards the borrowing: espeak reading a different number of syllables, or
a different consonant skeleton, means it built a different word and its stress is not
transferable. `Kwabena` comes back as /kwˈeɪbnə/, two nuclei against the lexicon's
three.
"""

from __future__ import annotations

import functools
import re
from typing import Dict, List, Optional, Sequence

from .mnemonics import (CONSONANTS, DIPHTHONGS, LONG_VOWELS, VOWELS,
                        MnemonicError, injection)

__all__ = ["FUNCTION_WORDS", "nuclei", "stress_index"]

# A number (keeping internal separators, so '3.30' is not read as two sentences),
# a word, whitespace, or a run of punctuation.
TOKEN = re.compile(
    r"\d+(?:[.,:/]\d+)*(?=\D|$)|[^\W_]+(?:['’][^\W_]+)*|\s+|[^\w\s]+",
    re.UNICODE,
)

# Punctuation espeak uses for phrasing. Anything else is dropped rather than
# passed through, since espeak reads an unknown symbol aloud as a word.
KEEP_PUNCT = set(";:,.!?—…\"()")

_STRESS_MARK = "ˈ"

# Function words are injected without a stress mark. Every injected word would
# otherwise take primary stress -- espeak adds none of its own to inline
# phonemes, so the mark we write is the only one there is -- and a sentence where
# 'at', 'the' and 'on' are all stressed has the rhythm of a word list. espeak's
# own English output leaves exactly this class unstressed.
FUNCTION_WORDS = frozenset("""
a an the this that these those
i me my we us our you your he him his she her it its they them their
am is are was were be been being do does did done have has had
and or but nor so yet if than as
at by for from in into of off on onto out over to up with within without
no not nor too very just also then there here
can could may might must shall should will would
""".split())


def _is_vowel(phone: str) -> bool:
    return phone.rstrip("ː") in VOWELS or phone in LONG_VOWELS


def nuclei(phones: Sequence[str]) -> List[int]:
    """Phone index of each syllable nucleus.

    A diphthong written as two phones counts once, so the count is comparable
    with a syllable count taken from espeak's IPA.
    """
    out: List[int] = []
    i = 0
    while i < len(phones):
        if i + 1 < len(phones) and (phones[i], phones[i + 1]) in DIPHTHONGS:
            out.append(i)
            i += 2
            continue
        if _is_vowel(phones[i]):
            out.append(i)
        i += 1
    return out


def stress_index(phones: Sequence[str], english_ipa: Optional[str] = None,
                 known_english: bool = False) -> Optional[int]:
    """Which phone to mark with primary stress, or None if there is no vowel.

    `english_ipa` is espeak's pronunciation of the word's spelling. It is used
    only when it agrees with the lexicon about the *shape* of the word -- same
    syllable count and same consonant skeleton -- which is a proxy for 'espeak
    recognised this word'. For a Ghanaian name espeak is guessing from spelling:
    'Kwabena' comes back /kwˈeɪbnə/, two nuclei against the lexicon's three, and
    'Okuapenhene' comes back /ˈoʊkjuːˌeɪpənhˌiːn/, which has the same five nuclei
    by coincidence but a /j/ where the lexicon has /w/. Syllable count alone
    would accept that one and stress OKuapenhene instead of okuapenhENE, so the
    skeleton is checked too. Declining leaves the penultimate rule in charge.
    """
    positions = nuclei(phones)
    if not positions:
        return None
    if len(positions) == 1:
        return positions[0]

    if english_ipa:
        mark = english_ipa.find(_STRESS_MARK)
        if mark >= 0:
            english = [i for i, ch in enumerate(english_ipa) if _english_nucleus(english_ipa, i)]
            # `known_english` means the word is in the English vocabulary, so
            # espeak is reading it from its own dictionary rather than guessing
            # from spelling, and its stress is the English stress we want to
            # keep. The skeleton test is a proxy for exactly that, and only a
            # proxy: it rejects 'yesterday' (espeak writes the diphthong with a
            # zero-width joiner, which no lexicon entry can match) and accepts
            # 'Kwabena' (whose consonants happen to line up while the vowels
            # are wrong). Asking the vocabulary directly gets both right.
            trusted = known_english or _same_shape(phones, english_ipa)
            if len(english) == len(positions) and trusted:
                ordinal = sum(1 for i in english if i < mark)
                if 0 <= ordinal < len(positions):
                    return positions[ordinal]

    return positions[-2]


_JOINERS = "‍‌͜͢͡"


def _consonants(phones: Sequence[str]) -> str:
    """The word's consonant skeleton, in the mnemonic alphabet."""
    out = []
    for phone in phones:
        base = phone.rstrip("ː")
        if _is_vowel(phone) or base in VOWELS:
            continue
        out.append(CONSONANTS.get(base, base))
    return "".join(out)


def _same_shape(phones: Sequence[str], english_ipa: str) -> bool:
    """Do the lexicon and espeak agree on the consonants of this word?

    Compared in one alphabet by segmenting espeak's IPA the same way the lexicon
    is written, so /ʃ/ from either side is 'S' and /ɹ/ and /ɾ/ are both 'r'.
    """
    from ghana_english_g2p import segment

    try:
        return _consonants(phones) == _consonants(segment(english_ipa))
    except Exception:      # a segmentation we cannot compare is not a match
        return False


_ENGLISH_VOWELS = frozenset("aɑɒæʌəɐɜɚeɛiɪɨoɔuʊyᵻ")


def _english_nucleus(ipa: str, i: int) -> bool:
    """Is position *i* the start of a vowel run in an espeak IPA string?"""
    if ipa[i] not in _ENGLISH_VOWELS:
        return False
    return i == 0 or ipa[i - 1] not in _ENGLISH_VOWELS


def _require_lexicon():
    """Import the lexicon package, or explain how to get it.

    `ghana-english-g2p` is an optional dependency because speaking does not need it:
    the dictionary is compiled into each voice, and the lexicon is what compiles it.
    Kept as a dependency rather than vendored so that a rebuild always uses the
    current lexicon rather than a copy that quietly falls behind.
    """
    try:
        import ghana_english_g2p  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "this needs the Ghanaian lexicon, an optional dependency used to build "
            "or inspect a dictionary:\n  pip install 'poto-tts[lexicon]'"
        ) from exc
    return ghana_english_g2p
