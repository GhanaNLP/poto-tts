"""Text -> text with `[[phonemes]]` injected, for both training and inference.

This is the module that has to be identical on both sides of the project. The
training CSV is built by running transcripts through `inject`, and at synthesis
time the same function prepares the text handed to sherpa-onnx. If they ever
diverge, the model is asked at inference for phone sequences it never saw.

Per word: in the Ghana lexicon -> its IPA becomes `[[mnemonics]]`; not in the
lexicon -> left as plain text for espeak to read as English. Numbers, dates and
abbreviations fall in the second group, which is the point of leaving espeak in
the loop rather than shipping a lexicon-only frontend.

Stress is supplied explicitly, because espeak will not supply it. An injected
word carries no lexical stress information, so espeak treats it as already
specified and marks nothing -- 'Kwame Nkrumah met Nana Yaa Asantewaa' comes back
with a single stress mark on the last word and a monotone before it. Where an
English pronunciation is available and agrees with the lexicon about the word's
shape, its stress is borrowed; otherwise the penultimate syllable is marked,
which is the Akan, Ewe, Ga and Dagbani pattern.
"""

from __future__ import annotations

import functools
import re
from typing import Dict, List, Optional, Sequence

from .mnemonics import (CONSONANTS, DIPHTHONGS, LONG_VOWELS, VOWELS,
                        MnemonicError, injection)

__all__ = ["GhanaInjector", "TOKEN", "nuclei", "stress_index"]

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


def stress_index(phones: Sequence[str], english_ipa: Optional[str] = None) -> Optional[int]:
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
            if len(english) == len(positions) and _same_shape(phones, english_ipa):
                ordinal = sum(1 for i in english if i < mark)
                if 0 <= ordinal < len(positions):
                    return positions[ordinal]

    return positions[-2]


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


class GhanaInjector:
    """Prepares text for espeak: Ghana lexicon words as phonemes, the rest as text.

    Args:
        lexicon: Extra word -> IPA overrides merged over the packaged Ghana
            lexicon, written the way `ghana-english-g2p` accepts it.
        english_stress: Borrow English stress where espeak agrees about the word.
            False marks the penultimate syllable of every injected word, which is
            more uniformly Ghanaian and less faithful to English word rhythm.
        verify: Read every injection back through the espeak binary and fall back
            to plain text if it does not match. Catches the one dangerous failure
            mode -- an invalid mnemonic makes espeak silently discard the rest of
            the injection -- at the cost of needing espeak-ng on PATH and being
            far slower. Worth it when building a training set, wasteful at
            synthesis time, so it defaults to off.
    """

    def __init__(
        self,
        lexicon: Optional[Dict[str, str]] = None,
        english_stress: bool = True,
        verify: bool = False,
    ):
        _require_lexicon()
        from ghana_english_g2p import GhanaEnglishG2P

        self._g2p = GhanaEnglishG2P(use_espeak=False, lexicon=lexicon)
        self.english_stress = english_stress
        self.verify = verify
        self._espeak = None
        self.skipped: List[str] = []      # words that failed verification

    def _english_ipa(self, word: str) -> Optional[str]:
        """espeak's reading of the spelling, for stress only. None if unavailable."""
        if not self.english_stress:
            return None
        if self._espeak is None:
            try:
                import espeak_english

                self._espeak = espeak_english
            except ImportError:  # pragma: no cover - optional at inference
                self._espeak = False
        if not self._espeak:
            return None
        ipa = self._espeak.phonemes(word)
        return ipa if " " not in ipa.strip() else None

    @functools.lru_cache(maxsize=200_000)
    def _word(self, word: str) -> str:
        phones = self._g2p.word(word)
        if not phones:
            return word                     # espeak reads it as English
        at = (None if word.lower() in FUNCTION_WORDS
              else stress_index(phones, self._english_ipa(word)))
        try:
            mnemonics = injection(phones, stress_at=at)
        except MnemonicError:
            return word
        if self.verify:
            from .mnemonics import verify as _verify

            if not _verify(phones, mnemonics):
                self.skipped.append(word)
                return word
        return f"[[{mnemonics}]]"

    def __call__(self, text: str) -> str:
        return self.inject(text)

    def inject(self, text: str) -> str:
        """Text with lexicon words replaced by inline phonemes."""
        out: List[str] = []
        for token in text.replace("-", " ").replace("/", " ").split(" "):
            if not token:
                continue
            out.append(self._token(token))
        return re.sub(r" +", " ", " ".join(out)).strip()

    def _token(self, token: str) -> str:
        """One whitespace-delimited token, whose punctuation must be kept outside.

        Punctuation cannot go inside `[[ ]]` -- espeak reads it as a phoneme and
        discards the rest -- so a trailing comma or full stop is split off and
        re-attached after the injection, where it still does its phrasing job.
        """
        head = "".join(TOKEN.findall(token)[:1])
        if not head or not head[:1].isalnum():
            return "".join(c for c in token if c in KEEP_PUNCT)
        rest = token[len(head):]
        tail = "".join(c for c in rest if c in KEEP_PUNCT)
        return self._word(head) + tail

    def coverage(self, text: str) -> float:
        """Fraction of words the Ghana lexicon covers, i.e. how much is injected."""
        words = [t for t in TOKEN.findall(text) if t[:1].isalpha()]
        return self._g2p.coverage(words)


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
