"""Ghanaian IPA -> Lingua Franca Nova spelling, the notation this library speaks in.

sherpa-onnx hands Kokoro *text*, runs espeak over it, and feeds the resulting phoneme
ids to the model. There is no way to pass phonemes directly, so espeak sits between
this library and the model and cannot be bypassed. Anything we want the model to say
has to be written as text that espeak turns into those phonemes.

Lingua Franca Nova is what makes that practical. Its orthography is strictly phonemic
-- one letter, one sound, no exceptions -- where English spelling is not (`through`,
`though`, `tough`). So a pronunciation can be *written* in lfn and read back
faithfully: `kwabina` comes back as /kwabˈina/. lfn is a codec, not a language
choice, and espeak's `lfn` voice is the decoder.

The letter values are lfn's own, established by asking espeak rather than reading a
spec: `xa` -> /ʃˈa/, `txa` -> /tʃˈa/, `gba` -> /ɡbˈa/, `nya` -> /njˈa/. What this
module owns is the mapping onto them, and every lossy decision in it:

  **Five vowels, not seven.** Akan has i e ɛ a ɔ o u; lfn has a e i o u. So ɛ folds
  into `e` and ɔ into `o`, and `Okuapɛnhɛnɛ` is spoken /okwapenhene/. This is the
  cost of the route and the reason the compiled-dictionary alternative exists.

  **No schwa.** ə becomes `a`, which is what Ghanaian English does with it anyway --
  full vowels where British and American English reduce.

  **No length.** Doubling a vowel in lfn reads as two syllables, which is worse than
  a short one, so length marks are dropped.

  **th-stopping.** θ and ð become `t` and `d`, the local realisation rather than a
  compromise.

  **ŋ needs care, and less of it than it first appears.** `ng` is read /ŋɡ/, so it
  inserts a /ɡ/ before a following stop -- measured: `bangka` -> /baŋɡka/, `bangpa`
  -> /baŋɡpa/, `bangta` -> /baŋɡta/. Plain `n` avoids that but stays alveolar
  (`banka` -> /banka/), so neither spelling gives a clean /ŋk/.
  
  `ng` word-finally, `n` everywhere else. Measured: `ng` is read /ŋɡ/ whenever
  anything follows it -- `bangka` /baŋɡka/, `bangpa` /baŋɡpa/, `bangta` /baŋɡta/ --
  and word-initially espeak prepends a vowel, giving `ngkruma` as /ˈenɡkɾuma/. Only
  at the end of a word does it come back clean: `sing` /sˈiŋ/.
  
  So `Nkrumah` is `nkruma` /nkɾˈuma/ and `Nyankpani` is `nyankpani` /njankpˈani/.
  The cost is that a medial ŋ becomes alveolar: `bank` is /bˈank/ rather than
  /bˈaŋk/. That is a place-of-articulation difference before a velar, which listeners
  barely register, where the alternative inserts a whole consonant nobody said.
  
  It took three attempts to land here, and the reason is worth recording: the lexicon
  is inconsistent about labial-velars, writing `Nyankpani` as ŋ + k + p but other
  words with a single `kp` phone. A rule keyed on "is the next phone kp" therefore
  fired for some words and not others -- which is exactly the kind of bug a table
  looks innocent of.
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence

__all__ = ["CONSONANTS", "VOWELS", "respell", "respell_ipa", "read_back"]

# -- vowels ----------------------------------------------------------------
VOWELS: Dict[str, str] = {
    "a": "a", "ɑ": "a", "ɒ": "a", "æ": "a", "ʌ": "a", "ə": "a", "ɐ": "a",
    "e": "e", "ɛ": "e", "ɜ": "e", "œ": "e",
    "i": "i", "ɪ": "i", "ɨ": "i", "ᵻ": "i", "y": "i",
    "o": "o", "ɔ": "o", "ø": "o",
    "u": "u", "ʊ": "u", "ɯ": "u",
    # Rhotic vowels keep their /r/: one phone in espeak's English, two letters here.
    "ɚ": "er", "ɝ": "er",
}

# -- consonants ------------------------------------------------------------
#
# The swaps worth noting: lfn's `j` is /ʒ/, so IPA /j/ has to be `i` before a vowel;
# `tx` is /tʃ/ because lfn reads `ch` as /kh/, which is why 'Achimota' unrespelled
# comes back /akhimota/; `x` is /ʃ/, which looks alarming and is correct.
CONSONANTS: Dict[str, str] = {
    "b": "b", "d": "d", "f": "f", "ɡ": "g", "g": "g", "h": "h", "j": "i",
    "k": "k", "l": "l", "m": "m", "n": "n", "p": "p", "r": "r", "ɹ": "r",
    "ɾ": "r", "s": "s", "t": "t", "v": "v", "w": "w", "z": "z",
    "ʃ": "x", "ʒ": "j", "tʃ": "tx", "ʧ": "tx", "dʒ": "dj", "ʤ": "dj",
    "ts": "tz", "dz": "dz", "tɕ": "tx", "dʑ": "dj", "ʨ": "tx", "ʥ": "dj",
    "θ": "t", "ð": "d", "ɲ": "ny", "ɟ": "i", "c": "k", "ɬ": "l",
    "kp": "kp", "ɡb": "gb", "gb": "gb", "ŋm": "nm",
    "x": "h", "ɣ": "g", "ç": "h", "ɕ": "x", "ʔ": "",
}

_DIACRITICS = "ːˑ̩̥̯̃ʰʷʲ͡"



def _strip(phone: str) -> str:
    for mark in _DIACRITICS:
        phone = phone.replace(mark, "")
    return phone


def _velar_nasal(phones: Sequence[str], i: int) -> str:
    """`ng` at the end of a word, `n` anywhere else. See the module docstring."""
    return "ng" if i == len(phones) - 1 else "n"


def respell(phones: Iterable[str]) -> str:
    """One word's IPA phones -> lfn spelling.

    Args:
        phones: Phones as `ghana-english-g2p` produces them, e.g.
            `['k', 'w', 'a', 'b', 'ɪ', 'n', 'a']`. Multi-character phones ('tʃ',
            'kp') and trailing diacritics are both understood.

    Returns:
        lfn spelling: `kwabina`. Empty for empty input.

    An unmappable phone is dropped rather than raising: this runs on every word of
    every utterance, where one missing phone is a small mispronunciation and an
    exception is a failed request.
    """
    phones = [p for p in phones if p]
    out: List[str] = []
    for i, phone in enumerate(phones):
        base = _strip(phone)
        if not base:
            continue
        if base == "ŋ":
            out.append(_velar_nasal(phones, i))
        elif base in VOWELS:
            out.append(VOWELS[base])
        elif base in CONSONANTS:
            out.append(CONSONANTS[base])
    return "".join(out)


def respell_ipa(ipa: str) -> str:
    """Respell an IPA string, run-together or space-separated.

    Segmentation is `ghana-english-g2p`'s, so 'tʃ' stays one phone and stress marks
    are dropped -- lfn assigns its own stress, and its rule is the penultimate one
    Ghanaian English already uses.
    """
    from ghana_english_g2p import segment

    return respell(segment(ipa))


@lru_cache(maxsize=4096)
def read_back(spelling: str, voice: str = "lfn") -> str:
    """What espeak's lfn voice makes of a spelling. For checking, not for synthesis.

    Uses the espeak-ng *binary*, because espeak initialises once per process inside
    sherpa-onnx: a check that shares a process with the synthesiser would be
    answering about whichever data directory was loaded first.
    """
    espeak = shutil.which("espeak-ng")
    if espeak is None:
        raise RuntimeError("checking a respelling needs the espeak-ng binary")
    out = subprocess.run([espeak, "-q", "--ipa=3", "-v", voice, spelling],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()
