"""Ghanaian IPA -> espeak-ng mnemonics, for building the espeak dictionary.

Not the path a Python caller takes -- that is respell.py, which writes lfn and
reaches every word. This table exists for `poto-tts dict`, which compiles Ghanaian
pronunciations into espeak's own English dictionary so that **deployments with no
Python** still get the names right: an Android or iOS app, a WASM build, a C++
service. Those runtimes load `espeak-ng-data` and send plain text, so the only place
a correction can live is inside that data.

The two routes differ in reach and fidelity. The dictionary corrects words espeak
mis-parses, keeps all seven Akan vowels, and leaves ordinary English alone. The lfn
respelling reaches every word, at the cost of five vowels. Python callers get the
second; everyone else gets the first.


espeak-ng reads `[[...]]` in its input as phonemes rather than letters, in the
mnemonics of whichever voice is active. That is the hook this module uses: a word
the Ghana lexicon knows becomes `[[kwab'ina]]`, everything else stays plain text
for espeak to phonemise as English, and sherpa-onnx's Piper frontend needs no
lexicon file and never drops a word it has not seen.

Why this beats the two alternatives:

  Against MeloTTS's lexicon (`MeloTtsLexicon`), which really does take a
  word-to-phones table: it has no espeak fallback, so a word missing from the
  table is dropped silently. Here the lexicon covers Ghanaian words and espeak
  covers the rest of the language.

  Against respelling in Lingua Franca Nova (see respell.py, the Kokoro route):
  lfn has five vowels, and Akan has seven. espeak-en's *rules* never produce
  [e] or [o] either -- but its phoneme table defines them, and an inline phoneme
  bypasses the rules. `[[b'et]]` is /bet/ and `[[b'ot]]` is /bot/. So ɛ/e and
  ɔ/o stay distinct, and `kp`, `gb` and `nj` are all available as well.

Two things to know before trusting an injection:

  **An invalid mnemonic truncates the rest of the string, silently.** `[[ny'ani]]`
  is not an error -- espeak returns "n" and discards the rest. Nothing warns you.
  So `injection` is always checked by `verify`, which phonemises the result and
  compares it against the target, and the caller falls back to plain text when it
  does not match.

  **espeak still applies its allophonic rules.** An intervocalic /t/ flaps:
  `[[atSim'ota]]` comes back /ætʃimˈoɾæ/. This is not corrected, because training
  and inference share this module: the model is trained on whatever espeak really
  produces, so a consistent flap is a label the model learns, not an error it
  inherits.

The vowel choices are the substance of the file; each mapping is a judgement
about which English vowel a Ghanaian one should share a token with, and the
comments give the reasoning.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from typing import Dict, Iterable, List, Optional, Sequence

__all__ = [
    "CONSONANTS",
    "LONG_VOWELS",
    "VOWELS",
    "MnemonicError",
    "injection",
    "phonemise",
    "verify",
]


class MnemonicError(ValueError):
    """A phone has no espeak-en mnemonic."""


# -- vowels ----------------------------------------------------------------
#
# The interesting decision is /a/, the lexicon's commonest phone by far (67,795
# occurrences). espeak-en offers three candidates and none of them is [a]:
#
#   'a'   -> æ. Front but too raised, and it collides with English /æ/ in 'bat',
#          forcing the model to average Ghanaian [a] against it.
#   '0'   -> ɑː, but *not stably*: before /r/ espeak realises it as ɔː, so
#          'sarakofe' comes back /sɔːɹɑːkofe/. That both mispronounces the word
#          and collides with genuine /ɔ/, which is a contrast Akan needs.
#   'A:'  -> ɑː in every context tested -- before /r/, between vowels, word-final.
#
# So 'A:' it is. The merger it implies is the one Ghanaian English already makes:
# 'bart' and 'Kwabena' both have [a] locally, so sharing a token with ɑː is free.
# Context-stability is the deciding property: a phone whose realisation shifts
# with its neighbours splits one lexicon phone across two model tokens.
#
# ɛ and ɔ map to 'E' and 'O', e and o to 'e' and 'o'. English rules never emit
# the latter two, so those tokens end up meaning 'the Ghanaian close-mid vowel'
# almost exclusively -- which is what we want them to mean.
VOWELS: Dict[str, str] = {
    # æ joins /a/ rather than keeping espeak's 'a'. Two reasons, and they agree:
    # Ghanaian English merges TRAP into [a] ('bat' is [bat]), and the lexicon
    # itself transcribes Ghanaian words with æ where [a] is meant -- 'Asantewaa'
    # is stored as 'æ s æ n t ɪ w æ', which read literally gives /æsæntɪwæ/.
    # Merging makes both come out right and costs an inventory slot nothing needs.
    "a": "A:", "ɑ": "A:", "ɒ": "A:", "æ": "A:", "ʌ": "V",
    "ə": "@", "ɐ": "@", "ɜ": "3:", "ɚ": "3",
    "e": "e", "ɛ": "E",
    "i": "i", "ɪ": "I", "ɨ": "I", "ᵻ": "I", "y": "i",
    "o": "o", "ɔ": "O",
    "u": "u", "ʊ": "U", "ɯ": "u",
    "ø": "o", "œ": "E",
}

# Long vowels are listed rather than derived by appending ':' to the short
# mnemonic, because appending is wrong twice over: 'A:' is already long, so
# 'A::' is meaningless, and 'a:' is read as two vowels (ææ) rather than one long
# one. Every entry below was checked against espeak individually.
LONG_VOWELS: Dict[str, str] = {
    "aː": "A:", "ɑː": "A:", "ɒː": "A:",
    "iː": "i:", "uː": "u:", "oː": "o:", "eː": "e:",
    "ɛː": "E:", "ɔː": "O:", "ɜː": "3:",
    "æː": "A:",
    "əː": "@",
}

# Diphthongs. The lexicon writes them as two phones; espeak has a single
# mnemonic for each ('eI', 'aI', 'oU', 'aU', 'OI'), and all five work inline.
#
# Only three are used, because Ghanaian English monophthongises FACE and GOAT:
# 'face' is [fes] and 'goat' is [got], while PRICE, MOUTH and CHOICE keep their
# glides. Mapping /eɪ/ to 'e' and /oʊ/ to 'o' therefore matches the training
# audio rather than fighting it -- and it is only possible because espeak's table
# has the monophthongs its English rules never emit. Anything else would ask the
# model to learn a glide the speakers do not produce, spending an inventory slot
# to make the vowel worse.
DIPHTHONGS = {
    ("e", "ɪ"): "e", ("eː", "ɪ"): "e",          # FACE, monophthongised
    ("o", "ʊ"): "o", ("oː", "ʊ"): "o",          # GOAT, monophthongised
    ("a", "ɪ"): "aI", ("aː", "ɪ"): "aI",        # PRICE
    ("a", "ʊ"): "aU", ("aː", "ʊ"): "aU",        # MOUTH
    ("ɔ", "ɪ"): "OI", ("ɔː", "ɪ"): "OI",        # CHOICE
}

# -- consonants ------------------------------------------------------------
#
# 'kp', 'gb' and 'nj' are in espeak's base tables even though English never uses
# them, so Akan and Ewe labial-velars survive as single phones rather than being
# split into their components.
CONSONANTS: Dict[str, str] = {
    "b": "b", "d": "d", "f": "f", "ɡ": "g", "g": "g", "h": "h", "j": "j",
    "k": "k", "l": "l", "m": "m", "n": "n", "ŋ": "N", "p": "p",
    "r": "r", "ɹ": "r", "ɾ": "r", "s": "s", "t": "t", "v": "v", "w": "w",
    "z": "z", "ʃ": "S", "ʒ": "Z", "tʃ": "tS", "ʧ": "tS", "dʒ": "dZ",
    "ʤ": "dZ", "θ": "T", "ð": "D",
    "ts": "ts", "dz": "dz", "tɕ": "tS", "dʑ": "dZ", "ʨ": "tS", "ʥ": "dZ",
    "ɲ": "nj", "ɟ": "j", "c": "k", "ɬ": "l",
    # The Twi digraphs mostly arrive already decomposed: the lexicon writes kw,
    # tw, dw and hw as two phones ('k w', 't w'), ny as ɲ, gy as dʒ and ky as tʃ
    # or tɕ. So the only ones needing an entry are the palatal fricatives, which
    # appear in hw- words: ç has its own espeak mnemonic, ɕ borrows ʃ's.
    "ç": "C", "ɕ": "S",
    "kp": "kp", "ɡb": "gb", "gb": "gb", "ŋm": "Nm",
    "x": "h", "ɣ": "g", "ʔ": "",
}

# Length is written ':' after the vowel; other diacritics have no mnemonic.
LENGTH = ":"
_DROPPED = "ˑ̩̥̯̃ʰʷʲ͡"

_STRESS = "'"
_SECONDARY = ","

_ESPEAK = shutil.which("espeak-ng")


def _strip(phone: str) -> tuple[str, bool]:
    """Split a phone from its length mark, dropping diacritics espeak cannot use."""
    long_ = "ː" in phone
    for d in _DROPPED + "ː":
        phone = phone.replace(d, "")
    return phone, long_


def injection(
    phones: Sequence[str],
    stress_at: Optional[int] = None,
    secondary_at: Optional[int] = None,
) -> str:
    """Mnemonics for one word, without the `[[ ]]` wrapper.

    Args:
        phones: Ghana IPA phones, e.g. `['k', 'w', 'a', 'b', 'ɪ', 'n', 'a']`.
        stress_at: Index of the phone carrying primary stress; the mark goes
            before the onset of that syllable. None leaves the word unmarked,
            which lets espeak place its own stress.
        secondary_at: As above, for secondary stress.

    Returns:
        A mnemonic string such as `kwab'ina`.

    Raises:
        MnemonicError: A phone has no mnemonic. Raised rather than dropped
            because a dropped phone here is a silent mispronunciation in every
            utterance containing the word, not a one-off.

    No spaces are emitted: espeak treats a space inside `[[ ]]` as a word
    boundary, which resets stress and breaks the injection into fragments.
    """
    out: List[str] = []
    i = 0
    while i < len(phones):
        if stress_at is not None and i == stress_at:
            out.append(_STRESS)
        elif secondary_at is not None and i == secondary_at:
            out.append(_SECONDARY)

        if i + 1 < len(phones):
            pair = (phones[i], phones[i + 1])
            if pair in DIPHTHONGS:
                out.append(DIPHTHONGS[pair])
                i += 2
                continue

        phone, long_ = _strip(phones[i])
        if not phone:
            i += 1
            continue
        if long_ and (phone + "ː") in LONG_VOWELS:
            out.append(LONG_VOWELS[phone + "ː"])
        elif phone in VOWELS:
            out.append(VOWELS[phone])
        elif phone in CONSONANTS:
            out.append(CONSONANTS[phone])
        else:
            raise MnemonicError(f"no espeak-en mnemonic for {phones[i]!r}")
        i += 1
    return "".join(out)


@functools.lru_cache(maxsize=200_000)
def phonemise(text: str, voice: str = "en-us") -> str:
    """What espeak-ng really says for *text*, as IPA. Needs the espeak-ng binary.

    Used to check injections, not at synthesis time -- sherpa-onnx calls its own
    bundled espeak. `--ipa=3` is the format Piper's phonemiser matches.
    """
    if _ESPEAK is None:
        raise RuntimeError("verifying injections needs the espeak-ng binary")
    out = subprocess.run(
        [_ESPEAK, "-q", "--ipa=3", "-v", voice, text],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


_TIE = "‍"          # espeak joins affricates and diphthongs with a ZWJ
_MARKS = "ˈˌ"


def _bare(ipa: str) -> str:
    return "".join(c for c in ipa.replace(_TIE, "") if c not in _MARKS and not c.isspace())


def verify(phones: Sequence[str], mnemonics: str, voice: str = "en-us") -> bool:
    """Did espeak read the injection as the phones we meant?

    This is not belt-and-braces. An invalid mnemonic makes espeak discard the
    remainder of the injection and return a fragment, with no error and no
    warning -- `[[ny'ani]]` returns 'n'. Every injection is therefore read back
    and compared, and a caller that cannot verify should send plain text instead.

    The comparison ignores stress marks, ties and spacing, and applies the
    mergers the mapping deliberately makes (ɑː for /a/, and espeak's own
    allophony such as intervocalic flapping), so a match means 'espeak produced
    the phone sequence this mapping asked for', not 'espeak produced the
    lexicon's exact transcription'.
    """
    got = _bare(phonemise(f"[[{mnemonics}]]", voice))
    want = _bare("".join(_expected_ipa(phones)))
    return _fold(got) == _fold(want)


# What each mnemonic comes back as, so `verify` compares like with like.
_MNEMONIC_IPA = {
    "A:": "ɑː", "0": "ɑː", "a": "æ", "V": "ʌ", "@": "ə", "3:": "ɜː", "3": "ɚ",
    "e": "e", "e:": "eː", "E": "ɛ", "E:": "ɛː",
    "i": "i", "i:": "iː", "I": "ɪ",
    "o": "o", "o:": "oː", "O": "ɔ", "O:": "ɔː",
    "u": "u", "u:": "uː", "U": "ʊ", "C": "ç",
    "eI": "eɪ", "aI": "aɪ", "oU": "oʊ", "aU": "aʊ", "OI": "ɔɪ",
    "N": "ŋ", "S": "ʃ", "Z": "ʒ", "tS": "tʃ", "dZ": "dʒ", "T": "θ",
    "D": "ð", "nj": "nj", "r": "ɹ", "Nm": "ŋm",
}

# Differences that are espeak being espeak rather than the mapping being wrong,
# so they are not held against a match:
#   ɾ -> t   intervocalic flapping. espeak-en renders /r/ as ɹ, never ɾ, so a ɾ
#            in the output can only be a flapped /t/ and folding it is safe.
#   ɡ -> g   espeak emits U+0261 LATIN SMALL LETTER SCRIPT G; the tables here use
#            ASCII 'g'. Purely a codepoint difference.
#   ɫ, ɐ, ʲ  velarised /l/, the reduced /a/ espeak uses unstressed, and
#            palatalisation espeak inserts before front vowels.
_FOLD = str.maketrans({"ɾ": "t", "ɡ": "g", "ɫ": "l", "ɐ": "ə", "ʲ": ""})


def _fold(ipa: str) -> str:
    return ipa.translate(_FOLD)


def _expected_ipa(phones: Sequence[str]) -> Iterable[str]:
    """The IPA the mnemonics for *phones* should read back as."""
    i = 0
    while i < len(phones):
        if i + 1 < len(phones) and (phones[i], phones[i + 1]) in DIPHTHONGS:
            yield _MNEMONIC_IPA[DIPHTHONGS[(phones[i], phones[i + 1])]]
            i += 2
            continue
        phone, long_ = _strip(phones[i])
        i += 1
        if not phone:
            continue
        mnemonic = ""
        if long_:
            mnemonic = LONG_VOWELS.get(phone + "ː", "")
        mnemonic = mnemonic or VOWELS.get(phone) or CONSONANTS.get(phone, "")
        if not mnemonic:
            continue
        yield _MNEMONIC_IPA.get(mnemonic, mnemonic)
