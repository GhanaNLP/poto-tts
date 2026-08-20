"""Text -> lfn spelling: the front-end every utterance passes through.

Per word, in order:

  1. **The Ghana lexicon** (`ghana-english-g2p`, 104,623 words) for anything it holds.
     This is what makes the output Ghanaian, and it covers ordinary English as well as
     names -- `convention` is stored /kɔnvɛnʃən/, not /kənvˈɛnʃən/ -- so the accent
     reaches the whole sentence rather than only the Ghanaian words.
  2. **espeak-en** for the rest: numbers, dates, abbreviations, and vocabulary outside
     Ghanaian usage. Its IPA is respelled the same way, so a lexicon miss still gets a
     plausible pronunciation.
  3. **Respelling** into lfn (see respell.py), which is the only notation sherpa-onnx
     will carry into Kokoro.

The fallback is worth understanding, because it is where quality drops. espeak-en
answers in *American* English, so a lexicon miss inherits American vowels -- flattened
by lfn's five-vowel inventory, which removes schwa, length and r-colouring and takes
much of the American quality with them, but not all of it. espeak also flaps
intervocalic /t/, and lfn has no flap, so a flapped /t/ is written `r` and returns as a
real /r/: unhandled, `citing` becomes `sairing` and `meeting` `miring`. Flaps are
therefore undone before respelling. Lexicon coverage on ordinary Ghanaian text runs
85-100%, so the fallback is a minority path, but it is the one that sounds wrong when
it goes wrong.

Why the whole sentence must be respelled, not only the lexicon words: the espeak voice
is chosen once per utterance, not per word. Under `lang=lfn` espeak reads every word
with lfn letter values, so an English spelling left in the string is read literally --
`the` as /thˈe/, `to` as /tˈo/. To ride in the same string, a word has to be in the
same alphabet.
"""

from __future__ import annotations

import functools
import re
from typing import Dict, List, Optional

from .respell import respell

__all__ = ["GhanaFrontend"]

# A number (keeping internal separators, so '3.30' is not read as two sentences), a
# word, whitespace, or a run of punctuation.
_TOKEN = re.compile(
    r"\d+(?:[.,:/]\d+)*(?=\D|$)|[^\W_]+(?:['’][^\W_]+)*|\s+|[^\w\s]+",
    re.UNICODE,
)

# Punctuation espeak uses for phrasing. Anything else is dropped: an unknown symbol
# reaches espeak as a word and gets spelled out loud.
_KEEP_PUNCT = set(";:,.!?—…\"()")

_TO_SPACE = str.maketrans({"-": " ", "–": " ", "—": " ", "/": " ", "_": " "})

# espeak-en's flap, and the vowel it flaps around. Turned back into /t/ before
# respelling, because lfn has no flap and would spell it `r`.
_UNFLAP = str.maketrans({"ɾ": "t"})


class GhanaFrontend:
    """Turns text into the lfn spelling sherpa-onnx should be given.

    Args:
        lexicon: Extra word -> IPA entries, merged over the packaged Ghana lexicon.
            Written the way `ghana-english-g2p` accepts it, space-separated
            ('k w a b e n a') or run-together ('kwabena'). Use this for a name the
            lexicon lacks or gets wrong.
        fallback: Ask espeak-en for words the lexicon does not have. False leaves them
            as their own spelling, which lfn then reads with Latin letter values --
            occasionally what you want for a Ghanaian-spelled word, usually not.
    """

    def __init__(self, lexicon: Optional[Dict[str, str]] = None, fallback: bool = True):
        from ghana_english_g2p import GhanaEnglishG2P

        # use_espeak=False so the two sources stay distinguishable here: a lexicon hit
        # is used as it stands, a miss goes through the flap repair below first.
        self._g2p = GhanaEnglishG2P(use_espeak=False, lexicon=lexicon)
        self.fallback = fallback
        self._espeak = None

    def _english_ipa(self, word: str) -> str:
        if self._espeak is None:
            try:
                import espeak_english

                self._espeak = espeak_english
            except ImportError as exc:  # pragma: no cover - install-time
                raise RuntimeError(
                    "words outside the Ghana lexicon need espeak: "
                    "pip install espeak-english"
                ) from exc
        return self._espeak.phonemes(word)

    @functools.lru_cache(maxsize=100_000)
    def _word(self, word: str) -> str:
        """One token -> its lfn spelling. Cached: news text repeats heavily."""
        phones = self._g2p.word(word)
        if phones:
            return respell(phones)
        if not self.fallback:
            return word

        from ghana_english_g2p import segment

        # espeak may answer with several words for one token ('2026' is four), and
        # each needs its own respelling or they fuse into one unpronounceable word.
        ipa = self._english_ipa(word).translate(_UNFLAP)
        return " ".join(respell(segment(chunk)) for chunk in ipa.split())

    def __call__(self, text: str) -> str:
        return self.respell_text(text)

    def respell_text(self, text: str) -> str:
        """Respell a whole text, keeping the punctuation espeak uses for phrasing."""
        out: List[str] = []
        for token in _TOKEN.findall(text.translate(_TO_SPACE)):
            if token.isspace():
                out.append(" ")
            elif token[0].isalnum():
                out.append(self._word(token))
            else:
                out.extend(c for c in token if c in _KEEP_PUNCT)
        return re.sub(r" +", " ", "".join(out)).strip()

    def coverage(self, text: str) -> float:
        """Fraction of the words the Ghana lexicon covers.

        A low number means the utterance is mostly espeak's American English pushed
        through lfn, rather than Ghanaian pronunciation.
        """
        words = [t for t in _TOKEN.findall(text) if t[:1].isalpha()]
        return self._g2p.coverage(words)

    def is_ghanaian(self, word: str) -> bool:
        """Is this word in the Ghana lexicon, rather than espeak's English?"""
        return word.lower() in self._g2p
