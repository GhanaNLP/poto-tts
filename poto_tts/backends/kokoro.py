"""Kokoro on sherpa-onnx, with the Ghanaian pronunciations in espeak's dictionary.

The text reaches the model unchanged. Pronunciation comes from the voice's
`espeak-ng-data`: 44,321 Ghanaian words compiled into espeak's own English dictionary,
carrying the lexicon's IPA, read as British English. A word with an entry is spoken
the Ghanaian way; every other word is spoken as British English would say it.

Nothing in this library rewrites the text, which is why an Android or C++ runtime
gets the identical result -- see examples/bare_sherpa_onnx.py.
"""

from __future__ import annotations

import re
from typing import List, Union

from ..dictionary import ghanaian_words as _ghanaian_words
from .. import voices as _voices
from .base import Backend

__all__ = ["KokoroBackend"]


class KokoroBackend(Backend):
    """Kokoro v1.0 on sherpa-onnx, speaking Ghanaian English.

    There is nothing to configure about the front-end, because there is no
    front-end here: the text is passed through untouched and the pronunciation
    comes from the voice's `espeak-ng-data` -- the Ghana lexicon compiled into
    espeak's dictionary and read as British English. That is what makes the
    result identical on Android, iOS and WebAssembly, which cannot run Python.

    To change a pronunciation, change the lexicon or the voice file and rebuild the
    dictionary: see docs/CUSTOMISING.md. Args are as `Backend`.
    """

    name = "kokoro"

    def annotate(self, text: str) -> List[tuple]:
        """`text` split into (word, is_ghanaian) pairs.

        Which words are Ghanaian vocabulary -- names, places, titles, Twi and Ga
        loans, food, money -- and which are ordinary English. Useful for showing a
        reader what the front-end is actually for.

        Note what this does *not* report. Asking "did the lexicon supply this word?"
        would mark almost every word true: the lexicon covers the whole language,
        including the Ghanaian accent of English words, so almost every word would
        answer true. That is accurate and tells nobody anything. The classification here is
        `data/ghanaian-words.txt`, built by tools/classify_lexicon.py.

        Reporting only -- `speak()` does not call it, so nothing about the audio
        depends on it.
        """
        known = _ghanaian_words()
        return [(part, part.lower() in known)
                for part in re.findall(r"[\w\u0300-\u036f']+|[^\w\s]+|\s+", text)
                if part.strip()]

    def coverage(self, text: str) -> float:
        """Share of the words in *text* that are Ghanaian vocabulary."""
        words = [(w, hit) for w, hit in self.annotate(text) if any(c.isalpha() for c in w)]
        return sum(hit for _, hit in words) / len(words) if words else 0.0

    def prepare_text(self, text: str) -> str:
        """What will actually be sent to sherpa-onnx.

        The text itself: this library does not rewrite it. Kept because the REST API
        and the CLI use it to show what the engine receives, and because for most of
        this project's life it did rewrite every word -- a reader who remembers the
        respeller should see plainly that it is gone.
        """
        return text

    # -- voices ------------------------------------------------------------

    @property
    def voices(self) -> List[str]:
        """The offered voices, British first. Kokoro's own names also work."""
        return [v.name for v in _voices.VOICES]

    @property
    def recommended(self) -> List[str]:
        return [v.name for v in _voices.VOICES if v.recommended]

    def describe_voices(self) -> List[dict]:
        """Every voice with its gender, accent, Kokoro name and speaker id."""
        return [v._asdict() for v in _voices.VOICES]

    def resolve(self, voice: Union[str, int]) -> int:
        """Ghanaian name, Kokoro name, or speaker id -> speaker id."""
        if isinstance(voice, bool):
            raise TypeError("voice must be a name or an id, not a bool")
        if isinstance(voice, int) or str(voice).strip().lstrip("-").isdigit():
            sid = int(voice)
            if not 0 <= sid < len(_voices.KOKORO_SPEAKERS):
                raise ValueError(
                    f"speaker id {sid} is outside 0..{len(_voices.KOKORO_SPEAKERS) - 1}")
            return sid
        found = _voices.by_name(str(voice))
        if found is not None:
            return found.sid
        # Kokoro's own name for a speaker we do not offer -- an American or an
        # other-language one. Honoured rather than refused: the model has 53
        # speakers and a caller naming one directly has said what they want. The
        # docs say why the offered set is British.
        if str(voice).strip() in _voices.KOKORO_SPEAKERS:
            return _voices.KOKORO_SPEAKERS.index(str(voice).strip())
        if found is None:
            close = [v.name for v in _voices.VOICES
                     if str(voice).strip().lower() in v.name.lower()]
            hint = (f"; did you mean {', '.join(close[:4])}?" if close
                    else f"; try {', '.join(self.recommended)}")
            raise ValueError(f"unknown voice {voice!r}{hint}")
        return found.sid

    def _default_voice(self) -> str:
        return self.recommended[0] if self.recommended else self.voices[0]

    def speak(self, text: str, **kw):
        if not text.strip():
            raise ValueError("nothing to speak")
        return super().speak(text, **kw)

    def _config(self, *, num_threads: int, provider: str, debug: bool):
        s = self._sherpa
        return s.OfflineTtsConfig(
            model=s.OfflineTtsModelConfig(
                kokoro=s.OfflineTtsKokoroModelConfig(
                    model=self._path("model"),
                    voices=self._path("voices"),
                    tokens=self._path("tokens"),
                    data_dir=str(self.espeak_data),
                    # Set explicitly rather than left to the model's metadata, so a
                    # future model naming a different voice cannot silently
                    # phonemise against a dictionary we are not patching.
                    lang=self.espeak_voice,
                ),
                provider=provider,
                num_threads=num_threads,
                debug=debug,
            ),
        )

    def _prepare(self, gen) -> None:
        # Belt and braces: the config carries lang already, but a per-call value
        # cannot be overridden by a metadata fallback.
        gen.extra = {"lang": self.espeak_voice}
