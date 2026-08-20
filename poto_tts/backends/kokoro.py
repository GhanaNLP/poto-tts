"""Kokoro: Apache-2.0 weights, Ghanaian pronunciation, no training.

Every utterance is respelled before it reaches the model. Each word's pronunciation
comes from the Ghanaian lexicon (or espeak-en for words the lexicon lacks), is
written in Lingua Franca Nova orthography, and espeak's `lfn` voice reads it back
into the phonemes Kokoro receives. See respell.py for why lfn, and frontend.py for
the per-word rules.

    text          Kwabena went to Achimota and met the Okuapenhene
    respelled     kwabina went tu atximota and met da okwapenhene
    Kokoro gets   kwabˈina wˈent tu ˌatʃimˈota ˈand mˈet dˈa ˌokwapenhˈene

Against stock Kokoro, which reads the same sentence /kwˈeɪbnə ... ˈoʊkjuːˌeɪpənhˌiːn/.

Two wrinkles that look like configuration options and are not. Kokoro v1.0 ships
`lexicon-us-en.txt`, and sherpa-onnx has a lexicon frontend that would consult it --
but that branch is unreachable for English: it runs only when `lang` is empty, `lang`
falls back to the model's `voice` metadata, and the metadata reader substitutes
'en-us' when that value is blank. And inline `[[phonemes]]`, which works on
sherpa-onnx's Piper frontend, is corrupted on this one: the Kokoro frontend rewrites
`:` to `,` in its input, which destroys length marks. Respelling is what is left, and
it turns out to be the better route anyway -- it reaches every word in the sentence
rather than only the ones a dictionary has entries for.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from .. import voices as _voices
from .base import Backend

__all__ = ["KokoroBackend"]


class KokoroBackend(Backend):
    """Kokoro v1.0 on sherpa-onnx, speaking Ghanaian English.

    Args:
        respell: Send every word through the Ghanaian front-end. False sends the text
            as it stands, which is worth doing once to hear the difference -- with
            `espeak_voice='en-us'` that is stock Kokoro.
        lexicon: Extra word -> IPA entries for the front-end, e.g.
            `{'Owusu': 'o w u s u'}` for a name the lexicon lacks or gets wrong.
        Everything else is as `Backend`.
    """

    name = "kokoro"

    def __init__(self, *args, respell: bool = True,
                 lexicon: Optional[Dict[str, str]] = None, **kw):
        # lfn unless told otherwise: it is the notation the front-end writes, and
        # pairing respelled text with an English voice would read the spellings as
        # English words.
        kw.setdefault("espeak_voice", "lfn" if respell else None)
        self.respell = respell
        self._lexicon_overrides = lexicon
        self._frontend = None
        super().__init__(*args, **kw)

    @property
    def frontend(self):
        """The Ghanaian front-end, built on first use (it loads a 104k-word lexicon)."""
        if self._frontend is None:
            from ..frontend import GhanaFrontend

            self._frontend = GhanaFrontend(lexicon=self._lexicon_overrides)
        return self._frontend

    def prepare_text(self, text: str) -> str:
        """What will actually be sent to sherpa-onnx. Useful for debugging a name."""
        return self.frontend(text) if self.respell else text

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
        """Respell, then synthesise. See `prepare_text` for what gets sent."""
        if not text.strip():
            raise ValueError("nothing to speak")
        return super().speak(self.prepare_text(text), **kw)

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
