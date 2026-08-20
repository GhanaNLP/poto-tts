"""What every backend has in common.

A backend is an engine that turns text into Ghanaian English speech through
sherpa-onnx. They differ in the model and its licence, not in the pronunciation:
every sherpa-onnx frontend used here phonemises English with espeak-ng, so all of
them read the same patched `espeak-ng-data`. That is the point of compiling the
lexicon into a dictionary rather than into code -- one dictionary serves every
engine, and adding an engine costs a config, not a front-end.

The voice repository's `config.json` supplies the file names, sample rate, speaker
list and licence, so a subclass only has to know how to hand those to sherpa-onnx.
"""

from __future__ import annotations

import json
import warnings
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

__all__ = ["Backend", "Synthesis", "write_wav"]


@dataclass
class Synthesis:
    """One utterance's audio, and enough context to report a bug about it."""

    samples: "object"          # numpy float32 array
    sample_rate: int
    text: str
    backend: str
    voice: Union[str, int]

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    def save(self, path: Union[str, Path]) -> Path:
        return write_wav(path, self.samples, self.sample_rate)


class Backend:
    """Base class: resolves the voice, configures sherpa-onnx, synthesises.

    Args:
        model_dir: A local voice directory. None downloads the published one.
        repo_id: A Hugging Face repo to use instead of the backend's default,
            for a voice this library has never heard of.
        espeak_data: The patched `espeak-ng-data`. None uses the one in the voice
            directory, and warns if it is not a poto-tts dictionary.
        voice: Speaker name or id. None takes the config's first recommendation.
        espeak_voice: Override the espeak voice the engine phonemises with. The
            voice's config names the right one and there is normally no reason to
            change it -- but the choice is what decides which dictionary applies,
            so it is exposed for experiments (e.g. 'lfn', whose five-vowel
            orthography reads Ghanaian spellings closer than English does).
        speed: Rate multiplier.
        num_threads: onnxruntime intra-op threads.
        provider: 'cpu', 'cuda' or 'coreml'.
        debug: Print sherpa-onnx diagnostics, including per-utterance token ids.
    """

    name: str = "base"

    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        repo_id: Optional[str] = None,
        espeak_data: Optional[Union[str, Path]] = None,
        voice: Optional[Union[str, int]] = None,
        espeak_voice: Optional[str] = None,
        speed: float = 1.0,
        num_threads: int = 2,
        provider: str = "cpu",
        debug: bool = False,
    ):
        import sherpa_onnx

        from ..download import ensure_voice, find_espeak_data

        self._sherpa = sherpa_onnx
        if model_dir is not None:
            self.model_dir = Path(model_dir)
            self.config = self._local_config(self.model_dir)
        else:
            self.model_dir, self.config = ensure_voice(
                backend=None if repo_id else self.name, repo_id=repo_id)

        self.speakers: Dict[str, int] = {
            name: i for i, name in enumerate(self.config.get("speakers") or [])}
        self.recommended: List[str] = list(self.config.get("recommended") or [])

        if espeak_data is not None:
            self.espeak_data = Path(espeak_data)
        else:
            found = find_espeak_data(self.model_dir)
            if found is None:
                warnings.warn(
                    f"{self.name}: no patched espeak-ng-data found, falling back to "
                    f"whatever ships with the model. Ghanaian names will come out as "
                    f"espeak's American English guesses (Kwabena as /kwˈeɪbnə/). "
                    f"Build one with tools/build_espeak_dict.py or set "
                    f"POTO_TTS_ESPEAK_DATA.", RuntimeWarning, stacklevel=2)
                found = self.model_dir / self.config["files"].get(
                    "espeak_data", "espeak-ng-data")
            self.espeak_data = found

        self.voice = voice if voice is not None else self._default_voice()
        self.espeak_voice = espeak_voice or self.config.get("espeak_voice", "en-us")
        self.speed = speed
        config = self._config(num_threads=num_threads, provider=provider, debug=debug)
        if not config.validate():
            raise RuntimeError(
                f"sherpa-onnx rejected the {self.name} voice in {self.model_dir}")
        self._tts = sherpa_onnx.OfflineTts(config)

    # -- config ------------------------------------------------------------

    @staticmethod
    def _local_config(model_dir: Path) -> dict:
        path = model_dir / "config.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. A voice directory needs the config.json that "
                f"describes it; tools/export_sherpa.py writes one.")
        return json.loads(path.read_text())

    def _path(self, key: str) -> str:
        """Absolute path to a file the config declares."""
        try:
            return str(self.model_dir / self.config["files"][key])
        except KeyError:
            raise KeyError(
                f"{self.name}: config.json declares no file for {key!r}; it has "
                f"{sorted(self.config.get('files', {}))}") from None

    def _default_voice(self) -> Union[str, int]:
        if self.recommended:
            return self.recommended[0]
        if self.speakers:
            return sorted(self.speakers)[0]
        return 0

    def _config(self, *, num_threads: int, provider: str, debug: bool):
        raise NotImplementedError

    # -- voices ------------------------------------------------------------

    @property
    def voices(self) -> List[str]:
        """Every usable voice, the recommended ones first."""
        rest = sorted(set(self.speakers) - set(self.recommended))
        return self.recommended + rest

    @property
    def licence(self) -> str:
        return (self.config.get("licence") or {}).get("model", "unspecified")

    @property
    def commercial_use(self) -> bool:
        return bool((self.config.get("licence") or {}).get("commercial_use", False))

    def resolve(self, voice: Union[str, int]) -> int:
        """Voice name or id -> speaker id."""
        if isinstance(voice, bool):
            raise TypeError("voice must be a name or an id, not a bool")
        if isinstance(voice, int):
            sid = voice
        elif str(voice).strip().lstrip("-").isdigit():
            sid = int(str(voice).strip())
        else:
            key = str(voice).strip()
            if key not in self.speakers:
                close = [v for v in self.speakers if key.lower() in v.lower()]
                hint = (f"; did you mean {', '.join(close[:5])}?" if close else
                        f"; try one of {', '.join(self.voices[:5])}")
                raise ValueError(f"unknown {self.name} voice {key!r}{hint}")
            return self.speakers[key]
        count = len(self.speakers) or 1
        if not 0 <= sid < count:
            raise ValueError(f"speaker id {sid} is outside 0..{count - 1}")
        return sid

    # -- synthesis ---------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate

    def speak(
        self,
        text: str,
        voice: Optional[Union[str, int]] = None,
        speed: Optional[float] = None,
    ) -> Synthesis:
        """Synthesise one text. Ghanaian pronunciation comes from `espeak_data`."""
        if not text.strip():
            raise ValueError("nothing to speak")
        chosen = self.voice if voice is None else voice
        gen = self._sherpa.GenerationConfig()
        gen.sid = self.resolve(chosen)
        gen.speed = self.speed if speed is None else speed
        self._prepare(gen)
        audio = self._tts.generate(text, gen)
        samples = audio.samples
        try:
            import numpy as np

            samples = np.asarray(samples, dtype=np.float32)
        except ImportError:  # pragma: no cover - numpy ships with sherpa-onnx
            pass
        return Synthesis(samples, audio.sample_rate, text, self.name, chosen)

    def _prepare(self, gen) -> None:
        """Last chance to set engine-specific generation options."""

    def save(self, text: str, path: Union[str, Path], **kw) -> Path:
        return self.speak(text, **kw).save(path)

    def __call__(self, text: str, **kw) -> Synthesis:
        return self.speak(text, **kw)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(voice={self.voice!r}, "
                f"sample_rate={self.sample_rate}, licence={self.licence!r})")


def write_wav(path: Union[str, Path], samples: Iterable[float], rate: int) -> Path:
    """16-bit PCM mono, written with the standard library.

    Clipped rather than cast directly: these models occasionally exceed +/-1.0 and
    an integer cast wraps around, which is heard as a click rather than distortion.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np

        data = np.asarray(samples, dtype=np.float32)
        pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    except ImportError:  # pragma: no cover
        import array
        import struct

        clipped = array.array("h", (int(max(-1.0, min(1.0, s)) * 32767) for s in samples))
        pcm = struct.pack(f"<{len(clipped)}h", *clipped)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return path
