"""The engines poto-tts can speak through.

Adding one is a config, not a front-end, because pronunciation lives in the
patched `espeak-ng-data` that all of them share.

    kokoro  Apache-2.0 weights, no training needed. Commercial use is fine.
    piper   Trained on Ghanaian speech, so the accent is in the weights.
            Non-commercial: see poto_tts/backends/piper.py.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import Backend, Synthesis, write_wav
from .kokoro import KokoroBackend
from .piper import PiperBackend

BACKENDS: Dict[str, Type[Backend]] = {
    KokoroBackend.name: KokoroBackend,
    PiperBackend.name: PiperBackend,
}

DEFAULT_BACKEND = KokoroBackend.name

__all__ = ["BACKENDS", "DEFAULT_BACKEND", "Backend", "KokoroBackend",
           "PiperBackend", "Synthesis", "write_wav", "load"]


def load(backend: str = DEFAULT_BACKEND, **kw) -> Backend:
    """Construct a backend by name.

    Kokoro is the default because it is the one anyone can use for anything: the
    Piper voice sounds more Ghanaian but its training data does not permit
    commercial use, and a library should not hand you a licence problem unless you
    asked for it.
    """
    try:
        cls = BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"unknown backend {backend!r}; choose from {', '.join(sorted(BACKENDS))}"
        ) from None
    return cls(**kw)
