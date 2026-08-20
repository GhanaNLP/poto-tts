"""The engine poto-tts speaks through.

One backend, deliberately: Kokoro v1.0, Apache-2.0, with Ghanaian pronunciation
supplied by the front-end rather than by training. A Piper backend existed here and
was removed -- a voice trained on Ghanaian broadcast speech sounded worse at 45k
steps than Kokoro does with a good front-end, and it carried a licence restriction
the corpus's speakers never consented to lift. The training pipeline is still in
`tools/` for anyone who wants to train on data they hold rights to.

`Backend` stays a base class rather than being folded into `KokoroBackend`, because
the interesting part of this project is the front-end and it should be attachable to
whatever engine comes next.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import Backend, Synthesis, write_wav
from .kokoro import KokoroBackend

BACKENDS: Dict[str, Type[Backend]] = {KokoroBackend.name: KokoroBackend}

DEFAULT_BACKEND = KokoroBackend.name

__all__ = ["BACKENDS", "DEFAULT_BACKEND", "Backend", "KokoroBackend",
           "Synthesis", "write_wav", "load"]


def load(backend: str = DEFAULT_BACKEND, **kw) -> Backend:
    """Construct a backend by name."""
    try:
        cls = BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"unknown backend {backend!r}; choose from {', '.join(sorted(BACKENDS))}"
        ) from None
    return cls(**kw)
