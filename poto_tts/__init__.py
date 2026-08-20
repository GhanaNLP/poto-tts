"""Ghanaian English speech synthesis on sherpa-onnx.

Your text reaches the model unchanged. A 44,321-word Ghanaian lexicon is compiled into
espeak's own dictionary, so pronunciation is data rather than code -- and the same
files give the same result wherever sherpa-onnx runs: Python, C++, Android, iOS,
WebAssembly.

    from poto_tts import load

    tts = load()                                   # Kokoro, Apache-2.0
    tts.save("Kwabena went to Achimota", "out.wav")

    tts = load(voice="Emmanuel")                   # British male
    tts.annotate("Yaw went to Kumasi by bus")      # which words the lexicon supplied

Changing a pronunciation is a change to the lexicon, not to code: docs/CUSTOMISING.md.
"""

from .backends import BACKENDS, DEFAULT_BACKEND, Backend, KokoroBackend, Synthesis, load
from .voices import VOICES, Voice, by_name, grouped

__version__ = "0.7.0"
__all__ = [
    "load",
    "KokoroBackend",
    "Backend",
    "Synthesis",
    "BACKENDS",
    "DEFAULT_BACKEND",
    "VOICES",
    "Voice",
    "by_name",
    "grouped",
]
