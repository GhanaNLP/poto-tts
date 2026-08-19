"""Ghanaian English speech synthesis on sherpa-onnx.

Pronunciation lives in a patched espeak-ng dictionary rather than in code, so the
same voice files run wherever sherpa-onnx runs -- Python, C++, Android, iOS,
WebAssembly -- with no front-end to port.

    from poto_tts import load

    tts = load("kokoro")                       # Apache-2.0, commercial use fine
    tts.save("Kwabena went to Achimota", "out.wav")

    tts = load("piper", voice="gh_00")         # trained on Ghanaian speech,
                                               # non-commercial
"""

from .backends import BACKENDS, DEFAULT_BACKEND, Backend, KokoroBackend, PiperBackend, Synthesis, load
from .inject import GhanaInjector
from .mnemonics import CONSONANTS, LONG_VOWELS, VOWELS, MnemonicError, injection, verify

__version__ = "0.3.0"
__all__ = [
    "load",
    "BACKENDS",
    "DEFAULT_BACKEND",
    "Backend",
    "KokoroBackend",
    "PiperBackend",
    "Synthesis",
    "GhanaInjector",
    "injection",
    "verify",
    "MnemonicError",
    "VOWELS",
    "LONG_VOWELS",
    "CONSONANTS",
]
