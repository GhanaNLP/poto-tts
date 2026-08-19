"""Ghanaian English speech synthesis: Piper on sherpa-onnx, with a Ghanaian espeak dictionary."""

from .inject import GhanaInjector, stress_index
from .mnemonics import CONSONANTS, LONG_VOWELS, VOWELS, MnemonicError, injection, verify

__version__ = "0.2.0"
__all__ = [
    "GhanaInjector",
    "injection",
    "verify",
    "stress_index",
    "MnemonicError",
    "VOWELS",
    "LONG_VOWELS",
    "CONSONANTS",
]
