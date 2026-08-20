"""Ghanaian English speech synthesis on sherpa-onnx.

Every utterance is respelled before it reaches the model: each word's pronunciation
comes from a 104,623-word Ghanaian lexicon, is written in Lingua Franca Nova
orthography, and espeak reads it back into the phonemes Kokoro receives. The
pronunciation is data, not code, so the same voice files work wherever sherpa-onnx
runs -- Python, C++, Android, iOS, WebAssembly.

    from poto_tts import load

    tts = load()                                   # Kokoro, Apache-2.0
    tts.save("Kwabena went to Achimota", "out.wav")

    tts = load(voice="Emmanuel")                   # British male
    tts.prepare_text("Nyankpani")                  # 'nyankpani' -- what espeak is given
"""

from .backends import BACKENDS, DEFAULT_BACKEND, Backend, KokoroBackend, Synthesis, load
from .frontend import GhanaFrontend
from .respell import CONSONANTS, VOWELS, respell, respell_ipa
from .voices import VOICES, Voice, by_name, grouped

__version__ = "0.5.0"
__all__ = [
    "load",
    "KokoroBackend",
    "Backend",
    "Synthesis",
    "BACKENDS",
    "DEFAULT_BACKEND",
    "GhanaFrontend",
    "respell",
    "respell_ipa",
    "VOWELS",
    "CONSONANTS",
    "VOICES",
    "Voice",
    "by_name",
    "grouped",
]
