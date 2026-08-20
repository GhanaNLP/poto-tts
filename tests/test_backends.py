"""Backend plumbing: the registry, voice resolution and audio output.

These run without a model. Whether sherpa-onnx accepts the config, and whether the
audio sounds Ghanaian, belong to the export smoke test and to listening -- not to a
unit test that would download 380 MB in CI.
"""

from __future__ import annotations

import pytest

from poto_tts import voices as voices_mod
from poto_tts.backends import BACKENDS, DEFAULT_BACKEND, KokoroBackend, load
from poto_tts.backends.base import Synthesis, write_wav


def make(**kw):
    """A KokoroBackend without touching sherpa-onnx, the network, or the filesystem.

    Built through __new__ deliberately: __init__ downloads a voice and loads a native
    library, while the logic under test here is bookkeeping.
    """
    backend = object.__new__(KokoroBackend)
    backend.config = {"files": {"model": "onnx/model.onnx", "tokens": "tokens.txt"},
                      "licence": {"model": "Apache-2.0", "commercial_use": True}}
    backend.respell = kw.get("respell", True)
    backend._frontend = kw.get("frontend")
    backend._lexicon_overrides = None
    return backend


# -- registry --------------------------------------------------------------


def test_only_kokoro_is_registered():
    """The Piper backend was removed: it sounded worse than Kokoro with a good
    front-end, and carried a licence its speakers never consented to lift."""
    assert set(BACKENDS) == {"kokoro"}
    assert DEFAULT_BACKEND == "kokoro"


def test_unknown_backend_lists_the_known_ones():
    with pytest.raises(ValueError, match="kokoro"):
        load("melo")


# -- voices ----------------------------------------------------------------


def test_only_english_voices_are_offered():
    """Kokoro's 25 other-language speakers exist in the model but are not listed:
    a Ghanaian English library cannot vouch for a Japanese speaker."""
    assert len(voices_mod.VOICES) == 28
    assert all(v.accent in ("British", "American") for v in voices_mod.VOICES)
    assert voices_mod.by_name("jf_alpha") is None


def test_ghanaian_aliases_and_kokoro_names_both_resolve():
    backend = make()
    assert backend.resolve("Emmanuel") == backend.resolve("bm_george")
    assert backend.resolve("emmanuel") == backend.resolve("Emmanuel")   # case-insensitive


def test_speaker_ids_still_work():
    """Kokoro's ids are what the model card and other sherpa-onnx tools use."""
    assert make().resolve(26) == 26
    assert make().resolve("26") == 26


def test_an_id_outside_the_model_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        make().resolve(99)


def test_unknown_voice_suggests_a_near_match():
    with pytest.raises(ValueError, match="Emmanuel"):
        make().resolve("emman")


def test_resolve_rejects_bool():
    """True is an int in Python and would silently mean speaker 1."""
    with pytest.raises(TypeError):
        make().resolve(True)


def test_british_voices_come_first_and_are_recommended():
    backend = make()
    assert backend.voices[0] == "Grace"
    assert set(backend.recommended) == {"Grace", "Comfort", "Mercy", "Patience",
                                        "Emmanuel", "Isaac", "Ebenezer", "Bright"}


def test_describe_voices_exposes_gender_and_accent():
    """So a UI can group them instead of decoding the af_/bm_ prefixes."""
    described = {v["name"]: v for v in make().describe_voices()}
    assert described["Emmanuel"]["gender"] == "male"
    assert described["Emmanuel"]["accent"] == "British"
    assert described["Gifty"]["kokoro"] == "af_heart"


def test_default_voice_is_a_recommended_one():
    assert make()._default_voice() in make().recommended


# -- respelling ------------------------------------------------------------


def test_prepare_text_respells_by_default():
    backend = make(frontend=lambda text: "kwabina")
    assert backend.prepare_text("Kwabena") == "kwabina"


def test_respell_can_be_turned_off():
    """With espeak_voice='en-us' this is stock Kokoro, which is how the comparison
    in samples/ was made."""
    backend = make(respell=False, frontend=lambda text: "SHOULD NOT BE USED")
    assert backend.prepare_text("Kwabena") == "Kwabena"


# -- audio output ----------------------------------------------------------


def test_write_wav_roundtrip(tmp_path):
    import wave

    path = write_wav(tmp_path / "a.wav", [0.0, 0.5, -0.5, 0.25], 16000)
    with wave.open(str(path)) as fh:
        assert (fh.getframerate(), fh.getnchannels(), fh.getnframes()) == (16000, 1, 4)


def test_write_wav_clips_instead_of_wrapping(tmp_path):
    """A sample above 1.0 cast straight to int16 wraps to a negative value, heard as
    a click. It must saturate."""
    import wave

    path = write_wav(tmp_path / "b.wav", [2.0, -2.0], 16000)
    with wave.open(str(path)) as fh:
        assert fh.readframes(2) == b"\xff\x7f\x01\x80"      # +32767, -32767


def test_synthesis_duration():
    assert Synthesis([0.0] * 24000, 24000, "hi", "kokoro", "Grace").duration == 1.0
