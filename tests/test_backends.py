"""Backend plumbing: the registry, config handling and voice resolution.

These run without a model. The parts that need one -- does sherpa-onnx accept the
config, does the audio sound Ghanaian -- belong to the export smoke test and to
listening, not to a unit test that would download 380 MB in CI.
"""

from __future__ import annotations

import json

import pytest

from poto_tts.backends import BACKENDS, DEFAULT_BACKEND, KokoroBackend, PiperBackend, load
from poto_tts.backends.base import Backend, Synthesis, write_wav


def make(cls=KokoroBackend, **config):
    """A backend instance without touching sherpa-onnx or the filesystem.

    Constructed through __new__ on purpose: __init__ downloads a voice and loads a
    native library, and the logic under test here is pure bookkeeping.
    """
    backend = object.__new__(cls)
    backend.config = {
        "files": {"model": "onnx/model.onnx", "tokens": "tokens.txt"},
        "speakers": ["bf_alice", "bm_george", "af_heart"],
        "recommended": ["bm_george"],
        "licence": {"model": "Apache-2.0", "commercial_use": True},
        **config,
    }
    backend.speakers = {n: i for i, n in enumerate(backend.config["speakers"])}
    backend.recommended = list(backend.config["recommended"])
    return backend


# -- registry --------------------------------------------------------------


def test_both_backends_registered():
    assert set(BACKENDS) == {"kokoro", "piper"}


def test_kokoro_is_the_default():
    """The Apache-2.0 one, because a library should not hand you a licence
    problem unless you asked for it."""
    assert DEFAULT_BACKEND == "kokoro"


def test_unknown_backend_lists_the_known_ones():
    with pytest.raises(ValueError, match="kokoro"):
        load("melo")


# -- config-driven behaviour ----------------------------------------------


def test_voices_put_recommended_first():
    assert make().voices == ["bm_george", "af_heart", "bf_alice"]


def test_default_voice_is_the_first_recommendation():
    assert make()._default_voice() == "bm_george"


def test_default_voice_falls_back_to_alphabetical():
    assert make(recommended=[])._default_voice() == "af_heart"


def test_licence_comes_from_the_config():
    assert make().licence == "Apache-2.0"
    assert make().commercial_use is True
    piper = make(PiperBackend, licence={"model": "non-commercial",
                                        "commercial_use": False})
    assert piper.commercial_use is False


def test_missing_file_key_names_what_is_available():
    backend = make()
    backend.model_dir = __import__("pathlib").Path("/nowhere")
    with pytest.raises(KeyError, match="voices"):
        backend._path("voices")


# -- voice resolution ------------------------------------------------------


def test_resolve_by_name_and_id():
    backend = make()
    assert backend.resolve("bm_george") == 1
    assert backend.resolve(2) == 2
    assert backend.resolve("2") == 2


def test_resolve_suggests_a_near_match():
    with pytest.raises(ValueError, match="george"):
        make().resolve("george")


def test_resolve_rejects_out_of_range_id():
    with pytest.raises(ValueError, match="outside"):
        make().resolve(99)


def test_resolve_rejects_bool():
    """True is an int in Python, and would silently mean speaker 1."""
    with pytest.raises(TypeError):
        make().resolve(True)


# -- audio output ----------------------------------------------------------


def test_write_wav_roundtrip(tmp_path):
    import wave

    path = write_wav(tmp_path / "a.wav", [0.0, 0.5, -0.5, 0.25], 16000)
    with wave.open(str(path)) as fh:
        assert fh.getframerate() == 16000
        assert fh.getnchannels() == 1
        assert fh.getsampwidth() == 2
        assert fh.getnframes() == 4


def test_write_wav_clips_instead_of_wrapping(tmp_path):
    """A sample above 1.0 cast straight to int16 wraps to a negative value, which
    is heard as a click. It must saturate instead."""
    import wave

    path = write_wav(tmp_path / "b.wav", [2.0, -2.0], 16000)
    with wave.open(str(path)) as fh:
        frames = fh.readframes(2)
    assert frames == b"\xff\x7f\x01\x80"          # +32767, -32767


def test_synthesis_duration():
    s = Synthesis([0.0] * 24000, 24000, "hi", "kokoro", "bf_alice")
    assert s.duration == pytest.approx(1.0)
