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


def test_only_british_voices_are_offered():
    """Kokoro's 25 other-language speakers exist in the model but are not listed:
    a Ghanaian English library cannot vouch for a Japanese speaker."""
    assert len(voices_mod.VOICES) == 8
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
    assert described["Grace"]["kokoro"] == "bf_alice"


def test_default_voice_is_a_recommended_one():
    assert make()._default_voice() in make().recommended


# -- the text reaches the model untouched ----------------------------------


def test_prepare_text_changes_nothing():
    """The text reaches the model as written. Pronunciation lives in the voice's
    espeak-ng-data -- the lexicon compiled into espeak's dictionary -- which is why a
    runtime without Python gets the same result. A rewrite here would be a
    Python-only behaviour and a divergence."""
    text = "Kwabena went to Achimota"
    assert make().prepare_text(text) == text


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


def test_annotate_marks_which_words_the_lexicon_supplied():
    """The audio cannot show it: a name from the lexicon and a name espeak guessed at
    sound equally confident, so a caller needs a way to ask."""
    pairs = dict(make().annotate("Yaw went to Kumasi by bus"))
    assert pairs["Yaw"] is True
    assert pairs["Kumasi"] is True
    assert pairs["bus"] is False
    assert pairs["went"] is False


def test_coverage_counts_only_words():
    backend = make()
    assert backend.coverage("Kumasi!") == 1.0
    assert backend.coverage("bus") == 0.0
    assert backend.coverage("") == 0.0


def test_unoffered_kokoro_speakers_are_still_reachable_by_their_own_name():
    """Only British voices are offered -- the pronunciations are shaped for the
    British phoneme table, so handing them to an American speaker is a mismatch.
    That is a default, not a prohibition: a caller who names `af_heart` outright
    has said what they want, and the model does have 53 speakers."""
    backend = make()
    assert backend.resolve("af_heart") == 3
    assert backend.resolve("bf_alice") == 20
    assert all(v.accent == "British" for v in voices_mod.VOICES)
