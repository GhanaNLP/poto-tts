"""The REST surface, exercised with a stub backend.

The point of these is the contract a non-Python caller depends on -- status codes,
headers, the shape of the JSON -- not the audio. A stub keeps them fast and means
CI never downloads a voice.
"""

from __future__ import annotations

import io
import wave

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from poto_tts.api import create_app  # noqa: E402
from poto_tts.backends.base import Synthesis  # noqa: E402


class StubBackend:
    name = "kokoro"
    licence = "Apache-2.0"
    commercial_use = True
    sample_rate = 16000
    voice = "bf_alice"
    recommended = ["bf_alice"]
    voices = ["bf_alice", "bm_george"]

    def speak(self, text, voice=None, speed=None):
        if voice is not None and voice not in self.voices:
            raise ValueError(f"unknown voice {voice!r}")
        return Synthesis([0.1] * 8000, self.sample_rate, text, self.name,
                         voice or self.voice)


@pytest.fixture
def client():
    app = create_app(backends={"kokoro": StubBackend()}, preload=False)
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["default_backend"] == "kokoro"


def test_backends_reports_licence_and_commercial_use(client):
    entry = next(b for b in client.get("/backends").json()["backends"]
                 if b["name"] == "kokoro")
    assert entry["licence"] == "Apache-2.0"
    assert entry["commercial_use"] is True


def test_voices_lists_recommended(client):
    body = client.get("/voices").json()
    assert body["recommended"] == ["bf_alice"]
    assert body["voices"] == ["bf_alice", "bm_george"]


def test_platforms_names_the_artefacts_and_the_trap(client):
    """A caller who only reads the API should still learn that the voice runs on
    device, and that swapping espeak-ng-data breaks pronunciation silently."""
    body = client.get("/platforms").json()
    assert "espeak-ng-data/" in body["artefacts"]
    assert "android" in body["runtimes"]
    assert "mispronounces" in body["warning"]


def test_speak_get_returns_a_wav(client):
    r = client.get("/speak", params={"text": "Kwabena went to Achimota"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.headers["X-Poto-Backend"] == "kokoro"
    with wave.open(io.BytesIO(r.content)) as fh:
        assert fh.getframerate() == 16000
        assert fh.getnframes() == 8000


def test_speak_post_returns_a_wav(client):
    r = client.post("/speak", json={"text": "Akwaaba", "voice": "bm_george"})
    assert r.status_code == 200
    assert r.headers["X-Poto-Voice"] == "bm_george"


def test_empty_text_is_422(client):
    assert client.get("/speak", params={"text": "   "}).status_code == 422


def test_unknown_voice_is_422_not_500(client):
    r = client.post("/speak", json={"text": "hi", "voice": "nope"})
    assert r.status_code == 422


def test_unknown_backend_is_404(client):
    assert client.get("/voices", params={"backend": "melo"}).status_code == 404


def test_overlong_text_is_refused_not_truncated(client):
    """Silently truncating would ship half an utterance; the 413 says what to do."""
    r = client.post("/speak", json={"text": "a" * 5000})
    assert r.status_code == 413
    assert "split" in r.json()["detail"].lower()
