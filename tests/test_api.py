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


# -- web interface and batch ----------------------------------------------


def test_index_is_self_contained(client):
    """No CDN, no build step: the page has to render on a box with no egress."""
    html = client.get("/").text
    assert "poto-tts" in html
    assert "http://" not in html and "https://" not in html


def test_ui_can_be_disabled():
    app = create_app(backends={"kokoro": StubBackend()}, preload=False, ui=False)
    assert TestClient(app).get("/").status_code == 404


def _zip_names(content):
    import io
    import zipfile

    return sorted(zipfile.ZipFile(io.BytesIO(content)).namelist())


def test_batch_with_a_text_column(client):
    csv = "text,voice\nKwabena went to Achimota,bm_george\nAkwaaba,\n"
    r = client.post("/batch", files={"file": ("in.csv", csv, "text/csv")})
    assert r.status_code == 200
    assert r.headers["X-Poto-Rows"] == "2"
    assert _zip_names(r.content) == ["0001.wav", "0002.wav", "manifest.csv"]


def test_batch_accepts_a_headerless_single_column(client):
    """What a one-column spreadsheet export looks like; rejecting it is pedantry."""
    r = client.post("/batch", files={"file": ("in.csv", "Akwaaba\nMedaase\n", "text/csv")})
    assert r.status_code == 200
    assert r.headers["X-Poto-Rows"] == "2"


def test_batch_honours_a_filename_column(client):
    csv = "text,filename\nAkwaaba,greeting\n"
    r = client.post("/batch", files={"file": ("in.csv", csv, "text/csv")})
    assert "greeting.wav" in _zip_names(r.content)


def test_batch_flattens_paths_in_filenames(client):
    """A filename column with a path would otherwise write outside the top level."""
    csv = "text,filename\nAkwaaba,../../escape.wav\n"
    r = client.post("/batch", files={"file": ("in.csv", csv, "text/csv")})
    assert _zip_names(r.content) == ["escape.wav", "manifest.csv"]


def test_batch_manifest_lists_every_row(client):
    import io
    import zipfile

    csv = "text\nAkwaaba\nMedaase\n"
    r = client.post("/batch", files={"file": ("in.csv", csv, "text/csv")})
    manifest = zipfile.ZipFile(io.BytesIO(r.content)).read("manifest.csv").decode()
    assert manifest.splitlines()[0] == "filename,voice,seconds,text"
    assert len(manifest.strip().splitlines()) == 3


def test_empty_csv_is_422(client):
    r = client.post("/batch", files={"file": ("in.csv", "\n\n", "text/csv")})
    assert r.status_code == 422


def test_batch_row_cap_is_refused_not_trimmed(client):
    from poto_tts import api

    rows = "text\n" + "Akwaaba\n" * (api.MAX_BATCH_ROWS + 1)
    r = client.post("/batch", files={"file": ("in.csv", rows, "text/csv")})
    assert r.status_code == 413
    assert "limit" in r.json()["detail"]
