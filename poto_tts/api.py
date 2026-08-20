"""A REST API over the voices, for callers that are not Python.

`poto-tts serve` starts this. It exists because the library's users are not all
Python programmes -- a PHP site, an Android app in development, a Twilio webhook,
a newsroom's CMS -- and because a server is the one deployment shape where the
model loads once and every request is cheap.

It is not, however, the way to ship to a phone. sherpa-onnx runs natively on
Android, iOS, WebAssembly and C++, and the voices this API serves are the same
files those runtimes load: `model.onnx`, `tokens.txt` and `espeak-ng-data`. On
device that means no network, no latency and no server bill. Use the API for
servers and prototypes; ship the files for apps. `GET /platforms` says the same
thing to anyone who only ever reads the API.

There is also a page at `/`: paste text and hear it, or upload a CSV and get a ZIP
back. One HTML file with no build step and no CDN, because a server that needs the
internet to render its own page is useless on a newsroom laptop, a Pi, or a container
with no egress.

Endpoints:

    GET  /                           the web interface
    POST /batch                      a CSV of rows -> a ZIP of WAVs and a manifest
    GET  /health                     liveness, and which voices are loaded
    GET  /backends                   engines, licences, commercial-use flags
    GET  /voices?backend=kokoro      speaker names, recommended ones first
    GET  /platforms                  how to run the same voice off-server
    POST /speak                      {"text": ..., "voice": ..., "speed": ...} -> WAV
    GET  /speak?text=...             the same, for a browser or curl
"""

from __future__ import annotations

import csv
import io
import os
import wave
import zipfile
from pathlib import Path
from typing import Dict, Optional

__all__ = ["create_app", "main"]

# Imported at module scope, not inside create_app, and that is load-bearing. This
# module uses `from __future__ import annotations`, so every annotation is a string
# that FastAPI resolves against the *module* globals -- a name imported inside the
# factory is not there. It cost two bugs to learn: a Pydantic body model read as a
# query parameter (every POST a 422), then an UploadFile that pydantic could not
# resolve at all. FastAPI stays optional by tolerating the ImportError here and
# failing in create_app with an instruction.
try:
    from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse, Response
except ImportError:  # pragma: no cover - the api extra is not installed
    FastAPI = None

# Requests longer than this are refused rather than truncated. A TTS endpoint is a
# convenient way to make a server do a lot of work for one cheap request, and the
# refusal tells the caller what to do instead.
MAX_CHARS = int(os.environ.get("POTO_TTS_MAX_CHARS", "2000"))

# Batch limits. A CSV is the easiest way to ask a server for an hour of compute by
# accident, so both are refused rather than trimmed: a caller who gets back fewer
# rows than they sent, silently, has a worse problem than one who gets an error.
MAX_BATCH_ROWS = int(os.environ.get("POTO_TTS_MAX_BATCH_ROWS", "500"))


def _read_batch_csv(raw: str, http_error) -> list:
    """Rows from an uploaded CSV, accepting the shapes people actually upload.

    A `text` column is the documented form. A single-column file with no header is
    accepted too, because that is what a spreadsheet export of one column looks like
    and rejecting it would be pedantry. `voice` and `filename` are optional.
    """
    reader = csv.reader(io.StringIO(raw))
    rows = [r for r in reader if any(field.strip() for field in r)]
    if not rows:
        raise http_error(status_code=422, detail="the CSV is empty")

    header = [h.strip().lower() for h in rows[0]]
    if "text" in header:
        index = {name: i for i, name in enumerate(header)}
        out = []
        for row in rows[1:]:
            text = row[index["text"]].strip() if index["text"] < len(row) else ""
            if not text:
                continue
            entry = {"text": text}
            for optional in ("voice", "filename"):
                if optional in index and index[optional] < len(row):
                    value = row[index[optional]].strip()
                    if value:
                        entry[optional] = value
            out.append(entry)
    else:
        # No header: treat the first column as the text, every row included.
        out = [{"text": row[0].strip()} for row in rows if row and row[0].strip()]

    if not out:
        raise http_error(
            status_code=422,
            detail="no rows with text found. Use a 'text' column, or a "
                   "single-column file with one utterance per line.")
    return out


def _wav_bytes(samples, sample_rate: int) -> bytes:
    """A WAV in memory, so nothing touches the filesystem per request."""
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.astype("<i2").tobytes())
    return buffer.getvalue()


def create_app(
    backends: Optional[Dict[str, object]] = None,
    default_backend: str = "kokoro",
    preload: bool = True,
    ui: bool = True,
    **backend_kwargs,
):
    """Build the FastAPI app.

    Args:
        backends: Already-constructed backends by name, for tests or for a caller
            that wants unusual settings. None constructs them on demand.
        default_backend: Used when a request names none.
        preload: Load the default backend at startup rather than on the first
            request. A cold first request on a 300 MB model is a timeout waiting
            to happen behind a load balancer.
        ui: Serve the web page at `/`. False leaves a JSON-only API, which is what
            you want behind someone else's front end.
        **backend_kwargs: Passed to every backend constructed here.
    """
    if FastAPI is None:
        raise RuntimeError("the REST API needs FastAPI: pip install 'poto-tts[api]'")

    from .backends import BACKENDS, load

    loaded: Dict[str, object] = dict(backends or {})

    def get_backend(name: Optional[str]):
        name = name or default_backend
        if name not in BACKENDS:
            raise HTTPException(
                status_code=404,
                detail=f"unknown backend {name!r}; have {sorted(BACKENDS)}")
        if name not in loaded:
            try:
                loaded[name] = load(name, **backend_kwargs)
            except Exception as exc:
                # A missing voice is a 503 rather than a 500: the server is fine,
                # the model is not there yet, and the message says which.
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return loaded[name]

    app = FastAPI(
        title="poto-tts",
        description="Ghanaian English speech synthesis over sherpa-onnx.",
        version=__import__("poto_tts").__version__,
    )

    @app.on_event("startup")
    def _startup():
        if preload and default_backend not in loaded:
            try:
                loaded[default_backend] = load(default_backend, **backend_kwargs)
            except Exception:
                # Deliberately not fatal: /health and /platforms stay useful, and
                # the first /speak returns a 503 explaining what is missing.
                pass

    @app.get("/health")
    def health():
        return {"status": "ok", "loaded": sorted(loaded),
                "default_backend": default_backend}

    @app.get("/backends")
    def backends_route():
        out = []
        for name in sorted(BACKENDS):
            entry = {"name": name, "loaded": name in loaded}
            if name in loaded:
                b = loaded[name]
                entry.update(licence=b.licence, commercial_use=b.commercial_use,
                             sample_rate=b.sample_rate, voices=len(b.voices))
            out.append(entry)
        return {"backends": out, "default": default_backend}

    @app.get("/voices")
    def voices(backend: Optional[str] = Query(None)):
        b = get_backend(backend)
        described = b.describe_voices() if hasattr(b, "describe_voices") else [
            {"name": v, "gender": "", "accent": "", "recommended": v in b.recommended}
            for v in b.voices]
        return {"backend": b.name, "default": b.voice, "recommended": b.recommended,
                "voices": b.voices, "described": described}

    @app.get("/platforms")
    def platforms():
        """The same voice, off the server. See this module's docstring."""
        return {
            "note": "The voices served here are sherpa-onnx models. The same three "
                    "artefacts run on device with no server and no network.",
            "artefacts": ["model.onnx", "tokens.txt", "espeak-ng-data/"],
            "warning": "Ship espeak-ng-data from the voice directory. Substituting "
                       "a stock espeak-ng-data leaves a working voice that "
                       "mispronounces every Ghanaian name.",
            "runtimes": {
                "android": "sherpa-onnx AAR, Kotlin/Java bindings",
                "ios": "sherpa-onnx XCFramework, Swift bindings",
                "web": "sherpa-onnx WebAssembly build",
                "desktop": "C++, C, Python, Go, C#, Java, Rust, Dart, Swift",
                "embedded": "Raspberry Pi and other arm64/arm32 Linux",
            },
            "docs": "https://k2-fsa.github.io/sherpa/onnx/",
        }

    def _speak(text: str, backend: Optional[str], voice: Optional[str],
               speed: Optional[float]):
        if not text or not text.strip():
            raise HTTPException(status_code=422, detail="text is empty")
        if len(text) > MAX_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f"text is {len(text)} characters; the limit is {MAX_CHARS}. "
                       f"Split it into sentences and concatenate the audio, or run "
                       f"the library directly for batch work.")
        b = get_backend(backend)
        try:
            result = b.speak(text, voice=voice, speed=speed)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        audio = _wav_bytes(result.samples, result.sample_rate)
        return Response(
            content=audio, media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="speech.wav"',
                "X-Poto-Backend": result.backend,
                "X-Poto-Voice": str(result.voice),
                "X-Poto-Duration": f"{result.duration:.3f}",
            },
        )

    # Declared as individual Body fields rather than as a Pydantic model, because
    # this module uses `from __future__ import annotations`: annotations become
    # strings, and FastAPI resolves them against the *module* globals. A model
    # defined inside this factory is not there, so FastAPI silently reads it as a
    # query parameter and every POST returns 422. Builtin and typing names resolve
    # fine, so these do.
    @app.post("/speak")
    def speak_post(
        text: str = Body(..., description="text to speak"),
        backend: Optional[str] = Body(None, description="synthesis engine"),
        voice: Optional[str] = Body(None, description="speaker name or id"),
        speed: Optional[float] = Body(None, gt=0.1, le=3.0),
    ):
        return _speak(text, backend, voice, speed)

    @app.get("/speak")
    def speak_get(text: str = Query(..., description="text to speak"),
                  backend: Optional[str] = Query(None),
                  voice: Optional[str] = Query(None),
                  speed: Optional[float] = Query(None, gt=0.1, le=3.0)):
        return _speak(text, backend, voice, speed)

    if ui:
        @app.get("/", response_class=HTMLResponse)
        def index():
            from importlib import resources

            return resources.files("poto_tts").joinpath("static/index.html").read_text(
                encoding="utf-8")

    @app.post("/batch")
    async def batch(
        file: UploadFile = File(..., description="CSV with a 'text' column"),
        backend: Optional[str] = Form(None),
        voice: Optional[str] = Form(None),
        speed: Optional[float] = Form(None),
    ):
        """A CSV in, a ZIP of WAVs and a manifest out.

        Synchronous on purpose. A job queue would be the right answer for thousands
        of rows, and the wrong answer for the newsroom running this on a laptop to
        voice twenty headlines -- which is the case this exists for. The row cap is
        what keeps the two apart.
        """
        raw = (await file.read()).decode("utf-8-sig", errors="replace")
        rows = _read_batch_csv(raw, HTTPException)
        if len(rows) > MAX_BATCH_ROWS:
            raise HTTPException(
                status_code=413,
                detail=f"{len(rows)} rows; the limit is {MAX_BATCH_ROWS}. Split the "
                       f"file, raise POTO_TTS_MAX_BATCH_ROWS, or use the library "
                       f"directly for a job this size.")
        b = get_backend(backend)

        buffer = io.BytesIO()
        manifest = ["filename,voice,seconds,text"]
        total = 0.0
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for i, row in enumerate(rows, start=1):
                text = row["text"]
                if len(text) > MAX_CHARS:
                    raise HTTPException(
                        status_code=413,
                        detail=f"row {i} is {len(text)} characters; the limit is "
                               f"{MAX_CHARS}")
                try:
                    result = b.speak(text, voice=row.get("voice") or voice, speed=speed)
                except ValueError as exc:
                    raise HTTPException(status_code=422,
                                        detail=f"row {i}: {exc}") from exc
                name = row.get("filename") or f"{i:04d}.wav"
                if not name.lower().endswith(".wav"):
                    name += ".wav"
                # Flattened: a filename column with a path in it would otherwise
                # write outside the archive's top level.
                name = Path(name).name
                archive.writestr(name, _wav_bytes(result.samples, result.sample_rate))
                manifest.append(
                    f'{name},{result.voice},{result.duration:.3f},"{text.replace(chr(34), chr(34) * 2)}"')
                total += result.duration
            archive.writestr("manifest.csv", "\n".join(manifest) + "\n")

        return Response(
            content=buffer.getvalue(), media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="poto-tts-batch.zip"',
                "X-Poto-Rows": str(len(rows)),
                "X-Poto-Seconds": f"{total:.1f}",
            },
        )

    @app.exception_handler(404)
    def _not_found(_request, exc):
        return JSONResponse(status_code=404, content={"detail": str(exc.detail)})

    return app


def main(argv=None) -> int:
    """`poto-tts serve` entry point."""
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="poto-tts serve",
                                 description="Serve the voices over HTTP.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to accept connections from other machines")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--backend", default="kokoro", help="backend used by default")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--provider", default="cpu")
    ap.add_argument("--no-preload", action="store_true",
                    help="load the model on the first request instead of at startup")
    ap.add_argument("--no-ui", action="store_true",
                    help="JSON API only, without the page at /")
    args = ap.parse_args(argv)

    try:
        import uvicorn
    except ImportError:  # pragma: no cover
        print("the server needs uvicorn: pip install 'poto-tts[api]'")
        return 1

    kwargs = {"num_threads": args.threads, "provider": args.provider}
    if args.voice:
        kwargs["voice"] = args.voice
    app = create_app(default_backend=args.backend, preload=not args.no_preload,
                     ui=not args.no_ui, **kwargs)
    where = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    if not args.no_ui:
        print(f"web interface: {where}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0
