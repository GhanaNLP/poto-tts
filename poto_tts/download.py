"""Fetch voices from the Hugging Face Hub, letting the model's own config drive it.

Two things shape this module.

**Downloads are tracked.** Assets come through `huggingface_hub`, so every fetch
counts against the model repository the way it would for any Hub model. A raw
`urllib` GET against a release asset works just as well technically and tells the
people who published the voice nothing about whether anyone uses it.

**The repository describes itself.** `config.json` in each voice repo names its own
files, sample rate, speakers and licence, and this module downloads what that file
lists. So publishing a new voice -- another speaker set, a Twi model, a 22 kHz
rebuild -- needs no release of this library, and a voice can carry metadata the
library did not anticipate.

    ghananlpcommunity/poto-tts-kokoro-gh      Apache-2.0 weights, no training
    ghananlpcommunity/poto-tts-piper-gh-16k   trained on Ghanaian speech

The patched `espeak-ng-data` travels *inside* the voice repository rather than
being built locally, because a voice and the dictionary it was trained with are one
artefact: a Piper voice trained on one phoneme inventory and served with another
will speak, badly, and nothing will report an error.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

__all__ = ["DownloadError", "MODELS", "MARKER", "cache_dir", "ensure_voice",
           "find_espeak_data", "load_config"]


class DownloadError(RuntimeError):
    pass


MODELS: Dict[str, str] = {
    "kokoro": "ghananlpcommunity/poto-tts-kokoro-gh",
    "piper": "ghananlpcommunity/poto-tts-piper-gh-16k",
}

# The marker tools/build_espeak_dict.py leaves in a patched espeak-ng-data.
# Searching for this rather than for en_dict is deliberate: every stock
# espeak-ng-data has an en_dict, so matching on that would quietly accept espeak's
# American English and mispronounce every Ghanaian name -- a silent version of the
# exact failure this project exists to prevent.
MARKER = "poto-tts-dictionary.json"

CONFIG_NAME = "config.json"


def cache_dir() -> Optional[Path]:
    """Where to keep downloads, or None for huggingface_hub's own cache.

    POTO_TTS_CACHE exists for the deployment case -- a container that wants model
    files on a mounted volume rather than in a home directory that vanishes.
    """
    env = os.environ.get("POTO_TTS_CACHE")
    return Path(env).expanduser() if env else None


def _hub():
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - install-time
        raise DownloadError(
            "fetching a voice needs huggingface_hub: pip install poto-tts[tts]"
        ) from exc
    return huggingface_hub


def load_config(repo_id: str, revision: str = "main") -> dict:
    """Just the config, without pulling several hundred megabytes first.

    Worth the separate request: it means a caller can inspect a voice's licence,
    speakers and sample rate, or fail on a typo in a repo name, before committing
    to the download.
    """
    hub = _hub()
    try:
        path = hub.hf_hub_download(repo_id, CONFIG_NAME, revision=revision,
                                   cache_dir=str(cache_dir()) if cache_dir() else None)
    except Exception as exc:
        raise DownloadError(f"could not read {CONFIG_NAME} from {repo_id}: {exc}") from exc
    config = json.loads(Path(path).read_text())
    if "files" not in config:
        raise DownloadError(
            f"{repo_id}/{CONFIG_NAME} has no 'files' section, so this library "
            f"cannot tell which files to download. Is it a poto-tts voice?")
    return config


def ensure_voice(
    backend: Optional[str] = None,
    repo_id: Optional[str] = None,
    revision: str = "main",
    quiet: bool = False,
) -> Tuple[Path, dict]:
    """Download a voice and return `(directory, config)`.

    Only the files `config.json` lists are fetched, so a repository can hold extra
    material -- sample audio, a quantised variant, training notes -- without every
    user paying for it.
    """
    if repo_id is None:
        if backend is None:
            raise ValueError("pass either backend= or repo_id=")
        if backend not in MODELS:
            raise ValueError(f"no published voice for backend {backend!r}; "
                             f"known: {', '.join(sorted(MODELS))}")
        repo_id = MODELS[backend]

    config = load_config(repo_id, revision)
    patterns = set()
    for value in config["files"].values():
        # A directory entry (espeak-ng-data) needs its contents, not just its name.
        patterns.add(f"{value}/*" if "." not in Path(value).name else value)
    patterns.add(CONFIG_NAME)

    hub = _hub()
    if not quiet:
        size = config.get("download_mb")
        note = f" (~{size} MB)" if size else ""
        print(f"fetching {repo_id}{note}", file=sys.stderr)
    try:
        local = hub.snapshot_download(
            repo_id, revision=revision, allow_patterns=sorted(patterns),
            cache_dir=str(cache_dir()) if cache_dir() else None,
        )
    except Exception as exc:
        raise DownloadError(f"could not download {repo_id}: {exc}") from exc

    directory = Path(local)
    missing = [v for v in config["files"].values() if not (directory / v).exists()]
    if missing:
        raise DownloadError(
            f"{repo_id} is missing files its config declares: {', '.join(missing)}")
    return directory, config


def find_espeak_data(model_dir: Optional[Path] = None) -> Optional[Path]:
    """A *patched* espeak-ng-data, or None.

    Searched in order of how deliberate the choice is: an explicit environment
    variable, then the voice directory (a published voice carries the dictionary it
    was built with, which is the one to trust), then a local build in this
    checkout.
    """
    candidates = []
    env = os.environ.get("POTO_TTS_ESPEAK_DATA")
    if env:
        candidates.append(Path(env).expanduser())
    if model_dir is not None:
        candidates.append(Path(model_dir) / "espeak-ng-data")
    candidates.append(Path(__file__).resolve().parent.parent / "build" / "espeak-ng-data")
    for path in candidates:
        if (path / MARKER).is_file():
            return path
    return None
