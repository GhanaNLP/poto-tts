"""Export a trained checkpoint into a sherpa-onnx voice directory.

Piper's own exporter writes `model.onnx` and a `config.json`; sherpa-onnx wants
a `tokens.txt` and a handful of ONNX metadata properties that Piper does not
write. This script produces the directory sherpa-onnx actually loads:

    voice/
      model.onnx          the exported generator, with sherpa's metadata added
      tokens.txt          'symbol id' per line, from config.json
      espeak-ng-data/     the patched dictionary -- this is the Ghanaian part
      config.json         Piper's config, kept for reference and speaker names
      MODEL_CARD

The metadata is what makes sherpa-onnx pick the right frontend. `comment=piper`
is how it decides the model is a Piper VITS at all; `has_espeak=1` and
`voice=en-us` send the text through espeak-ng with the bundled data directory,
which is where the Ghanaian pronunciations live. Get `voice` wrong and the voice
still speaks -- with espeak's American English pronunciations of Ghanaian names,
which is the bug this project exists to fix, and it fails silently.

The smoke test runs in a subprocess on purpose: espeak initialises once per
process inside sherpa-onnx, so a test that loads two data directories in one
interpreter silently tests the first one twice.

Usage:
    python tools/export_sherpa.py \
        --checkpoint runs/gh_en_16k/lightning_logs/version_0/checkpoints/last.ckpt \
        --config runs/gh_en_16k/config.json \
        --espeak-data build/espeak-ng-data \
        --out dist/poto-tts-gh_en-16k
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

SMOKE_TEXT = "Kwabena went to Achimota and met the Okuapenhene."


def export_onnx(checkpoint: Path, out_model: Path) -> None:
    """Run Piper's exporter. It writes the generator only, without the discriminator."""
    subprocess.run(
        [sys.executable, "-m", "piper.train.export_onnx",
         "--checkpoint", str(checkpoint), "--output-file", str(out_model)],
        check=True,
    )


def write_tokens(config: dict, path: Path) -> int:
    """tokens.txt from the phoneme id map.

    The space symbol is a real token with its own id, so lines are written as
    'symbol<space>id' and read back by splitting on the *last* space -- the same
    convention sherpa-onnx uses. Writing them sorted by id keeps the file
    diffable between exports.
    """
    id_map = config["phoneme_id_map"]
    by_id = {}
    for symbol, ids in id_map.items():
        for i in ids:
            by_id.setdefault(int(i), symbol)
    with open(path, "w", encoding="utf-8") as fh:
        for i in sorted(by_id):
            fh.write(f"{by_id[i]} {i}\n")
    return len(by_id)


def add_metadata(model: Path, config: dict) -> None:
    import onnx

    m = onnx.load(str(model))
    meta = {
        "model_type": "vits",
        # sherpa-onnx keys off this string to select the Piper frontend.
        "comment": "piper",
        "language": "English",
        # The espeak voice. The patched dictionary is an en-us dictionary, so
        # this has to stay en-us for the Ghanaian entries to be consulted.
        "voice": config.get("espeak", {}).get("voice", "en-us"),
        "has_espeak": "1",
        "n_speakers": str(config.get("num_speakers", 1)),
        "sample_rate": str(config["audio"]["sample_rate"]),
    }
    # Replace rather than append: an exported model may already carry keys, and
    # duplicates make onnxruntime return whichever it finds first.
    keep = [p for p in m.metadata_props if p.key not in meta]
    del m.metadata_props[:]
    m.metadata_props.extend(keep)
    for key, value in meta.items():
        entry = m.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(m, str(model))


SMOKE = """
import sys, wave, numpy as np, sherpa_onnx
d = sys.argv[1]
cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
        model=f"{d}/model.onnx", tokens=f"{d}/tokens.txt",
        data_dir=f"{d}/espeak-ng-data"),
    provider="cpu", num_threads=2))
assert cfg.validate(), "sherpa-onnx rejected the voice directory"
tts = sherpa_onnx.OfflineTts(cfg)
gen = sherpa_onnx.GenerationConfig()
gen.sid = int(sys.argv[3]) if len(sys.argv) > 3 else 0
audio = tts.generate(sys.argv[2], gen)
s = np.asarray(audio.samples)
assert s.size, "no audio produced"
with wave.open(f"{d}/smoke.wav", "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(audio.sample_rate)
    w.writeframes((np.clip(s, -1, 1) * 32767).astype("<i2").tobytes())
print(f"{s.size / audio.sample_rate:.2f}s at {audio.sample_rate} Hz -> {d}/smoke.wav")
"""


def smoke_test(voice_dir: Path, text: str, sid: int) -> bool:
    result = subprocess.run(
        [sys.executable, "-c", SMOKE, str(voice_dir), text, str(sid)],
        capture_output=True, text=True,
    )
    print(result.stdout.strip() or result.stderr.strip()[-600:])
    return result.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True, help="config.json written by training")
    ap.add_argument("--espeak-data", required=True, help="patched espeak-ng-data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sid", type=int, default=0, help="speaker id for the smoke test")
    ap.add_argument("--speakers", default=None,
                    help="speakers.json from prepare_dataset, for the voice shortlist")
    ap.add_argument("--skip-export", action="store_true",
                    help="reuse an existing model.onnx in --out")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())

    model = out / "model.onnx"
    if not args.skip_export:
        export_onnx(Path(args.checkpoint), model)
    if not model.exists():
        print(f"no model at {model}", file=sys.stderr)
        return 1

    n_tokens = write_tokens(config, out / "tokens.txt")
    add_metadata(model, config)
    shutil.copy(args.config, out / "config.json")

    espeak_src = Path(args.espeak_data)
    if not (espeak_src / "en_dict").is_file():
        print(f"no en_dict in {espeak_src} -- run build_espeak_dict.py", file=sys.stderr)
        return 1
    espeak_dst = out / "espeak-ng-data"
    if espeak_dst.exists():
        shutil.rmtree(espeak_dst)
    shutil.copytree(espeak_src, espeak_dst)

    speakers = config.get("speaker_id_map") or {}
    # The recommended shortlist, if the preparation pipeline recorded one. Every
    # speaker id in the model is usable, but the ones listed here came from
    # cohesive speaker clusters and are the ones worth putting in front of users;
    # the rest exist because their audio helped train the shared model.
    recommended, detail = [], {}
    speakers_json = Path(args.speakers) if args.speakers else None
    if speakers_json and speakers_json.is_file():
        info = json.loads(speakers_json.read_text())
        recommended = info.get("recommended", [])
        detail = info.get("detail", {})
    (out / "MODEL_CARD").write_text(textwrap.dedent(f"""\
        poto-tts -- Ghanaian English, Piper VITS on sherpa-onnx

        sample rate   {config['audio']['sample_rate']} Hz (native, not upsampled)
        speakers      {config.get('num_speakers', 1)}
        tokens        {n_tokens}
        phonemes      espeak-ng en-us, with a Ghanaian dictionary compiled in
        licence       MIT (this voice); espeak-ng data is GPL-3.0

        Pronunciation of Ghanaian names comes from the espeak-ng-data directory in
        this folder, not from any code. Ship that directory alongside the model on
        every platform: replacing it with a stock espeak-ng-data leaves a working
        voice that says Kwabena as /kwˈeɪbnə/.

        Speakers: {len(speakers)} ids. Every id speaks, but the shortlist below is
        the set worth exposing -- those clusters were cohesive enough to be one
        person rather than an average of several. The others contributed audio to
        the shared model, which is what carries the accent.

        {chr(10).join(f"          {n}  sid {speakers.get(n, '?')}  "
                      f"{detail.get(n, {}).get('hours', '?')} h  "
                      f"cohesion {detail.get(n, {}).get('cohesion', '?')}"
                      for n in recommended[:25]) or "          (no shortlist recorded)"}
        """))

    print(f"\nwrote {out}")
    for item in sorted(out.iterdir()):
        size = sum(f.stat().st_size for f in item.rglob("*")) if item.is_dir() else item.stat().st_size
        print(f"  {item.name:18s} {size/1e6:8.1f} MB")

    print("\nsmoke test")
    return 0 if smoke_test(out, SMOKE_TEXT, args.sid) else 3


if __name__ == "__main__":
    raise SystemExit(main())
