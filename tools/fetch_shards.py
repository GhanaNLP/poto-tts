"""Download the dataset shard by shard, extract WAVs, delete the parquet.

Shard-at-a-time because holding all 88 parquet files (108 GB) alongside the
extracted WAVs (~121 GB) is 230 GB of disk for no reason: once a shard's clips
are on disk as WAVs, the parquet is dead weight. Each shard is resumable -- a
shard whose marker file exists is skipped -- so an interrupted run costs at most
one shard.
"""
import io, json, os, sys, wave
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# Overridable, because this is the one script whose paths were written for one
# machine: POTO_TTS_DATA is where the WAVs land, POTO_TTS_HF_REPO is the dataset.
REPO = os.environ.get("POTO_TTS_HF_REPO", "ghanaopendata/ghana-english-tts-clean2")
ROOT = Path(os.environ.get("POTO_TTS_DATA",
                           "/mnt/volume_d2wey28/projects/poto-tts/data"))
WAV = ROOT / "wav"
CACHE = os.environ.get("HF_HOME_HUB", "/mnt/volume_d2wey28/hf_cache/hub")
N_SHARDS = 88

WAV.mkdir(parents=True, exist_ok=True)
(ROOT / "done").mkdir(exist_ok=True)

first, last = int(sys.argv[1]), int(sys.argv[2])
for shard in range(first, last + 1):
    marker = ROOT / "done" / f"{shard:05d}.json"
    if marker.exists():
        print(f"shard {shard:05d} already done", flush=True)
        continue
    name = f"data/data/train-{shard:05d}-of-{N_SHARDS:05d}.parquet"
    path = hf_hub_download(REPO, name, repo_type="dataset", cache_dir=CACHE)

    out_dir = WAV / f"{shard:05d}"
    out_dir.mkdir(exist_ok=True)
    manifest = ROOT / f"manifest-{shard:05d}.tsv"
    n = 0
    total_frames = 0
    with open(manifest, "w", encoding="utf-8") as man:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                text = (row["corrected_text"] or "").strip()
                raw = row["bytes"]
                if not text or not raw:
                    continue
                clip = f"{shard:05d}_{n:06d}.wav"
                with open(out_dir / clip, "wb") as fh:
                    fh.write(raw)
                with wave.open(io.BytesIO(raw)) as w:
                    frames, rate = w.getnframes(), w.getframerate()
                total_frames += frames / rate
                man.write(f"{shard:05d}/{clip}\t{rate}\t{frames}\t{text}\n")
                n += 1

    # The parquet is ~1.2 GB and of no further use; the HF cache keeps a blob
    # plus a symlink, so both go.
    real = os.path.realpath(path)
    for p in (path, real):
        try:
            os.remove(p)
        except OSError:
            pass
    marker.write_text(json.dumps({"clips": n, "hours": round(total_frames / 3600, 2)}))
    print(f"shard {shard:05d}: {n} clips, {total_frames/3600:.2f} h", flush=True)
