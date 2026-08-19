"""Drop segments whose text cannot be aligned to their audio, before training.

This exists because of a crash. VITS learns a monotonic alignment between phoneme
tokens and mel frames, and Piper's Cython kernel for that (`monotonic_align`)
indexes out of bounds -- a segmentation fault, not an exception -- when an
utterance has more tokens than frames, because no monotonic path exists. Training
ran 32 steps and died in `core.so`.

Measured on this corpus, 0.6% of utterances have frames < tokens outright and 1.8%
fall below a 1.1 ratio. That is rare enough to survive a sanity check and frequent
enough to kill a run within the first epoch.

Why it happens here at all is worth recording: the model trains at 16 kHz with
hop_length 256, which is 62.5 mel frames per second, where the 22.05 kHz
checkpoint the settings came from gives 86. Choosing the corpus's native sample
rate -- right for fidelity -- cut the frame budget per second by 28% and made the
token count the binding constraint for fast speech.

Estimating token counts from character counts is not good enough: the ratio is 2.2
tokens per character at the median but 3.5 at p99 and 4.9 at worst, because digits
expand ('35.1 71.3%' is ten characters and about fifty phonemes). A
character-based rule either lets those through or discards thousands of good
segments. So this phonemises for real, with Piper's own phonemiser and the same
patched dictionary training will use, and compares exact counts.

Usage:
    python tools/check_alignable.py --csv data/train.csv --out data/train.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Piper interspersed a PAD token between phonemes and adds BOS/EOS, so the id
# sequence is about twice the phoneme count. Taken from its `phonemes_to_ids`.
IDS_PER_PHONEME = 2
IDS_OVERHEAD = 3

# Frames must exceed ids by this factor. Equality is a degenerate alignment where
# every token gets exactly one frame; the model needs room to allocate durations.
MIN_RATIO = 1.15


def _duration(path: Path) -> float:
    with wave.open(str(path), "rb") as fh:
        return fh.getnframes() / fh.getframerate()


def _phoneme_counts(texts):
    """Phoneme count per text, in a worker process.

    The phonemiser is created once per worker: espeak initialises once per
    process, so building it per call would be both slow and pointless.
    """
    from piper.phonemize_espeak import EspeakPhonemizer

    phonemizer = EspeakPhonemizer()
    out = []
    for text in texts:
        try:
            sentences = phonemizer.phonemize("en-us", text)
            out.append(sum(len(s) for s in sentences))
        except Exception:
            out.append(-1)          # unphonemisable: dropped by the caller
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True, help="Piper CSV: wav|speaker|text")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True, help="where to write the filtered CSV")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--hop-length", type=int, default=256)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--min-ratio", type=float, default=MIN_RATIO)
    args = ap.parse_args()

    # Parsed with csv.reader, the way Piper parses it -- not with str.split("|").
    # Validating with your own parser instead of the consumer's is how the quote
    # bug survived a filter written specifically to catch impossible utterances:
    # every line looked short and sane on its own, while Piper was reading 68 of
    # them as multi-row blobs.
    line_count = sum(1 for _ in open(args.csv, encoding="utf-8"))
    with open(args.csv, encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh, delimiter="|") if r]
    print(f"{len(rows)} rows in {args.csv}", file=sys.stderr)
    if len(rows) != line_count:
        print(f"  ERROR: {line_count} lines but {len(rows)} parsed rows -- "
              f"{line_count - len(rows)} were swallowed, almost certainly by an "
              f"unmatched quote. Sanitise the text before training.",
              file=sys.stderr)
        return 2
    longest = max(rows, key=lambda r: len(r[-1]))
    print(f"  longest text {len(longest[-1])} chars", file=sys.stderr)

    texts = [r[-1] for r in rows]
    chunk = max(1, len(texts) // (args.workers * 4))
    batches = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    counts: list[int] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, part in enumerate(pool.map(_phoneme_counts, batches)):
            counts.extend(part)
            print(f"\r  phonemised {len(counts)}/{len(texts)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    frames_per_second = args.sample_rate / args.hop_length
    audio_dir = Path(args.audio_dir)
    kept, dropped_ratio, dropped_missing, dropped_phonemes = [], 0, 0, 0
    seconds_kept = seconds_dropped = 0.0

    for row, phonemes in zip(rows, counts):
        path = audio_dir / row[0]
        if not path.is_file():
            dropped_missing += 1
            continue
        if phonemes <= 0:
            dropped_phonemes += 1
            continue
        seconds = _duration(path)
        frames = seconds * frames_per_second
        ids = IDS_PER_PHONEME * phonemes + IDS_OVERHEAD
        if frames < args.min_ratio * ids:
            dropped_ratio += 1
            seconds_dropped += seconds
            continue
        kept.append(row)
        seconds_kept += seconds

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write("|".join(row) + "\n")

    print(f"kept {len(kept)} rows ({seconds_kept/3600:.1f} h) -> {args.out}",
          file=sys.stderr)
    print(f"  dropped {dropped_ratio} for frames < {args.min_ratio} x ids "
          f"({seconds_dropped/3600:.1f} h) -- these are what crash monotonic_align",
          file=sys.stderr)
    print(f"  dropped {dropped_missing} with no audio file", file=sys.stderr)
    print(f"  dropped {dropped_phonemes} that would not phonemise", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
