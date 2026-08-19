"""Turn the scraped corpus into a Piper training set: filter, then find speakers.

Runs on the GPU box against the WAVs `fetch_shards.py` extracted. Three stages,
each writing its own file so a later stage can be re-run without redoing an
earlier one.

**Stage 1, text and audio filters (CPU).** The corpus is broadcast and interview
speech, which is not TTS material as it stands: utterances start mid-phrase, carry
disfluencies, and some clips are two people talking. Every filter below is
dropping something that would teach the model a habit we do not want, and each
prints its own count so the cost of each rule is visible rather than buried in a
single 'kept N of M'.

**Stage 2, speaker embeddings (GPU).** The dataset has no speaker labels. Each
clip is embedded three times -- start, middle, end -- with a speaker-verification
model. Two things come out of that: a clip whose three windows disagree has more
than one speaker in it and is dropped, and the mean embedding of the rest is what
stage 3 clusters. Doing both from one pass is why the windows are embedded
separately rather than the clip as a whole.

**Stage 3, clustering into pseudo-speakers.** k-means over the clip embeddings,
then clusters with too little audio are dropped -- a voice with ten minutes of
speech trains badly and dilutes the ones that would have worked. What survives
becomes the speaker names in Piper's CSV, and each is a selectable voice.

**Stage 4, forced alignment (GPU).** The corpus arrives as 13-15 second chunks cut
on a timer, not on speech: they start mid-phrase, end mid-phrase, and carry no
punctuation at all (3.3% end in '.', '!' or '?'). Aligning the transcript to the
audio gives a timestamp per word, and the silences between those words are the
punctuation the transcript is missing -- a pause is where a comma or a full stop
belongs, and a long pause is where the utterance should have been cut. So each
chunk becomes several utterances that begin and end at real boundaries and carry
commas and full stops the speaker actually produced.

This is the difference between a voice that reads and a voice that recites. It is
also the only honest way to get punctuation here: guessing commas from syntax puts
pauses where the speaker did not pause, whereas a measured silence is a fact about
the recording.

Usage:
    python prepare_dataset.py filter
    python prepare_dataset.py embed --batch 64
    python prepare_dataset.py cluster --k 256 --min-minutes 45
    python prepare_dataset.py align --batch 8 --shard 0 --of 4
    python prepare_dataset.py segment
    python prepare_dataset.py csv --from-segments
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from collections import Counter
from pathlib import Path

ROOT = Path("/mnt/volume_d2wey28/projects/poto-tts")
DATA = ROOT / "data"
WAV = DATA / "wav"

# -- stage 1: filters ------------------------------------------------------

# Clips must be long enough to be worth a training step and short enough to
# batch. The corpus is capped at 15 s by whoever cut it, and 108k of its 123k
# clips are 13 s or longer, so an upper bound below 15.5 does not trim outliers --
# it discards the dataset. At 16 kHz and hop 256 a 15 s clip is 938 mel frames,
# which batches fine on an H200.
MIN_SECONDS = 2.0
MAX_SECONDS = 15.5

# Disfluencies. A transcript that is one-tenth filler teaches the model to fill.
FILLERS = re.compile(r"\b(uh|um|umm|uhh|erm|eh|mm|mhm|hmm|ah|ehm)\b", re.I)
MAX_FILLER_RATIO = 0.06

# Characters we are willing to send to espeak. Anything else -- other scripts,
# stray symbols, mojibake -- means the transcript is not what was said.
ALLOWED = re.compile(r"^[A-Za-z0-9 .,;:!?'’\-()%&/$£¢€\"]+$")

# Speech is roughly 10-22 characters per second. Far outside that band means the
# transcript and the audio do not match: a truncated transcript against a long
# clip, or a wall of text against a short one. This catches misalignment that no
# text-only rule can see.
MIN_CPS, MAX_CPS = 8.0, 26.0

# There is no fragment filter, and the reason is worth recording: it was written
# first and then measured. Across the 123,390 clips, 9.6% start with a capital,
# 3.3% end with '.', '!' or '?', and 0.6% do both. Requiring a well-formed
# sentence keeps one clip in 160 -- the transcripts are ASR-style, unpunctuated
# and cut mid-phrase, and no filter turns them into sentences.
#
# What follows from that is a real limitation of the resulting voice, not a
# preprocessing detail: a model trained on unpunctuated text learns little about
# what a comma or a full stop should do, so phrasing at inference will be weaker
# than a voice trained on read speech. Fixing it properly means forced alignment
# to cut the long clips at pauses and punctuate them, which is a separate job.
# `stage_csv` appends a full stop so that utterance-final phrasing at least has a
# consistent signal.


def stage_filter(args) -> int:
    manifests = sorted(DATA.glob("manifest-*.tsv"))
    if not manifests:
        print("no manifests; run fetch_shards.py first", file=sys.stderr)
        return 1

    reasons: Counter = Counter()
    kept = []
    total = 0
    for manifest in manifests:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            rel, rate, frames, text = parts
            total += 1
            seconds = int(frames) / int(rate)
            text = " ".join(text.split())

            if not (MIN_SECONDS <= seconds <= MAX_SECONDS):
                reasons["duration"] += 1
                continue
            if len(text) < 8:
                reasons["text too short"] += 1
                continue
            if not ALLOWED.match(text):
                reasons["stray characters"] += 1
                continue
            words = text.split()
            if len(FILLERS.findall(text)) / len(words) > MAX_FILLER_RATIO:
                reasons["disfluent"] += 1
                continue
            cps = len(text) / seconds
            if not (MIN_CPS <= cps <= MAX_CPS):
                reasons["text/audio mismatch"] += 1
                continue
            kept.append((rel, seconds, text))

    out = DATA / "filtered.tsv"
    with open(out, "w", encoding="utf-8") as fh:
        for rel, seconds, text in kept:
            fh.write(f"{rel}\t{seconds:.3f}\t{text}\n")

    hours = sum(s for _, s, _ in kept) / 3600
    print(f"kept {len(kept)} of {total} clips ({hours:.1f} h) -> {out}")
    for reason, n in reasons.most_common():
        print(f"  dropped {n:>7} {reason}")
    return 0


# -- stage 2: speaker embeddings -------------------------------------------

WINDOW_SECONDS = 3.0

# Minimum cosine similarity between a clip's own windows: below it, the windows are
# not the same person and the clip is a conversation rather than an utterance.
#
# This number is measured, not chosen. Over all 112,208 clips, the within-clip
# minimum pairwise similarity has median 0.546 and mean 0.482, while random pairs
# drawn from *different* clips -- usually different speakers -- have median 0.237
# and mean 0.244. The two distributions are clearly separated, so the embeddings
# are discriminative on this audio; what a threshold buys is where to sit on the
# overlap in the tails:
#
#     0.45  keeps 68% of clips, accepts  5.4% of random pairs as same-speaker
#     0.50  keeps 60%,                   2.4%
#     0.55  keeps 49%,                   1.2%
#     0.62  keeps 28%,                   0.7%
#
# 0.55 is the operating point: it still leaves ~208 hours, which is well past what
# this training needs, in exchange for a 1.2% contamination rate. An earlier guess
# of 0.62 discarded three quarters of the corpus and looked like evidence that the
# corpus was mostly conversations; it was evidence about the threshold.
MIN_SELF_SIMILARITY = 0.55


def _read_wav(path: Path):
    """16-bit PCM mono WAV -> (float32 tensor in [-1, 1], sample rate).

    `wave` rather than torchaudio: every clip in this corpus is 16 kHz 16-bit mono
    -- checked against the manifest -- and torchaudio.load in torch 2.11 raises
    unless TorchCodec is installed, which is a dependency worth avoiding for a
    format the standard library already reads.
    """
    import numpy as np
    import torch

    with wave.open(str(path), "rb") as fh:
        if fh.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {fh.getsampwidth()*8}-bit")
        frames = fh.readframes(fh.getnframes())
        rate, channels = fh.getframerate(), fh.getnchannels()
    data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(data.copy()), rate


def stage_embed(args) -> int:
    import numpy as np
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    rows = [l.split("\t") for l in (DATA / "filtered.tsv").read_text().splitlines()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ROOT / "models" / "ecapa"),
        run_opts={"device": device},
    )

    embeddings = np.zeros((len(rows), 192), dtype=np.float32)
    self_sim = np.zeros(len(rows), dtype=np.float32)
    done = 0
    batch_waves, batch_index = [], []

    def flush():
        nonlocal batch_waves, batch_index, done
        if not batch_waves:
            return
        # Three windows per clip, embedded as one batch: padding to the longest
        # window is cheap because every window is the same length by construction.
        stacked = torch.stack(batch_waves).to(device)
        with torch.no_grad():
            vecs = encoder.encode_batch(stacked).squeeze(1).cpu().numpy()
        for row_i, (start, count) in batch_index:
            group = vecs[start:start + count]
            group = group / (np.linalg.norm(group, axis=1, keepdims=True) + 1e-9)
            embeddings[row_i] = group.mean(axis=0)
            sims = [float(group[a] @ group[b])
                    for a in range(count) for b in range(a + 1, count)]
            self_sim[row_i] = min(sims) if sims else 1.0
        done += len(batch_index)
        if done % 5000 < len(batch_index):
            print(f"  embedded {done}/{len(rows)}", flush=True)
        batch_waves, batch_index = [], []

    for i, (rel, seconds, _text) in enumerate(rows):
        wav, sr = _read_wav(WAV / rel)
        window = int(WINDOW_SECONDS * sr)
        if wav.numel() < window:
            wav = torch.nn.functional.pad(wav, (0, window - wav.numel()))
        offsets = [0, max(0, (wav.numel() - window) // 2), max(0, wav.numel() - window)]
        start = len(batch_waves)
        for off in offsets:
            batch_waves.append(wav[off:off + window])
        batch_index.append((i, (start, len(offsets))))
        if len(batch_index) >= args.batch:
            flush()
    flush()

    np.save(DATA / "embeddings.npy", embeddings)
    np.save(DATA / "self_sim.npy", self_sim)
    multi = int((self_sim < MIN_SELF_SIMILARITY).sum())
    print(f"embedded {len(rows)} clips; {multi} look like more than one speaker "
          f"({multi/len(rows):.1%})")
    return 0


# -- stage 3: clustering ---------------------------------------------------


# Mean cosine similarity of a cluster's clips to its own centroid. A tight cluster
# is one speaker; a loose one is two or three who sound alike, and a voice trained
# on it is an average of them.
#
# The measurement that matters here: filtering by cohesion barely changes how much
# data each voice gets (1.26 -> 1.42 h at a 0.80 floor) but costs 40% of the total,
# because this corpus is a long tail -- the largest single speaker has 4.45 hours.
# So cohesion is not used to discard data. Every cluster keeps its own speaker id
# and all of it trains the shared model, which is what learns the accent; cohesion
# decides only which voices are worth *advertising*. Big clusters turn out to be
# tighter than small ones (0.878 against 0.805), so the prolific speakers are
# genuinely single people rather than merges.
COHESION_RECOMMEND = 0.80


def stage_cluster(args) -> int:
    """Cluster into pseudo-speakers, then cap how much each one contributes.

    The caps are not about disk or time. A corpus like this is dominated by a few
    presenters, and without a cap one voice brings 40 hours while the next brings
    40 minutes; the model then sounds like the presenter whatever speaker id you
    ask for, because that is where the gradient came from. Capping per speaker
    buys balance, and capping the total keeps an epoch short enough to see
    progress -- 100-200 hours is already far more than the 5-40 hours Piper's own
    released voices are trained on.
    """
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    rows = [l.split("\t") for l in (DATA / "filtered.tsv").read_text().splitlines()]
    embeddings = np.load(DATA / "embeddings.npy")
    self_sim = np.load(DATA / "self_sim.npy")

    keep = self_sim >= MIN_SELF_SIMILARITY
    print(f"dropping {int((~keep).sum())} multi-speaker clips")

    index = np.flatnonzero(keep)
    vecs = embeddings[index]
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    kmeans = MiniBatchKMeans(n_clusters=args.k, random_state=0, batch_size=4096,
                             n_init=10)
    labels = kmeans.fit_predict(vecs)

    centres = kmeans.cluster_centers_
    centres = centres / (np.linalg.norm(centres, axis=1, keepdims=True) + 1e-9)
    to_centre = (vecs * centres[labels]).sum(1)
    cohesion = {c: float(to_centre[labels == c].mean())
                for c in range(args.k) if (labels == c).any()}

    seconds = np.array([float(rows[i][1]) for i in index])
    per_cluster = {c: float(seconds[labels == c].sum()) for c in range(args.k)}
    big = sorted((c for c, s in per_cluster.items() if s >= args.min_minutes * 60),
                 key=lambda c: -per_cluster[c])
    print(f"{len(big)} of {args.k} clusters have >= {args.min_minutes} min")
    for rank, c in enumerate(big[:20]):
        print(f"  voice {rank:02d}  cluster {c:3d}  {per_cluster[c]/3600:5.2f} h")

    name_of = {c: f"gh_{rank:02d}" for rank, c in enumerate(big)}
    assignment = {}
    used = {c: 0.0 for c in big}
    cap = args.max_hours_per_speaker * 3600
    # Longest clips first within a speaker: they carry more phonetic context per
    # training step, and every clip here is a mid-phrase chunk anyway.
    order = sorted(range(len(index)), key=lambda j: -seconds[j])
    for j in order:
        label = labels[j]
        if label not in name_of or used[label] + seconds[j] > cap:
            continue
        used[label] += seconds[j]
        assignment[int(index[j])] = name_of[label]
        if args.max_total_hours and sum(used.values()) >= args.max_total_hours * 3600:
            break

    voices = sorted({v for v in assignment.values()})
    detail = {name_of[c]: {"hours": round(used[c] / 3600, 2),
                           "cohesion": round(cohesion.get(c, 0.0), 3)}
              for c in big if used[c]}
    recommended = sorted(
        (n for n, d in detail.items() if d["cohesion"] >= COHESION_RECOMMEND),
        key=lambda n: -detail[n]["hours"])

    (DATA / "speakers.json").write_text(json.dumps({
        "min_self_similarity": MIN_SELF_SIMILARITY,
        "k": args.k,
        "min_minutes": args.min_minutes,
        "max_hours_per_speaker": args.max_hours_per_speaker,
        "cohesion_recommend": COHESION_RECOMMEND,
        "hours_kept": sum(used.values()) / 3600,
        "voices": voices,
        "recommended": recommended,
        "detail": detail,
        "assignment": assignment,
    }))
    print(f"kept {len(assignment)} clips across {len(voices)} voices "
          f"({sum(used.values())/3600:.1f} h)")
    print(f"{len(recommended)} of them are cohesive enough to advertise "
          f"(>= {COHESION_RECOMMEND}); the rest still train the shared model")
    for name in recommended[:15]:
        d = detail[name]
        print(f"  {name}  {d['hours']:5.2f} h  cohesion {d['cohesion']:.3f}")
    return 0


# -- stage 4: forced alignment, segmentation and punctuation ----------------

# Silence long enough to be a boundary rather than a stop closure. Below COMMA a
# gap is ordinary articulation; between COMMA and PERIOD it is a phrase break;
# above PERIOD the speaker finished a thought and the utterance can be cut there.
#
# Measured against the corpus rather than assumed: inter-word gaps here run p50
# 0.06 s, p75 0.12 s, p90 0.32 s, p95 0.50 s. PERIOD at 0.35 s therefore selects
# roughly the top decile of gaps, and the yield is what settles it -- over a
# 200-clip sample, 0.30/0.35/0.45 keep 71%/69%/60% of the audio as closed
# segments, at mean lengths 4.2/4.6/5.7 s. 0.35 buys sentence-like boundaries for
# a few points of data.
COMMA_PAUSE = 0.20
PERIOD_PAUSE = 0.35

# Segment length bounds. The short end is what makes the segmentation worth doing:
# training on 3-8 second phrases rather than 15 second chunks means the model sees
# utterance beginnings and endings, which is where prosody lives.
SEG_MIN = 2.5
SEG_MAX = 12.0

# Keep a little audio either side of the words, so a cut does not clip the release
# of the last consonant or the onset of the first.
SEG_PAD = 0.06

# Mean per-word alignment score below which transcript and audio disagree enough
# that the clip is not worth keeping. Alignment gives this for free and it catches
# mismatches no text rule can see.
#
# Note the scale: these are log-probabilities, not a 0-1 confidence. Over a
# 250-clip sample the mean is -0.88 with p5 at -1.98, so a threshold of -1.8 drops
# the badly-aligned tail and keeps about 92%. An earlier value of 0.35, written
# before the distribution was checked, would have discarded the entire corpus.
MIN_ALIGN_SCORE = -1.8

# A crude SNR: the 95th percentile of frame energy over the 10th, in dB. The
# corpus turns out to be clean -- p50 31 dB, p5 21 dB, no clipped frames -- so this
# is a safety net for the worst 1-2% rather than a filter that shapes the set.
MIN_SNR_DB = 15.0


def _segment_words(words, min_s=SEG_MIN, max_s=SEG_MAX):
    """Group aligned words into utterances, keeping only those closed by a pause.

    Greedy: walk the words and close the current segment at a gap of at least
    PERIOD_PAUSE once it is long enough, or force a close at a phrase-level gap if
    it has run past max_s.

    Whatever is left over at the end of the clip is discarded, and that is the
    important part. These clips were cut on a timer, not on speech: measured over
    250 of them, trailing silence is 0.02 s at every percentile from p1 to p99, so
    a clip's final words are always mid-phrase. Keeping that remainder and giving
    it a full stop would teach the model that '.' means stop abruptly -- corrupting
    the one prosodic cue that matters most -- and it only costs about 30% of the
    audio to drop it, since each clip yields two or three properly closed segments
    first.

    Mid-phrase *starts* are tolerated: leading silence is 0.04 s median, so every
    first segment begins in the middle of a phrase too, and dropping those as well
    would halve the corpus to fix a much softer problem.
    """
    segments, current = [], []
    for i, word in enumerate(words):
        current.append(word)
        gap = (words[i + 1]["start"] - word["end"]) if i + 1 < len(words) else None
        if gap is None:
            break                      # the remainder ends at the clip boundary
        span = current[-1]["end"] - current[0]["start"]
        if (gap >= PERIOD_PAUSE and span >= min_s) or (span >= max_s and gap >= COMMA_PAUSE):
            segments.append(current)
            current = []
    return [s for s in segments if s and (s[-1]["end"] - s[0]["start"]) >= min_s]


# Words a comma will not be written after, however long the pause. A speaker who
# stops after 'and' or 'into' is hesitating, not closing a phrase, and the sample
# output was full of it: "serve the present age, and, a calling to fulfill",
# "get, to the link, sports stadium". The pause is real, so marking it is not
# dishonest -- but a comma in text a user types is syntactic, and training the model
# that a comma may follow 'and' spends the cue on hesitation instead of structure.
# The audio keeps the pause; only the comma is withheld.
HESITATION_BEFORE = frozenset("""
a an the this that these those and or but nor so yet if than as at by for from in
into of off on onto out over to up with i we you he she it they my our your his
her their is are was were be been being have has had do does did will would can
could may might must shall very just also then there here no not too
""".split())


# Disfluencies. A segment containing one is dropped rather than cleaned: the filler
# is in the audio, so removing it from the text alone would misalign them, and
# cutting it out of the middle of an utterance leaves an audible splice. They are
# only 10% of segments.
SEGMENT_FILLERS = frozenset("uh um umm uhh erm eh mm mhm hmm ehm ah".split())

# The fewest words worth a training step. Two-word fragments carry almost no
# phonetic context and their prosody is mostly boundary effects.
MIN_WORDS = 4


def _clean_segment(segment):
    """Trim a segment to something that ends where a sentence could end.

    A pause is not always a sentence boundary: a speaker who hesitates after 'and'
    or trails off on 'I'm' produces a gap that passes the PERIOD_PAUSE test, and
    the segment then gets a full stop after a function word. Left alone that
    teaches the model that '.' can mean 'stop mid-phrase' -- the same error as
    keeping the clip's trailing remainder, arriving by a different route.

    Trailing function words are trimmed rather than the segment discarded, because
    the timings make it exact: drop the words, drop the audio after the new last
    word, and what remains still ends at a real pause. Over this corpus that
    recovers most of the 22 hours that dropping the whole segment would cost.

    Returns None if what is left is not worth keeping.
    """
    words = list(segment)
    while words and words[-1]["text"].strip(",.?!;:'\"").lower() in HESITATION_BEFORE:
        words.pop()
    if len(words) < MIN_WORDS:
        return None
    if any(w["text"].strip(",.?!;:'\"").lower() in SEGMENT_FILLERS for w in words):
        return None
    if words[-1]["end"] - words[0]["start"] < SEG_MIN:
        return None
    return words


# Characters that must never reach the CSV. A double quote is the dangerous one:
# Piper reads its metadata with `csv.reader`, whose default quotechar is '"', so a
# single unmatched quote makes the parser swallow every following line until it
# finds a closing one. Sixty-eight rows here carried a quote from the transcript,
# and they ate 1,837 other rows, producing "utterances" of 56,722 characters whose
# phonemes were the WAV filenames of everything they had absorbed. That is what
# crashed training twice: as more tokens than mel frames it walked off the end of
# the monotonic_align kernel, and as a 250,000-token attention matrix it asked
# CUDA for 7,761 GiB.
#
# The pipe is stripped for the same class of reason -- it is the field separator --
# and control characters because a newline would split a row in half.
CSV_UNSAFE = str.maketrans({c: None for c in '"\u201c\u201d|\r\n\t'})


def csv_safe(text: str) -> str:
    """Text that cannot corrupt Piper's CSV parsing."""
    return " ".join(text.translate(CSV_UNSAFE).split())


def _punctuate(segment):
    """Text for one segment: commas at its phrase breaks, a full stop at the end.

    The full stop is earned rather than assumed -- `_segment_words` only returns
    segments that were closed by a pause of at least PERIOD_PAUSE.
    """
    parts = []
    for i, word in enumerate(segment):
        text = word["text"]
        if i + 1 < len(segment):
            gap = segment[i + 1]["start"] - word["end"]
            bare = text.strip(",.?!;:'\"").lower()
            if (gap >= COMMA_PAUSE and bare not in HESITATION_BEFORE
                    and not text.endswith((",", ".", "?", "!", ";", ":"))):
                text += ","
        parts.append(text)
    out = csv_safe(" ".join(parts)).strip().rstrip(",")
    return out + "." if out and out[-1] not in ".?!" else out


def stage_segment(args) -> int:
    """Cut the aligned clips into utterances and write the segment manifest.

    Reads the timings `align` stored, so changing a pause threshold or the comma
    policy costs a couple of minutes rather than another GPU hour.
    """
    align_root = DATA / "align"
    files = sorted(align_root.rglob("*.json"))
    if not files:
        print("no alignments; run the align stage first", file=sys.stderr)
        return 1

    out_root = DATA / "seg"
    out_root.mkdir(exist_ok=True)
    kept = dropped_short = dropped_unclean = 0
    seconds_in = seconds_out = 0.0
    commas = periods = 0
    lengths = []

    with open(DATA / "segments.tsv", "w", encoding="utf-8") as manifest:
        for n, path in enumerate(files):
            info = json.loads(path.read_text())
            words = info["words"]
            seconds_in += info["duration"]
            segments = _segment_words(words)
            if not segments:
                continue
            wav, sr = _read_wav(WAV / info["clip"])
            shard = info["clip"].split("/")[0]
            stem = Path(info["clip"]).stem
            (out_root / shard).mkdir(parents=True, exist_ok=True)
            for k, raw in enumerate(segments):
                segment = _clean_segment(raw)
                if segment is None:
                    dropped_unclean += 1
                    continue
                start = max(0.0, segment[0]["start"] - SEG_PAD)
                end = min(info["duration"], segment[-1]["end"] + SEG_PAD)
                if end - start < SEG_MIN:
                    dropped_short += 1
                    continue
                text = _punctuate(segment)
                name = f"{shard}/{stem}_{k:02d}.wav"
                _write_wav(out_root / name, wav[int(start * sr):int(end * sr)].numpy(), sr)
                manifest.write(f"{name}\t{end - start:.3f}\t{info['speaker']}\t{text}\n")
                kept += 1
                seconds_out += end - start
                lengths.append(end - start)
                commas += text.count(",")
                periods += 1
            if n % 5000 == 0:
                print(f"  {n}/{len(files)} clips -> {kept} segments "
                      f"({seconds_out/3600:.1f} h)", flush=True)

    import statistics
    print(f"\n{kept} segments, {seconds_out/3600:.1f} h of {seconds_in/3600:.1f} h "
          f"aligned ({seconds_out/max(seconds_in,1e-9):.0%} kept)")
    if lengths:
        print(f"  length mean {statistics.mean(lengths):.1f}s  "
              f"median {statistics.median(lengths):.1f}s  "
              f"min {min(lengths):.1f}s  max {max(lengths):.1f}s")
    print(f"  {commas} commas over {periods} utterances "
          f"({commas/max(periods,1):.2f} per utterance)")
    print(f"  dropped {dropped_short} segments shorter than {SEG_MIN}s")
    print(f"  dropped {dropped_unclean} segments with a filler, too few words, or "
          f"nothing left after trimming trailing function words")
    return 0


def stage_align(args) -> int:
    import numpy as np
    import torch
    from ctc_forced_aligner import (generate_emissions, get_alignments, get_spans,
                                    load_alignment_model, postprocess_results,
                                    preprocess_text)

    rows = [l.split("\t") for l in (DATA / "filtered.tsv").read_text().splitlines()]
    speakers = json.loads((DATA / "speakers.json").read_text())["assignment"]
    todo = [(i, rows[i]) for i in range(len(rows)) if str(i) in speakers]
    # Sharded by stride rather than by block, so every worker sees the same mix of
    # speakers and shards and their progress lines are comparable.
    if args.of > 1:
        todo = todo[args.shard::args.of]
    print(f"aligning {len(todo)} clips (shard {args.shard + 1}/{args.of})", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_alignment_model(
        device, dtype=torch.float16 if device == "cuda" else torch.float32)

    # Timings are stored per clip and the cutting happens in `segment`, a separate
    # stage. Alignment costs an hour on four GPU workers; deciding where a comma
    # belongs is a policy that wants revisiting, and it should not cost an hour to
    # change one threshold.
    out_root = DATA / "align"
    out_root.mkdir(exist_ok=True)
    kept = dropped_score = dropped_snr = 0
    seconds_out = 0.0

    for n, (row_i, (rel, _seconds, text)) in enumerate(todo):
        path = WAV / rel
        try:
            wav, sr = _read_wav(path)
            if _snr_db(wav.numpy()) < MIN_SNR_DB:
                dropped_snr += 1
                continue
            audio = wav.to(device).to(model.dtype)
            emissions, stride = generate_emissions(model, audio, batch_size=args.batch)
            tokens_starred, text_starred = preprocess_text(
                text, romanize=True, language="eng")
            segments_ctc, scores, blank = get_alignments(emissions, tokens_starred, tokenizer)
            spans = get_spans(tokens_starred, segments_ctc, blank)
            words = postprocess_results(text_starred, spans, stride, scores)
        except Exception as exc:                     # a clip that will not align
            if n < 5:
                print(f"  {rel}: {type(exc).__name__}: {exc}", flush=True)
            dropped_score += 1
            continue

        words = [w for w in words if w.get("text")]
        if not words:
            dropped_score += 1
            continue
        mean_score = float(np.mean([w.get("score", 0.0) for w in words]))
        if mean_score < MIN_ALIGN_SCORE:
            dropped_score += 1
            continue

        shard = rel.split("/")[0]
        (out_root / shard).mkdir(parents=True, exist_ok=True)
        stem = Path(rel).stem
        (out_root / shard / f"{stem}.json").write_text(json.dumps({
            "clip": rel,
            "speaker": speakers[str(row_i)],
            "duration": len(wav) / sr,
            "score": mean_score,
            "words": [{"text": w["text"], "start": round(w["start"], 3),
                       "end": round(w["end"], 3), "score": round(w.get("score", 0.0), 3)}
                      for w in words],
        }))
        kept += 1
        seconds_out += len(wav) / sr

        if n % 2000 == 0:
            print(f"  {n}/{len(todo)} clips aligned ({seconds_out/3600:.1f} h)",
                  flush=True)

    print(f"\n{kept} clips aligned ({seconds_out/3600:.1f} h) -> {out_root}")
    print(f"  dropped {dropped_score} clips on alignment score or failure")
    print(f"  dropped {dropped_snr} clips below {MIN_SNR_DB} dB")
    return 0


def _snr_db(x) -> float:
    """Loud-to-quiet frame energy ratio in dB, as a stand-in for SNR.

    25 ms frames, 95th percentile over 10th: speech sits in the former and the
    noise floor in the latter. Crude, but it needs no model and it separates a
    clean studio feed from a phone-in recorded in a market.
    """
    import numpy as np

    if x.size < 4000:
        return 0.0
    frames = x[:x.size // 400 * 400].reshape(-1, 400)
    energy = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
    return float(20 * np.log10(np.percentile(energy, 95) / np.percentile(energy, 10)))


def _write_wav(path: Path, samples, rate: int) -> None:
    """16-bit PCM mono, the format the rest of the pipeline expects."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(pcm)


# -- stage 5: the CSV Piper trains from -----------------------------------


def stage_csv(args) -> int:
    """Write Piper's `wav|speaker|text`.

    Text stays as text on purpose: pronunciation comes from the patched espeak
    dictionary (tools/build_espeak_dict.py), which is also what runs at inference.
    Injecting phonemes here would work too, but then training text and the deployed
    front-end are two things to keep in step instead of one.
    """
    out = Path(args.out)
    n = 0

    if args.from_segments:
        # The aligned path: audio already cut at real pauses, text already
        # punctuated from the silences, fillers dropped, trailing function words
        # trimmed, speaker inherited from the parent clip. Paths are relative to
        # data/seg, so training needs --data.audio_dir pointing there.
        segments = DATA / "segments.tsv"
        if not segments.is_file():
            print(f"no {segments}; run the align and segment stages first",
                  file=sys.stderr)
            return 1
        with open(out, "w", encoding="utf-8") as fh:
            for line in segments.read_text(encoding="utf-8").splitlines():
                rel, _seconds, speaker, text = line.split("\t")
                # Sanitised again here, not only in `_punctuate`, so an older
                # segments.tsv cannot poison a fresh CSV.
                fh.write(f"{rel}|{speaker}|{csv_safe(text)}\n")
                n += 1
        print(f"wrote {n} rows from aligned segments -> {out}")
        print(f"train with --data.audio_dir {DATA / 'seg'}")
        return 0

    rows = [l.split("\t") for l in (DATA / "filtered.tsv").read_text().splitlines()]
    speakers = json.loads((DATA / "speakers.json").read_text())["assignment"]
    with open(out, "w", encoding="utf-8") as fh:
        for i, (rel, _seconds, text) in enumerate(rows):
            name = speakers.get(str(i))
            if not name:
                continue
            # Unaligned fallback: a terminal full stop is the only phrasing cue
            # that can be added honestly without timestamps -- the clip does end
            # there -- and these clips keep their fillers and mid-phrase edges.
            if text[-1:] not in ".!?":
                text += "."
            fh.write(f"{rel}|{name}|{text}\n")
            n += 1
    print(f"wrote {n} rows of whole clips -> {out}")
    print("note: unsegmented text keeps fillers and mid-phrase boundaries; "
          "pass --from-segments to use the aligned set")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("filter")
    p = sub.add_parser("embed")
    p.add_argument("--batch", type=int, default=64, help="clips per GPU batch")
    p = sub.add_parser("cluster")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--min-minutes", type=float, default=30.0)
    p.add_argument("--max-hours-per-speaker", type=float, default=6.0)
    p.add_argument("--max-total-hours", type=float, default=150.0)
    p = sub.add_parser("align")
    p.add_argument("--batch", type=int, default=8, help="emission batch per clip")
    p.add_argument("--shard", type=int, default=0, help="this worker's index")
    p.add_argument("--of", type=int, default=1, help="number of workers")
    sub.add_parser("segment")
    p = sub.add_parser("csv")
    p.add_argument("--out", default=str(DATA / "train.csv"))
    p.add_argument("--from-segments", action="store_true",
                   help="use the aligned segments rather than the whole clips")
    args = ap.parse_args()
    return {"filter": stage_filter, "embed": stage_embed, "cluster": stage_cluster,
            "align": stage_align, "segment": stage_segment,
            "csv": stage_csv}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
