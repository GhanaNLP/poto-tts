#!/usr/bin/env bash
# Train the Ghanaian English Piper voice at native 16 kHz on the H200.
#
# Why 16 kHz and not 22.05 kHz: the corpus is 16 kHz mono throughout. Upsampling
# it into a 22.05 kHz model would ask the decoder to synthesise an 8-11 kHz band
# that contains no signal, spending capacity to produce nothing. The reason that
# is normally a hard trade -- Piper's maintained trainer only resumes cleanly from
# `medium` checkpoints, which are 22.05 kHz -- does not apply here, because the
# architecture and the sample rate are separate settings. The trainer's defaults
# already *are* the medium architecture:
#
#     hop_length 256, filter_length 1024, win_length 1024, mel_channels 80,
#     upsample_rates (8, 8, 4), upsample_initial_channel 256, resblock '2'
#
# None of those tensors encodes a sample rate, so setting `--model.sample_rate
# 16000` and leaving the rest alone gives a native-16k model whose weights are
# shape-identical to a medium checkpoint. mel_fmax stays at its default (None =
# Nyquist), which is 8000 Hz here rather than 11025, so the mel target matches
# the real bandwidth of the audio.
#
# `--model.warmstart_ckpt` rather than `--ckpt_path`: warmstart copies the
# parameters that match and skips the rest, which is what makes a single-speaker
# checkpoint usable as the starting point for a multi-speaker model -- the speaker
# embedding table does not exist in the source and is left randomly initialised.
# `--ckpt_path` would try to restore optimiser state and speaker count too, and
# fail.
#
# `--data.trim_silence false` because the segments are already trimmed: each one
# starts and ends at a pause the aligner found, with 60 ms of padding kept
# deliberately so a cut does not clip a consonant release. Piper's VAD trim would
# take that padding back off, and it costs a second full decode of every file at
# 16 kHz -- the preparation pass is single-threaded, so that is the difference
# between forty minutes and a couple of hours before the first training step.
#
# The espeak dictionary is the other half of the contract. Piper phonemises with
# the espeak-ng data bundled in its own package, so the patched dictionary built
# by tools/build_espeak_dict.py has to be installed there before training -- see
# install_dict below. Train on stock espeak and the model learns anglicised
# Ghanaian names, which is the whole problem this project exists to fix.

set -euo pipefail

ROOT=/mnt/volume_d2wey28/projects/poto-tts
PIPER="$ROOT/piper1-gpl"
VENV="$PIPER/.venv-train"
DICT="$ROOT/build/espeak-ng-data"          # from tools/build_espeak_dict.py
CSV="$ROOT/data/train.csv"                 # from tools/prepare_dataset.py csv
# The aligned segments, not the raw clips: cut at real pauses, punctuated from the
# silences, 2.5-12 s instead of 13-15 s. Set AUDIO=$ROOT/data/wav to train on the
# unsegmented corpus instead.
AUDIO="${AUDIO:-$ROOT/data/seg}"
CACHE="$ROOT/cache"
OUT="$ROOT/runs/gh_en_16k"
BASE="$ROOT/base/lessac-medium.ckpt"

BATCH="${BATCH:-48}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"

install_dict() {
    # Piper's phonemiser reads `espeak-ng-data` from inside its own package
    # (src/piper/espeak-ng-data, created when the espeakbridge extension builds
    # espeak-ng from source).
    #
    # The *whole* directory is replaced, not just en_dict. Piper compiles its own
    # espeak-ng, so its phondata and phontab are not byte-identical to the ones
    # the shipped voice will carry, and those files define the phoneme inventory
    # and its allophonic rules -- the same text can come out with different
    # phones under two builds. Training on one inventory and synthesising with
    # another is a mismatch that no test in this repo would catch, so both sides
    # get the same directory and the difference cannot arise.
    local target
    target="$("$VENV/bin/python" - <<'PY'
import pathlib, piper
print(pathlib.Path(piper.__file__).parent / "espeak-ng-data")
PY
)"
    if [ ! -d "$target" ]; then
        echo "piper's espeak-ng-data not found at $target" >&2
        exit 1
    fi
    if [ ! -f "$DICT/en_dict" ]; then
        echo "no patched dictionary at $DICT/en_dict -- run build_espeak_dict.py" >&2
        exit 1
    fi
    if [ ! -d "$target.stock" ]; then
        cp -r "$target" "$target.stock"        # keep piper's original, once
    fi
    rm -rf "$target"
    cp -r "$DICT" "$target"
    echo "installed patched espeak-ng-data into $target"

    # Prove it took effect, in the same interpreter that will do the training.
    # The check is on a name espeak mis-parses without the dictionary: stock
    # espeak says /kwˈeɪbnə/, the dictionary says /kwɑːbˈɪnɑː/. Comparing against
    # the expected string rather than "did anything change" means a partially
    # applied dictionary fails here rather than three hours into training.
    "$VENV/bin/python" - <<'PY'
from piper.phonemize_espeak import EspeakPhonemizer
got = "".join(p for chunk in EspeakPhonemizer().phonemize("en-us", "Kwabena") for p in chunk)
print("Kwabena ->", got)
assert "ɪ" in got and "eɪ" not in got, (
    f"the patched dictionary is not in use: got {got!r}, expected the Ghanaian "
    f"/kwɑːbˈɪnɑː/. Training now would teach the model anglicised names."
)
PY
}

speakers() {
    cut -d'|' -f2 "$CSV" | sort -u | wc -l
}

main() {
    source "$VENV/bin/activate"
    install_dict
    local n
    n="$(speakers)"
    echo "training $n speakers from $(wc -l < "$CSV") utterances"
    mkdir -p "$CACHE" "$OUT"

    python3 -m piper.train fit \
        --data.voice_name gh_en \
        --data.csv_path "$CSV" \
        --data.audio_dir "$AUDIO" \
        --data.espeak_voice en-us \
        --data.cache_dir "$CACHE" \
        --data.config_path "$OUT/config.json" \
        --data.batch_size "$BATCH" \
        --data.num_workers 8 \
        --data.trim_silence false \
        --model.sample_rate 16000 \
        --model.num_speakers "$n" \
        --model.warmstart_ckpt "$BASE" \
        --trainer.default_root_dir "$OUT" \
        --trainer.max_epochs "$MAX_EPOCHS" \
        --trainer.precision bf16-mixed \
        --trainer.accelerator gpu \
        --trainer.devices 1 \
        --trainer.log_every_n_steps 50 \
        "$@"
}

main "$@"
