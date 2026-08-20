"""Synthesis with sherpa-onnx alone. No poto_tts anywhere in this file.

There is no front-end code, so any sherpa-onnx runtime -- Kotlin, Swift, C++,
WebAssembly, Rust -- gets the same pronunciations. This script is in Python only
because that is what is installed here; it uses nothing from the library, just the four
files on disk.

    python examples/bare_sherpa_onnx.py <voice-dir> "Kwabena went to Achimota" en out.wav

The voice directory is what `poto_tts.download.ensure_voice()` fetches, or a
`huggingface-cli download ghananlpcommunity/poto-tts-kokoro-gh`, or any copy of those
files in an app bundle. `lang` is `en`: British English, reading the Ghanaian entries
out of espeak's dictionary. There is nothing else to configure.

What still needs Python -- and the espeak-ng binary -- is *authoring*: `poto-tts dict`
compiles the lexicon into espeak-ng-data. That is a build step, run once, not something
a device does.
"""
import sys, wave
import numpy as np, sherpa_onnx

d, text, lang, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(
        kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=f"{d}/onnx/model.onnx", voices=f"{d}/voices.bin",
            tokens=f"{d}/tokens.txt", data_dir=f"{d}/espeak-ng-data", lang=lang),
        provider="cpu", num_threads=2))
assert cfg.validate(), "sherpa-onnx rejected the voice directory"
tts = sherpa_onnx.OfflineTts(cfg)
gen = sherpa_onnx.GenerationConfig(); gen.sid = 20          # Grace / bf_alice
audio = tts.generate(text, gen)
s = np.asarray(audio.samples)
with wave.open(out, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(audio.sample_rate)
    w.writeframes((np.clip(s, -1, 1) * 32767).astype("<i2").tobytes())
print(f"  {lang:6s} {s.size/audio.sample_rate:.2f}s @ {audio.sample_rate} Hz -> {out}")
