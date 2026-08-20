---
title: poto-tts — Ghanaian English on Kokoro
emoji: 🗣️
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
---

# poto-tts — samples

English text-to-speech that pronounces Ghanaian words properly.

**[GhanaNLP/poto-tts on GitHub](https://github.com/GhanaNLP/poto-tts)** ·
`pip install poto-tts` ·
[the voice](https://huggingface.co/ghananlpcommunity/poto-tts-kokoro-gh) ·
[Android & iOS guide](https://github.com/GhanaNLP/poto-tts/blob/main/docs/MOBILE.md)

Thirty-two real sentences from a Ghanaian English news corpus, each synthesised twice —
standard Kokoro and poto-tts — across all eight voices. Tabs switch voice; each reads a
different four sentences. Underlined words are in the Ghanaian lexicon.

The Kokoro column is standard Kokoro: its own package, its PyTorch weights and misaki
for grapheme-to-phoneme. Not this pipeline with the dictionary removed.

The voices are Kokoro's British speakers, so this is **not** a Ghanaian accent — what
changes is what they say.

Regenerate with `python tools/build_space_audio.py && python tools/build_space.py` in
the repo.
