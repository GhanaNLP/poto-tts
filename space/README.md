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

# poto-tts

English text-to-speech that pronounces Ghanaian words properly. Thirty-two real
sentences from a Ghanaian English news corpus, each synthesised twice — Kokoro, and
poto-tts. Same model, same speaker, same text; only the front-end differs.

Tabs switch voice. Each of the eight British speakers reads a different four
sentences, so switching brings new material as well as a new voice.

The voices are Kokoro's British speakers, so this is **not** a Ghanaian accent.
What changes is what they say.

Both modes get the Ghanaian words right. `gh` says them with Ghanaian vowel
qualities and a tapped r; because its rules run on every word, ordinary English picks
up a little of the same accent (`late` as /let/). `en` uses no voice file, so the
names carry English vowels and every other word is exactly what any English TTS
would say.

Regenerate this page with `python tools/build_space.py` in
[GhanaNLP/poto-tts](https://github.com/GhanaNLP/poto-tts).
