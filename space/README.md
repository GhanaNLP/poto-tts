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

English text-to-speech that pronounces Ghanaian words properly. Eight sentences,
synthesised three ways — ordinary Kokoro, and poto-tts in each of its two modes.
Same model, same speaker, same text; only the front-end differs.

The voices are Kokoro's British speakers, so this is **not** a Ghanaian accent.
What changes is what they say.

Ordinary English is identical in both modes: the lexicon holds only Ghanaian words,
so English words are left to espeak. The modes differ on the Ghanaian words — `gh`
gives them Ghanaian vowel qualities and a tapped r, `en` gives them English ones.

Regenerate this page with `python tools/build_space.py` in
[GhanaNLP/poto-tts](https://github.com/GhanaNLP/poto-tts).
