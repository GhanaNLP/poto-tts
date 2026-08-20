# Samples

The listenable comparison lives on the Hub:
**https://huggingface.co/spaces/ghananlpcommunity/poto-tts** — thirty-two real
sentences from a Ghanaian English news corpus, each synthesised twice, Kokoro against
poto-tts, across all eight voices, with the lexicon's words marked in the text.

Regenerate it from a checkout:

```bash
poto-tts dict --out build/direct/espeak-ng-data --ghanaian-stress
python tools/build_space_audio.py      # 64 files into space/audio/
python tools/build_space.py            # space/index.html
```

The page is static and self-contained, so opening `space/index.html` locally works
without a server.
