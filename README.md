# poto-tts

**English text-to-speech that pronounces Ghanaian words properly.** Kwabena, Achimota,
the Okuapenhene, Nyankpani, Gyasi — names, places and titles that every general TTS
mangles.

To be clear about what this is not: **the voices do not have a Ghanaian accent.** They
are Kokoro's speakers — British and American — and this library changes what they say,
not who they sound like. A Ghanaian-sounding *voice* needs a model trained on Ghanaian
speech, which is a different problem. What you get here is an English voice that no
longer stumbles over Ghanaian words.

```
                 general TTS                    poto-tts
Kwabena          kwˈeɪbnə                       kwabˈina
Achimota         ɐtʃɪmˈoʊɾə                     atʃimˈota
Okuapenhene      ˈoʊkjuːˌeɪpənhˌiːn             okwapenhˈene
Nyankpani        nˌaɪɐŋkpˈɑːni                  njankpˈani
Gyasi            dʒaɪʲˈɑːsi                     dʒˈasi
```

```bash
pip install poto-tts
poto-tts "Kwabena went to Achimota" -o out.wav
```

```python
from poto_tts import load

tts = load()                               # downloads the voice on first use
tts.save("The Okuapenhene met Nana Bawumia", "out.wav")
```

## How it works

<img src="docs/pipeline.svg" alt="text, then a per-word lookup in the Ghanaian lexicon
with an espeak fallback, then respelling into lfn letters, then espeak reads it back,
then Kokoro speaks it" width="100%">

The middle step is the one that needs explaining. sherpa-onnx hands Kokoro *text* and
runs espeak over it — there is no way to pass phonemes in — so a pronunciation can only
reach the model as spelling espeak will read correctly. Lingua Franca Nova is used as
that notation because its spelling is strictly phonemic, one letter per sound, where
English spelling is not (`through`, `though`, `tough`). So /kwabɪna/ is written
`kwabina`, espeak reads it back as /kwabˈina/, and Kokoro says it. lfn is a codec, not
a language.

Because every word goes through the lexicon, the Ghanaian pronunciation reaches
ordinary English too — `convention` is /kɔnvɛnʃən/ rather than /kənvˈɛnʃən/ — and that
is most of what makes the result sound local. On name-heavy news text the lexicon covers
about nine words in ten; the rest fall back to espeak's English, respelled the same way.

Roughly six times more Ghanaian words come out right than with stock Kokoro. That is
measurable — `tools/measure_coverage.py` — but the honest summary is simpler: the names
work now.

## Voices

Twenty-eight English speakers, named so you can choose one without decoding a prefix:

| | female | male |
|---|---|---|
| **British** ★ | Grace, Comfort, Mercy, Patience | Emmanuel, Isaac, Ebenezer, Bright |
| American | Gifty, Beatrice, Esther, Vida, Felicia, Priscilla, Charity, Regina, Cynthia, Georgina, Adelaide | Samuel, Prince, Godfred, Wisdom, Justice, Solomon, Nathaniel, Cephas, Desmond |

★ British voices sit closest to educated Ghanaian English, so they come first. A
listening judgement, not a measurement.

```python
tts = load(voice="Emmanuel")      # "bm_george" and 26 work too
```

```bash
poto-tts --voices                 # the full list with genders and ids
```

The names are aliases and do not change a voice's timbre. Kokoro's 25 other-language
speakers exist in the model but are not offered here — a Ghanaian English library cannot
vouch for a Japanese speaker.

## Cross-platform

sherpa-onnx runs on **Android, iOS, WebAssembly, C++, C, Go, C#, Java, Kotlin, Rust,
Dart and Swift**. Those runtimes have no Python, so they cannot run the respeller — and
they still get the names right, because the voice ships an `espeak-ng-data` directory
with 69,198 Ghanaian pronunciations compiled into espeak's own English dictionary.

| | pronunciation comes from | reaches |
|---|---|---|
| Python | the respeller (lfn) | every word in the sentence |
| Android, iOS, WASM, C++ | the compiled espeak dictionary | Ghanaian names, places, titles |

Ship these four files and send plain text:

```
onnx/model.onnx      the generator
voices.bin           speaker embeddings
tokens.txt           phoneme → id
espeak-ng-data/      the Ghanaian part — ship this one
```

> Swap in a stock `espeak-ng-data` and you get a working voice that mispronounces every
> Ghanaian name, with no error. That directory is the deliverable.

Want the full respelling on device? It is a lookup and a join, and the voice repo
carries the table: `lfn-lexicon.tsv.gz`, 104,623 words mapped to their lfn spelling.
Look each word up, join with spaces, synthesise with `lang=lfn`. What a port must decide
is what to do with a word the table lacks — under `lang=lfn` an unrespelled English word
is read with Latin letter values, so either fall back to the dictionary route for that
utterance or accept the odd word.

[sherpa-onnx docs](https://k2-fsa.github.io/sherpa/onnx/) ·
[the voice on the Hub](https://huggingface.co/ghananlpcommunity/poto-tts-kokoro-gh)

## Web interface and REST API

```bash
pip install 'poto-tts[api]'
poto-tts serve                    # then open http://localhost:8080
```

Paste text and hear it, or upload a CSV for batch work and get back a ZIP of WAVs with a
manifest. One HTML file, no build step, no CDN — it renders on a laptop or a Pi with no
internet.

```csv
text,voice,filename
"Kwabena went to Achimota",Grace,kwabena
"The Okuapenhene met Nana Bawumia",Emmanuel,durbar
```

`text` is the only required column, and a single-column file with no header works too.
Batches are capped at 500 rows and refused rather than trimmed.

| endpoint | |
|---|---|
| `GET /` | the web interface |
| `GET /speak?text=…` · `POST /speak` | WAV |
| `POST /batch` | CSV → ZIP of WAVs plus a manifest |
| `GET /voices` · `GET /backends` · `GET /health` | |
| `GET /platforms` | how to run the same voice off-server |

## Changing how a word is said

The lexicon will still miss your grandmother's name.

```python
tts = load(lexicon={"Owusu": "o w u s u", "Tetteh": "t ɛ t ɛ"})
```

Values are Ghanaian IPA. That fixes the Python path immediately. To fix it for Android
and iOS as well, rebuild the dictionary they read — needs the `espeak-ng` binary and the
lexicon extra:

```bash
pip install 'poto-tts[lexicon]'
poto-tts dict --out my-espeak-data --extra my_words.tsv
```

```tsv
Owusu	o w u s u
Kufuor	=kufu'or
```

A leading `=` passes raw espeak mnemonics through, and is checked — an invalid mnemonic
makes espeak silently discard the rest of an entry, so the build rejects it rather than
shipping half a word. Your entries override the packaged lexicon, so the file corrects
as well as adds.

To see what will be sent to espeak, without loading a model:

```bash
poto-tts --phonemes "Nana Addo met Kwame Nkrumah"
```

## Models

Fetched from the Hub through `huggingface_hub`, so downloads are counted:
[ghananlpcommunity/poto-tts-kokoro-gh](https://huggingface.co/ghananlpcommunity/poto-tts-kokoro-gh).
Each repo's `config.json` names its own files, speakers and licence, so a new voice needs
no release of this library:

```python
tts = load(repo_id="your-org/your-voice")
```

`POTO_TTS_CACHE` moves model files off `~/.cache`.

## Training a voice

`tools/` holds a full Piper training pipeline — dataset fetch, filtering, speaker
clustering, forced alignment, punctuation from measured pauses, training, export. It was
used to train a Ghanaian voice on 85 hours of broadcast speech, and that voice is **not**
shipped: at 45,000 steps it sounded worse than Kokoro with a good front-end, and its
corpus is broadcast recordings whose speakers never consented to being modelled. The
pipeline is kept because it works on any dataset you hold rights to.

Thresholds in it are measured rather than chosen, and the comments record the
distributions they came from — including the ones that were wrong first.

## Credits

[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (k2-fsa) ·
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (hexgrad, Apache-2.0) ·
[ghana-english-g2p](https://github.com/GhanaNLP/ghana-english-g2p) (GhanaNLP) ·
[espeak-ng](https://github.com/espeak-ng/espeak-ng) ·
[Lingua Franca Nova](https://en.wikipedia.org/wiki/Lingua_Franca_Nova), whose
orthography is doing work its designers did not plan for

MIT, except the bundled `espeak-ng-data`, which is GPL-3.0.
