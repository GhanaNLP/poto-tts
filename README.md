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

Pronunciation comes from a Ghanaian lexicon compiled into espeak's own dictionary.
Your text is sent to the model unchanged:

<img src="docs/pipeline.svg" width="100%" alt="your text, unchanged, goes into
espeak-ng: the Ghanaian lexicon supplies the words it knows and espeak's
letter-to-sound rules cover the rest; the phonemes go to Kokoro, which speaks it">

The dictionary holds **44,321 Ghanaian words** -- names, places, titles, Twi and Ga
loans, food, money, everyday coinage. Every other word is left to espeak's ordinary
English. So `Kwabena`, `Achimota` and `Okuapenhene` come from the lexicon, while
`bus`, `passed` and `through` are pronounced as any English TTS would.

That division is deliberate and was arrived at the hard way. The upstream lexicon
covers the whole language -- `bus` and `way` have entries too, recording the Ghanaian
*accent* of an English word rather than a pronunciation that cannot be derived. Using
all of it made every word of every sentence Ghanaian: `from` as /frɔm/, `on` as /an/.
That is a different product from an English voice that says Ghanaian names properly.
`poto_tts/data/ghanaian-words.txt` is the subset used, and
`tools/classify_lexicon.py` regenerates it.

To see what will be sent to espeak, without loading a model:

```bash
poto-tts --phonemes "Nana Addo met Kwame Nkrumah"
```

Changing any of it is [docs/CUSTOMISING.md](docs/CUSTOMISING.md).

## Voices

Eight British speakers:

| female | male |
|---|---|
| Grace, Comfort, Mercy, Patience | Emmanuel, Isaac, Ebenezer, Bright |

```python
tts = load(voice="Emmanuel")
tts.annotate("Yaw went to Kumasi by bus")   # which words the lexicon supplied
```

```bash
poto-tts --voices
```

British, and only British, for two reasons. It is the variety Ghanaian English is
closest to -- non-rhotic, with vowels in roughly the same places. And the
pronunciations shipped here are shaped for it: entries are read with espeak's British
phoneme table, so the phonemes Kokoro receives are non-rhotic and use /a/ where
American English has /æ/. Kokoro's American speakers were trained on American
phonemes, and handing them these is a mismatch users should not have to find by ear.

Kokoro's twenty American speakers and twenty-five other-language speakers are still
in the model and still reachable by their own names -- `load(voice="af_heart")` -- for
anyone who wants to try. They are simply not offered.

## Cross-platform

sherpa-onnx runs on **Android, iOS, WebAssembly, C++, C, Go, C#, Java, Kotlin, Rust,
Dart and Swift**, and they all get the same pronunciations as Python, because the
front-end is data rather than code. There is no Python-only path to fall back from:
the dictionary and the voice file *are* the front-end.

Ship four things and send plain text with `lang=en`:

```
onnx/model.onnx      the model
voices.bin           speaker embeddings
tokens.txt           phoneme -> id
espeak-ng-data/      the dictionary and the voice -- the Ghanaian part
```

> Swap in a stock `espeak-ng-data` and you get a working voice that mispronounces
> every Ghanaian name, with no error. That directory is the deliverable.

Send plain text with `lang=en`. `examples/bare_sherpa_onnx.py` does exactly that with
no `poto_tts` import at all -- it is there to keep this claim testable rather than
merely stated.

What does need Python is *authoring* a pronunciation: `poto-tts dict` compiles the
lexicon into `espeak-ng-data` and wants the `espeak-ng` binary as well. That is a
build step you run once; nothing on the device does it.

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

## Contributing a pronunciation

The lexicon will always be missing somebody's name, and the fastest way to fix that is
for the person who knows the name to say so.

- **a Ghanaian word missing or wrong** -> [ghana-english-g2p](https://github.com/GhanaNLP/ghana-english-g2p),
  where pronunciations live so ASR and TTS share one answer. An issue is enough; you
  do not need IPA to report that a name is wrong.
- **a Ghanaian word read as if it were English**, or an English word said the Ghanaian
  way -> here: it is in or out of `poto_tts/data/ghanaian-words.txt` when it should not be.

Please listen to an entry before opening a PR -- [CONTRIBUTING.md](CONTRIBUTING.md)
has the commands. An entry that reads correctly and sounds wrong is worse than a
missing one, because the missing word is obviously missing.

## Changing how a word is said

The lexicon will still miss your grandmother's name. Put it in a TSV -- Ghana IPA,
space-separated -- and rebuild the dictionary:

```tsv
Owusu	o w u s u
Tetteh	t ɛ t ɛ
```

```bash
pip install 'poto-tts[lexicon]'
poto-tts dict --out build/espeak-ng-data --ghanaian-stress --extra my_words.tsv
```

Where each kind of change belongs -- a single word, a sound across the whole accent,
or the lexicon upstream -- and why a rule must never overrule the lexicon:
[docs/CUSTOMISING.md](docs/CUSTOMISING.md).

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
[espeak-ng](https://github.com/espeak-ng/espeak-ng), whose voice files turn out to
be a perfectly good place to keep an accent

MIT, except the bundled `espeak-ng-data`, which is GPL-3.0.
