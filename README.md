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

Pronunciation comes from a 104,623-word Ghanaian lexicon, compiled into espeak's own
dictionary and paired with an accent voice file. Your text is sent to the model
unchanged:

<img src="docs/pipeline.svg" width="100%" alt="your text, unchanged, goes into
espeak-ng: the Ghana lexicon supplies every word it knows and espeak's
letter-to-sound rules cover only what it does not; accent rules are applied to
both, and may not overrule the lexicon; the phonemes go to Kokoro, which speaks
it">

Three layers, in order. The **dictionary** decides how a word is pronounced, and it
holds the whole lexicon -- names, places, titles, Twi and Ga loans, and ordinary
English, all treated alike. Words it does not have fall to espeak's **letter-to-sound
rules**. Then the **voice file** applies the accent to whatever those produced.

The dictionary wins over the rules, and the voice is applied on top of both -- which
is why a voice rule may not contradict the lexicon. A rule collapsing ɛ to e would
override the lexicon on every word it knows, and `Okuapɛnhɛnɛ` would come back
/okwapenhene/ with nothing in the output to show why. `tests/test_espeak_voice.py`
fails if one appears.

Because the lexicon covers ordinary English too, the Ghanaian pronunciation reaches
whole sentences and not only the names: `convention` is /kɔnvɛnʃən/ rather than
/kənvˈɛnʃən/. Stress stays where English puts it inside a word, while the vowels are
Ghanaian.

Roughly six times more Ghanaian words come out right than with stock Kokoro
(`tools/measure_coverage.py`), but the honest summary is simpler: the names work now.

To see what will be sent to espeak, without loading a model:

```bash
poto-tts --phonemes "Nana Addo met Kwame Nkrumah"
```

Changing any of it -- one word, or the accent as a whole -- is
[docs/CUSTOMISING.md](docs/CUSTOMISING.md).

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
Dart and Swift**, and they all get the same pronunciations as Python, because the
front-end is data rather than code. There is no Python-only path to fall back from:
the dictionary and the voice file *are* the front-end.

Ship four things and send plain text with `lang=en-gh`:

```
onnx/model.onnx      the model
voices.bin           speaker embeddings
tokens.txt           phoneme -> id
espeak-ng-data/      the dictionary and the voice -- the Ghanaian part
```

> Swap in a stock `espeak-ng-data` and you get a working voice that mispronounces
> every Ghanaian name, with no error. That directory is the deliverable.

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
