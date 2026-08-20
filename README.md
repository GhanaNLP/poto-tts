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
Your text reaches the model unchanged:

<img src="docs/pipeline.svg" width="100%" alt="your text, unchanged, goes into
espeak-ng: the Ghanaian lexicon supplies the words it knows and espeak's British
English rules cover the rest; the phonemes go to Kokoro, which speaks it">

The dictionary holds **44,321 Ghanaian words** -- names, places, titles, Twi and Ga
loans, food, money, everyday coinage -- each carrying the lexicon's own IPA. Every
other word is left to espeak's British English. So `Kwabena`, `Achimota` and
`Okuapenhene` come from the lexicon, while `bus`, `passed` and `through` are
pronounced as any British English voice would.

All seven Akan vowels reach the model: `Okuapɛnhɛnɛ` is /ˌokwapɛnhˈɛnɛ/, with ɛ
distinct from e. British English for the rest, because Ghanaian English is closer to it
than to American.

On a sample of 400 Ghanaian words, checked against the lexicon: stock Kokoro says 2.8%
of them correctly, poto-tts 98.5% (`tools/measure_coverage.py`). The honest summary is
simpler though -- the names work.

Two things are doing the work, and they compose.

**The phonemiser.** Standard Kokoro converts text to phonemes with
[misaki](https://github.com/hexgrad/misaki), whose English lexicon has no Ghanaian
words in it — ask it for `Kwabena`, `Achimota`, `Okuapenhene` or `Akple` and it returns
a placeholder, then falls back. sherpa-onnx phonemises with espeak-ng instead, which at
least reads Ghanaian spelling as spelling: `Ewe` comes out as a word rather than as the
English "you". That difference is free, before any lexicon is involved.

**The lexicon.** espeak still guesses, and its guesses are English ones. The dictionary
is where a guess gets replaced by the recorded pronunciation, one word at a time —
which is also the part you can extend for a name it has never seen.

**Nothing in this library rewrites your text.** Pronunciation is data, so an Android or
C++ app gets exactly the same result.

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

## Which words came from the lexicon

```python
tts = load()
tts.annotate("Yaw went to Kumasi by bus")
# [('Yaw', True), ('went', False), ('to', False), ('Kumasi', True), ('by', False), ('bus', False)]
tts.coverage("Yaw went to Kumasi by bus")      # 0.33
```

Useful because the audio cannot show it: a name from the lexicon and a name espeak
guessed at sound equally confident, so when one is wrong you cannot tell whether the
entry is missing or the entry is wrong.

Neither method affects synthesis. A port that wants the same information reads
`poto_tts/data/ghanaian-words.txt` and does one set lookup per word.

## Cross-platform

sherpa-onnx runs on **Android, iOS, Flutter, Kotlin, Swift, Java, C, C++, C#, Go, Rust,
Dart and WebAssembly**, and every one of them gets the same pronunciations as Python,
because there is no front-end code to port. Ship four files and send plain text with
`lang=en`:

```
onnx/model.onnx      the model
voices.bin           speaker embeddings
tokens.txt           phoneme -> id
espeak-ng-data/      the dictionary -- the Ghanaian part
```

> Swap in a stock `espeak-ng-data` and you get a working voice that mispronounces every
> Ghanaian name, with no error. That directory is the deliverable.

**[docs/MOBILE.md](docs/MOBILE.md)** has the integration guide: Kotlin, Swift and C++
snippets, the speaker ids, how to trim `espeak-ng-data` from 28 MB to 2.3 MB, the two
settings that fail silently, and the licensing to check before an app-store submission.

`examples/bare_sherpa_onnx.py` is a working example with no `poto_tts` import at all.
This library is not needed on the device.

Adding or changing a pronunciation does need Python and the `espeak-ng` binary --
`poto-tts dict` compiles the lexicon into `espeak-ng-data`. That is a build step, run
once; nothing on the device does it.

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

The lexicon will always be missing somebody's name. Put it in a TSV -- Ghana IPA,
space-separated -- and rebuild the dictionary:

```tsv
Owusu	o w u s u
Tetteh	t ɛ t ɛ
```

```bash
pip install 'poto-tts[lexicon]'
poto-tts dict --out build/espeak-ng-data --ghanaian-stress --extra my_words.tsv
poto-tts --espeak-data build/espeak-ng-data "Owusu and Tetteh arrived" -o out.wav
```

Pronunciation is data, so a change made this way applies everywhere the voice is used
-- Python, the REST API, Android, iOS -- rather than only where this library runs.

More detail: [docs/CUSTOMISING.md](docs/CUSTOMISING.md).

## Models

Fetched from the Hub through `huggingface_hub`, so downloads are counted:
[ghananlpcommunity/poto-tts-kokoro-gh](https://huggingface.co/ghananlpcommunity/poto-tts-kokoro-gh).
Each repo's `config.json` names its own files, speakers and licence, so a new voice needs
no release of this library:

```python
tts = load(repo_id="your-org/your-voice")
```

`POTO_TTS_CACHE` moves model files off `~/.cache`.

## Credits

[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (k2-fsa) ·
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (hexgrad, Apache-2.0) ·
[ghana-english-g2p](https://github.com/GhanaNLP/ghana-english-g2p) (GhanaNLP) ·
[espeak-ng](https://github.com/espeak-ng/espeak-ng), whose voice files turn out to
be a perfectly good place to keep an accent

MIT, except the bundled `espeak-ng-data`, which is GPL-3.0.
