# poto-tts

Ghanaian English speech synthesis. Several engines, one pronunciation: Ghanaian
names, places and titles are compiled into an **espeak-ng dictionary**, so they come
out right from plain text — with no lexicon file for the caller to ship, no Python
front-end, and nothing dropped for words the dictionary has never seen.

```
                stock espeak                  poto-tts
Kwabena         kwˈeɪbnə                      kwɑːbˈɪnɑː
Achimota        ɐtʃɪmˈoʊɾə                    ɑːtʃimˈoɾɑː
Okuapenhene     ˈoʊkjuːˌeɪpənhˌiːn            okwɑːpɛnhˈɛnɛ
Nyankpani       nˌaɪɐŋkpˈɑːni                 njɑːŋkpˈɑːni
Gyasi           dʒaɪʲˈɑːsi                    dʒˈɑːsi
Nyame           nˈaɪeɪm                       njˈɑːmi
```

Ordinary English is untouched — `yesterday` keeps its own stress — because the
dictionary corrects words espeak *mis-parses* rather than re-spelling the language.

```bash
pip install poto-tts
poto-tts "Kwabena went to Achimota" -o out.wav
```

```python
from poto_tts import load

tts = load("kokoro")                     # downloads the voice on first use
tts.save("The Okuapenhene met Nana Bawumia at Manhyia", "out.wav")
```

## Backends

| backend | model | commercial use | accent comes from |
|---|---|---|---|
| `kokoro` (default) | [Kokoro v1.0](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0 | **yes** | the dictionary |
| `piper` | trained on Ghanaian broadcast speech | **no** | the weights *and* the dictionary |

Both read the same `espeak-ng-data`, so either pronounces a Ghanaian name the same
way. What differs is the voice around it — and the licence.

`kokoro` is the default deliberately: it is the one anyone can use for anything.
The `piper` voice is trained on Ghanaian broadcast and interview recordings, whose
speakers did not consent to having their voices modelled, so it is for research and
non-commercial use. If you need a commercial Ghanaian voice, use `kokoro`, or train
`piper` on recordings you hold rights to — the pipeline in `tools/` takes any
dataset.

```python
tts = load("piper", voice="gh_00")       # research use
print(tts.licence, tts.commercial_use)
```

## Cross-platform: this is the point

sherpa-onnx runs natively on **Android, iOS, WebAssembly, C++, C, Go, C#, Java,
Kotlin, Rust, Dart and Swift**, on x86, arm64, arm32 and RISC-V. Because
pronunciation lives in a data directory rather than in Python, a voice published
here is a *file set*, and every one of those runtimes loads it directly — offline,
with no server:

| file | what it is |
|---|---|
| `onnx/model.onnx` | the generator |
| `tokens.txt` | phoneme → id |
| `voices.bin` | speaker embeddings (Kokoro only) |
| `espeak-ng-data/` | **the Ghanaian part. Ship this one.** |

> Substituting a stock `espeak-ng-data` leaves a voice that works and mispronounces
> every Ghanaian name, with no error and no warning. That directory *is* the
> deliverable.

Download the files from the model repo (below) and point your platform's
sherpa-onnx TTS API at them — Kotlin `OfflineTts`, Swift `SherpaOnnxOfflineTts`,
the WASM build, or `sherpa-onnx-offline-tts` on the command line. Upstream docs:
<https://k2-fsa.github.io/sherpa/onnx/>.

## Models

Hosted on the Hub, fetched through `huggingface_hub` so downloads are counted
against the voice that earned them:

| backend | repo |
|---|---|
| `kokoro` | [ghananlpcommunity/poto-tts-kokoro-gh](https://huggingface.co/ghananlpcommunity/poto-tts-kokoro-gh) |
| `piper` | `ghananlpcommunity/poto-tts-piper-gh-16k` *(training)* |

Each repo carries a `config.json` that names **its own files**, sample rate,
speakers, dictionary provenance and licence. The library downloads what that file
lists, so a new voice — another speaker set, a Twi model, a 22 kHz rebuild — needs
no release of this library:

```python
tts = load("kokoro", repo_id="your-org/your-voice")
```

Set `POTO_TTS_CACHE` to keep model files on a mounted volume rather than in
`~/.cache`.

## REST API

For callers that are not Python — a PHP site, a CMS, a Twilio webhook — and the one
shape where the model loads once and each request is cheap:

```bash
pip install 'poto-tts[api]'
poto-tts serve --host 0.0.0.0 --port 8080
curl "localhost:8080/speak?text=Kwabena+went+to+Achimota" -o out.wav
```

| endpoint | |
|---|---|
| `GET /health` | liveness, and which voices are loaded |
| `GET /backends` | engines, licences, commercial-use flags |
| `GET /voices?backend=kokoro` | speaker names, recommended first |
| `GET /platforms` | how to run the same voice off-server |
| `POST /speak` | `{"text": …, "backend": …, "voice": …, "speed": …}` → WAV |
| `GET /speak?text=…` | the same, for a browser or curl |

Requests over 2,000 characters are refused rather than truncated (`POTO_TTS_MAX_CHARS`);
the response says to split into sentences or use the library directly for batch work.
For a phone app, prefer shipping the files: no network, no latency, no server bill.

## Changing how a word is said

The dictionary holds 69,198 entries from a 104,623-word lexicon, and it will still
miss your grandmother's name. Two ways to fix that, and the first works on every
platform because it changes the data rather than the code.

**Rebuild the dictionary with your own entries.** Needs the `espeak-ng` binary
(`apt install espeak-ng`):

```tsv
# my_words.tsv — word<TAB>pronunciation
Owusu	o w u s u
Tetteh	t ɛ t ɛ
Kufuor	=kuf.'uor
```

```bash
poto-tts dict --out my-espeak-data --extra my_words.tsv
POTO_TTS_ESPEAK_DATA=./my-espeak-data poto-tts "Owusu met Tetteh" -o out.wav
```

Values are Ghanaian IPA, mapped by the same tables as the packaged lexicon. A
leading `=` means the value is already in espeak's mnemonics and is passed through
untouched — the escape hatch for a pronunciation this project's mapping cannot
express. **Your entries override the packaged lexicon**, so the file corrects
entries as well as adding them. Ship `my-espeak-data/` in place of the voice's
`espeak-ng-data/` and your Android and iOS builds get the same corrections.

**Or override in Python, without rebuilding**, for a quick check:

```python
from poto_tts import GhanaInjector
injector = GhanaInjector(lexicon={"Owusu": "o w u s u"})
print(injector("Owusu came home"))     # [[ow'usu]] came home
```

This wraps the word in espeak's inline-phoneme syntax, which the `piper` backend
accepts as text. It is a debugging tool rather than a deployment route: it is
Python-only, and the Kokoro frontend rewrites `:` in its input, which corrupts
length marks. For anything you ship, rebuild the dictionary.

Inspect what the front-end does to any text, without loading a model:

```bash
poto-tts --phonemes "Nana Addo met Kwame Nkrumah"
```

## How the dictionary is built

`ghana-english-g2p`'s Ghanaian IPA is mapped to espeak-ng mnemonics
(`poto_tts/mnemonics.py`) and compiled into espeak's `en_extra`. Entries are written
**only for words that are not ordinary English**, checked against a 370k-word list:
espeak already pronounces English correctly, including its stress, and the Ghanaian
realisation of an English word is the acoustic model's job. An earlier version that
entered every word whose pronunciation differed selected 93,526 of them and moved
the stress in *yesterday*, *January* and *Wednesday*.

The build verifies what it produces: no truncated entries — an invalid mnemonic
makes espeak silently discard the rest of an entry — and espeak's own English
unchanged, byte for byte.

The vowel decisions are the substance, and they are in the module's comments: `/a/`
uses `A:` rather than `0` because espeak reads `0` as a different vowel before `/r/`
and it would collide with a contrast Akan needs; `æ` merges into `/a/` because
Ghanaian English merges TRAP; FACE and GOAT map to the monophthongs espeak's
English rules never emit but its tables define.

## Training a voice

`tools/` holds the pipeline used for the `piper` backend, and it takes any dataset:

| stage | |
|---|---|
| `prepare_dataset.py filter` | duration, character set, disfluency, transcript/audio mismatch |
| `prepare_dataset.py embed` | ECAPA speaker embeddings, on GPU |
| `prepare_dataset.py cluster` | pseudo-speakers, with per-speaker and total caps |
| `prepare_dataset.py align` | forced alignment, storing word timings |
| `prepare_dataset.py segment` | cut at real pauses; punctuate from the silences |
| `check_alignable.py` | drop utterances that would crash `monotonic_align` |
| `train.sh` | Piper VITS at the corpus's native sample rate |
| `export_sherpa.py` | package a voice directory for sherpa-onnx |

Thresholds in that pipeline are measured rather than chosen, and the comments record
the distributions they came from — including the ones that were wrong first.

## Credits

- Engine: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (k2-fsa)
- Kokoro: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0
- Trainer: [piper1-gpl](https://github.com/OHF-voice/piper1-gpl) (OHF Voice)
- Lexicon: [ghana-english-g2p](https://github.com/GhanaNLP/ghana-english-g2p) (GhanaNLP)
- Phonemiser: [espeak-ng](https://github.com/espeak-ng/espeak-ng)

MIT, except the bundled `espeak-ng-data` (GPL-3.0) and the `piper` voice
(non-commercial).
