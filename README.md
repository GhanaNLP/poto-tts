# poto-tts

Ghanaian English speech synthesis on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).
A 104,623-word Ghanaian lexicon is **compiled into an espeak-ng dictionary**, so
names, places and titles come out right from plain text.

```
                stock espeak                  poto-tts
Kwabena         kwˈeɪbnə                      kwɑːbˈɪnɑː
Achimota        ɐtʃɪmˈoʊɾə                    ɑːtʃimˈoɾɑː
Okuapenhene     ˈoʊkjuːˌeɪpənhˌiːn            okwɑːpɛnhˈɛnɛ
Nyankpani       nˌaɪɐŋkpˈɑːni                 njɑːŋkpˈɑːni
Gyasi           dʒaɪʲˈɑːsi                    dʒˈɑːsi
```

Ordinary English is untouched: `yesterday` keeps its own stress. Compiling the
lexicon in, rather than looking it up at runtime, means callers ship no lexicon file
and words the lexicon never had are still spoken — espeak's own rules handle them.

```bash
pip install poto-tts
poto-tts "Kwabena went to Achimota" -o out.wav
```

```python
from poto_tts import load

tts = load("kokoro")            # downloads the voice on first use
tts.save("The Okuapenhene met Nana Bawumia", "out.wav")
```

## Backends

| backend | model | commercial use | accent from |
|---|---|---|---|
| `kokoro` (default) | [Kokoro v1.0](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0 | **yes** | the dictionary |
| `piper` | trained on Ghanaian broadcast speech | **no** | the weights and the dictionary |

Both read the same `espeak-ng-data`, so either pronounces a Ghanaian name the same
way; what differs is the voice and the licence. The `piper` corpus is broadcast
recordings whose speakers did not consent to being modelled — research and
non-commercial use only. For a commercial Ghanaian voice use `kokoro`, or train
`piper` on recordings you hold rights to (`tools/` takes any dataset).

```python
tts = load("piper", voice="gh_00")
print(tts.licence, tts.commercial_use)
```

## Models

Voices live on the Hub and are fetched through `huggingface_hub`, so downloads are
counted:

| backend | repo |
|---|---|
| `kokoro` | [ghananlpcommunity/poto-tts-kokoro-gh](https://huggingface.co/ghananlpcommunity/poto-tts-kokoro-gh) |
| `piper` | `ghananlpcommunity/poto-tts-piper-gh-16k` *(training)* |

Each repo's `config.json` names its own files, speakers and licence, so a new voice
needs no release of this library:

```python
tts = load("kokoro", repo_id="your-org/your-voice")
```

`POTO_TTS_CACHE` moves model files off `~/.cache`.

## Cross-platform

sherpa-onnx runs on **Android, iOS, WebAssembly, C++, C, Go, C#, Java, Kotlin, Rust,
Dart and Swift**, across x86, arm64, arm32 and RISC-V. Pronunciation lives in a data
directory, not in Python, so a voice is a file set that any of those runtimes loads
directly — offline, no server:

| file | |
|---|---|
| `onnx/model.onnx` | the generator |
| `tokens.txt` | phoneme → id |
| `voices.bin` | speaker embeddings (Kokoro only) |
| `espeak-ng-data/` | **the Ghanaian part. Ship this one.** |

> A stock `espeak-ng-data` leaves a working voice that mispronounces every Ghanaian
> name, with no error. That directory is the deliverable.

Point your platform's sherpa-onnx TTS API at those files — Kotlin `OfflineTts`,
Swift `SherpaOnnxOfflineTts`, the WASM build, or `sherpa-onnx-offline-tts`.
[Upstream docs](https://k2-fsa.github.io/sherpa/onnx/).

## Web interface and REST API

```bash
pip install 'poto-tts[api]'
poto-tts serve                    # then open http://localhost:8080
```

The page takes pasted text, or a CSV for batch work, and hands back a ZIP of WAVs
with a manifest. One HTML file, no build step and no CDN — it renders on a laptop or
a Pi with no internet.

```csv
text,voice,filename
"Kwabena went to Achimota",bf_alice,kwabena
"The Okuapenhene met Nana Bawumia",bm_george,durbar
```

A `text` column is the documented form; `voice` and `filename` are optional, and a
single-column file with no header works too. Batches are capped at 500 rows
(`POTO_TTS_MAX_BATCH_ROWS`) and refused rather than trimmed — fewer rows back than
you sent, silently, is worse than an error. `--no-ui` leaves a JSON-only API.

Same server, from the command line:

```bash
curl "localhost:8080/speak?text=Kwabena+went+to+Achimota" -o out.wav
curl -X POST localhost:8080/batch -F file=@rows.csv -o batch.zip
```

| endpoint | |
|---|---|
| `GET /health` | liveness, loaded voices |
| `GET /backends` | engines, licences, commercial-use flags |
| `GET /voices?backend=kokoro` | speaker names, recommended first |
| `GET /platforms` | how to run the same voice off-server |
| `GET /` | the web interface |
| `POST /batch` | a CSV of rows → a ZIP of WAVs and a manifest |
| `POST /speak` | `{"text", "backend", "voice", "speed"}` → WAV |
| `GET /speak?text=…` | the same, for a browser or curl |

Requests over 2,000 characters are refused rather than truncated
(`POTO_TTS_MAX_CHARS`). For phone apps, ship the files instead — no network, no
latency, no server.

## Changing how a word is said

The dictionary will still miss your grandmother's name. Rebuilding is the supported
fix: it changes the data every runtime loads, so Android and iOS get it too. Needs
the `espeak-ng` binary and the lexicon extra:

```bash
pip install 'poto-tts[lexicon]'
```

```tsv
# my_words.tsv — word<TAB>pronunciation
Darkoa	d a r k o a
Kufuor	=kufu'or
```

```bash
poto-tts dict --out my-espeak-data --extra my_words.tsv
POTO_TTS_ESPEAK_DATA=./my-espeak-data poto-tts "Darkoa met Kufuor" -o out.wav
```

Values are Ghanaian IPA, mapped by the same tables as the packaged lexicon. A
leading `=` passes raw espeak mnemonics through, and is checked: an invalid mnemonic
makes espeak silently discard the rest of an entry, so the build rejects it rather
than shipping half a word. **Your entries override the packaged lexicon**, so the
file corrects entries as well as adding them.

For a quick check without rebuilding:

```bash
poto-tts --phonemes "Nana Addo met Kwame Nkrumah"
```

```python
from poto_tts import GhanaInjector
GhanaInjector(lexicon={"Darkoa": "d a r k o a"})("Darkoa came home")
# "[[dA:rk'oA:]] came home"
```

That is a debugging tool, not a deployment route: Python-only, and Kokoro's frontend
rewrites `:` in its input, which corrupts length marks.

## How the dictionary is built

`ghana-english-g2p`'s Ghanaian IPA is mapped to espeak mnemonics
(`poto_tts/mnemonics.py`) and compiled into espeak's `en_extra`. Entries are written
**only for words that are not ordinary English**, checked against a 370k-word list:
espeak already handles English, including its stress. An earlier version that
entered every word whose pronunciation differed took 93,526 of them and moved the
stress in *yesterday*, *January* and *Wednesday*.

The build verifies its own output — no truncated entries, and espeak's English
unchanged byte for byte. The vowel decisions are in the module's comments: `/a/` uses
`A:` rather than `0`, which espeak reads as a different vowel before `/r/`; `æ`
merges into `/a/`; FACE and GOAT map to the monophthongs espeak's English rules
never emit but its tables define.

## Training a voice

`tools/` holds the pipeline behind the `piper` backend, and takes any dataset:

| stage | |
|---|---|
| `fetch_shards.py` | download the dataset shard by shard, extract WAVs, drop the parquet |
| `prepare_dataset.py filter` | duration, character set, disfluency, transcript/audio mismatch |
| `prepare_dataset.py embed` | ECAPA speaker embeddings on GPU |
| `prepare_dataset.py cluster` | pseudo-speakers, with per-speaker and total caps |
| `prepare_dataset.py align` | forced alignment, storing word timings |
| `prepare_dataset.py segment` | cut at real pauses, punctuate from the silences |
| `check_alignable.py` | drop utterances that crash `monotonic_align` |
| `train.sh` | Piper VITS at the corpus's native sample rate |
| `export_sherpa.py` | package a voice for sherpa-onnx |

Thresholds there are measured, and the comments record the distributions they came
from — including the ones that were wrong first.

`fetch_shards.py` takes `POTO_TTS_HF_REPO` and `POTO_TTS_DATA`; the rest read paths
from the top of `prepare_dataset.py`.

## Credits

[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (k2-fsa) ·
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (hexgrad, Apache-2.0) ·
[piper1-gpl](https://github.com/OHF-voice/piper1-gpl) (OHF Voice) ·
[ghana-english-g2p](https://github.com/GhanaNLP/ghana-english-g2p) (GhanaNLP) ·
[espeak-ng](https://github.com/espeak-ng/espeak-ng)

MIT, except the bundled `espeak-ng-data` (GPL-3.0) and the `piper` voice
(non-commercial).
