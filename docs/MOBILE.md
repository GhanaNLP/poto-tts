# poto-tts on mobile and other platforms

The speech works anywhere [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) runs —
**Android, iOS, Flutter, Kotlin, Swift, Java, C, C++, C#, Go, Rust, Dart, WebAssembly**
— and gets the same pronunciations as Python. There is no front-end to port: the
Ghanaian pronunciations live in a data file that espeak reads, not in code.

The Python package is not needed on the device. It downloads the voice, maps `Grace` to
a speaker id, and serves a CLI and REST API — all things an app does itself.

## What to ship

Four things, from
[ghananlpcommunity/poto-tts-kokoro-gh](https://huggingface.co/ghananlpcommunity/poto-tts-kokoro-gh):

| file | | size |
|---|---|---|
| `onnx/model.onnx` | the model | 311 MB |
| `voices.bin` | speaker embeddings | 27 MB |
| `tokens.txt` | phoneme → id | 2 KB |
| `espeak-ng-data/` | **the Ghanaian pronunciations** | 28 MB, trimmable to 2.3 MB |

```bash
pip install huggingface_hub
huggingface-cli download ghananlpcommunity/poto-tts-kokoro-gh --local-dir voice
```

Then put `voice/` in your app bundle, or download it on first run.

> Swapping in a stock `espeak-ng-data` leaves a working voice that mispronounces every
> Ghanaian name, with no error and no warning. That directory is the deliverable.

### Trimming espeak-ng-data to 2.3 MB

26 of the 28 MB is other languages' dictionaries — Russian alone is 8.4 MB. Keep only:

```
espeak-ng-data/
  en_dict  phontab  phondata  phonindex  intonations
  voices/  lang/gmw/en
```

The phonemes come out byte-identical. Verified with `examples/bare_sherpa_onnx.py`.

## Calling it

Two settings matter, and both are easy to get wrong silently:

- **`lang` must be `en`** — British English. That is what the entries were built
  against. `en-us` still speaks, and mispronounces: American English maps the vowel in
  `stop` and `cannot` differently, and flaps /t/ between vowels.
- **`data_dir` must point at your `espeak-ng-data`**, not at a system one.

### Kotlin, Android

```kotlin
val config = OfflineTtsConfig(
    model = OfflineTtsModelConfig(
        kokoro = OfflineTtsKokoroModelConfig(
            model = "$dir/onnx/model.onnx",
            voices = "$dir/voices.bin",
            tokens = "$dir/tokens.txt",
            dataDir = "$dir/espeak-ng-data",   // the Ghanaian part
            lang = "en",                        // British; not en-us
        ),
        numThreads = 2,
    ),
)
val tts = OfflineTts(config = config)
val audio = tts.generate(text = "Kwabena went to Achimota", sid = 20, speed = 1.0f)
audio.save("out.wav")
```

`sherpa-onnx` publishes an Android AAR and a Kotlin API; see
[its docs](https://k2-fsa.github.io/sherpa/onnx/android/index.html).

### Swift, iOS

```swift
var kokoro = sherpaOnnxOfflineTtsKokoroModelConfig(
  model: "\(dir)/onnx/model.onnx",
  voices: "\(dir)/voices.bin",
  tokens: "\(dir)/tokens.txt",
  dataDir: "\(dir)/espeak-ng-data",
  lang: "en")
var model = sherpaOnnxOfflineTtsModelConfig(kokoro: kokoro, numThreads: 2)
var config = sherpaOnnxOfflineTtsConfig(model: model)
let tts = SherpaOnnxOfflineTtsWrapper(config: &config)
let audio = tts.generate(text: "Kwabena went to Achimota", sid: 20, speed: 1.0)
```

### C++

```cpp
SherpaOnnxOfflineTtsConfig config{};
config.model.kokoro.model     = "voice/onnx/model.onnx";
config.model.kokoro.voices    = "voice/voices.bin";
config.model.kokoro.tokens    = "voice/tokens.txt";
config.model.kokoro.data_dir  = "voice/espeak-ng-data";
config.model.kokoro.lang      = "en";
config.model.num_threads      = 2;

const SherpaOnnxOfflineTts *tts = SherpaOnnxCreateOfflineTts(&config);
const SherpaOnnxGeneratedAudio *audio =
    SherpaOnnxOfflineTtsGenerate(tts, "Kwabena went to Achimota", /*sid=*/20, 1.0f);
```

### Python, for reference

`examples/bare_sherpa_onnx.py` is the same thing with no `poto_tts` import — useful for
checking your data directory before wiring up a device build.

## Voices

Eight British speakers. `sid` is what sherpa-onnx wants:

| name | sid | | name | sid |
|---|---|---|---|---|
| Grace | 20 | | Emmanuel | 26 |
| Comfort | 21 | | Isaac | 27 |
| Mercy | 22 | | Ebenezer | 24 |
| Patience | 23 | | Bright | 25 |

Grace, Comfort, Mercy and Patience are female; Emmanuel, Isaac, Ebenezer and Bright are
male. The model holds 53 speakers and any id in 0–52 will speak, but the others are
American or other-language voices: the entries are read with espeak's British table, so
the phonemes are non-rhotic and use /a/ where American English has /æ/, which those
speakers were not trained on.

Output is **24 kHz mono float**.

## Adding a pronunciation

Not on the device. `espeak-ng-data` is compiled on a workstation and shipped:

```bash
pip install 'poto-tts[lexicon]'          # needs the espeak-ng binary
printf 'Owusu\to w u s u\n' > my_words.tsv
poto-tts dict --out espeak-ng-data --ghanaian-stress --extra my_words.tsv
```

Ship the result. See [CUSTOMISING.md](CUSTOMISING.md).

## Licensing

Read this before an app-store submission rather than during one.

- **Kokoro weights: Apache-2.0.** Commercial use is fine.
- **`espeak-ng-data`: GPL-3.0.** This is the part carrying the Ghanaian pronunciations.

That applies to any espeak-based deployment, including sherpa-onnx's own releases, so
poto-tts does not add an obligation you would otherwise escape. But GPL-3.0 has known
friction with app-store distribution terms, and it is a licence whose reciprocity may
reach further into your app than you want.

If that is a blocker, the pronunciations themselves are **MIT**:
[ghana-english-g2p](https://github.com/GhanaNLP/ghana-english-g2p). What is GPL is
espeak's engine and data files, not the lexicon.

## Checking you got it right

Synthesise this and read the phonemes:

```
Kwabena went to Achimota and met the Okuapenhene at Nyankpani.
```

Correct, with the Ghanaian dictionary and `lang=en`:

```
kwabˈɪna wɛnt tʊ ˌatʃimˈota and mˈɛt ðɪʲ ˌokwapɛnhˈɛnɛ at njaŋkpˈani
```

If you hear `kwˈeɪbnə` for Kwabena, `espeak-ng-data` is not being found — the voice is
working and the dictionary is not, which is the failure mode that looks like success.
