# poto-tts

Ghanaian English speech synthesis. A Piper (VITS) voice trained on Ghanaian
speech, served by sherpa-onnx, with Ghanaian pronunciation compiled into an
espeak-ng dictionary — so a deployment needs **no Python at inference** and works
on Android, iOS, WebAssembly, C++ and the desktop from the same three files.

```
Kwabena went to Achimota and met the Okuapenhene
  stock espeak    kwˈeɪbnə … ɐtʃɪmˈoʊɾə … ˈoʊkjuːˌeɪpənhˌiːn
  poto-tts        kwɑːbˈɪnɑː … ɑːtʃimˈoɾɑː … okwɑːpɛnhˈɛnɛ
```

## How it works

Two halves, and it matters which half does what.

**The dictionary fixes words espeak mis-parses.** `ghana-english-g2p`'s 104k-word
Ghanaian lexicon is compiled into espeak-ng's `en_extra`, producing a patched
`espeak-ng-data` directory. Entries are written only where espeak's syllable
count or consonant skeleton disagrees with the lexicon — the signature of a
mis-parse, like reading *Kwabena* as /kwˈeɪbnə/ with a syllable missing. Ordinary
English keeps espeak's own pronunciation, and therefore its own stress.

**The acoustic model supplies the accent.** A voice trained on Ghanaian speech
renders espeak's phones with Ghanaian phonetics whether or not the dictionary
intervened; that is what accent adaptation is. So the dictionary does not need to
re-spell the English language, and an early version that tried to — 93,526
entries — moved the stress in *yesterday*, *January* and *Wednesday* while adding
nothing the model was not already doing.

Because pronunciation lives in a data directory rather than in code, any
sherpa-onnx binding gets it for free:

```
voice/
  model.onnx          the trained Piper voice
  tokens.txt
  espeak-ng-data/     ← the Ghanaian part. Ship it, or Kwabena reverts.
```

## Why Piper and not MeloTTS

Three reasons, in order of how much they decided it:

1. **sherpa-onnx's MeloTTS export zeroes BERT** (`bert = torch.zeros(...)` in
   `export-onnx-en.py`). MeloTTS's advantage over a plain VITS is its BERT
   prosody, and that is exactly what the export discards — leaving the complexity
   without the benefit.
2. **MeloTTS English is lexicon-only.** Its frontend (`MeloTtsLexicon`) has no
   espeak fallback: a word missing from the lexicon is dropped silently
   (`"Ignore OOV"`). Piper phonemises everything through espeak, so novel words,
   numbers and dates are never lost.
3. **Sample rate.** The corpus is 16 kHz. MeloTTS is 44.1 kHz only, so it would
   mean upsampling scrape audio and asking the decoder to synthesise an 8–11 kHz
   band that holds no signal.

## Training

Native 16 kHz, which costs nothing here. Piper's maintained trainer only resumes
cleanly from `medium` checkpoints, and those are 22.05 kHz — but its *defaults*
already are the medium architecture (hop 256, filter 1024, mel 80, upsample
(8,8,4), 192 channels), and no tensor in a checkpoint encodes a sample rate. So
`--model.sample_rate 16000` gives a medium-size model at the corpus's own rate
whose weights are shape-identical to the medium checkpoint, warm-started with
`--model.warmstart_ckpt` (which copies what matches and skips the rest — that is
what lets a single-speaker checkpoint seed a multi-speaker model).

Data preparation, in `tools/prepare_dataset.py`:

| stage | what it does |
|---|---|
| `filter` | duration, character set, disfluency ratio, and a text-length-vs-duration check that catches transcript/audio mismatch |
| `embed` | ECAPA speaker embeddings over three windows per clip, on GPU |
| `cluster` | k-means into pseudo-speakers, with per-speaker and total hour caps |
| `csv` | `wav\|speaker\|text` for Piper |

The dataset has no speaker labels, so speakers are found rather than read. Each
clip is embedded at its start, middle and end: clips whose own windows disagree
are conversations rather than utterances and are dropped, and the mean embedding
of the rest is clustered. Both thresholds in that pipeline are measured, not
guessed — see the comments, which record the distributions they came from.

## Known limitations

**No punctuation in the training text.** Measured across the corpus: 9.6% of
transcripts start with a capital, 3.3% end with `.`/`!`/`?`, 0.6% do both. They
are ASR-style, unpunctuated, cut mid-phrase. A terminal full stop is appended
(the clip does end there) but internal commas are not invented, so the voice
learns less about phrasing than one trained on read speech. Forced alignment to
cut the long clips at pauses and punctuate them is the largest quality win still
on the table.

**espeak initialises once per process.** Inside sherpa-onnx, the first
`espeak_Initialize` wins; a second data directory in the same process is silently
ignored. Any A/B between two `espeak-ng-data` directories has to run in separate
processes, or it tests the first one twice — a mistake that produced two
identical-looking results here before it was caught.

**Some Ghanaian phones are approximated.** `kp`, `gb` and `nj` survive intact
because espeak's tables define them, but `ky`/`gy` reach the model as `tʃ`/`dʒ`
(the lexicon already writes them that way), and Ghanaian /a/ shares a token with
ɑː. See the reasoning in `poto_tts/mnemonics.py`.

## The Python side

Not needed for deployment; it is what builds the dictionary and what inspects
pronunciations while working.

```python
from poto_tts import GhanaInjector, injection

injection(["k", "w", "a", "b", "ɪ", "n", "a"], stress_at=4)   # "kwA:b'InA:"
GhanaInjector()("Kwame Nkrumah met the Okuapenhene")
# "[[kw'A:me]] [[Nkr'umA:]] [[mEt]] [[D@]] [[okwA:pEnh'EnE]]"
```

`injection` maps Ghanaian IPA to espeak mnemonics; `GhanaInjector` wraps
lexicon words in `[[…]]` so they can be checked without compiling a dictionary.
`verify` reads an injection back through espeak, which is not decoration: an
invalid mnemonic makes espeak discard the rest of the string and return a
fragment, with no error at all.

## Tools

```bash
python tools/build_espeak_dict.py --out build/espeak-ng-data   # the dictionary
python tools/prepare_dataset.py filter|embed|cluster|csv        # the training set
bash   tools/train.sh                                           # train (H200)
python tools/export_sherpa.py --checkpoint … --config … --out … # package a voice
```

## Credits

- Trainer: [piper1-gpl](https://github.com/OHF-voice/piper1-gpl) (OHF Voice).
- Engine: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (k2-fsa).
- Lexicon: [ghana-english-g2p](https://github.com/GhanaNLP/ghana-english-g2p) (GhanaNLP).
- Phonemiser: [espeak-ng](https://github.com/espeak-ng/espeak-ng).

MIT, except the bundled `espeak-ng-data`, which is GPL-3.0.
