# Changing how poto-tts pronounces a word

Every pronunciation decision happens in one of three places. Finding the right one
first saves undoing work later, because a fix in the wrong layer looks correct on
the word you tested and quietly breaks others.

```
  your text
     |
 [1] espeak dictionary  <- built from the Ghana lexicon: 104,623 words.
     |                     If the word is here, this decides its pronunciation.
     |
 [2] espeak's letter-to-sound rules  <- only for words step 1 does not have.
     |
 [3] the voice file, espeak/en-gh  <- accent rules, applied to whatever
     |                                steps 1 and 2 produced.
  phonemes -> Kokoro -> audio
```

The order matters: **the dictionary wins over the rules, and the voice file is
applied on top of both.** So a voice rule affects every word, including ones the
dictionary got right.

## Which layer is your problem in?

Ask what espeak is being sent, before touching anything:

```bash
poto-tts --phonemes "Nana Addo met Kwame Nkrumah"
```

| symptom | layer | what to edit |
|---|---|---|
| one word is wrong, others fine | 1 | the lexicon |
| a word poto-tts has never heard of is wrong | 2 | add it to the lexicon |
| the same sound is wrong in *every* word | 3 | the voice file |
| it is right in Python but wrong on Android | — | rebuild the dictionary |

That third row is the one to be careful about, and the test suite enforces it —
see "The rule that constrains rules" below.

## 1. The lexicon: one word's pronunciation

This is where almost every fix belongs. The lexicon is
[ghana-english-g2p](https://github.com/GhanaNLP/ghana-english-g2p), a separate
package, so that pronunciations have one home rather than one per project.

There is deliberately no Python-only way to do this. A `lexicon=` argument used to
exist and was removed: it changed pronunciation for Python callers and for nobody
else, so the same code gave two different answers depending on the platform. The
pronunciation is data, and changing it means changing the data.

Put your words in a TSV — Ghana IPA, space-separated — and rebuild:

```tsv
Owusu	o w u s u
Kufuor	=kufu'or
```

```bash
pip install 'poto-tts[lexicon]'
poto-tts dict --out my-espeak-data --ghanaian-stress --extra my_words.tsv
```

`--extra` entries override the packaged lexicon, so the file corrects as well as
adds. A leading `=` passes raw espeak mnemonics straight through; those are
validated, because an invalid mnemonic makes espeak silently discard the rest of
the entry and ship half a word.

If the word is one other people will need too, fix it upstream in
`ghana-english-g2p` instead and everything downstream inherits it.

## 2. Words the lexicon does not have

There is nothing to edit here — this layer is espeak's own English letter-to-sound
rules, and it is the fallback. `inflationary` and `drove` are not in the lexicon;
they are pronounced by rule, then given a Ghanaian accent by layer 3.

If a word matters, the fix is to add it to the lexicon, not to lean on the rules.

## 3. The voice file: how a sound is realised everywhere

`espeak/en-gh` is the accent. It is a plain espeak voice file, and its `replace`
lines rewrite phonemes after everything else has run:

```
replace 00 t# t   // undo espeak's intervocalic flap
replace 00 r  *   // /r/ as a tap, not an approximant
replace 00 oU o   // GOAT monophthong, for words the lexicon lacks
```

Edit this only for something true of the accent as a whole — every /oʊ/, not one
word's /oʊ/. After editing, reinstall it:

```bash
cp espeak/en-gh path/to/espeak-ng-data/lang/gmw/en-gh
```

No recompilation: voice files are read as text at load time. Only the dictionary
is compiled.

### The rule that constrains rules

**A `replace` rule may not contradict the lexicon.** Rules run on every word after
the dictionary is consulted, so a rule whose source phoneme our own entries can
emit does not fill a gap — it overwrites what the lexicon said, on every word the
lexicon knows, with nothing in the output to show it happened.

This was learned from a real bug. The voice file used to collapse `E` to `e` and
`@` to `a`, so `Okuapɛnhɛnɛ` was flattened to /okwapenhene/ and `the` came out
/da/, though the lexicon plainly records `[ð, ə]`. Both looked like lexicon errors
and were not.

`tests/test_espeak_voice.py` fails if such a rule appears. Three exemptions exist
and each states its reason in the voice file: `t# -> t` undoes an allophone espeak
adds *after* lookup, `r -> *` picks a realisation of the lexicon's own `/r/`, and
`A: -> a` reconciles two espeak phoneme tables. If you add an exemption, write down
why it is a reconciliation rather than an override.

Two limits of the mechanism, both because espeak substitutes before it applies its
own allophony:

- `replace` has no positional condition, so "ŋ word-finally, n elsewhere" cannot be
  expressed. `Nyankpani` keeps its ŋ.
- espeak silently ignores a rule whose source is `0`, the LOT vowel.

## Rebuilding

```bash
pip install 'poto-tts[lexicon]'                     # needs the espeak-ng binary
poto-tts dict --out build/espeak-ng-data --ghanaian-stress
```

Takes a few minutes: it phonemises the whole lexicon twice, once to write entries
and once to read them back and check none was truncated. It also installs
`espeak/en-gh` into the result, because the dictionary and the voice are one
deliverable — the dictionary alone gives you a working voice that mispronounces
everything slightly.

What `--ghanaian-stress` means: every one of the 104,623 lexicon words is treated as
a Ghanaian word, entered in the dictionary and stressed by the Ghanaian penultimate
rule, whether or not English spells it the same way. Without it, words English also
has are left to espeak — which is why `Yaw` was once read as the nautical term
/jɔː/ although the lexicon holds `[j, a, w]`.

Function words (`the`, `to`, `of`, `and`) are entered but left unstressed, so their
weight still depends on the sentence. Marked, every one takes a beat and a sentence
reads like a list.

Point a run at the result to hear it:

```python
tts = load(espeak_data="build/espeak-ng-data", espeak_voice="en-gh")
```

Then check that ordinary English did not move:

```bash
poto-tts --phonemes "yesterday January Wednesday government"
```

Stress should stay where English puts it inside the word; the vowels should be
Ghanaian. The build prints this comparison itself, under
`checking espeak's own English is unchanged`.

## Using it without Python

The dictionary and voice file are the whole front-end, so any sherpa-onnx runtime —
Android, iOS, WebAssembly, C++, Kotlin, Swift, Rust — gets the same pronunciations
with no Python involved. Ship four things and send plain text with `lang=en-gh`:

```
onnx/model.onnx                  the model
voices.bin                       speaker embeddings
tokens.txt                       phoneme -> id
espeak-ng-data/                  the dictionary and the voice -- the Ghanaian part
```

Swap in a stock `espeak-ng-data` and you still get a working voice that
mispronounces every Ghanaian name, silently. That directory is the deliverable.

## What is not customisable here

- **Kokoro's phoneme inventory.** Pronunciations are approximated into the token set
  the model was trained on, so Akan phones that English lacks land on their nearest
  English neighbour. No amount of lexicon editing changes that.
- **The voices' accent.** These are Kokoro's British and American speakers. poto-tts
  changes what they say, not who they sound like.
