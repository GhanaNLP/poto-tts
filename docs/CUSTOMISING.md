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
  phonemes -> Kokoro -> audio
```

The order matters: **the dictionary wins over the rules.** A word with an entry never
reaches espeak's letter-to-sound rules at all.

## Which layer is your problem in?

Ask what espeak is being sent, before touching anything:

```bash
poto-tts --phonemes "Nana Addo met Kwame Nkrumah"
```

| symptom | layer | what to edit |
|---|---|---|
| one word is wrong, others fine | 1 | the lexicon |
| a word poto-tts has never heard of is wrong | 2 | add it to the lexicon |
| the same sound is wrong in *every* word | — | espeak's own English; not ours to change |
| it is right in Python but wrong on Android | — | rebuild the dictionary |

That third row is not a poto-tts problem: it is how espeak says English, and the
lexicon is not the place to correct an accent one word at a time.

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

## 3. There is no third layer any more

There used to be an accent voice file here, `espeak/en-gh`, whose `replace` rules
rewrote phonemes after everything else had run. It is gone, and the reason is worth
recording: those rules fired on every word, so a rule whose source phoneme our own
entries could emit did not fill a gap in the lexicon -- it overwrote what the lexicon
said, on every word the lexicon knew, invisibly. `the` came out /da/ while the lexicon
plainly recorded [ð, ə].

Once the lexicon held only Ghanaian words, the rules had nothing left to do that was
not already the dictionary's job, and every remaining one was a preference rather than
a fix. Deleting the file deleted the whole question. If you want a Ghanaian accent
applied to English words too, that is an espeak voice file of your own -- espeak ships
`en-029` as a worked example -- but it is not what this library does.

## Rebuilding

```bash
pip install 'poto-tts[lexicon]'                     # needs the espeak-ng binary
poto-tts dict --out build/espeak-ng-data --ghanaian-stress
```

Takes a few minutes: it phonemises the lexicon twice, once to write the entries and
once to read them back and check none was truncated.

Two things it does that are easy to miss:

**It uses only the Ghanaian subset.** `poto_tts/data/ghanaian-words.txt` lists the
44,321 entries a Ghanaian speaker pronounces by local rules, out of the upstream
lexicon's 104,623. The rest -- `bus`, `passed`, `way`, `yesterday` -- record the
Ghanaian *accent* of an English word, which is right for a G2P library and wrong here:
compiling all of them made every word of every sentence Ghanaian. Regenerate that list
with `tools/classify_lexicon.py`, or edit it by hand.

**`--ghanaian-stress` treats every entry as a Ghanaian word**, stressing it by the
penultimate rule rather than borrowing espeak's. That is correct now that only
Ghanaian words are entered; without the flag, words English also spells are handed
back to espeak, which is what once made `Yaw` read as the nautical term /jɔː/.

Function words (`the`, `to`, `of`, `and`) are entered but left unstressed, so their
weight still depends on the sentence. Marked, every one takes a beat and a line reads
like a list.

Point a run at the result to hear it:

```python
tts = load(espeak_data="build/espeak-ng-data")
```

Then check that ordinary English did not move:

```bash
poto-tts --phonemes "yesterday January Wednesday government"
```

Nothing should change: those words have no entries. The build prints its own version
of this check as `checking espeak's own English is unchanged`, and with the Ghanaian
subset it should report `unchanged`. If it lists altered words, the word list has
English in it.

## Using it without Python

The dictionary and voice file are the whole front-end, so any sherpa-onnx runtime —
Android, iOS, WebAssembly, C++, Kotlin, Swift, Rust — gets the same pronunciations
with no Python involved. Ship four things and send plain text with `lang=en`:

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
