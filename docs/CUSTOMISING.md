# Changing how poto-tts pronounces a word

Every pronunciation decision happens in one of two places.

```
  your text
     |
 [1] espeak dictionary  <- 44,321 Ghanaian words, carrying the lexicon's IPA.
     |                     If the word is here, this decides its pronunciation.
     |
 [2] espeak's British English rules  <- every word step 1 does not have.
     |
  phonemes -> Kokoro -> audio
```

The dictionary wins: a word with an entry never reaches the letter-to-sound rules.

## Which layer is your problem in?

Ask what espeak is being sent, before touching anything:

```bash
poto-tts --phonemes "Nana Addo met Kwame Nkrumah"
```

| symptom | where it belongs |
|---|---|
| one Ghanaian word is wrong | the lexicon |
| a Ghanaian word poto-tts has never heard of | add it to the lexicon |
| a Ghanaian word is read as if it were English | it is missing from `poto_tts/data/ghanaian-words.txt` |
| an English word is said the Ghanaian way | it should not be in that list |
| an English word is wrong | espeak's British English; not ours to correct one word at a time |

## 1. The lexicon: one word's pronunciation

Almost every fix belongs here. The lexicon is
[ghana-english-g2p](https://github.com/GhanaNLP/ghana-english-g2p), a separate package,
so that pronunciations have one home rather than one per project — a word fixed there
reaches ASR and alignment too, not only this.

Put your words in a TSV, Ghana IPA, space-separated:

```tsv
Owusu	o w u s u
Kufuor	k u f u ɔ r
```

```bash
pip install 'poto-tts[lexicon]'                   # needs the espeak-ng binary
poto-tts dict --out build/espeak-ng-data --ghanaian-stress --extra my_words.tsv
poto-tts --espeak-data build/espeak-ng-data "Owusu and Kufuor arrived" -o out.wav
```

`--extra` entries override the packaged lexicon, so the file corrects as well as adds.

A change made this way applies everywhere the voice is used — Python, the REST API,
Android, iOS — because pronunciation is data rather than code.

**Judge it by ear, not by IPA.** An entry that reads correctly and sounds wrong is
worse than a missing one, because a missing word is obviously missing and a wrong one
is confidently wrong.

### Writing the pronunciation

Multi-character phones are single tokens: `kp`, `ɡb`, `tʃ`, `dʒ`, `ɲ`, `ŋm`.

```
vowels       a e i o u ɛ ɔ ɪ ʊ ʌ ə æ ɑ ɜ ɐ ɨ y ø œ ɯ ᵻ
long         aː eː iː oː uː ɑː ɔː ɛː
consonants   b d f g h j k l m n p r s t v w z
             kp ɡb ɲ ŋ ŋm tʃ dʒ ts dz ʃ ʒ θ ð ɾ ɹ ɣ x ç ɬ ʔ
```

A phone outside that set cannot be compiled and the entry is dropped, silently, so stay
inside it.

All seven Akan vowels reach the model: ɛ and ɔ stay distinct from e and o.

What the model cannot do is not worth encoding. Kokoro's inventory is 113 tokens with
no `kp` or `ɡb`, so labial-velars land on their nearest neighbours whatever you write,
and tone is not carried at all.

## 2. Words the lexicon does not have

Nothing to edit: this is espeak's own British English, and it is the fallback. If a
word matters, add it to the lexicon rather than leaning on the rules.

British and not American, deliberately: Ghanaian English is much closer to it. British
also flaps no /t/ between vowels and has no r-coloured vowels, both of which would
otherwise turn up in words the lexicon does not cover.

## Rebuilding

```bash
pip install 'poto-tts[lexicon]'
poto-tts dict --out build/espeak-ng-data --ghanaian-stress
```

A few minutes: it phonemises the lexicon twice, once to write the entries and once to
read them back and check none was truncated.

Two things it does that are easy to miss:

**It uses only the Ghanaian subset.** `poto_tts/data/ghanaian-words.txt` lists the
44,321 entries a Ghanaian speaker pronounces by local rules, out of the upstream
lexicon's 104,623. The rest — `bus`, `passed`, `way`, `yesterday` — record the Ghanaian
*accent* of an English word, which is right for a general G2P library but would make
every word of every sentence Ghanaian here. Regenerate that list with
`tools/classify_lexicon.py`, or edit it by hand; `--whole-lexicon` compiles everything
if you want to hear the difference.

**`--ghanaian-stress` treats every entry as a Ghanaian word**, stressing it by the
penultimate rule rather than borrowing espeak's. Without the flag, words English also
spells are handed back to espeak, so `Yaw` is read as the nautical term /jɔː/.

Function words (`the`, `to`, `of`, `and`) are entered but left unstressed, so their
weight still depends on the sentence. Marked, every one takes a beat and a line reads
like a list.

Then check that ordinary English did not move:

```bash
poto-tts --phonemes "yesterday January Wednesday government"
```

Nothing should change: those words have no entries. The build prints its own version of
this check as `checking espeak's own English is unchanged`, and with the Ghanaian
subset it should report `unchanged`. If it lists altered words, the word list has
English in it.

## Using it without Python

The dictionary is the whole front-end, so any sherpa-onnx runtime — Android, iOS,
WebAssembly, C++, Kotlin, Swift, Rust — gets the same pronunciations. Ship four files
and send plain text with `lang=en`:

```
onnx/model.onnx      voices.bin      tokens.txt      espeak-ng-data/
```

`examples/bare_sherpa_onnx.py` does exactly that with no `poto_tts` import, so the
claim stays testable. [MOBILE.md](MOBILE.md) has the Kotlin, Swift and C++ snippets.

## What is not customisable here

- **Kokoro's phoneme inventory.** 113 tokens. Akan phones English lacks land on their
  nearest neighbour, and no amount of lexicon editing changes that.
- **The voices' accent.** These are Kokoro's British speakers. poto-tts changes what
  they say, not who they sound like.
