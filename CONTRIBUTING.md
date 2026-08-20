# Contributing

The most useful contribution is a pronunciation: a Ghanaian word we do not have, or
one we have wrong. Both are welcome, and both are worth a minute of routing first,
because pronunciations and code live in different repositories.

## Where does your change belong?

| what you found | where it goes |
|---|---|
| a Ghanaian word missing from the lexicon | **[ghana-english-g2p](https://github.com/GhanaNLP/ghana-english-g2p)** — a PR adding it |
| a Ghanaian word pronounced wrongly | **ghana-english-g2p** — an issue, or a PR fixing the entry |
| a Ghanaian word read as if it were English | **this repo** — it is missing from `poto_tts/data/ghanaian-words.txt` |
| an ordinary English word said the Ghanaian way | **this repo** — it should not be in that list |
| a bug in the library, CLI, API or docs | **this repo** |

The lexicon is a separate package on purpose. It is a pronunciation dictionary for
Ghanaian English generally, useful to ASR and to other TTS systems, not only to us --
so a word added there reaches everything, and a word added here reaches only poto-tts.

`poto_tts/data/ghanaian-words.txt` is a different decision: which of the lexicon's
entries a *TTS front-end* should use. The lexicon records the Ghanaian accent of
ordinary English words too (`bus`, `passed`, `way`), and compiling those made every
word of every sentence Ghanaian, which is not what this library is for.

## Please listen to it first

An entry that looks right and sounds wrong is worse than a missing entry, because a
missing word is obvious and a wrong one is confidently wrong. So test before opening
a PR.

Write the word and its pronunciation in a TSV, one per line, phones separated by
spaces:

```tsv
Owusu	o w u s u
Tetteh	t ɛ t ɛ
```

Build a dictionary with it and listen:

```bash
pip install 'poto-tts[lexicon]'                   # needs the espeak-ng binary
poto-tts dict --out build/espeak-ng-data --ghanaian-stress --extra my_words.tsv
poto-tts --phonemes --espeak-data build/espeak-ng-data "Owusu and Tetteh arrived"
poto-tts --espeak-data build/espeak-ng-data "Owusu and Tetteh arrived" -o out.wav
```

`--phonemes` shows what espeak will say without loading a model; the second command
produces audio. Judge by the audio. The phonemes tell you whether the entry was
understood, not whether it sounds like the word.

Two things to check while listening:

- **Stress.** Entries are stressed by the Ghanaian penultimate rule, which is right
  for most Ghanaian words and not for all. If the stress is wrong, say so in the PR --
  the lexicon records segments, not stress, so this cannot be fixed by editing the
  entry alone.
- **Nothing else moved.** The build prints `checking espeak's own English is
  unchanged`. If your entry alters an ordinary English word, that will show up there.

## Writing the pronunciation

Ghanaian IPA, space-separated. Multi-character phones are single tokens: `kp`, `ɡb`,
`tʃ`, `dʒ`, `ɲ`, `ŋm`.

These are the phones that can be carried through to the model:

```
vowels       a e i o u ɛ ɔ ɪ ʊ ʌ ə æ ɑ ɒ ɜ ɐ ɨ y ø œ ɯ ᵻ ɚ
long         aː eː iː oː uː ɑː ɒː ɔː ɛː ɜː əː æː
consonants   b d f g h j k l m n p r s t v w z
             kp ɡb ɲ ŋ ŋm tʃ dʒ ts dz ʃ ʒ θ ð ɾ ɹ ɣ x ç ɬ ʔ
diphthongs   written as two phones: a ɪ / a ʊ / ɔ ɪ / e ɪ / o ʊ
```

A phone outside that set means the entry cannot be compiled and is dropped, silently,
so stay inside it. `e ɪ` and `o ʊ` are deliberately monophthongised on the way to the
model, because Ghanaian English says `face` as [fes] and `goat` as [got].

What the model can produce is narrower than what you can write: pronunciations are
approximated into Kokoro's own token set, so Akan phones English lacks land on their
nearest English neighbour. `kp` and `ɡb` survive; a tone does not. Do not spend
effort on distinctions the audio cannot carry.

## Reporting a wrong pronunciation

An issue is enough, and more useful than you might expect. Include:

- the word, and what it should sound like — a rhyme or a syllable breakdown is fine,
  IPA is not required
- what poto-tts currently says: `poto-tts --phonemes "<the word>"`
- where the word is from, if it is regional: Akan, Ga, Ewe, Dagbani, Hausa, a place,
  a family name

You do not need to know IPA to report that a name is wrong. Knowing that it *is*
wrong is the part we cannot do from here.

## Code

`python -m pytest` — the lexicon tests need `poto-tts[lexicon]` and skip without it.

Comments in this codebase explain *why*, especially where something looks odd: the
zero-width joiner that broke stress on every English word with a diphthong, the
`replace` rules that silently overrode the lexicon, the espeak data directory that
must be named exactly `espeak-ng-data`. If you fix something subtle, leave the reason
behind — several of those bugs were found twice.
