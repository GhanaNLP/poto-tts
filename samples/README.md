# Samples

What each file says, so you can listen without opening a manifest.

| id | text |
|---|---|
| `names` | Kwabena went to Achimota and met the Okuapenhene at Nyankpani. |
| `news` | The Bank of Ghana raised the policy rate, citing pressure on the cedi. |
| `english` | The convention discussed inflationary policy yesterday afternoon. |
| `mixed` | Yaw Mensah drove from Adenta through Madina to Kotoka. |
| `question` | Have you told Ama that the meeting at Ridge is postponed? |

> **Only `kokoro_engh/` is how poto-tts works.** The other directories are the
> record of getting there -- respelling through lfn, the names-only dictionary, two
> Piper checkpoints. None of them is a supported option any more and the code for
> them has been removed: `respell.py`, `frontend.py` and the `respell=` argument are
> gone, because keeping three front-ends meant three ways for a user to get a worse
> result. The audio stays because the decisions were made by listening, and someone
> should be able to check that.

## Sets

| folder | what it is |
|---|---|
| `kokoro/` | Kokoro + the Ghanaian dictionary, espeak `en-us`. The shipped route. |
| `kokoro_lfn_plain/` | Kokoro with espeak `lfn` and plain English text |
| `kokoro_lfn_respelled/` | Kokoro with espeak `lfn`, every word respelled from the Ghana lexicon |
| `kokoro_lfn_english/` | Kokoro with espeak `lfn`, every word respelled from **espeak-EN** IPA — no lexicon at all |
| `piper_e06/` | the trained Piper voice at epoch 6 (~11k steps, val_mos 2.245) |
| `piper_e21/` | the same at epoch 21 (~38k steps, val_mos 2.306) |

Three voices per Piper set (`gh_00`, `gh_01`, `gh_02` — the largest, most
cohesive speaker clusters), one voice for Kokoro (`bf_alice`).

### kokoro

| file | voice | seconds | rms |
|---|---|---|---|
| `bf_alice_english.wav` | bf_alice |  |  |
| `bf_alice_mixed.wav` | bf_alice |  |  |
| `bf_alice_names.wav` | bf_alice |  |  |
| `bf_alice_news.wav` | bf_alice |  |  |
| `bf_alice_question.wav` | bf_alice |  |  |

### piper, epoch 6

| file | voice | seconds | rms |
|---|---|---|---|
| `gh_00_names.wav` | gh_00 | 2.93 | 0.0396 |
| `gh_00_news.wav` | gh_00 | 3.74 | 0.0480 |
| `gh_00_english.wav` | gh_00 | 3.02 | 0.0508 |
| `gh_00_mixed.wav` | gh_00 | 2.64 | 0.0504 |
| `gh_00_question.wav` | gh_00 | 2.88 | 0.0514 |
| `gh_01_names.wav` | gh_01 | 3.15 | 0.0525 |
| `gh_01_news.wav` | gh_01 | 3.12 | 0.0629 |
| `gh_01_english.wav` | gh_01 | 3.04 | 0.0649 |
| `gh_01_mixed.wav` | gh_01 | 2.72 | 0.0544 |
| `gh_01_question.wav` | gh_01 | 2.86 | 0.0570 |
| `gh_02_names.wav` | gh_02 | 2.98 | 0.0514 |
| `gh_02_news.wav` | gh_02 | 3.47 | 0.0650 |
| `gh_02_english.wav` | gh_02 | 3.23 | 0.0576 |
| `gh_02_mixed.wav` | gh_02 | 2.32 | 0.0591 |
| `gh_02_question.wav` | gh_02 | 2.82 | 0.0615 |

### piper, epoch 21

| file | voice | seconds | rms |
|---|---|---|---|
| `gh_00_names.wav` | gh_00 | 3.57 | 0.0372 |
| `gh_00_news.wav` | gh_00 | 3.74 | 0.0481 |
| `gh_00_english.wav` | gh_00 | 2.99 | 0.0501 |
| `gh_00_mixed.wav` | gh_00 | 2.59 | 0.0372 |
| `gh_00_question.wav` | gh_00 | 2.64 | 0.0521 |
| `gh_01_names.wav` | gh_01 | 3.46 | 0.0577 |
| `gh_01_news.wav` | gh_01 | 3.71 | 0.0662 |
| `gh_01_english.wav` | gh_01 | 3.10 | 0.0726 |
| `gh_01_mixed.wav` | gh_01 | 3.25 | 0.0523 |
| `gh_01_question.wav` | gh_01 | 2.91 | 0.0472 |
| `gh_02_names.wav` | gh_02 | 3.25 | 0.0455 |
| `gh_02_news.wav` | gh_02 | 3.74 | 0.0607 |
| `gh_02_english.wav` | gh_02 | 2.82 | 0.0611 |
| `gh_02_mixed.wav` | gh_02 | 2.35 | 0.0544 |
| `gh_02_question.wav` | gh_02 | 2.27 | 0.0392 |

## The three Kokoro routes

Lingua Franca Nova has a five-vowel phonemic orthography that reads Ghanaian
spellings closer than English does, which is why it was the first approach tried.
What each route does to the same sentence:

```
1 en-us + dictionary   kwɑːbˈɪnɑː wɛnt tʊ ˌɑːtʃimˈoɾɑː ... ˌokwɑːpɛnhˈɛnɛ  æt njɑːŋkpˈɑːni
2 lfn + plain text     kwabˈena   wˈent tˈo ˌakhimˈota  ... ˌokuˌapenhˈene ˈat njankpˈani
3 lfn + lexicon        kwabˈina   wˈent tu  ˌatʃimˈota  ... ˌokwapenhˈene  ˈat njaŋɡkpˈani
4 lfn, no lexicon      kweˈibna   wˈent tu  ˌatʃimˈoɾa  ... ˌokiwˌeipanhˈin ˈat nˌaɪaŋɡkpˈani
```

Route 4 is the codec with nothing behind it: every word's IPA from espeak-EN, respelled
into lfn, read back. It shows what the alphabet alone contributes, and what it cannot
do. The English words come out plausibly Ghanaian -- `the` as /dˈa/, `yesterday` as
`iesterdei`, `policy` as `palisi` -- because collapsing to five vowels removes schwa,
length and r-colouring, which is much of what makes a vowel sound American. The names
do not: `Kwabena` becomes /kweˈibna/ and `Okuapenhene` /okiwˌeipanhˈin/, since espeak-EN
was guessing from spelling and the codec faithfully carries the guess.

It also inherits American *allophones*. espeak-EN flaps intervocalic /t/, and lfn has no
flap, so the flap is written `r` and comes back as a real /r/: `citing` becomes `sairing`,
`meeting` `miring`, `Achimota` `atximoura`. So the lexicon is not only protecting the
names -- it is keeping American articulation out of the ordinary words too.

Reading across: lfn gets Ghanaian names close on the first try, and mangles ordinary
English -- `to` becomes /tˈo/, `the` becomes /thˈe/, and every function word takes a
stress it should not have, because lfn has no notion of English weak forms.
Respelling fixes the words (`da` for `the`) at the cost of a five-vowel inventory,
so Akan's ɛ/e and ɔ/o distinctions collapse: `Okuapɛnhɛnɛ` comes back
/okwapenhene/. `Achimota` also shows lfn's `ch` rule reading /kh/ unless respelled.

The dictionary route keeps seven vowels and English weak forms at once, which is why
it is what ships. These samples are here so that judgement can be heard rather than
taken on trust.

Loudness note: the training audio averages rms 0.085 (-21 dBFS). The Piper
outputs sit at 0.04-0.07, i.e. quieter than what they learned from, which is
what an undertrained VITS does. Kokoro sits near 0.22. Judge the voices, not
the volume -- or normalise before comparing.


## kokoro_engh/ -- the lexicon, on any platform, without Python

Compare against `kokoro_lfn_respelled/`. Same texts, same speaker. Plain text,
`lang=en-gh`, a dictionary and a voice file: no respelling, no lfn, no Python.
Built with `poto-tts dict --ghanaian-stress`.

**One lexicon, one rule.** All 104,623 lexicon words are entered and stressed by the
Ghanaian penultimate rule, whether or not English spells them the same way. There is
no membership test in the build and no second class of entry. That replaced a split
which excluded the 35,425 words English also has -- which is why `Yaw` was read as
the nautical term /jˈɔː/ though the lexicon plainly holds [j, a, w].

**The lexicon decides pronunciation; rules may not contradict it.** This is the
constraint that shaped the voice file, and it was learned the hard way. `replace`
rules fire on every word *after* dictionary lookup, so a rule whose source is a
phoneme our entries can emit does not fill a gap in the lexicon -- it overwrites
what the lexicon said, on every word the lexicon knows, invisibly. An earlier
version collapsed E to e, O: to o, I to i and @ to a, and so overrode the lexicon's
ɛ, ɔ, ɪ and ə everywhere: `Okuapɛnhɛnɛ` flattened to /okwapenhene/, and `the` came
out /da/ while the lexicon records [ð, ə]. Those rules are gone, and
`tests/test_espeak_voice.py` fails if one comes back.

So the vowels are now the lexicon's own -- something the Python route cannot manage,
because lfn has five vowels and Akan has seven:

    Okuapenhene   python /ˌokwapenhˈene/   en-gh /ˌokwapɛnhˈɛnɛ/
    Kwabena       python /kwabˈina/        en-gh /kwabˈɪna/
    the           python /dˈa/             en-gh /ðə/
    through       python /tɾˈu/            en-gh /θɾˈuː/

The last two are the lexicon asserting itself against a rule I had invented:
th-stopping is real in Ghanaian English, but the lexicon spells `the` with ð and
`through` with θ, and that call is the lexicon's to make word by word.

Three rules do touch known words, and none changes a phoneme the lexicon chose:
`t# -> t` restores the /t/ in `Achimota` and `meeting` after espeak flaps it
post-lexically; `r -> *` reads the lexicon's /r/ as a tap rather than espeak's
approximant ɹ; `A: -> a` reconciles two espeak phoneme tables, since mnemonics.py
writes Ghanaian /a/ for the American inventory and this voice reads it with the
British one.

Known artefacts, neither of them a lexicon problem:

- **Intrusive r.** British rules insert a linking `r-` after ə or ɑː before a
  vowel-initial word, so `the Okuapenhene` is /ðəɹ okwapɛnhɛnɛ/. `r-` is inserted
  after the replace stage, so no rule can remove it. `A: -> a` removes the ɑː
  case; the ə case remains, and `ghana` really is [ɡ, ɑː, n, ə] in the lexicon.
  The American table has no intrusive r but flaps every intervocalic /t/, also
  post-lexically and also unremovable, which is why the base here is British.
- **The velar nasal.** `replace` has no positional condition, so the rule that
  writes `ng` word-finally and `n` elsewhere has no equivalent. `Nyankpani` keeps
  its ŋ.

Two words are wrong because the lexicon has them wrong, and are left that way
deliberately: `have` is [h, a, v, ɛ] so it reads /hˈavɛ/, and `Kofi` is
[k, k, o, f, i] so it reads /kkˈofi/ in every route, Python included. Both are fixes
for `ghana-english-g2p`.

## vowel_fix/ -- /a/ written as `a` rather than `A:`

Three sentences, three ways each: `N.kokoro.wav` is ordinary Kokoro, `N.before.wav`
is what the Space served before this fix, `N.after.wav` is with it.

The bug was in the encoder, not in the lexicon or the dictionary. `mnemonics.py`
wrote Ghanaian /a/ as the espeak mnemonic `A:`, decided when the target was espeak's
*American* table where plain `a` is /æ/ and `A:` was the only context-stable way to
get a back [a]. The voice later moved to the British table, where `a` **is** [a], and
`A:` became wrong in two ways at once: it says a long back ɑː where the lexicon says
[a], and ɑː before a vowel triggers British linking-r.

    Otanka     kokoro ɑːtˈæŋkə     before otˈɑːŋkɑːɹ      after otˈaŋka
    Okaija     kokoro ɑːkˈeɪdʒə    before ˌokɑːɹˈidʒɑːɹ   after ˌokaˈidʒa
    Akraman    kokoro ˈækɹæmən     before ɑːkɹˈɑːmɑːn     after akɹˈaman

`Okaija` is the clearest: an r inserted *inside* the name and another after it, in a
word that has no r. It was found by listening -- the IPA looked different from stock
Kokoro, so the entries were plainly being used, but the audio was barely
distinguishable, which is what prompted a closer look.

The same change fixed a second bug behind it. `verify()` and the build's readback
check both defaulted to `en-us`, so entries were verified against the American
inventory while production read them as British: `kwæbɪnæ` would have passed as a
correct reading of `kwabIna`. They use `en` now, the voice the entries are actually
read with.

What this does not fix, and cannot: Kokoro's inventory is 113 tokens, and `kp` and
`ɡb` are not among them. The labial-velars in `Nyankpani` and `Gbedemah` have no
token to land on. Every sound available is an English sound, so what a lexicon entry
changes is which English sounds are used and where the stress falls -- `Kwabena` from
/kwˈeɪbnə/ to /kwabˈɪna/, two syllables to three -- not the inventory itself. Going
past that needs a model trained on Ghanaian speech, which is what `tools/` is for.
