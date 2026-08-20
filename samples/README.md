# Samples

What each file says, so you can listen without opening a manifest.

| id | text |
|---|---|
| `names` | Kwabena went to Achimota and met the Okuapenhene at Nyankpani. |
| `news` | The Bank of Ghana raised the policy rate, citing pressure on the cedi. |
| `english` | The convention discussed inflationary policy yesterday afternoon. |
| `mixed` | Yaw Mensah drove from Adenta through Madina to Kotoka. |
| `question` | Have you told Ama that the meeting at Ridge is postponed? |

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

