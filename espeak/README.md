# `en-gh` -- the Ghanaian accent, as espeak data

The accent is a plain espeak voice file. espeak can rewrite the phonemes its own
letter-to-sound rules produce -- it ships accent variants like `en-029` (Caribbean)
built exactly this way -- so poto-tts needs no front-end code at all, and Android,
iOS, WebAssembly and C++ get the same pronunciations as Python.

`poto-tts dict` installs this file into the data directory it builds, because the
dictionary and the voice are one deliverable: the dictionary alone gives you a
working voice that mispronounces everything slightly. To install it by hand:

    cp espeak/en-gh <espeak-ng-data>/lang/gmw/en-gh    # then synthesise lang=en-gh

Voice files are read as text at load time, so editing one needs no recompilation.
Only the dictionary is compiled.

## The rule that constrains rules

**A `replace` rule may not contradict the lexicon.** Rules run on every word after
the dictionary is consulted, so a rule whose source phoneme our entries can emit
does not fill a gap -- it silently overwrites what the lexicon chose. An earlier
version of this file collapsed `E` to `e` and `@` to `a`, and so flattened
`Okuapɛnhɛnɛ` to /okwapenhene/ and turned `the` into /da/ though the lexicon plainly
records `[ð, ə]`. `tests/test_espeak_voice.py` fails if such a rule reappears.

Three rules touch words the lexicon knows, and each says why it is a reconciliation
rather than an override: `t# -> t` undoes the flap espeak adds *after* lookup,
`r -> *` reads the lexicon's own `/r/` as a tap rather than espeak's approximant,
and `A: -> a` reconciles two espeak phoneme tables.

## What cannot be expressed

Both because espeak substitutes before applying its own allophony:

- **The flap.** Under the American table, `t` between vowels becomes `t#` *after*
  the replace stage, so `meeting` stays /miɾiŋ/ however you rewrite `t#`. Hence
  `phonemes en`, the British table.
- **The intrusive r.** The British table appends `r-` after ə or ɑː before a
  vowel-initial word, so `the Okuapenhene` is /ðəɹ okwapɛnhɛnɛ/. `A: -> a` removes
  the ɑː case; the ə case remains, and the lexicon really does record `ghana` as
  `[ɡ, ɑː, n, ə]`.

And one for a different reason: `replace` has no positional condition, so "ŋ
word-finally, n elsewhere" has no equivalent. `Nyankpani` keeps its ŋ. espeak also
silently ignores a rule whose source is `0`, the LOT vowel.

Full guide: [../docs/CUSTOMISING.md](../docs/CUSTOMISING.md).
