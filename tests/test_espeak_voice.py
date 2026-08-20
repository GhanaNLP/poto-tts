"""The espeak voice file may not overrule the lexicon.

`replace` rules in a voice file are applied to every word, after the dictionary
has been consulted. So a rule whose source phoneme our own entries can emit does
not fill a gap in the lexicon -- it overwrites what the lexicon said, on every
word the lexicon knows, invisibly.

That is how `the` came out as /da/ while the lexicon plainly recorded [ð, ə], and
how `Okuapɛnhɛnɛ` was flattened to /okwapenhene/ by a rule collapsing E to e. The
pronunciation of a word is the lexicon's decision; a wrong pronunciation is a
lexicon fix, not a rule.

Three exemptions are allowed, and each is exempt for a stated reason rather than
because it is convenient -- see the comments in `espeak/en-gh`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from poto_tts import mnemonics

VOICE = Path(__file__).resolve().parent.parent / "espeak" / "en-gh"

# `t#` is not in this set, so the flap rule needs no exemption: espeak creates the
# flap itself, after lookup, and rewriting it restores what the entry asked for.
EXEMPT = {
    "r": "the lexicon writes /r/; espeak's `r` is the approximant ɹ and `*` the "
         "tap, which is the nearer reading of the same phoneme",
    "A:": "a table translation, not an accent rule: mnemonics.py writes Ghanaian "
          "/a/ as `A:` for the American inventory, and this voice reads those "
          "entries with the British one where `a` is /a/",
}


def emitted_mnemonics() -> set:
    """Every mnemonic a dictionary entry can contain."""
    out = set()
    for table in ("VOWELS", "LONG_VOWELS", "CONSONANTS", "DIPHTHONGS"):
        out.update(getattr(mnemonics, table, {}).values())
    return out


def replace_rules():
    if not VOICE.is_file():
        pytest.skip(f"no voice file at {VOICE}")
    for line in VOICE.read_text().splitlines():
        match = re.match(r"\s*replace\s+\d+\s+(\S+)\s+(\S+)", line)
        if match:
            yield match.group(1), match.group(2)


def test_no_rule_overrides_a_lexicon_phoneme():
    emitted = emitted_mnemonics()
    offenders = [src for src, _ in replace_rules() if src in emitted and src not in EXEMPT]
    assert not offenders, (
        f"these replace rules rewrite phonemes the lexicon chose: {offenders}. "
        f"A word pronounced wrongly is a lexicon fix. If a rule really is a "
        f"reconciliation rather than an override, add it to EXEMPT with the reason."
    )


def test_the_voice_still_has_rules():
    """Guards the guard: an empty or unparsed voice file must not pass silently."""
    assert len(list(replace_rules())) >= 5


def test_exemptions_are_actually_used():
    """An exemption nobody needs is a licence left lying around."""
    sources = {src for src, _ in replace_rules()}
    unused = set(EXEMPT) - sources
    assert not unused, f"remove these stale exemptions: {sorted(unused)}"
