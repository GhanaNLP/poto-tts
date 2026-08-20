"""Kokoro's speakers, named so a Ghanaian user recognises them.

Kokoro ships 53 speakers called `af_heart`, `bm_george`, `zf_xiaoni` and so on --
a prefix for language and gender, then an arbitrary first name. Asking someone to
choose from that list means asking them to decode a naming scheme first, so the
English speakers are aliased and the metadata is exposed rather than hidden in a
prefix.

The names are English ones as Ghana uses them: Grace, Comfort, Gifty, Emmanuel,
Ebenezer, Bright, Wisdom. Familiar to a Ghanaian reader without being anyone's
day name, and they sit naturally beside the English these voices actually speak.
The Kokoro names still work everywhere a voice is accepted, as do the numeric ids,
since that is what the model card and other sherpa-onnx tools use -- nothing is
renamed, an alias is added.

**Only the 28 English speakers are offered.** Kokoro's Spanish, French, Hindi,
Italian, Japanese, Portuguese and Chinese speakers exist in the model and will
read Ghanaian phonemes in their own timbre, but a Ghanaian English library that
lists them is offering something it cannot vouch for, so they are left out. Pass a
raw Kokoro speaker id if you want to experiment with one.

An alias says nothing about timbre. No Kokoro speaker is Ghanaian, and this
library changes pronunciation, not the voice.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

__all__ = ["VOICES", "KOKORO_SPEAKERS", "Voice", "by_name", "english", "grouped"]


class Voice(NamedTuple):
    name: str            # the Ghanaian alias, or the Kokoro name for non-English
    kokoro: str          # the upstream speaker name
    sid: int             # speaker id in the model
    gender: str          # 'female' | 'male'
    accent: str          # 'British' | 'American' | a language name
    recommended: bool    # worth offering first for Ghanaian English

    @property
    def label(self) -> str:
        return f"{self.name} ({self.accent} {self.gender})"


# Kokoro v1.0's speakers in metadata order; the index is the speaker id.
KOKORO_SPEAKERS = (
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "em_alex", "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    "pf_dora", "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
)

# The aliases. British speakers are listed first and marked recommended: they are
# non-rhotic and their vowels sit closer to educated Ghanaian English than the
# American ones do. That is a listening judgement, not a measurement, and every
# other voice works.
_ALIASES = (
    # alias        kokoro         gender    accent       recommended
    ("Grace",      "bf_alice",    "female", "British",   True),
    ("Comfort",    "bf_emma",     "female", "British",   True),
    ("Mercy",      "bf_isabella", "female", "British",   True),
    ("Patience",   "bf_lily",     "female", "British",   True),
    ("Emmanuel",   "bm_george",   "male",   "British",   True),
    ("Isaac",      "bm_lewis",    "male",   "British",   True),
    ("Ebenezer",   "bm_daniel",   "male",   "British",   True),
    ("Bright",     "bm_fable",    "male",   "British",   True),
    ("Gifty",      "af_heart",    "female", "American",  False),
    ("Beatrice",   "af_bella",    "female", "American",  False),
    ("Esther",     "af_sarah",    "female", "American",  False),
    ("Vida",       "af_nicole",   "female", "American",  False),
    ("Felicia",    "af_sky",      "female", "American",  False),
    ("Priscilla",  "af_nova",     "female", "American",  False),
    ("Charity",    "af_aoede",    "female", "American",  False),
    ("Regina",     "af_kore",     "female", "American",  False),
    ("Cynthia",    "af_alloy",    "female", "American",  False),
    ("Georgina",   "af_jessica",  "female", "American",  False),
    ("Adelaide",   "af_river",    "female", "American",  False),
    ("Samuel",     "am_michael",  "male",   "American",  False),
    ("Prince",     "am_adam",     "male",   "American",  False),
    ("Godfred",    "am_eric",     "male",   "American",  False),
    ("Wisdom",     "am_liam",     "male",   "American",  False),
    ("Justice",    "am_onyx",     "male",   "American",  False),
    ("Solomon",    "am_puck",     "male",   "American",  False),
    ("Nathaniel",  "am_echo",     "male",   "American",  False),
    ("Cephas",     "am_fenrir",   "male",   "American",  False),
    ("Desmond",    "am_santa",    "male",   "American",  False),
)

def _build() -> List[Voice]:
    sid_of = {name: i for i, name in enumerate(KOKORO_SPEAKERS)}
    return [Voice(alias, kokoro, sid_of[kokoro], gender, accent, rec)
            for alias, kokoro, gender, accent, rec in _ALIASES]


VOICES: List[Voice] = _build()

_BY_NAME: Dict[str, Voice] = {}
for _v in VOICES:
    _BY_NAME[_v.name.lower()] = _v
    _BY_NAME[_v.kokoro.lower()] = _v


def by_name(name: str) -> Optional[Voice]:
    """Look up a voice by Ghanaian alias or Kokoro name, case-insensitively."""
    return _BY_NAME.get(name.strip().lower())


def english() -> List[Voice]:
    """Every offered voice. All 28 are English; the rest are not exposed."""
    return list(VOICES)


def grouped() -> Dict[str, List[Voice]]:
    """Voices by accent and gender, for a list a person has to read.

    Ordered so the first thing offered is a British voice: those sit closest to
    educated Ghanaian English.
    """
    out: Dict[str, List[Voice]] = {}
    for voice in VOICES:
        out.setdefault(f"{voice.accent} {voice.gender}", []).append(voice)
    return out


def sids() -> List[int]:
    """Speaker ids of the offered voices, for validating a raw id."""
    return [v.sid for v in VOICES]
