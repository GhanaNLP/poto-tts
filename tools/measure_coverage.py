"""Measure what poto-tts changes about Kokoro's Ghanaian English.

Two questions, on the same sample:

  **Pronunciation.** Take a Ghanaian word the lexicon records. What does stock Kokoro
  say, and what does poto-tts say? Both are compared against the lexicon as ground
  truth, on the bare phone string -- stress and length aside, and with espeak's own
  allophony allowed for, since it assimilates /n/ to ŋ before /k/ whatever we write.

  Note what is *not* folded away. An earlier version collapsed ɛ/e, ɔ/o, ɪ/i and ʊ/u
  before comparing, because the respelling route wrote pronunciations in a five-vowel
  orthography that could not carry those distinctions -- holding them against it would
  have measured the alphabet rather than the pipeline. The entries now carry the
  lexicon's IPA directly, so the distinctions survive and the comparison is stricter.

  **Coverage.** What fraction of the words in real Ghanaian text does the dictionary
  hold? That decides how much of an utterance is pronounced from the lexicon rather
  than by espeak's British English rules.

    python tools/measure_coverage.py --sample 400 --espeak-data build/espeak-ng-data
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poto_tts.dictionary import ghanaian_words  # noqa: E402

# Text a Ghanaian TTS actually gets asked to read: news, names, institutions.
SENTENCES = [
    "Kwabena went to Achimota and met the Okuapenhene at Nyankpani.",
    "The Bank of Ghana raised the policy rate, citing pressure on the cedi.",
    "Nana Addo Dankwa Akufo-Addo commissioned the interchange at Kwame Nkrumah Circle.",
    "Asante Kotoko and Hearts of Oak drew nil-nil at the Baba Yara Stadium.",
    "Auntie Adjoa sells kenkey and tilapia at Makola Market in Accra.",
    "The Okuapenhene poured libation at the durbar while the Asantehene's delegation arrived.",
    "Doctor Yaa Asantewaa Mensah published her findings on cholera in peri-urban Kumasi.",
    "Kwesi drove from Adenta through Madina and Legon to Kotoka International Airport.",
]

FOLD = str.maketrans({"ɛ": "e", "ɪ": "i", "ʊ": "u", "ə": "a", "ɐ": "a", "ʌ": "a",
                      "ɑ": "a", "æ": "a", "ɔ": "o", "ɜ": "e", "ɚ": "a", "ᵻ": "i",
                      "ɹ": "r", "ɾ": "r", "ɡ": "g", "ɒ": "a"})
BARE = str.maketrans("", "", "ˈˌː ʲʰ‍")


def espeak(text: str, voice: str, path=None) -> str:
    cmd = ["espeak-ng", "-q", "--ipa=3", "-v", voice]
    # --path takes the directory *containing* espeak-ng-data, by that exact name.
    if path:
        cmd.append(f"--path={path}")
    out = subprocess.run(cmd + [text], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def fold(ipa: str) -> str:
    return ipa.translate(BARE).translate(FOLD)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--espeak-data", default=None,
                    help="a built espeak-ng-data; without it only stock Kokoro and "
                         "coverage can be measured")
    args = ap.parse_args()

    if not shutil.which("espeak-ng"):
        print("needs the espeak-ng binary", file=sys.stderr)
        return 1

    from ghana_english_g2p import GhanaEnglishG2P
    from ghana_english_g2p.core import _load_lexicon

    lexicon = _load_lexicon()
    lex = GhanaEnglishG2P(use_espeak=False)

    # -- pronunciation of Ghanaian words -----------------------------------
    random.seed(args.seed)
    # Ghanaian words only. Sampling the whole lexicon measured something else: the
    # other 60,000 entries are ordinary English and deliberately have no dictionary
    # entry, so poto-tts scored 47% on a population where it is *meant* to agree with
    # espeak rather than with the lexicon.
    ghanaian = ghanaian_words()
    words = [w for w in lexicon
             if w.isalpha() and 4 <= len(w) <= 14 and w.lower() in ghanaian]
    sample = random.sample(words, args.sample)

    # espeak assimilates /n/ before /k/ and inserts a glide between vowels; neither is
    # a mispronunciation, so both sides are compared with those normalised.
    def bare(ipa):
        """Strip everything that is notation rather than pronunciation.

        The lexicon and espeak spell several of the same sounds differently, and
        counting those as errors measured the transcription convention rather than
        the pipeline: `makro` scored as wrong because the lexicon writes plain `r`
        where espeak writes the approximant ɹ, and `nyanu` because ɲ is one symbol
        to the lexicon and `nj` to espeak. espeak's own allophony is allowed for
        too -- it assimilates /n/ to ŋ before /k/ whatever the entry says.
        """
        ipa = ipa.translate(BARE).replace("\u200d", "").replace("ʲ", "")
        for a, b in (("ŋk", "nk"), ("ŋɡ", "nɡ"),          # espeak's assimilation
                     ("ɹ", "r"), ("ɾ", "r"),               # one rhotic
                     ("ɑ", "a"),                            # one open vowel
                     ("nj", "ɲ"),                           # one palatal nasal
                     ("tɕ", "tʃ"), ("dʑ", "dʒ"),           # one affricate pair
                     ("c", "k"), ("ɕ", "ʃ"), ("ʑ", "ʒ")):
            ipa = ipa.replace(a, b)
        return ipa

    stock_hits = poto_hits = 0
    root = Path(args.espeak_data).resolve().parent if args.espeak_data else None
    for word in sample:
        target = bare("".join(lexicon[word][0]))
        if bare(espeak(word, "en-us")) == target:
            stock_hits += 1
        if root and bare(espeak(word, "en", path=root)) == target:
            poto_hits += 1

    stock_pron = 100 * stock_hits / len(sample)
    poto_pron = 100 * poto_hits / len(sample)

    # -- coverage of real text ---------------------------------------------
    all_words = [w for s in SENTENCES for w in s.replace(",", " ").replace(".", " ").split()]
    covered = sum(1 for w in all_words if w.strip("'-").lower() in ghanaian)
    coverage = 100 * covered / len(all_words)

    print(f"sample: {len(sample)} lexicon words, seed {args.seed}")
    print(f"  Ghanaian words pronounced correctly")
    print(f"    stock Kokoro (espeak en-us)   {stock_pron:5.1f}%")
    if args.espeak_data:
        print(f"    poto-tts                      {poto_pron:5.1f}%")
    else:
        print(f"    poto-tts                      (pass --espeak-data to measure)")
    print(f"  dictionary coverage of Ghanaian news text {coverage:5.1f}% "
          f"({covered}/{len(all_words)} words)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
