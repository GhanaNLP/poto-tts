"""Measure what poto-tts changes about Kokoro's Ghanaian English, and draw it.

Two questions, answered on the same sample:

  **Pronunciation.** Take a Ghanaian word whose pronunciation the lexicon records.
  What does stock Kokoro say, and what does poto-tts say? Compared against the
  lexicon as ground truth, folded to the five vowels the lfn route can express --
  ɛ/e, ɔ/o, ɪ/i, ʊ/u and ə/a are distinctions the notation cannot carry, so holding
  them against it would measure the alphabet rather than the pipeline.

  **Coverage.** What fraction of the words in real Ghanaian text does the lexicon
  hold? That decides how much of an utterance gets a Ghanaian pronunciation rather
  than espeak's American guess.

Prints the numbers. It does not draw them: the README's diagram is about how data
moves through the pipeline, which is the thing a reader actually needs, and two
percentages belong in a sentence rather than in a chart.

    python tools/measure_coverage.py --sample 400
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poto_tts.respell import respell  # noqa: E402

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


def espeak(text: str, voice: str) -> str:
    out = subprocess.run(["espeak-ng", "-q", "--ipa=3", "-v", voice, text],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def fold(ipa: str) -> str:
    return ipa.translate(BARE).translate(FOLD)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
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
    words = [w for w in lexicon if w.isalpha() and 4 <= len(w) <= 14]
    sample = random.sample(words, args.sample)

    stock_hits = poto_hits = 0
    for word in sample:
        target = fold("".join(lexicon[word][0]))
        if fold(espeak(word, "en-us")) == target:
            stock_hits += 1
        if fold(espeak(respell(lexicon[word][0]), "lfn")) == target:
            poto_hits += 1

    stock_pron = 100 * stock_hits / len(sample)
    poto_pron = 100 * poto_hits / len(sample)

    # -- coverage of real text ---------------------------------------------
    all_words = [w for s in SENTENCES for w in s.replace(",", " ").replace(".", " ").split()]
    covered = sum(1 for w in all_words if w.strip("'-").lower() in lex)
    coverage = 100 * covered / len(all_words)

    print(f"sample: {len(sample)} lexicon words, seed {args.seed}")
    print(f"  Ghanaian words pronounced correctly")
    print(f"    stock Kokoro (espeak en-us)   {stock_pron:5.1f}%")
    print(f"    poto-tts                      {poto_pron:5.1f}%")
    print(f"  lexicon coverage of Ghanaian news text  {coverage:5.1f}% "
          f"({covered}/{len(all_words)} words)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
