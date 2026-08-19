#!/usr/bin/env python3
"""Build the Ghanaian espeak-ng dictionary.

The implementation lives in `poto_tts/dictionary.py` so that it ships in the wheel:
users who `pip install poto-tts` need to be able to add their own pronunciations,
and a tool that only exists in a git checkout cannot help them. Run it either way:

    python tools/build_espeak_dict.py --out build/espeak-ng-data
    poto-tts dict --out build/espeak-ng-data --extra my_words.tsv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poto_tts.dictionary import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
