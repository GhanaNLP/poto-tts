"""Ask Gemini which lexicon words are genuinely Ghanaian, and keep only those.

Every word is sent, not only the ones an English word list also has. Restricting it
to those was wrong: the 69,198 words absent from the English list are absent because
many are malformed rather than because they are Ghanaian -- `macfamous`, `preevent`,
`groundstone`, `notfor`, `kollage`, `dkc` -- and a "Ghanaian words only" lexicon that
keeps those has not been cleaned, only shrunk.

This exists because no mechanical signal worked. Subtracting an English word list
deletes `Yaw`, `cedi`, `Ghana`, `Accra`, `Ama` and `Tema`, all of which English also
spells. Comparing pronunciations flags every non-rhotic word -- `backdoor`,
`airspeed`, `awardee` -- because Ghanaian English drops /r/ and espeak's English
does not. Spelling patterns match `twin` and `dwell`. The question is about what a
word *is*, not how it looks, so it needs a judgement rather than a rule.

Two safety nets, because a model saying "not Ghanaian" about a Ghanaian word is a
silent mispronunciation:

  * a word whose IPA contains kp, ɡb or ɲ is kept regardless of the verdict --
    English has no such sounds, so the entry cannot be an English word
  * KNOWN below must survive, and the run fails if any is dropped

Results are cached per batch, so a re-run costs nothing for work already done.

    GEMINI_API_KEY=... python tools/classify_lexicon.py --out build/classified
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "gemini-3.5-flash"   # overridden by --model
URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
       "{model}:generateContent?key={key}")

GH_PHONES = ("kp", "ɡb", "gb", "ɲ")

# Words that must survive. Ghanaian names, days, currency, places, food, slang --
# every one of them also an English word, which is why the list exists.
KNOWN = ["yaw", "cedi", "ghana", "ghanaian", "accra", "ama", "tema", "kumasi",
         "adenta", "madina", "kotoka", "mensah", "adjoa", "kwesi", "kojo"]

PROMPT = """You are building a Ghanaian English pronunciation dictionary. For each
word, decide whether a Ghanaian speaker pronounces it by LOCAL GHANAIAN rules (1) or
by STANDARD ENGLISH rules (0).

Answer 1 when the word is:
  - from a Ghanaian language: Akan/Twi/Fante, Ga, Ewe, Dagbani, Nzema, Hausa
  - a Ghanaian personal name, family name, or day-name
  - a Ghanaian place, river, region, or neighbourhood
  - a Ghanaian chieftaincy or civic title
  - Ghanaian food, money, transport, or everyday coinage

Answer 0 when the word is an ordinary English word, a misspelling, an abbreviation,
two words run together, or anything you do not recognise.

The hard cases are words English also spells. Judge what the word IS to a Ghanaian
speaker, not how it looks. Worked examples:

  yaw 1          a day-name (born Thursday), not the nautical term
  cedi 1         Ghana's currency
  accra 1        the capital
  tema 1         a city
  ama 1          a day-name (born Saturday)
  mensah 1       a family name
  adjoa 1        a day-name
  kwabena 1      a day-name
  banku 1        a food
  kenkey 1       a food
  trotro 1       a minibus
  asantehene 1   a title
  mantse 1       a Ga title
  gari 1         a food
  chale 1        Ghanaian address term
  mole 0         the animal; the national park shares the spelling but the word is English
  chop 0         English, even though "chop bar" is Ghanaian usage
  dash 0         English, even though "dash" means a tip in Ghana
  bus 0          English
  way 0          English
  passed 0       English
  yesterday 0    English
  government 0   English
  through 0      English
  macfamous 0    not a word
  preevent 0     not a word
  notfor 0       two words run together

Reply with one line per word, in this exact form and nothing else:

  word 1
  word 0

Use the same spelling as the input, one line per word, all %d of them.

Words:
%s"""


def ask(words: list, key: str, attempts: int = 4, model: str = MODEL) -> dict:
    """Verdicts for one batch, as one bit per word.

    A bit string is the cheapest reply that still carries a verdict per word. Batch
    size is a throughput trade rather than a cost one: at 250 words a majority of
    replies came back one or two characters short of the count and had to be halved
    and retried, so the extra calls outweighed the larger batch. The cost is that
    it is positional: one missing character silently shifts every verdict after it
    onto the neighbouring word. So the length is checked, and a batch that comes back
    the wrong length is split rather than trusted -- halving until it aligns or
    reaches a single word.
    """
    n = len(words)
    body = json.dumps({
        "contents": [{"parts": [{"text": PROMPT % (n, "\n".join(words))}]}],
        # Each verdict names its own word, so alignment is verified rather than
        # assumed. The earlier format -- one bit per word, positional -- let the
        # model emit a plausible-looking string that had drifted: it misclassified
        # `bus`, `way` and `cedi` even though the prompt names all three with their
        # answers, and 3% of replies came back the wrong length outright. A reply
        # that has to repeat the word cannot drift silently.
        "generationConfig": {"temperature": 0,
                             "thinkingConfig": {"thinkingBudget": 0},
                             "maxOutputTokens": max(4096, n * 12 + 2048)},
    }).encode()
    last = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(
                URL.format(model=model, key=key), data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                payload = json.load(r)
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            got = {}
            for line in text.splitlines():
                parts = line.strip().rsplit(None, 1)
                if len(parts) == 2 and parts[1] in ("0", "1"):
                    got[parts[0].strip().lower()] = parts[1] == "1"
            wanted = {w.lower() for w in words}
            missing = wanted - set(got)
            if not missing:
                return {w: got[w.lower()] for w in words}
            last = f"{len(missing)} of {n} words unanswered (e.g. {sorted(missing)[:3]})"
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(min(60, 2 ** attempt * 3))
                continue
            raise
        except Exception as exc:
            last = exc
            time.sleep(min(30, 2 ** attempt * 2))
    if n == 1:
        print(f"\n  giving up on {words[0]!r} ({last}); treating as English",
              file=sys.stderr)
        return {words[0]: False}
    mid = n // 2
    print(f"\n  {last}; splitting {n} into {mid} + {n - mid}", file=sys.stderr)
    out = ask(words[:mid], key, attempts, model)
    out.update(ask(words[mid:], key, attempts, model))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="only the first N words (pilot)")
    ap.add_argument("--model", default=MODEL,
                    help="a second pass with a different model gives an error "
                         "estimate: where two disagree, one of them is wrong")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("set GEMINI_API_KEY", file=sys.stderr)
        return 2

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache_dir = out / "batches"; cache_dir.mkdir(exist_ok=True)

    from ghana_english_g2p.core import _load_lexicon

    from poto_tts.dictionary import english_vocabulary

    lexicon = _load_lexicon()
    english = english_vocabulary(out / "words_alpha.txt")
    words = sorted(w for w in lexicon if w.isalpha() and len(w) > 1)
    contested = list(words)
    if args.limit:
        contested = contested[:args.limit]
    print(f"{len(words)} lexicon words ({sum(w in english for w in words)} of them "
          f"also English) -> asking {MODEL}", file=sys.stderr)

    batches = [contested[i:i + args.batch] for i in range(0, len(contested), args.batch)]
    done = 0

    def run(i_batch):
        nonlocal done
        i, batch = i_batch
        path = cache_dir / f"{i:05d}.json"
        if path.is_file():
            verdicts = json.loads(path.read_text())
        else:
            verdicts = ask(batch, key, model=args.model)
            path.write_text(json.dumps(verdicts, ensure_ascii=False))
        done += 1
        print(f"\r  {done}/{len(batches)} batches", end="", file=sys.stderr, flush=True)
        return verdicts

    verdicts = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(run, enumerate(batches)):
            verdicts.update(part)
    print(file=sys.stderr)

    missing = [w for w in contested if w not in verdicts]
    if missing:
        print(f"  {len(missing)} words got no verdict, treating as English: "
              f"{missing[:8]}", file=sys.stderr)

    keep_gh, forced = [], []
    for w in contested:
        phones = lexicon[w][0]
        if any(p in GH_PHONES for p in phones):
            keep_gh.append(w)
            if not verdicts.get(w):
                forced.append(w)
        elif verdicts.get(w):
            keep_gh.append(w)

    (out / "ghanaian_contested.txt").write_text("\n".join(keep_gh) + "\n")
    # `set(keep_gh)` once, not once per word: inside the comprehension it rebuilt a
    # 40,000-element set for each of 104,623 words and the write never finished.
    kept_set = set(keep_gh)
    (out / "english_dropped.txt").write_text(
        "\n".join(w for w in contested if w not in kept_set) + "\n")
    print(f"\nof {len(contested)} contested words:", file=sys.stderr)
    print(f"  {len(keep_gh)} judged Ghanaian (kept)", file=sys.stderr)
    print(f"  {len(contested) - len(keep_gh)} judged English (dropped)", file=sys.stderr)
    if forced:
        print(f"  {len(forced)} kept over the verdict for kp/gb/ɲ: {forced[:8]}",
              file=sys.stderr)

    kept = set(keep_gh)
    lost = [w for w in KNOWN if w in contested and w not in kept]
    print("\nknown Ghanaian words that are also English:", file=sys.stderr)
    for w in KNOWN:
        state = ("not contested" if w not in contested
                 else "kept" if w in kept else "DROPPED")
        print(f"  {w:12s} {state}", file=sys.stderr)
    if lost:
        print(f"\nFAILED: {lost} were judged English. Fix the prompt before using "
              f"this.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
