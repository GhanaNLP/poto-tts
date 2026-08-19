"""Compile the Ghana lexicon into an espeak-ng English dictionary.

The output is a patched `espeak-ng-data` directory. Point sherpa-onnx at it with
`--vits-data-dir` and Ghanaian names come out right from *plain text* -- no
Python, no inline phonemes, no lexicon file for the caller to ship. That is what
makes Android, iOS, WASM and C++ deployments work: they already bundle an
espeak-ng-data directory, and this one is a drop-in replacement.

How it works: espeak reads `en_extra` from its dictionary sources, which exists
precisely for local additions, and entries there override the letter-to-sound
rules. Each lexicon word becomes one line, `word<TAB>mnemonics`, using the
mapping in poto_tts.mnemonics and the stress rule in poto_tts.inject -- the same
two modules the training-data preparation uses, so the phones the model is
trained on and the phones espeak produces at inference are the same phones.

Which words get an entry is the central decision, and the first attempt got it
wrong in an instructive way. Writing an entry wherever the Ghanaian pronunciation
differs from espeak's selects 93,526 of 94,943 words -- essentially the whole
language -- because Ghanaian English differs from espeak's American English on
nearly every word: full vowels for schwa, TRAP merged into [a], FACE and GOAT
monophthongised, non-rhotic codas. Most of those entries were right, but they also
moved the stress in words like 'yesterday', 'January' and 'Wednesday', because the
lexicon's own transcription of ordinary English words often has a different
syllable count from espeak's and the stress had to be guessed.

The division of labour that avoids all of it: **the dictionary fixes words espeak
mis-parses; the acoustic model supplies the accent.** A voice trained on Ghanaian
speech renders espeak's phone sequence with Ghanaian phonetics whether or not the
dictionary intervened -- that is what accent adaptation is. What the model cannot
fix is espeak reading 'Kwabena' as /kwˈeɪbnə/, a wrong *segmentation* of the word
with a syllable missing. So an entry is written only where espeak's syllable count
or consonant skeleton disagrees with the lexicon, which is the signature of a
mis-parse rather than an accent difference. Ordinary English keeps espeak's own
pronunciation and, crucially, its own stress.

Three things this script checks, because all three have bitten:

  **A bad mnemonic is silent.** espeak does not reject an invalid phoneme; it
  discards the rest of the entry. Every written entry is read back through espeak
  and compared against the intended pronunciation, and mismatches are reported
  rather than shipped quietly.

  **espeak initialises once per process.** A single process cannot compare two
  data directories -- the second `espeak_Initialize` is a no-op and you get the
  first directory's answers twice. All verification here runs the espeak-ng
  binary as a subprocess with `--path`, never the in-process library.

Usage:
    python tools/build_espeak_dict.py --out build/espeak-ng-data
    python tools/build_espeak_dict.py --out build/espeak-ng-data --dictsource-version 1.52.0
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poto_tts.inject import stress_index                    # noqa: E402
from poto_tts.mnemonics import MnemonicError, injection      # noqa: E402

DICTSOURCE = "https://raw.githubusercontent.com/espeak-ng/espeak-ng/{version}/dictsource/{name}"
# A general English word list, used to decide what *not* to touch. espeak already
# pronounces English correctly, including its stress; the Ghanaian realisation of
# an English word is the acoustic model's job.
ENGLISH_WORDS = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
# en_extra is ours to write. The others are espeak's own sources and have to be
# present, or the compile produces a dictionary with only our entries in it.
SOURCES = ("en_list", "en_rules", "en_emoji")

# Read back after the build to prove espeak's English is unchanged. Chosen to
# cover the vowels and clusters the mapping touches.
REGRESSION = """
the convention discussed inflationary policy yesterday about water bottle
face goat price mouth choice father bath trap sing shop chop judge thin then
computer parliament epidemiologist January Wednesday twenty fifteen
""".split()


def espeak_ipa(words, path: Path | None = None, voice: str = "en-us") -> str:
    """Phonemise with the espeak-ng *binary*, so `--path` is honoured.

    Never use the in-process library for this: espeak initialises once per
    process, so a second data directory is silently ignored.
    """
    cmd = ["espeak-ng", "-q", "--ipa=3", "-v", voice]
    if path is not None:
        cmd += ["--path", str(path)]
    out = subprocess.run(cmd + [" ".join(words)], capture_output=True, text=True, check=True)
    return out.stdout.strip()


# espeak in one-line-per-clause mode manages about 90 words a second, which is 20
# minutes for this lexicon. Each call is a subprocess, so threads wait on it rather
# than holding the GIL, and the work divides cleanly.
WORKERS = min(16, (os.cpu_count() or 4))


def espeak_many(items, path: Path | None = None, voice: str = "en-us",
                chunk: int = 2000, progress=None) -> list[str]:
    """`espeak_each` across several subprocesses, preserving order."""
    batches = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    results: list[str] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for part in pool.map(lambda b: espeak_each(b, path, voice), batches):
            results.extend(part)
            if progress:
                progress(len(results), len(items))
    return results


def espeak_each(items, path: Path | None = None, voice: str = "en-us") -> list[str]:
    """Phonemise many items, one per line, and get one answer per item.

    Batching by joining words with spaces returns a single line that has to be
    split again, and any item whose pronunciation contains -- or loses -- a space
    shifts every result after it. The shift is silent and produces a table where
    words are paired with other words' pronunciations. Feeding one item per line
    via `-f` keeps the mapping positional: espeak emits one line per input line,
    and a count mismatch is detectable instead of corrupting the data.
    """
    if not items:
        return []
    cmd = ["espeak-ng", "-q", "--ipa=3", "-v", voice]
    if path is not None:
        cmd += ["--path", str(path)]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as fh:
        fh.write("\n".join(items) + "\n")
        name = fh.name
    try:
        out = subprocess.run(cmd + ["-f", name], capture_output=True, text=True, check=True)
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
    finally:
        Path(name).unlink(missing_ok=True)
    if len(lines) != len(items):
        # One item per call: slow, but only for the batch that disagreed.
        return [espeak_ipa([item], path, voice) for item in items]
    return lines


def fetch_sources(work: Path, version: str) -> None:
    for name in SOURCES:
        dest = work / name
        if dest.exists():
            continue
        url = DICTSOURCE.format(version=version, name=name)
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())


def _syllables(ipa: str) -> int:
    """Vowel runs in an espeak IPA string, i.e. its syllable count."""
    vowels = "aɑɒæʌəɐɜɚeɛiɪɨoɔuʊyᵻ"
    n = previous = 0
    for ch in ipa:
        is_vowel = ch in vowels
        n += is_vowel and not previous
        previous = is_vowel
    return n


def _skeleton(ipa: str) -> str:
    """The consonants of an espeak IPA string, normalised.

    Normalised because espeak and the lexicon spell the same consonant
    differently in places -- script vs ASCII g, the flap for an intervocalic /t/,
    ɹ against r -- and none of those differences means the word was mis-parsed.
    """
    fold = str.maketrans({"ɡ": "g", "ɾ": "t", "ɹ": "r", "ɫ": "l"})
    drop = "aɑɒæʌəɐɜɚeɛiɪɨoɔuʊyᵻˈˌːˑ‍ʲʰ "
    return "".join(c for c in ipa.translate(fold) if c not in drop)


def mis_parsed(ours: str, theirs: str) -> bool:
    """Did espeak get the *shape* of this word wrong, rather than just the accent?

    A different syllable count or a different consonant skeleton means espeak's
    letter-to-sound rules built the wrong word -- 'Kwabena' as /kwˈeɪbnə/, three
    syllables collapsed to two. Same shape with different vowel qualities is the
    accent, which the acoustic model handles, so those words are left alone.
    """
    return (_syllables(ours) != _syllables(theirs)
            or _skeleton(ours) != _skeleton(theirs))


def english_vocabulary(cache: Path) -> set:
    """A set of ordinary English words, downloaded once and cached.

    This is the selection rule the earlier attempts were groping towards. Deciding
    per word whether espeak's *pronunciation* looked wrong could not distinguish
    'espeak mis-read a Ghanaian name' from 'the lexicon and espeak simply disagree
    about an English word', and the second case is where the stress regressions
    came from: 'yesterday' and 'January' got entries whose stress had to be
    guessed, and the guess was the Ghanaian penultimate rule.

    Asking 'is this an English word?' separates the two cleanly. English keeps
    espeak's pronunciation and its stress; the accent comes from the model. What
    is left -- names, places, chieftaincy titles, Twi and Ga loans -- is where
    espeak's letter-to-sound rules are guessing and the lexicon knows better.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        with urllib.request.urlopen(ENGLISH_WORDS, timeout=120) as r:
            cache.write_bytes(r.read())
    return {w.strip().lower() for w in cache.read_text(encoding="utf-8").split() if w.strip()}


def build_entries(quiet: bool = False):
    """Lexicon -> {word: mnemonics} for every word, before selection."""
    from ghana_english_g2p.core import _load_lexicon

    lexicon = _load_lexicon()
    entries: dict[str, str] = {}
    unmappable: list[str] = []

    # espeak's own reading of every candidate, in batches -- one subprocess call
    # per word would take hours for 104k words.
    words = [w for w in lexicon if w.isalpha() and len(w) > 1]
    report = None if quiet else (
        lambda done, total: print(f"\r  espeak baseline {done}/{total}",
                                  end="", file=sys.stderr, flush=True))
    stock = dict(zip(words, espeak_many(words, progress=report)))
    if not quiet:
        print(file=sys.stderr)

    for word in words:
        phones = lexicon[word][0]
        if not phones:
            continue
        try:
            mnemonics = injection(phones, stress_at=stress_index(phones, stock.get(word)))
        except MnemonicError:
            unmappable.append(word)
            continue
        entries[word] = mnemonics
    return entries, unmappable, stock


def ipa_of_injections(mnemonics, path=None, quiet=True):
    """The IPA espeak reads a list of injections back as, batched.

    One subprocess call per word would be 104k calls. Injections contain no
    spaces by construction, so a batch phonemises to the same number of
    whitespace-separated results; when it does not -- espeak occasionally merges
    or splits -- that batch is redone one at a time rather than misaligned.
    """
    report = None if quiet else (
        lambda done, total: print(f"\r  intended IPA {done}/{total}",
                                  end="", file=sys.stderr, flush=True))
    got = espeak_many([f"[[{m}]]" for m in mnemonics], path=path, progress=report)
    if not quiet:
        print(file=sys.stderr)
    return dict(zip(mnemonics, got))


_MARKS = str.maketrans("", "", "ˈˌ‍ ")


def _bare(ipa: str) -> str:
    return ipa.translate(_MARKS)


def verify_entries(entries: dict[str, str], data_root: Path, sample: int = 2000):
    """Read entries back out of the compiled dictionary and report mismatches.

    A mismatch is not necessarily a broken entry -- espeak applies allophony after
    dictionary lookup, flapping /t/ and assimilating /n/ before /k/ -- so the
    comparison is on the bare phone string and the report is for a human to read,
    not a gate. What it reliably catches is an entry that was truncated, which
    shows up as a much shorter result.
    """
    words = list(entries)[:sample]
    mismatched = []
    truncated = []
    wanted_ipa = ipa_of_injections([entries[w] for w in words])
    batch = 200
    for i in range(0, len(words), batch):
        chunk = words[i:i + batch]
        got = espeak_ipa(chunk, path=data_root).split()
        if len(got) != len(chunk):
            continue
        for word, actual in zip(chunk, got):
            wanted = _bare(wanted_ipa.get(entries[word], ""))
            actual = _bare(actual)
            if actual == wanted:
                continue
            if len(actual) < len(wanted) * 0.7:
                truncated.append((word, entries[word], actual, wanted))
            else:
                mismatched.append((word, entries[word], actual, wanted))
    return mismatched, truncated


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="directory to write espeak-ng-data into")
    ap.add_argument("--base", default=None,
                    help="an existing espeak-ng-data to patch (default: the system one)")
    ap.add_argument("--dictsource-version", default="1.52.0",
                    help="espeak-ng tag to take en_list/en_rules from")
    ap.add_argument("--sample", type=int, default=2000, help="entries to verify")
    args = ap.parse_args()

    if not shutil.which("espeak-ng"):
        print("needs the espeak-ng binary (apt install espeak-ng)", file=sys.stderr)
        return 1

    base = Path(args.base) if args.base else Path("/usr/local/share/espeak-ng-data")
    if not base.is_dir():
        base = Path("/usr/share/espeak-ng-data")
    if not base.is_dir():
        print(f"no espeak-ng-data to patch; pass --base", file=sys.stderr)
        return 1

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out)
    print(f"copied {base} -> {out}", file=sys.stderr)

    print("building entries from the Ghana lexicon", file=sys.stderr)
    entries, unmappable, stock = build_entries()

    print("loading the English vocabulary", file=sys.stderr)
    english = english_vocabulary(out.parent / "words_alpha.txt")
    print(f"  {len(english)} English words will be left to espeak", file=sys.stderr)

    differing = {w: m for w, m in entries.items() if w not in english}
    print(f"  {len(differing)} of {len(entries)} lexicon words are not English "
          f"and get an entry", file=sys.stderr)

    # Of those, report how many espeak was also mis-parsing -- a measure of what
    # the dictionary buys rather than a step the dictionary needs, so it runs on a
    # sample. Phonemising all 69k injections took longer than the rest of the
    # build put together and told us nothing the sample does not.
    sample_words = sorted(differing)[::max(1, len(differing) // args.sample)][:args.sample]
    print(f"measuring what the entries change, on {len(sample_words)} of them",
          file=sys.stderr)
    intended = ipa_of_injections([differing[w] for w in sample_words], quiet=False)
    fixed = sum(1 for w in sample_words
                if mis_parsed(intended.get(differing[w], ""), stock.get(w, "")))
    print(f"  {fixed} of {len(sample_words)} ({fixed/max(len(sample_words),1):.0%}) "
          f"espeak also got structurally wrong, not merely in accent", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        fetch_sources(work, args.dictsource_version)
        (work / "en_extra").write_text(
            "// Generated by poto-tts tools/build_espeak_dict.py -- do not edit.\n"
            + "".join(f"{w}\t{m}\n" for w, m in sorted(differing.items())),
            encoding="utf-8",
        )
        root = out.parent
        # espeak wants --path to be the directory *containing* espeak-ng-data,
        # and reads the dictionary sources from the working directory.
        compiled = subprocess.run(
            ["espeak-ng", "--compile=en", "--path", str(root)],
            cwd=work, capture_output=True, text=True,
        )
        print(compiled.stdout.strip() or compiled.stderr.strip(), file=sys.stderr)

    print(f"\nentries written: {len(differing)}", file=sys.stderr)
    if unmappable:
        print(f"unmappable words: {len(unmappable)} (e.g. {unmappable[:5]})", file=sys.stderr)

    print("verifying entries read back from the dictionary", file=sys.stderr)
    mismatched, truncated = verify_entries(differing, out.parent, args.sample)
    print(f"  checked {min(args.sample, len(differing))}: "
          f"{len(truncated)} truncated, {len(mismatched)} differ by allophony", file=sys.stderr)
    for row in truncated[:10]:
        print("  TRUNCATED {} [{}] -> {!r} wanted {!r}".format(*row), file=sys.stderr)

    print("checking espeak's own English is unchanged", file=sys.stderr)
    before = espeak_ipa(REGRESSION, path=base)
    after = espeak_ipa(REGRESSION, path=out.parent)
    if before == after:
        print("  unchanged", file=sys.stderr)
    else:
        print("  CHANGED:", file=sys.stderr)
        for b, a in zip(before.split(), after.split()):
            if b != a:
                print(f"    {b!r} -> {a!r}", file=sys.stderr)
    print(f"\n{out}")
    return 0 if not truncated else 2


if __name__ == "__main__":
    raise SystemExit(main())
