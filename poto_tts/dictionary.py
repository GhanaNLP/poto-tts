"""Compile the Ghana lexicon into an espeak-ng English dictionary.

For deployments that cannot run this library: Android, iOS, WebAssembly, C++. They
load `espeak-ng-data` and send plain text, so a patched dictionary is the only place
Ghanaian pronunciations can live for them. Python callers do not need it -- their path
is the lfn respelling in respell.py, which reaches every word rather than only the
ones with entries.


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
    poto-tts dict --out build/espeak-ng-data
    poto-tts dict --out my-dictionary --extra my_words.tsv

`--extra` takes a caller's own pronunciations, one `word<TAB>IPA` per line, and they
override the packaged lexicon -- so the file corrects entries as well as adding
them. That is the supported way to fix a name this project gets wrong, and it works
on every platform, because it changes the data every sherpa-onnx runtime loads
rather than any code.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from typing import Optional
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .inject import stress_index
from .inject import FUNCTION_WORDS
from .mnemonics import MnemonicError, injection

DICTSOURCE = "https://raw.githubusercontent.com/espeak-ng/espeak-ng/{version}/dictsource/{name}"
# A general English word list, used to decide what *not* to touch. espeak already
# pronounces English correctly, including its stress; the Ghanaian realisation of
# an English word is the acoustic model's job.
VOICE_FILE = "en-gh"

ENGLISH_WORDS = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
# en_extra is ours to write. The others are espeak's own sources and have to be
# present, or the compile produces a dictionary with only our entries in it.
SOURCES = ("en_list", "en_rules", "en_emoji")

# Written into the output directory to mark it as ours. poto_tts.download looks for
# this rather than for en_dict, which every stock espeak-ng-data also has.
MARKER = "poto-tts-dictionary.json"

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


def build_entries(quiet: bool = False, english: Optional[set] = None,
                  uniform_stress: bool = False):
    """Lexicon -> {word: mnemonics} for every word, before selection.

    Two ways to place stress, and they encode different claims about what the
    lexicon is.

    `english` is the English vocabulary, and passing it means: a word that English
    also has keeps English rhythm, because espeak read it from its own dictionary.
    'yesterday' stays YES-ter-day and only its vowels become Ghanaian.

    `uniform_stress` means the opposite, and simpler: every word in the lexicon is
    a Ghanaian word, so every word takes the Ghanaian penultimate rule, whether or
    not English happens to spell it the same way. 'yesterday' becomes
    /jɛstadˈei/. One lexicon, one rule, no membership test anywhere in the build --
    which is the point, since the alternative is two classes of entry that behave
    differently for reasons a reader of the dictionary cannot see.
    """
    from .inject import _require_lexicon

    _require_lexicon()
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
            # Function words carry no stress mark, the same way GhanaInjector
            # leaves them alone. They are grammar rather than vocabulary: their
            # weight depends on the sentence, and a dictionary entry can only
            # record the citation form. Marked, every 'and', 'at' and 'to' in a
            # sentence takes a beat of its own and the line reads like a list.
            if word.lower() in FUNCTION_WORDS:
                at = None
            else:
                # `None` for the English IPA is what forces the penultimate rule:
                # stress_index only borrows espeak's stress when it is given
                # espeak's pronunciation to borrow from.
                at = stress_index(
                    phones,
                    None if uniform_stress else stock.get(word),
                    known_english=(not uniform_stress
                                   and bool(english) and word in english))
            mnemonics = injection(phones, stress_at=at)
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


def read_extra(path: Path) -> dict:
    """A user's own pronunciations: `word<TAB>IPA` or `word<TAB>=mnemonics`.

    Two notations, because the two audiences differ. IPA is what a linguist has to
    hand and what the lexicon itself uses, so `Owusu\to w u s u` is mapped through
    the same tables as everything else. A leading '=' means the value is already in
    espeak's mnemonics and is passed through untouched, which is the escape hatch
    for a pronunciation this project's mapping cannot express.

    Overrides win over the packaged lexicon, so a user can correct an entry as well
    as add one -- that is the point of the file.
    """
    entries = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t") if "\t" in line else line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(
                f"{path}:{number}: expected 'word<TAB>pronunciation', got {line!r}")
        word, value = parts[0].strip().lower(), parts[1].strip()
        if value.startswith("="):
            # Raw mnemonics bypass the mapping, so they also bypass the checks the
            # mapping provides -- and espeak answers an invalid mnemonic by
            # discarding the rest of the entry and returning a fragment, with no
            # error. `=kuf.'uor` becomes /kˈuf/. Checked here, because the
            # verification pass later cannot catch it: it compares the dictionary
            # against the same string, so a broken entry agrees with itself.
            mnemonics = value[1:].strip()
            spoken = _bare(espeak_ipa([f"[[{mnemonics}]]"]))
            # Proportional, not a fixed floor: a truncated entry still returns
            # something ('kuf.\'uor' returns /kˈuf/, three characters), so what
            # gives it away is how much of the string went missing. Counting
            # mnemonic characters overestimates phones -- 'tS' and 'A:' are one
            # phone each -- so a valid entry lands near 0.8-1.0 and a truncated one
            # well below.
            units = len([c for c in mnemonics if c not in "',.:%"])
            if units and len(spoken) / units < 0.7:
                raise ValueError(
                    f"{path}:{number}: espeak read [[{mnemonics}]] as {spoken!r} -- "
                    f"too short, so one of those mnemonics is invalid and espeak "
                    f"discarded the rest. Write the pronunciation as IPA instead, "
                    f"or check it with: espeak-ng -q --ipa=3 \"[[{mnemonics}]]\"")
            entries[word] = mnemonics
            continue
        from ghana_english_g2p import segment

        phones = segment(value)
        if not phones:
            raise ValueError(f"{path}:{number}: no phones in {value!r}")
        entries[word] = injection(phones, stress_at=stress_index(phones))
    return entries


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="poto-tts dict",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="directory to write espeak-ng-data into")
    ap.add_argument("--ghanaian-stress", action="store_true",
                    help="stress every entry by the Ghanaian penultimate rule, "
                         "including words English also has (implies --all-words)")
    ap.add_argument("--all-words", action="store_true",
                    help="write entries for ordinary English words too, giving the "
                         "whole utterance Ghanaian vowels rather than only the "
                         "Ghanaian words. Stress is taken from espeak's own "
                         "pronunciation so English keeps its rhythm.")
    ap.add_argument("--extra", default=None,
                    help="your own pronunciations: word<TAB>IPA per line, or "
                         "word<TAB>=mnemonics to bypass the mapping. These "
                         "override the packaged lexicon.")
    ap.add_argument("--base", default=None,
                    help="an existing espeak-ng-data to patch (default: the system one)")
    ap.add_argument("--dictsource-version", default="1.52.0",
                    help="espeak-ng tag to take en_list/en_rules from")
    ap.add_argument("--sample", type=int, default=2000, help="entries to verify")
    args = ap.parse_args(argv)

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

    # Everything is built in a staging directory that is *named* espeak-ng-data,
    # whatever the caller wants the result called. espeak's --path takes the
    # directory *containing* espeak-ng-data and finds the data by that exact name,
    # so building directly into `--out my-dictionary` compiles into a path espeak
    # cannot see. Staging then moving keeps --out free.
    staging_parent = Path(tempfile.mkdtemp(dir=str(out.parent), prefix=".build-"))
    staging = staging_parent / "espeak-ng-data"
    shutil.copytree(base, staging)
    print(f"copied {base} -> {out}", file=sys.stderr)

    # Loaded before the entries are built, not after: an entry for an English word
    # has to take espeak's stress rather than the Ghanaian penultimate rule, and
    # that decision is made while the mnemonic is written.
    print("loading the English vocabulary", file=sys.stderr)
    english = english_vocabulary(out.parent / "words_alpha.txt")
    print(f"  {len(english)} English words", file=sys.stderr)

    print("building entries from the Ghana lexicon", file=sys.stderr)
    entries, unmappable, stock = build_entries(
        english=english, uniform_stress=args.ghanaian_stress)

    if args.all_words or args.ghanaian_stress:
        # Every lexicon word gets an entry, so ordinary English is spoken with
        # Ghanaian vowels too -- /kɔnvɛnʃən/ rather than /kənvˈɛnʃən/. This is the
        # accent applied to the whole utterance rather than only to Ghanaian words.
        #
        # It is off by default because of what it costs if done carelessly. The
        # first attempt at it moved the stress in 'yesterday', 'January' and
        # 'Wednesday': the lexicon records segments, not stress, so every entry had
        # to guess, and the Ghanaian penultimate rule is wrong for English. Here the
        # stress comes from espeak's own pronunciation of the word, mapped by
        # syllable position, so the vowels change and the rhythm does not.
        differing = dict(entries)
        print(f"  --all-words: entries for all {len(differing)} lexicon words",
              file=sys.stderr)
    else:
        print(f"  {len(english)} English words will be left to espeak", file=sys.stderr)
        differing = {w: m for w, m in entries.items() if w not in english}
        print(f"  {len(differing)} of {len(entries)} lexicon words are not English "
              f"and get an entry", file=sys.stderr)

    # User overrides last, so they win -- including over the English-word rule. A
    # caller who writes an entry for 'record' means it, and second-guessing them
    # would make the file useless for the case it exists for.
    if args.extra:
        extra = read_extra(Path(args.extra))
        overridden = sum(1 for w in extra if w in differing)
        differing.update(extra)
        print(f"  {len(extra)} entries from {args.extra} "
              f"({overridden} overriding the packaged lexicon)", file=sys.stderr)

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
        # espeak reads the dictionary sources from the working directory and
        # writes the compiled dictionary into <path>/espeak-ng-data.
        compiled = subprocess.run(
            ["espeak-ng", "--compile=en", "--path", str(staging_parent)],
            cwd=work, capture_output=True, text=True,
        )
        print(compiled.stdout.strip() or compiled.stderr.strip(), file=sys.stderr)

    # The voice file goes in beside the dictionary, because the two are one
    # deliverable: the entries decide pronunciation and the voice decides how
    # espeak reads them back. Shipping the dictionary without the voice leaves a
    # data directory that works and mispronounces everything slightly, which is
    # the failure mode this project exists to avoid.
    voice_src = Path(__file__).resolve().parent.parent / "espeak" / VOICE_FILE
    if voice_src.is_file():
        voice_dst = staging / "lang" / "gmw" / VOICE_FILE
        voice_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(voice_src, voice_dst)
        print(f"installed voice {VOICE_FILE} -> lang/gmw/{VOICE_FILE}", file=sys.stderr)
    else:
        print(f"WARNING: no voice file at {voice_src}; the dictionary alone will be "
              f"read with espeak's own English vowels", file=sys.stderr)

    print(f"\nentries written: {len(differing)}", file=sys.stderr)
    if unmappable:
        print(f"unmappable words: {len(unmappable)} (e.g. {unmappable[:5]})", file=sys.stderr)

    print("verifying entries read back from the dictionary", file=sys.stderr)
    mismatched, truncated = verify_entries(differing, staging_parent, args.sample)
    print(f"  checked {min(args.sample, len(differing))}: "
          f"{len(truncated)} truncated, {len(mismatched)} differ by allophony", file=sys.stderr)
    for row in truncated[:10]:
        print("  TRUNCATED {} [{}] -> {!r} wanted {!r}".format(*row), file=sys.stderr)

    print("checking espeak's own English is unchanged", file=sys.stderr)
    before = espeak_ipa(REGRESSION, path=base.parent)
    after = espeak_ipa(REGRESSION, path=staging_parent)
    if before == after:
        print("  unchanged", file=sys.stderr)
    else:
        print("  CHANGED:", file=sys.stderr)
        for b, a in zip(before.split(), after.split()):
            if b != a:
                print(f"    {b!r} -> {a!r}", file=sys.stderr)
    # A marker, so a patched directory can be told apart from a stock one. Without
    # it "a directory containing en_dict" matches the espeak data that ships inside
    # every Kokoro and Piper release, and a caller who never built a dictionary
    # would get espeak's American English silently -- the one failure this project
    # exists to prevent.
    (staging / MARKER).write_text(json.dumps({
        "generator": "poto-tts tools/build_espeak_dict.py",
        "entries": len(differing),
        "lexicon_words": len(entries),
        "dictsource_version": args.dictsource_version,
        "espeak_ng": subprocess.run(["espeak-ng", "--version"], capture_output=True,
                                    text=True).stdout.strip()[:80],
    }, indent=1))

    if out.exists():
        shutil.rmtree(out)
    shutil.move(str(staging), str(out))
    shutil.rmtree(staging_parent, ignore_errors=True)

    print(f"\n{out}")
    return 0 if not truncated else 2


if __name__ == "__main__":
    raise SystemExit(main())
