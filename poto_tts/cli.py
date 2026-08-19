"""Command line.

    poto-tts "Kwabena went to Achimota" -o out.wav
    poto-tts --backend piper --voice gh_00 "The Okuapenhene spoke" -o out.wav
    poto-tts --voices
    poto-tts --phonemes "Nana Addo met Kwame Nkrumah"
    poto-tts serve --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import sys


def _entry() -> int:
    return main(sys.argv[1:])


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `serve` is a subcommand rather than a flag, because its options are the
    # server's and not synthesis's; keeping them apart avoids a help screen where
    # half the flags do nothing.
    if argv and argv[0] == "serve":
        from .api import main as serve_main

        return serve_main(argv[1:])
    if argv and argv[0] == "dict":
        from .dictionary import main as dict_main

        return dict_main(argv[1:])

    p = argparse.ArgumentParser(
        prog="poto-tts",
        description="Ghanaian English speech synthesis. Subcommands: serve, dict.",
        epilog="The voices also run on Android, iOS, WebAssembly and C++ through "
               "sherpa-onnx: ship model.onnx, tokens.txt and espeak-ng-data.",
    )
    p.add_argument("text", nargs="*", help="text to speak; '-' reads stdin")
    p.add_argument("-o", "--output", default="out.wav")
    p.add_argument("-b", "--backend", default="kokoro", help="kokoro or piper")
    p.add_argument("-v", "--voice", default=None)
    p.add_argument("-s", "--speed", type=float, default=1.0)
    p.add_argument("--model-dir", default=None, help="a local voice directory")
    p.add_argument("--repo-id", default=None, help="a Hugging Face voice repo")
    p.add_argument("--espeak-data", default=None, help="patched espeak-ng-data")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--provider", default="cpu", help="cpu, cuda or coreml")
    p.add_argument("--voices", action="store_true", help="list voices and exit")
    p.add_argument("--licence", action="store_true",
                   help="print the backend's licence terms and exit")
    p.add_argument("--phonemes", action="store_true",
                   help="show what the Ghana lexicon does to the text, without "
                        "loading a model")
    p.add_argument("--debug", action="store_true", help="sherpa-onnx diagnostics")
    args = p.parse_args(argv)

    text = " ".join(args.text)
    if text == "-" or (not text and not sys.stdin.isatty()):
        text = sys.stdin.read()

    if args.phonemes:
        # Deliberately model-free: this is for checking a pronunciation, and
        # loading 300 MB to answer it would be absurd.
        from .inject import GhanaInjector
        from .mnemonics import phonemise

        if not text.strip():
            p.error("--phonemes needs some text")
        injector = GhanaInjector()
        injected = injector(text)
        print(f"injected: {injected}")
        try:
            print(f"espeak:   {phonemise(injected)}")
        except RuntimeError as exc:
            print(f"espeak:   ({exc})")
        print(f"coverage: {injector.coverage(text):.0%} of words are in the lexicon")
        return 0

    from .backends import load

    kwargs = dict(model_dir=args.model_dir, repo_id=args.repo_id,
                  espeak_data=args.espeak_data, num_threads=args.threads,
                  provider=args.provider, debug=args.debug, speed=args.speed)
    if args.voice:
        kwargs["voice"] = args.voice
    backend = load(args.backend, **kwargs)

    if args.licence:
        licence = backend.config.get("licence") or {}
        print(f"backend        {backend.name}")
        for key, value in licence.items():
            print(f"{key:15s}{value}")
        if not backend.commercial_use:
            print("\nThis voice is not licensed for commercial use. The kokoro "
                  "backend is.")
        return 0

    if args.voices:
        for name in backend.voices:
            mark = " *" if name in backend.recommended else ""
            print(f"{backend.resolve(name):4d}  {name}{mark}")
        if backend.recommended:
            print("\n* recommended", file=sys.stderr)
        return 0

    if not text.strip():
        p.error("no text given")

    result = backend.speak(text)
    path = result.save(args.output)
    print(f"{path}  {result.duration:.2f}s  {result.backend}/{result.voice}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_entry())
