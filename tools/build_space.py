"""Generate the HF Space comparison page from space/samples.json.

A static page, no framework and no CDN: a Space with `sdk: static` serves this
directory as-is, and the audio is already sitting next to it.

Regenerate the audio and the JSON with the block at the bottom of this docstring,
then run this to rebuild the page:

    python tools/build_space.py

The page deliberately shows the phonemes as well as the audio. The claim being made
is about pronunciation rather than voice quality, and a reader who cannot hear the
difference between /kwˈeɪbnə/ and /kwabˈɪna/ on a laptop speaker can still read it.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "space"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>poto-tts &mdash; Ghanaian English on Kokoro</title>
<style>
  :root {{
    --bg: #fbfaf8; --fg: #1a1a1a; --muted: #5c5c5c; --line: #e3e0da;
    --card: #ffffff; --accent: #1a6b4f; --accent-soft: #eaf3ef;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #16181a; --fg: #eceae6; --muted: #9c9a95; --line: #2b2e31;
      --card: #1d2022; --accent: #6cc4a1; --accent-soft: #1d2b26;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #16181a; --fg: #eceae6; --muted: #9c9a95; --line: #2b2e31;
    --card: #1d2022; --accent: #6cc4a1; --accent-soft: #1d2b26;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--fg); line-height: 1.55;
    font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    -webkit-text-size-adjust: 100%;
  }}
  .wrap {{ max-width: 900px; margin: 0 auto; padding: 40px 20px 80px; }}
  h1 {{ font-size: 1.7rem; margin: 0 0 .3em; letter-spacing: -.01em; }}
  .lede {{ color: var(--muted); max-width: 62ch; margin: 0 0 1.6em; }}
  .note {{
    background: var(--accent-soft); border-left: 3px solid var(--accent);
    padding: .85em 1.1em; margin: 0 0 2.2em; font-size: .93rem; max-width: 70ch;
  }}
  .note strong {{ color: var(--accent); }}
  .item {{ border: 1px solid var(--line); background: var(--card);
           border-radius: 10px; padding: 18px 20px; margin: 0 0 16px; }}
  .txt {{ font-size: 1.03rem; margin: 0 0 14px; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
  @media (max-width: 860px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .side h3 {{
    font-size: .72rem; text-transform: uppercase; letter-spacing: .09em;
    margin: 0 0 .5em; color: var(--muted); font-weight: 600;
  }}
  .side.poto h3 {{ color: var(--accent); }}
  audio {{ width: 100%; height: 34px; }}
  .ipa {{
    font-size: .84rem; color: var(--muted); margin: .6em 0 0; word-break: break-word;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  }}
  .side.poto .ipa {{ color: var(--fg); }}
  footer {{ margin-top: 44px; padding-top: 20px; border-top: 1px solid var(--line);
            color: var(--muted); font-size: .88rem; }}
  a {{ color: var(--accent); }}
</style>
</head>
<body>
<div class="wrap">
  <h1>poto-tts</h1>
  <p class="lede">English text-to-speech that pronounces Ghanaian words properly.
  Same model, same speaker, same text &mdash; the only difference is the front-end.</p>

  <div class="note"><strong>Not Ghanaian-accented voices.</strong> All three columns
  are Kokoro's British speaker &ldquo;Grace&rdquo;: what changes is what she says, not
  who she sounds like. Both poto-tts modes pronounce the Ghanaian words correctly and
  differ only in the accent &mdash; <strong>gh</strong> gives them Ghanaian vowels and
  a tapped r, and lets a little of that reach the English too; <strong>en</strong>
  gives them English vowels and leaves every other word exactly as an English TTS
  would say it.</div>

{items}

  <footer>
    Kokoro v1.0 via sherpa-onnx, speaker Grace (bf_alice). Pronunciation from a
    104,623-word lexicon compiled into espeak's dictionary and read by the
    <code>en-gh</code> accent voice &mdash; data, not code, so Android, iOS and
    WebAssembly get the same result.
    <br>
    <a href="https://github.com/GhanaNLP/poto-tts">GhanaNLP/poto-tts</a> &middot;
    <a href="https://huggingface.co/ghananlpcommunity/poto-tts-kokoro-gh">the voice on the Hub</a>
    &middot; <code>pip install poto-tts</code>
  </footer>
</div>
</body>
</html>
"""

ITEM = """  <div class="item">
    <p class="txt">{text}</p>
    <div class="grid">
      <div class="side">
        <h3>Ordinary Kokoro</h3>
        <audio controls preload="none" src="audio/{kokoro_file}"></audio>
        <p class="ipa">{kokoro_ipa}</p>
      </div>
      <div class="side poto">
        <h3>poto-tts &middot; gh mode</h3>
        <audio controls preload="none" src="audio/{gh_file}"></audio>
        <p class="ipa">{gh_ipa}</p>
      </div>
      <div class="side poto">
        <h3>poto-tts &middot; en mode</h3>
        <audio controls preload="none" src="audio/{en_file}"></audio>
        <p class="ipa">{en_ipa}</p>
      </div>
    </div>
  </div>
"""


def main() -> int:
    data = json.loads((SPACE / "samples.json").read_text())
    by: dict[str, dict[str, dict]] = {}
    for row in data["rows"]:
        by.setdefault(row["key"], {})[row["route"]] = row
    items = []
    for key, text in data["texts"].items():
        pair = by.get(key, {})
        if {"gh", "en", "kokoro"} - set(pair):
            print(f"  skipping {key}: needs all three routes")
            continue
        items.append(ITEM.format(
            text=html.escape(text),
            kokoro_file=html.escape(pair["kokoro"]["file"]),
            gh_file=html.escape(pair["gh"]["file"]),
            en_file=html.escape(pair["en"]["file"]),
            kokoro_ipa=html.escape(pair["kokoro"]["phonemes"]),
            gh_ipa=html.escape(pair["gh"]["phonemes"]),
            en_ipa=html.escape(pair["en"]["phonemes"]),
        ))
    (SPACE / "index.html").write_text(PAGE.format(items="\n".join(items)))
    print(f"wrote {SPACE / 'index.html'} with {len(items)} comparisons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
