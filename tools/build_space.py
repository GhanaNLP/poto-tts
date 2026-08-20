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
    --card: #ffffff; --accent: #1a6b4f;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #16181a; --fg: #eceae6; --muted: #9c9a95; --line: #2b2e31;
      --card: #1d2022; --accent: #6cc4a1;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #16181a; --fg: #eceae6; --muted: #9c9a95; --line: #2b2e31;
    --card: #1d2022; --accent: #6cc4a1;
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
  .tabs {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 22px; }}
  .tabs button {{
    font: inherit; font-size: .88rem; padding: .42em .9em; cursor: pointer;
    background: var(--card); color: var(--muted);
    border: 1px solid var(--line); border-radius: 999px;
  }}
  .tabs button[aria-selected="true"] {{
    background: var(--accent); border-color: var(--accent); color: var(--bg);
    font-weight: 600;
  }}
  mark {{
    background: none; color: var(--accent); font-weight: 600;
    border-bottom: 2px solid var(--accent); padding-bottom: 1px;
  }}
  .legend {{ font-size: .84rem; color: var(--muted); margin: 0 0 26px; }}
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


  <div class="tabs" role="tablist" id="tabs"></div>
  <p class="legend"><mark>Underlined</mark> words are in the Ghanaian lexicon, so
  poto-tts pronounces them from it. Everything else is left to espeak's English and is
  said exactly as any English TTS would. Each voice reads different sentences, drawn
  from a Ghanaian English news corpus. <strong>gh</strong> reads the lexicon's words
  with Ghanaian vowels and a tapped r; <strong>en</strong> reads them with English
  vowels.</p>

{items}

  <script>
    const VOICES = {voices};
    const tabs = document.getElementById("tabs");
    function show(voice) {{
      for (const b of tabs.children)
        b.setAttribute("aria-selected", String(b.dataset.voice === voice));
      for (const s of document.querySelectorAll("section[data-voice]")) {{
        const on = s.dataset.voice === voice;
        s.hidden = !on;
        if (!on) for (const a of s.querySelectorAll("audio")) a.pause();
      }}
    }}
    for (const v of VOICES) {{
      const b = document.createElement("button");
      b.type = "button"; b.textContent = v; b.dataset.voice = v;
      b.setAttribute("role", "tab"); b.onclick = () => show(v);
      tabs.appendChild(b);
    }}
    show(VOICES[0]);
  </script>

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
        <h3>poto-tts &middot; gh</h3>
        <audio controls preload="none" src="audio/{gh_file}"></audio>
        <p class="ipa">{gh_ipa}</p>
      </div>
      <div class="side poto">
        <h3>poto-tts &middot; en</h3>
        <audio controls preload="none" src="audio/{en_file}"></audio>
        <p class="ipa">{en_ipa}</p>
      </div>
    </div>
  </div>
"""

SECTION = """  <section data-voice="{voice}" hidden>
{items}  </section>
"""


def marked(annotation, fallback):
    """The sentence with lexicon words wrapped in <mark>.

    Built from the annotation the library produced rather than re-derived here, so the
    page cannot claim a word came from the lexicon when the dictionary disagrees.
    """
    if not annotation:
        return html.escape(fallback)
    out = []
    for word, hit in annotation:
        safe = html.escape(word)
        out.append(f"<mark>{safe}</mark>" if hit else safe)
    text = " ".join(out)
    for mark in (".", ",", "?", "!", ";", ":", "'s", "'"):
        text = text.replace(f" {mark}", mark)
    return text


def main() -> int:
    data = json.loads((SPACE / "samples.json").read_text())
    voices = data["voices"]
    files = {(r["key"], r["route"]): r["file"] for r in data["rows"]}
    sections = []
    for voice in voices:
        items = []
        for key in data["scripts"][voice]:
            if any((key, r) not in files for r in ("gh", "en", "kokoro")):
                print(f"  skipping {key}: needs all three routes")
                continue
            ipa = data["phonemes"].get(key, {})
            items.append(ITEM.format(
                text=marked(data.get("annotations", {}).get(key), data["texts"][key]),
                kokoro_file=html.escape(files[(key, "kokoro")]),
                gh_file=html.escape(files[(key, "gh")]),
                en_file=html.escape(files[(key, "en")]),
                kokoro_ipa=html.escape(ipa.get("kokoro", "")),
                gh_ipa=html.escape(ipa.get("gh", "")),
                en_ipa=html.escape(ipa.get("en", "")),
            ))
        sections.append(SECTION.format(voice=html.escape(voice), items="".join(items)))
    (SPACE / "index.html").write_text(PAGE.format(
        items="".join(sections), voices=json.dumps(voices)))
    print(f"wrote {SPACE / 'index.html'}: {len(voices)} voices, "
          f"{len(data['texts'])} sentences, 3 routes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
