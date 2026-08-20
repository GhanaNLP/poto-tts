"""Generate the Space's audio: every sentence in three routes, per voice.

The thirty-two sentences below are real, taken from a Ghanaian English news corpus of
1.7M lines by scoring each candidate against the shipped lexicon and keeping prose
with two to four Ghanaian words in it. They are inlined rather than re-extracted
because the corpus is not part of this repository -- it lives on an external drive --
and because the selection is a fixed part of the demo, not something to redo.

Written examples flatter the system: you reach for the names you already know it
handles. Selecting by lexicon coverage instead showed the real distribution, and the
first attempt also showed why filtering matters -- the highest-scoring sentences were
all name lists ("Sefwi, Fante, Ewe, Ashanti, Ga-Dangbe, Krobo..." at 94% coverage),
technically ideal and useless to listen to. Requiring a verb and at most one comma is
what produced usable prose.

Four sentences per voice rather than the same four repeated eight times, so switching
tab brings new material as well as a new speaker.

Three routes per sentence, all saying the same words:

    kokoro   stock espeak-ng English, no Ghanaian dictionary
    poto     the Ghanaian dictionary, read as British English

Needs a built dictionary -- see docs/CUSTOMISING.md -- then:

    python tools/build_space_audio.py && python tools/build_space.py
"""
import json, subprocess, wave, sys
from pathlib import Path
sys.path.insert(0, "/home/owusus/Documents/GitHub/poto-tts")
from poto_tts import load, VOICES

SCRIPTS = {
 "Grace": [
  "Akple and Ademe are traditional Ghanaian dishes, particularly popular among the Ewe people of the Volta Region.",
  "Ayisha Modi has issued a stern warning to popular Ghanaian fetish priestess, Nana Agradaa.",
  "Following his installation, he has assumed the stool name Nii Otanka Okaija Akraman.",
  "Team Ghana were disqualified on infringement in the final changeover from Joseph Oduro Manu and Joseph Amoah."
 ],
 "Comfort": [
  "They are skipper Christian Amoah, Winifred Ntumi and Sandra Mensimah Owusu.",
  "He is the son of the late Emma Naa Ameley Tagoe and hails from the Adanse Royal lineage.",
  "Didi began his youth career with Bolga Soccer Masters at Bolgatanga in the Upper East Region of Ghana.",
  "The arrests occurred in Blorkorfe and Awakpedome, both suburbs of Adidome in the Central Tongu District."
 ],
 "Mercy": [
  "This ruling was widely interpreted as a validation of Nii Adama Latse's claim to the Ga Mantse title.",
  "Adams Atidana Oliver won gold while Ofori Appiah Joel and Samuel Owusu won silver and bronze respectively.",
  "Juabenhene Nana Otuo Sereboe II was among the chiefs who pleaded on their behalf.",
  "Chirano is a mining community in the Bibiani Ahweso Bekai Municipality in the Western North."
 ],
 "Patience": [
  "The consignment originated from Kwahu Fodoa and was en route to Togo through Aflao.",
  "Following the fight, Ghanaians have been recalling the similar annihilation Bastie Samir handed to Bukom Banku.",
  "The NPP's Solomon Kwame Asumadu is contesting the Akwatia seat against the NDC's Bernard Bediako.",
  "Esa Payin gave birth to Bediako Ntim, who became the successor of Opon Payin."
 ],
 "Emmanuel": [
  "Volivo to Dorfor Adidome Bridge across the Volta River has been revived.",
  "Shatta Bandle was born Idris Firdaus, a Northerner from a small village called Karaga.",
  "The Saviour Church of Ghana has named a school at Bonwire after her, Nana Konadu Saviour School.",
  "Bofoakwa Tano have severed ties with coach Frimpong Manso after four months at the helm, the club announced."
 ],
 "Isaac": [
  "Member of Parliament for Bawku Central, Mahama Ayariga has reported that three more people were killed in Bawku.",
  "Ghanaian musician Stonebwoy has taken legal action against politician and entertainment critic, Baba Sadiq.",
  "Ghana legend and former Asante Kotoko star Opoku Afriyie has passed away.",
  "Ghanaians on social media are eulogising veteran Highlife musician, Nana Kwame Ampadu."
 ],
 "Ebenezer": [
  "The Information Minister, Kojo Oppong Nkrumah says government is working to stabilise the cedi.",
  "The Adom TV Fufuo Party is underway at the forecourt of The Multimedia Group at Kokomlemle in Accra.",
  "A court has ordered Marwako Fast Food in Accra to pay over a million Ghana cedis in damages.",
  "Sarkodie, Shatta Wale and Stonebwoy are undoubtedly three of Ghana's biggest musicians."
 ],
 "Bright": [
  "Odododiodio MP Edwin Nii Lante Vanderpuye, will not return to Parliament after the expiration of his current term.",
  "Ghanaian singer Kofi Mante has acknowledged Bisa Kdei's influence on the highlife music genre.",
  "Former President, John Dramani Mahama has condemned the killing of Al Jazeera journalist Shireen Abu Akleh.",
  "Ghanaian legal luminary Tsatsu Tsikata has proposed a solution to Ghana's recurring Cedi depreciation."
 ]
}

ROUTES = {"poto":   dict(espeak_voice="en",    espeak_data="build/direct/espeak-ng-data"),
          "kokoro": dict(espeak_voice="en-us", espeak_data="/usr/local/share/espeak-ng-data")}
out = Path("space/audio"); out.mkdir(parents=True, exist_ok=True)
rows, texts, annot, phon = [], {}, {}, {}

for v in VOICES:
    lines = SCRIPTS[v.name]
    for route, kw in ROUTES.items():
        tts = load(voice=v.name, **kw)
        root = Path(kw["espeak_data"]).resolve().parent
        for i, text in enumerate(lines):
            key = f"{v.name}-{i}"
            texts[key] = text
            p = out / f"{key}.{route}.wav"
            tts.save(text, str(p))
            with wave.open(str(p)) as fh:
                n, sr = fh.getnframes(), fh.getframerate()
            rows.append({"key": key, "route": route, "voice": v.name,
                         "file": p.name, "seconds": round(n / sr, 2)})
            phon.setdefault(key, {})[route] = " ".join(subprocess.run(
                ["espeak-ng", f"--path={root}", "-v", kw["espeak_voice"],
                 "--ipa=3", "-q", text], capture_output=True, text=True).stdout.split())
            if route == "poto":
                annot[key] = tts.annotate(text)
    print(f"  {v.name}", flush=True)

Path("space/samples.json").write_text(json.dumps({
    "texts": texts, "rows": rows, "phonemes": phon,
    "voices": [v.name for v in VOICES],
    "scripts": {v.name: [f"{v.name}-{i}" for i in range(len(SCRIPTS[v.name]))]
                for v in VOICES},
    "annotations": {k: [[w, bool(h)] for w, h in a] for k, a in annot.items()}},
    ensure_ascii=False, indent=1))
print(f"{len(rows)} files, {len(texts)} distinct sentences")
