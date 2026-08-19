"""Kokoro backend: an Apache-2.0 voice with Ghanaian pronunciation, untrained.

No training is involved. Kokoro v1.0 already speaks good English, and sherpa-onnx
phonemises its English with espeak-ng -- so pointing it at the patched
`espeak-ng-data` is the entire intervention. Decoded token ids for 'Kwabena went to
Achimota and met the Okuapenhene':

    stock     k w ˈe ɪ b n ə  ...  ˈo ʊ k j uː ˌe ɪ p ə n h ˌiː n
    patched   k w ɑː b ˈɪ n ɑː  ...  ˌo k w ɑː p ɛ n h ˈɛ n ɛ

Kokoro's token inventory is misaki's, and every phone the Ghana dictionary produces
has a token there -- ɑ, ɛ, ɔ, e, o, ː, ŋ, ɲ, ɡ, and the components of kp and gb --
so nothing is dropped on the way in.

A wrinkle that looks like a configuration option and is not: Kokoro v1.0 ships
`lexicon-us-en.txt`, and sherpa-onnx has a lexicon frontend that would consult it.
That branch is unreachable for English. It runs only when `lang` is empty; `lang`
falls back to the model's `voice` metadata; and the metadata reader substitutes
'en-us' when that value is blank. Nothing can empty it -- which is why Ghanaian
pronunciation lives in the espeak dictionary rather than in a lexicon file.
"""

from __future__ import annotations

from .base import Backend

__all__ = ["KokoroBackend"]


class KokoroBackend(Backend):
    """Kokoro v1.0 on sherpa-onnx, reading the Ghanaian espeak dictionary."""

    name = "kokoro"

    def _config(self, *, num_threads: int, provider: str, debug: bool):
        s = self._sherpa
        return s.OfflineTtsConfig(
            model=s.OfflineTtsModelConfig(
                kokoro=s.OfflineTtsKokoroModelConfig(
                    model=self._path("model"),
                    voices=self._path("voices"),
                    tokens=self._path("tokens"),
                    data_dir=str(self.espeak_data),
                    # Set explicitly rather than left to the model's metadata, so a
                    # future model naming a different voice cannot silently
                    # phonemise against a dictionary we are not patching.
                    lang=self.config.get("espeak_voice", "en-us"),
                ),
                provider=provider,
                num_threads=num_threads,
                debug=debug,
            ),
        )

    def _prepare(self, gen) -> None:
        # Belt and braces: the config carries lang already, but a per-call value
        # cannot be overridden by a metadata fallback.
        gen.extra = {"lang": self.config.get("espeak_voice", "en-us")}
