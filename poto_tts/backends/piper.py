"""Piper backend: a voice trained on Ghanaian speech.

Where the Kokoro backend borrows a model that speaks English and only corrects its
pronunciation of Ghanaian words, this one is trained on Ghanaian English, so the
accent is in the weights rather than only in the dictionary. It should sound more
Ghanaian. It also carries a restriction the other does not.

**Not for commercial use.** The voice is trained on Ghanaian broadcast and
interview recordings. Whatever a dataset card permits, those speakers did not
consent to having their voices modelled, and a voice model carries speaker
identity in a way a text corpus does not. Research, evaluation and non-commercial
use. Anyone needing a commercial Ghanaian voice should use the Kokoro backend, or
train this one on recordings they hold rights to -- the pipeline in `tools/` takes
any dataset.

Both backends read the same `espeak-ng-data`, so either pronounces a Ghanaian name
the same way. What differs is the voice around it.
"""

from __future__ import annotations

from .base import Backend

__all__ = ["PiperBackend"]


class PiperBackend(Backend):
    """A Piper (VITS) voice on sherpa-onnx, reading the Ghanaian espeak dictionary.

    Voice names come from the config's `speakers`, and `recommended` names the
    subset whose speaker clusters were cohesive enough to be one person rather
    than an average of several; the rest exist because their audio improved the
    shared model.
    """

    name = "piper"

    def _config(self, *, num_threads: int, provider: str, debug: bool):
        s = self._sherpa
        return s.OfflineTtsConfig(
            model=s.OfflineTtsModelConfig(
                vits=s.OfflineTtsVitsModelConfig(
                    model=self._path("model"),
                    tokens=self._path("tokens"),
                    data_dir=str(self.espeak_data),
                ),
                provider=provider,
                num_threads=num_threads,
                debug=debug,
            ),
        )
