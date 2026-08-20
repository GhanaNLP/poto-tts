

# The lexicon is an optional dependency (`poto-tts[lexicon]`): needed to build or
# inspect a dictionary, not to speak. Tests that read it should say so and skip,
# rather than failing and looking like a broken build.
import pytest


def pytest_collection_modifyitems(config, items):
    try:
        import ghana_english_g2p  # noqa: F401
        return
    except ImportError:
        pass
    skip = pytest.mark.skip(reason="needs poto-tts[lexicon] (ghana-english-g2p)")
    for item in items:
        if item.fspath.basename in ("test_inject.py", "test_mnemonics.py"):
            item.add_marker(skip)
