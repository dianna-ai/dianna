"""Browser stand-in for spaCy (no Pyodide build): `load(name).tokenizer` splits words and punctuation."""
import re
from types import SimpleNamespace


def load(name):
    """Return an object whose `tokenizer(text)` gives tokens with a `.text`, like a spaCy pipeline."""
    return SimpleNamespace(
        tokenizer=lambda text:
        [SimpleNamespace(text=t) for t in re.findall(r"\w+|[^\w\s]", text)])
