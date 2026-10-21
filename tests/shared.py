"""One full catalog + semantic engine (no LLM) shared by every test module in a run."""
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = os.getenv('CATALOG_PATH', str(ROOT / 'data/products.json'))


@lru_cache(maxsize=1)
def catalog():
    import search
    from engine.llm import NullLLM
    return search.Catalog.load(CATALOG_PATH, llm=NullLLM())
