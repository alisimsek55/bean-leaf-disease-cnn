"""Test yardimcilari: src/ klasorunu import yoluna ekler ve rakamla baslayan betikleri yukler."""
import importlib
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def yukle(ad: str):
    """'01_on_isleme' gibi rakamla baslayan modulleri importlib ile yukler."""
    return importlib.import_module(ad)