# Smoke test mirror for examples/quickstart.py.
from pathlib import Path


def test_quickstart_exists():
    root = Path(__file__).resolve().parents[2]
    assert (root / "examples" / "quickstart.py").is_file()
