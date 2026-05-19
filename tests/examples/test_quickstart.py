# Smoke test mirror for examples/quickstart.py.
from pathlib import Path


def test_quickstart_exists_root_examples_quickstart_py_is_file():
    # Arrange
    # Act
    root = Path(__file__).resolve().parents[2]
    # Assert
    assert (root / "examples" / "quickstart.py").is_file()
