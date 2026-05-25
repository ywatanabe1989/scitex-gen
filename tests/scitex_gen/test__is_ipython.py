#!/usr/bin/env python3
"""Tests for scitex_gen._is_ipython module.

`is_ipython()` inspects its own module globals for `__IPYTHON__`, and
`is_script()` is its logical inverse. The test process is a plain
(non-IPython) interpreter, so the real, honest observable behavior is
`is_ipython() is False` / `is_script() is True`. We assert against that
real environment rather than faking the interpreter — no mocks.
"""

import pytest

pytest.importorskip("torch")

from scitex_gen import is_ipython, is_script


class TestIsIPython:
    """Test cases for is_ipython function."""

    def test_is_ipython_returns_false_in_script(self):
        # Arrange
        # Act
        result = is_ipython()
        # Assert
        assert result is False

    def test_is_ipython_returns_boolean_type(self):
        # Arrange
        # Act
        result = is_ipython()
        # Assert
        assert isinstance(result, bool)

    def test_is_ipython_is_consistent_across_calls(self):
        # Arrange
        first = is_ipython()
        # Act
        second = is_ipython()
        # Assert
        assert first == second


class TestIsScript:
    """Test cases for is_script function."""

    def test_is_script_returns_true_in_script(self):
        # Arrange
        # Act
        result = is_script()
        # Assert
        assert result is True

    def test_is_script_returns_boolean_type(self):
        # Arrange
        # Act
        result = is_script()
        # Assert
        assert isinstance(result, bool)

    def test_is_script_is_inverse_of_is_ipython(self):
        # Arrange
        ipython_result = is_ipython()
        # Act
        script_result = is_script()
        # Assert
        assert script_result == (not ipython_result)


class TestIsIPythonScriptIntegration:
    """Integration tests for is_ipython and is_script."""

    def test_is_ipython_and_is_script_are_mutually_exclusive(self):
        # Arrange
        ipython = is_ipython()
        # Act
        script = is_script()
        # Assert
        assert ipython != script

    def test_exactly_one_of_ipython_or_script_is_true(self):
        # Arrange
        ipython = is_ipython()
        # Act
        script = is_script()
        # Assert
        assert ipython or script

    def test_branching_on_environment_selects_script_mode(self):
        # Arrange
        # Act
        if is_ipython():
            mode = "interactive"
        else:
            mode = "script"
        # Assert
        assert mode == "script"


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_is_ipython.py
# --------------------------------------------------------------------------------
# def is_ipython():
#     try:
#         __IPYTHON__
#         ipython_mode = True
#     except NameError:
#         ipython_mode = False
#
#     return ipython_mode
#
#
# def is_script():
#     return not is_ipython()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_is_ipython.py
# --------------------------------------------------------------------------------
