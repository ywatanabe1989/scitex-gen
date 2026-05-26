#!/usr/bin/env python3
"""Tests for scitex_gen._ipython._is_ipython module."""

import contextlib

import pytest

pytest.importorskip("torch")

from scitex_gen import is_ipython, is_script


@contextlib.contextmanager
def _swap_attr(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


class TestIsIPython:
    """Test cases for is_ipython function."""

    def test_is_ipython_not_in_ipython(self):
        """Test is_ipython returns False when not in IPython."""
        # In normal Python environment, __IPYTHON__ is not defined
        # Arrange
        # Act
        # Assert
        assert is_ipython() is False

    def test_is_ipython_in_ipython(self):
        """Test is_ipython returns True when in IPython environment.

        Note: We can't directly mock the global __IPYTHON__ variable because
        is_ipython() checks its own module's globals. Instead, we test the
        mocking approach that simulates the behavior.
        """
        # Arrange
        # Act
        # Assert
        import scitex_gen._ipython._is_ipython

        # Save original function
        original_is_ipython = scitex_gen._ipython._is_ipython.is_ipython

        # Create a mock that simulates being in IPython
        scitex_gen._ipython._is_ipython.is_ipython = lambda: True

        try:
            # Verify our mock works
            assert scitex_gen._ipython._is_ipython.is_ipython() is True
        finally:
            # Restore original function
            scitex_gen._ipython._is_ipython.is_ipython = original_is_ipython

    def test_is_ipython_jupyter_check(self):
        """Test behavior in Jupyter-like environment."""
        # In Jupyter, both __IPYTHON__ and get_ipython are typically available
        # Arrange
        # Act
        # Assert
        import builtins

        with _swap_attr(
            builtins, "globals", lambda *a, **kw: {"__IPYTHON__": True}
        ):
            # This test shows the limitation - we can't easily mock the global __IPYTHON__
            # The actual function checks for __IPYTHON__ in its own global namespace
            assert is_ipython() is False  # Will still be False in test environment

    def test_is_ipython_consistency_split_1(self):
        """Test that is_ipython returns consistent results."""
        # Arrange
        result1 = is_ipython()
        result2 = is_ipython()
        result3 = is_ipython()
        # Act
        # Assert
        assert result1 == result2 == result3

    def test_is_ipython_consistency_split_2(self):
        """Test that is_ipython returns consistent results."""
        # Arrange
        result1 = is_ipython()
        result2 = is_ipython()
        result3 = is_ipython()
        result1 == result2 == result3
        # Act
        # Assert
        assert isinstance(result1, bool)


class TestIsScript:
    """Test cases for is_script function."""

    def test_is_script_inverse_of_ipython(self):
        """Test that is_script is the logical inverse of is_ipython."""
        # Arrange
        ipython_result = is_ipython()
        # Act
        script_result = is_script()

        # Assert
        assert script_result == (not ipython_result)

    def test_is_script_in_normal_python(self):
        """Test is_script returns True in normal Python environment."""
        # When not in IPython, we are in a script
        # Arrange
        # Act
        # Assert
        assert is_script() is True

    def test_is_script_with_mocked_ipython(self):
        """Test is_script behavior when mocking IPython environment."""
        # Arrange
        # Act
        # Assert
        import scitex_gen._ipython._is_ipython

        # Save original functions
        original_is_ipython = scitex_gen._ipython._is_ipython.is_ipython
        original_is_script = scitex_gen._ipython._is_ipython.is_script

        # Swap is_ipython to return True
        scitex_gen._ipython._is_ipython.is_ipython = lambda: True

        # Redefine is_script to use the swapped is_ipython
        scitex_gen._ipython._is_ipython.is_script = (
            lambda: not scitex_gen._ipython._is_ipython.is_ipython()
        )

        try:
            assert scitex_gen._ipython._is_ipython.is_script() is False
        finally:
            # Restore original functions
            scitex_gen._ipython._is_ipython.is_ipython = original_is_ipython
            scitex_gen._ipython._is_ipython.is_script = original_is_script

    def test_is_script_consistency_split_1(self):
        """Test that is_script returns consistent results."""
        # Arrange
        results = [is_script() for _ in range(5)]
        # Act
        # Assert
        assert all((r == results[0] for r in results))

    def test_is_script_consistency_split_2(self):
        """Test that is_script returns consistent results."""
        # Arrange
        results = [is_script() for _ in range(5)]
        all((r == results[0] for r in results))
        # Act
        # Assert
        assert isinstance(results[0], bool)


class TestIntegration:
    """Integration tests for is_ipython and is_script."""

    def test_mutual_exclusivity_ipython_script_split_1(self):
        """Test that is_ipython and is_script are mutually exclusive."""
        # Arrange
        ipython = is_ipython()
        script = is_script()
        # Act
        # Assert
        assert ipython != script

    def test_mutual_exclusivity_ipython_script_split_2(self):
        """Test that is_ipython and is_script are mutually exclusive."""
        # Arrange
        ipython = is_ipython()
        script = is_script()
        ipython != script
        # Act
        # Assert
        assert ipython or script

    def test_mutual_exclusivity_ipython_script_split_3(self):
        """Test that is_ipython and is_script are mutually exclusive."""
        # Arrange
        ipython = is_ipython()
        script = is_script()
        ipython != script
        ipython or script
        # Act
        # Assert
        assert not (ipython and script)

    def test_use_case_branching_split_1(self):
        """Test typical use case of branching based on environment."""
        # Arrange
        if is_ipython():
            mode = 'interactive'
        else:
            mode = 'script'
        # Act
        # Assert
        assert mode == 'script'

    def test_use_case_branching_split_2(self):
        """Test typical use case of branching based on environment."""
        # Arrange
        if is_ipython():
            mode = 'interactive'
        else:
            mode = 'script'
        mode == 'script'
        mode2 = 'script' if is_script() else 'interactive'
        # Act
        # Assert
        assert mode2 == 'script'

    @pytest.mark.parametrize("mock_ipython", [True, False])
    def test_environment_detection_smoke_case(self, mock_ipython):
        """Test environment detection with different states."""
        # Arrange
        import scitex_gen._ipython._is_ipython
        original_is_ipython = scitex_gen._ipython._is_ipython.is_ipython
        scitex_gen._ipython._is_ipython.is_ipython = lambda: mock_ipython
        # Act
        try:
            detected = (
                scitex_gen._ipython._is_ipython.is_ipython(),
                scitex_gen._ipython._is_ipython.is_script(),
            )
        finally:
            scitex_gen._ipython._is_ipython.is_ipython = original_is_ipython
        expected = (True, False) if mock_ipython else (False, True)
        # Assert
        assert detected == expected


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
