#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-03 02:55:22 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/gen/test__less.py

"""Test suite for scitex_gen._less module.

`less()` writes its argument to a real temporary file and asks the active
IPython shell to page it. We inject a hand-rolled `RecordingShell` in place
of the real shell (no mocks): its `system()` reads the real temp file that
production just wrote, so the assertions observe genuine filesystem state and
the real `os.remove` cleanup.
"""

import os

import pytest

pytest.importorskip("torch")

from scitex_gen import less


class RecordingShell:
    """Stand-in IPython shell that records the `less <path>` command it is
    asked to run and reads back the real temp file's contents."""

    def __init__(self, error=None):
        self.commands = []
        self.paths = []
        self.contents = []
        self._error = error

    def system(self, command):
        self.commands.append(command)
        path = command.split(" ", 1)[1]
        self.paths.append(path)
        with open(path) as handle:
            self.contents.append(handle.read())
        if self._error is not None:
            raise self._error


def _run_less(output, error=None):
    shell = RecordingShell(error=error)
    less(output, get_ipython=lambda: shell)
    return shell


def test_less_writes_single_line_output_to_temp_file():
    # Arrange
    # Act
    shell = _run_less("Hello, World!")
    # Assert
    assert shell.contents == ["Hello, World!"]


def test_less_writes_multiline_output_to_temp_file():
    # Arrange
    text = "Line 1\nLine 2\nLine 3\nThis is a longer line\nLast line"
    # Act
    shell = _run_less(text)
    # Assert
    assert shell.contents == [text]


def test_less_writes_special_characters_to_temp_file():
    # Arrange
    text = "Special chars: @#$%^&*() Unicode: 你好世界 Émojis: 🚀💻"
    # Act
    shell = _run_less(text)
    # Assert
    assert shell.contents == [text]


def test_less_writes_empty_output_to_temp_file():
    # Arrange
    # Act
    shell = _run_less("")
    # Assert
    assert shell.contents == [""]


def test_less_writes_large_output_to_temp_file():
    # Arrange
    text = "x" * 10000 + "\n" + "y" * 10000
    # Act
    shell = _run_less(text)
    # Assert
    assert shell.contents == [text]


def test_less_invokes_less_command_on_temp_file():
    # Arrange
    # Act
    shell = _run_less("Command test")
    # Assert
    assert shell.commands[0].startswith("less ")


def test_less_removes_temp_file_after_display():
    # Arrange
    # Act
    shell = _run_less("Cleanup test")
    # Assert
    assert not os.path.exists(shell.paths[0])


def test_less_propagates_error_from_shell():
    # Arrange
    shell = RecordingShell(error=RuntimeError("System command failed"))
    ctx = pytest.raises(RuntimeError, match="System command failed")
    # Act
    # Assert
    with ctx:
        less("Test output", get_ipython=lambda: shell)


def test_less_keeps_temp_file_when_shell_fails():
    # Arrange
    shell = RecordingShell(error=RuntimeError("System command failed"))
    try:
        less("Test output", get_ipython=lambda: shell)
    except RuntimeError:
        pass
    # Act
    leftover_exists = os.path.exists(shell.paths[0])
    os.remove(shell.paths[0])
    # Assert
    assert leftover_exists is True


def test_less_raises_attribute_error_without_a_shell():
    # Arrange
    ctx = pytest.raises(AttributeError)
    # Act
    # Assert
    with ctx:
        less("Test output", get_ipython=lambda: None)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_less.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-03 02:11:18 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/gen/_less.py
#
# # Functions
# def less(output, *, get_ipython=None):
#     """
#     Print the given output using `less` in an IPython or IPdb session.
#     """
#     import os
#     import tempfile
#
#     if get_ipython is None:
#         from IPython import get_ipython as _default_get_ipython
#
#         get_ipython = _default_get_ipython
#
#     # Create a temporary file to hold the output
#     with tempfile.NamedTemporaryFile(delete=False, mode="w+t") as tmpfile:
#         # Write the output to the temporary file
#         tmpfile.write(output)
#         tmpfile_name = tmpfile.name
#
#     # Use IPython's system command access to pipe the content to `less`
#     get_ipython().system(f"less {tmpfile_name}")
#
#     # Clean up the temporary file
#     os.remove(tmpfile_name)
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_less.py
# --------------------------------------------------------------------------------
