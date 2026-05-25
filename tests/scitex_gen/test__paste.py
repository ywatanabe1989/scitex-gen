#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-03 02:55:27 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/gen/test__paste.py

"""Test suite for scitex_gen._paste module.

`paste()` reads code from the clipboard, dedents it, and executes it. We
inject the clipboard reader and the executor (no mocks): a recording executor
captures the exact (dedented) code string production produced, and the real
built-in `exec` is used to prove genuine execution and real error handling,
observed via the real `capsys` fixture.
"""

import textwrap

import pytest

pytest.importorskip("torch")

from scitex_gen import paste


def _raising_getter(exc):
    def getter():
        raise exc

    return getter


def _capture_executor():
    captured = []

    def executor(code):
        captured.append(code)

    return captured, executor


def test_paste_passes_clipboard_content_to_executor():
    # Arrange
    captured, executor = _capture_executor()
    content = "print('Hello from clipboard')"
    # Act
    paste(clipboard_getter=lambda: content, executor=executor)
    # Assert
    assert captured == [content]


def test_paste_dedents_indented_code():
    # Arrange
    captured, executor = _capture_executor()
    content = "\n            def hello():\n                return 42\n        "
    # Act
    paste(clipboard_getter=lambda: content, executor=executor)
    # Assert
    assert captured == [textwrap.dedent(content)]


def test_paste_handles_multiline_code():
    # Arrange
    captured, executor = _capture_executor()
    content = "\nx = 10\ny = 20\nz = x + y\n"
    # Act
    paste(clipboard_getter=lambda: content, executor=executor)
    # Assert
    assert captured == [textwrap.dedent(content)]


def test_paste_handles_empty_clipboard():
    # Arrange
    captured, executor = _capture_executor()
    # Act
    paste(clipboard_getter=lambda: "", executor=executor)
    # Assert
    assert captured == [""]


def test_paste_dedents_whitespace_only_content():
    # Arrange
    captured, executor = _capture_executor()
    content = "   \n\t  \n   "
    # Act
    paste(clipboard_getter=lambda: content, executor=executor)
    # Assert
    assert captured == [textwrap.dedent(content)]


def test_paste_preserves_unicode_characters():
    # Arrange
    captured, executor = _capture_executor()
    content = "print('Hello 世界! 🚀')"
    # Act
    paste(clipboard_getter=lambda: content, executor=executor)
    # Assert
    assert captured == [content]


def test_paste_dedents_complex_indentation():
    # Arrange
    captured, executor = _capture_executor()
    content = (
        "\n            class MyClass:\n"
        "                def __init__(self):\n"
        "                    self.value = 42\n        "
    )
    # Act
    paste(clipboard_getter=lambda: content, executor=executor)
    # Assert
    assert captured == [textwrap.dedent(content)]


def test_paste_preserves_unix_line_endings():
    # Arrange
    captured, executor = _capture_executor()
    content = "line1\nline2\nline3"
    # Act
    paste(clipboard_getter=lambda: content, executor=executor)
    # Assert
    assert captured == [content]


def test_paste_dedents_windows_line_endings():
    # Arrange
    captured, executor = _capture_executor()
    content = "line1\r\nline2\r\nline3"
    # Act
    paste(clipboard_getter=lambda: content, executor=executor)
    # Assert
    assert captured == [textwrap.dedent(content)]


def test_paste_executes_clipboard_code_for_real():
    # Arrange
    namespace = {}
    # Act
    paste(
        clipboard_getter=lambda: "result = 7 * 6",
        executor=lambda code: exec(code, namespace),
    )
    # Assert
    assert namespace["result"] == 42


def test_paste_reports_syntax_error(capsys):
    # Arrange
    # Act
    paste(clipboard_getter=lambda: "print('unterminated")
    # Assert
    assert capsys.readouterr().out.startswith("Could not execute clipboard content:")


def test_paste_reports_runtime_error(capsys):
    # Arrange
    # Act
    paste(clipboard_getter=lambda: "1 / 0")
    # Assert
    assert (
        capsys.readouterr().out
        == "Could not execute clipboard content: division by zero\n"
    )


def test_paste_reports_import_error(capsys):
    # Arrange
    # Act
    paste(clipboard_getter=lambda: "import nonexistent_module_xyz")
    # Assert
    assert (
        capsys.readouterr().out
        == "Could not execute clipboard content: No module named 'nonexistent_module_xyz'\n"
    )


def test_paste_reports_clipboard_access_error(capsys):
    # Arrange
    getter = _raising_getter(Exception("Clipboard access denied"))
    # Act
    paste(clipboard_getter=getter)
    # Assert
    assert (
        capsys.readouterr().out
        == "Could not execute clipboard content: Clipboard access denied\n"
    )


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_paste.py
# --------------------------------------------------------------------------------
# def paste(*, clipboard_getter=None, executor=None):
#     import textwrap
#
#     if clipboard_getter is None:
#         import pyperclip
#
#         clipboard_getter = pyperclip.paste
#     if executor is None:
#         executor = exec
#
#     try:
#         clipboard_content = clipboard_getter()
#         clipboard_content = textwrap.dedent(clipboard_content)
#         executor(clipboard_content)
#     except Exception as e:
#         print(f"Could not execute clipboard content: {e}")
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_paste.py
# --------------------------------------------------------------------------------
