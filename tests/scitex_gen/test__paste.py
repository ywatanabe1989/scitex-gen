#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-03 02:55:27 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/gen/test__paste.py

"""Test suite for scitex_gen._paste module."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("pyperclip")
import builtins
import contextlib
import textwrap

import pyperclip

from scitex_gen import paste


@contextlib.contextmanager
def _swap_attr(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


class _Recorder:
    """Minimal recorder that captures positional and keyword call arguments."""

    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
        self.calls = []  # list of (args, kwargs)

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.side_effect is not None:
            if isinstance(self.side_effect, BaseException) or (
                isinstance(self.side_effect, type)
                and issubclass(self.side_effect, BaseException)
            ):
                raise self.side_effect
            return self.side_effect(*args, **kwargs)
        return self.return_value

    @property
    def call_count(self):
        return len(self.calls)

    @property
    def last_args(self):
        return self.calls[-1][0]

    @property
    def last_kwargs(self):
        return self.calls[-1][1]


class TestPaste:
    """Test cases for the paste function."""

    def test_paste_basic_functionality(self):
        """Test basic paste and execution functionality."""
        clipboard_content = "print('Hello from clipboard')"
        fake_paste = _Recorder(return_value=clipboard_content)
        fake_exec = _Recorder()

        with _swap_attr(pyperclip, "paste", fake_paste), _swap_attr(
            builtins, "exec", fake_exec
        ):
            paste()

        assert fake_paste.call_count == 1
        assert fake_exec.call_count == 1
        assert fake_exec.last_args == (clipboard_content,)

    def test_paste_with_indented_code(self):
        """Test paste with indented code (dedenting)."""
        clipboard_content = """
            def hello():
                print('Hello')
                return 42

            result = hello()
        """
        fake_paste = _Recorder(return_value=clipboard_content)
        fake_exec = _Recorder()
        expected_dedented = textwrap.dedent(clipboard_content)

        with _swap_attr(pyperclip, "paste", fake_paste), _swap_attr(
            builtins, "exec", fake_exec
        ):
            paste()

        assert fake_exec.call_count == 1
        assert fake_exec.last_args == (expected_dedented,)

    def test_paste_multiline_code(self):
        """Test paste with multiline code."""
        clipboard_content = """
x = 10
y = 20
z = x + y
print(f'Result: {z}')
"""
        fake_paste = _Recorder(return_value=clipboard_content)
        fake_exec = _Recorder()

        with _swap_attr(pyperclip, "paste", fake_paste), _swap_attr(
            builtins, "exec", fake_exec
        ):
            paste()

        assert fake_paste.call_count == 1
        assert fake_exec.call_count == 1
        assert fake_exec.last_args == (textwrap.dedent(clipboard_content),)

    def test_paste_syntax_error(self):
        """Test paste with code that has syntax errors."""
        clipboard_content = "print('Missing closing quote"
        fake_paste = _Recorder(return_value=clipboard_content)
        fake_exec = _Recorder(
            side_effect=SyntaxError("EOL while scanning string literal")
        )
        fake_print = _Recorder()

        with _swap_attr(pyperclip, "paste", fake_paste), _swap_attr(
            builtins, "exec", fake_exec
        ), _swap_attr(builtins, "print", fake_print):
            paste()

        assert fake_print.call_count == 1
        error_msg = fake_print.last_args[0]
        assert "Could not execute clipboard content:" in error_msg
        assert "EOL while scanning string literal" in error_msg

    def test_paste_runtime_error(self):
        """Test paste with code that raises runtime errors."""
        clipboard_content = "1 / 0"
        fake_paste = _Recorder(return_value=clipboard_content)
        fake_exec = _Recorder(side_effect=ZeroDivisionError("division by zero"))
        fake_print = _Recorder()

        with _swap_attr(pyperclip, "paste", fake_paste), _swap_attr(
            builtins, "exec", fake_exec
        ), _swap_attr(builtins, "print", fake_print):
            paste()

        assert fake_print.call_count == 1
        error_msg = fake_print.last_args[0]
        assert "Could not execute clipboard content:" in error_msg
        assert "division by zero" in error_msg

    def test_paste_empty_clipboard(self):
        """Test paste with empty clipboard."""
        fake_paste = _Recorder(return_value="")
        fake_exec = _Recorder()

        with _swap_attr(pyperclip, "paste", fake_paste), _swap_attr(
            builtins, "exec", fake_exec
        ):
            paste()

        assert fake_exec.call_count == 1
        assert fake_exec.last_args == ("",)

    def test_paste_whitespace_only(self):
        """Test paste with whitespace-only content."""
        clipboard_content = "   \n\t  \n   "
        fake_paste = _Recorder(return_value=clipboard_content)
        fake_exec = _Recorder()

        with _swap_attr(pyperclip, "paste", fake_paste), _swap_attr(
            builtins, "exec", fake_exec
        ):
            paste()

        assert fake_exec.call_count == 1
        assert fake_exec.last_args == (textwrap.dedent(clipboard_content),)

    def test_paste_clipboard_access_error(self):
        """Test paste when clipboard access fails."""
        fake_paste = _Recorder(side_effect=Exception("Clipboard access denied"))
        fake_print = _Recorder()

        with _swap_attr(pyperclip, "paste", fake_paste), _swap_attr(
            builtins, "print", fake_print
        ):
            paste()

        assert fake_print.call_count == 1
        error_msg = fake_print.last_args[0]
        assert "Could not execute clipboard content:" in error_msg
        assert "Clipboard access denied" in error_msg


class TestPasteEdgeCases:
    """Test edge cases for the paste function."""

    def test_paste_with_unicode(self):
        """Test paste with unicode characters."""
        clipboard_content = "print('Hello 世界! 🚀')"
        fake_paste = _Recorder(return_value=clipboard_content)
        fake_exec = _Recorder()

        with _swap_attr(pyperclip, "paste", fake_paste), _swap_attr(
            builtins, "exec", fake_exec
        ):
            paste()

        assert fake_exec.call_count == 1
        assert fake_exec.last_args == (clipboard_content,)

    def test_paste_with_complex_indentation(self):
        """Test paste with complex mixed indentation."""
        clipboard_content = """
            class MyClass:
                def __init__(self):
                    self.value = 42

                def method(self):
                    if self.value > 0:
                        print("Positive")
                    else:
                        print("Non-positive")
        """
        fake_paste = _Recorder(return_value=clipboard_content)
        fake_exec = _Recorder()

        with _swap_attr(pyperclip, "paste", fake_paste), _swap_attr(
            builtins, "exec", fake_exec
        ):
            paste()

        expected = textwrap.dedent(clipboard_content)
        assert fake_exec.call_count == 1
        assert fake_exec.last_args == (expected,)

    def test_paste_with_import_error(self):
        """Test paste with code that raises ImportError."""
        clipboard_content = "import nonexistent_module"
        fake_paste = _Recorder(return_value=clipboard_content)
        fake_exec = _Recorder(
            side_effect=ImportError("No module named 'nonexistent_module'")
        )
        fake_print = _Recorder()

        with _swap_attr(pyperclip, "paste", fake_paste), _swap_attr(
            builtins, "exec", fake_exec
        ), _swap_attr(builtins, "print", fake_print):
            paste()

        assert fake_print.call_count == 1
        error_msg = fake_print.last_args[0]
        assert "Could not execute clipboard content:" in error_msg
        assert "No module named 'nonexistent_module'" in error_msg


class TestPasteIntegration:
    """Integration tests for paste function."""

    def test_paste_executes_in_correct_namespace(self):
        """Test that pasted code executes in the correct namespace."""
        clipboard_content = "test_var = 123"
        fake_paste = _Recorder(return_value=clipboard_content)
        fake_exec = _Recorder()

        with _swap_attr(pyperclip, "paste", fake_paste), _swap_attr(
            builtins, "exec", fake_exec
        ):
            paste()

        assert fake_exec.call_count == 1
        assert fake_exec.last_args == (clipboard_content,)

    def test_paste_preserves_line_endings(self):
        """Test that paste preserves different line ending styles."""
        # Test with Unix line endings
        unix_content = "line1\nline2\nline3"
        fake_paste = _Recorder(return_value=unix_content)
        fake_exec = _Recorder()

        with _swap_attr(pyperclip, "paste", fake_paste), _swap_attr(
            builtins, "exec", fake_exec
        ):
            paste()
        assert fake_exec.last_args == (unix_content,)

        # Test with Windows line endings
        windows_content = "line1\r\nline2\r\nline3"
        fake_paste2 = _Recorder(return_value=windows_content)

        with _swap_attr(pyperclip, "paste", fake_paste2), _swap_attr(
            builtins, "exec", fake_exec
        ):
            paste()
        # textwrap.dedent should handle this correctly
        assert fake_exec.last_args == (textwrap.dedent(windows_content),)


def test_main_calls_main():
    """Main function for running tests."""
    # Arrange
    # Act
    # Assert
    pytest.main([__file__, "-xvs"])


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_paste.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-03 02:13:54 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/gen/_paste.py
# def paste():
#     import textwrap
#
#     import pyperclip
#
#     try:
#         clipboard_content = pyperclip.paste()
#         clipboard_content = textwrap.dedent(clipboard_content)
#         exec(clipboard_content)
#     except Exception as e:
#         print(f"Could not execute clipboard content: {e}")
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_paste.py
# --------------------------------------------------------------------------------
