#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-03 02:13:54 (ywatanabe)"
# File: ./scitex_repo/src/scitex/gen/_paste.py
def paste(*, clipboard_getter=None, executor=None):
    """Read code from the clipboard, dedent it, and execute it.

    Parameters
    ----------
    clipboard_getter : callable, optional
        Zero-argument callable returning the clipboard text. Defaults to
        ``pyperclip.paste``; injectable so callers (and tests) can supply text
        without touching a real clipboard.
    executor : callable, optional
        Single-argument callable that executes the (dedented) code string.
        Defaults to the built-in ``exec``.
    """
    import textwrap

    if clipboard_getter is None:
        import pyperclip

        clipboard_getter = pyperclip.paste
    if executor is None:
        executor = exec

    try:
        clipboard_content = clipboard_getter()
        clipboard_content = textwrap.dedent(clipboard_content)
        executor(clipboard_content)
    except Exception as e:
        print(f"Could not execute clipboard content: {e}")


# EOF
