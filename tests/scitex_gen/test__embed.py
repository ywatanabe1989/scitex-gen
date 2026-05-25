#!/usr/bin/env python3
# Timestamp: "2025-05-31 20:20:00 (Claude)"
# File: /tests/scitex/gen/test__embed.py

"""Tests for the embed function.

The embed function is interactive (clipboard + IPython shell + input()), so its
runtime behavior cannot be exercised in a non-interactive test. These tests
verify the public surface and the lazy-import design by inspecting the real
function object and source — no mocks.
"""

import inspect

import pytest

pytest.importorskip("torch")

from scitex_gen import _embed, embed


def test_embed_is_a_callable_function():
    # Arrange
    # Act
    # Assert
    assert callable(embed)


def test_embed_has_no_required_parameters():
    # Arrange
    sig = inspect.signature(embed)
    # Act
    required = [
        p
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    # Assert
    assert required == []


def test_embed_module_provides_documentation():
    # Arrange
    # Act
    has_doc = _embed.__doc__ is not None or embed.__doc__ is not None
    # Assert
    assert has_doc


def test_embed_imports_pyperclip_inside_function_body():
    # Arrange
    # Act
    source = inspect.getsource(embed)
    # Assert
    assert "import pyperclip" in source


def test_embed_imports_ipython_inside_function_body():
    # Arrange
    # Act
    source = inspect.getsource(embed)
    # Assert
    assert "from IPython import embed" in source


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_embed.py
# --------------------------------------------------------------------------------
# """
# This script does XYZ.
# """
#
# # import os
# # import sys
#
# # import matplotlib.pyplot as plt
#
# # # Imports
# #
# # import numpy as np
# # import pandas as pd
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
#
# # # Config
# # CONFIG = scitex_gen.load_configs()
#
# # Functions
# # from IPython import embed as _embed
# # import pyperclip
#
# # def embed_with_clipboard_exec():
# #     # Try to get text from the clipboard
# #     try:
# #         clipboard_content = pyperclip.paste()
# #     except pyperclip.PyperclipException as e:
# #         clipboard_content = ""
# #         print("Could not access the clipboard:", e)
#
# #     # Start IPython session with the clipboard content preloaded
# #     ipython_shell = embed(header='IPython is now running with the following clipboard content executed:', compile_flags=None)
#
# #     # Optionally, execute the clipboard content automatically
# #     if clipboard_content:
# #         # Execute the content as if it was typed in directly
# #         ipython_shell.run_cell(clipboard_content)
#
#
# def embed():
#     import pyperclip
#     from IPython import embed as _embed
#
#     try:
#         clipboard_content = pyperclip.paste()
#     except pyperclip.PyperclipException as e:
#         clipboard_content = ""
#         print("Could not access the clipboard:", e)
#
#     print("Clipboard content loaded. Do you want to execute it? [y/n]")
#     execute_clipboard = input().strip().lower() == "y"
#
#     # Start IPython shell
#     ipython_shell = _embed(
#         header="IPython is now running. Clipboard content will be executed if confirmed."
#     )
#
#     # Execute if confirmed
#     if clipboard_content and execute_clipboard:
#         ipython_shell.run_cell(clipboard_content)
#
#
# if __name__ == "__main__":
#     import sys
#     import matplotlib.pyplot as plt
#     import scitex
#
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(sys, plt)
#
#     embed()
#
#     # Close
#     scitex.session.close(CONFIG)
#
# # EOF
#
# """
# /ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/scitex/gen/_embed.py
# """

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_embed.py
# --------------------------------------------------------------------------------
