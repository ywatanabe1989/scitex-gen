#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-03 02:55:33 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/gen/test__src.py

"""Test suite for scitex_gen._src module.

`src()` retrieves an object's source with the real `inspect.getsource` and
hands it to a pager. We inject a recording pager (a plain list's `append`)
instead of spawning `less` — no mocks — so assertions observe the real source
string that production extracted. Error branches are driven by a hand-rolled
pager that raises, and observed via the real `capsys` fixture.
"""

import inspect

import pytest

pytest.importorskip("torch")

from scitex_gen import src


def sample_function():
    """A sample function for source retrieval."""
    return 42


class SampleClass:
    """A sample class for source retrieval."""

    def method(self):
        return "test"


def _raise(exc):
    def pager(_source):
        raise exc

    return pager


def test_src_passes_function_source_to_pager():
    # Arrange
    captured = []
    # Act
    src(sample_function, pager=captured.append)
    # Assert
    assert captured == [inspect.getsource(sample_function)]


def test_src_passes_class_source_to_pager():
    # Arrange
    captured = []
    # Act
    src(SampleClass, pager=captured.append)
    # Assert
    assert captured == [inspect.getsource(SampleClass)]


def test_src_uses_class_source_for_instance():
    # Arrange
    captured = []
    # Act
    src(SampleClass(), pager=captured.append)
    # Assert
    assert captured == [inspect.getsource(SampleClass)]


def test_src_passes_method_source_to_pager():
    # Arrange
    captured = []
    # Act
    src(SampleClass.method, pager=captured.append)
    # Assert
    assert captured == [inspect.getsource(SampleClass.method)]


def test_src_passes_lambda_source_to_pager():
    # Arrange
    captured = []
    sample_lambda = lambda x: x * 2  # noqa: E731
    # Act
    src(sample_lambda, pager=captured.append)
    # Assert
    assert captured == [inspect.getsource(sample_lambda)]


def test_src_passes_nested_class_source_to_pager():
    # Arrange
    captured = []

    class OuterClass:
        class InnerClass:
            def inner_method(self):
                return "inner"

    # Act
    src(OuterClass.InnerClass, pager=captured.append)
    # Assert
    assert captured == [inspect.getsource(OuterClass.InnerClass)]


def test_src_preserves_source_formatting():
    # Arrange
    captured = []

    def formatted_function():
        """Docstring."""
        # Comment
        if True:
            return 42
        return 0

    # Act
    src(formatted_function, pager=captured.append)
    # Assert
    assert captured[0] == inspect.getsource(formatted_function)


def test_src_source_contains_function_definition():
    # Arrange
    captured = []
    # Act
    src(sample_function, pager=captured.append)
    # Assert
    assert "def sample_function" in captured[0]


def test_src_source_contains_function_body():
    # Arrange
    captured = []
    # Act
    src(sample_function, pager=captured.append)
    # Assert
    assert "return 42" in captured[0]


def test_src_reports_oserror_from_pager(capsys):
    # Arrange
    # Act
    src(sample_function, pager=_raise(OSError("could not get source code")))
    # Assert
    assert capsys.readouterr().out.startswith("Cannot retrieve source code:")


def test_src_reports_typeerror_from_pager(capsys):
    # Arrange
    # Act
    src(sample_function, pager=_raise(TypeError("unsupported object")))
    # Assert
    assert capsys.readouterr().out.startswith("TypeError:")


def test_src_reports_unexpected_error_from_pager(capsys):
    # Arrange
    # Act
    src(sample_function, pager=_raise(RuntimeError("unexpected")))
    # Assert
    assert capsys.readouterr().out.startswith("Error:")


def test_src_reports_typeerror_for_builtin(capsys):
    # Arrange
    # Act
    src(print, pager=lambda _source: None)
    # Assert
    assert capsys.readouterr().out.startswith("TypeError:")


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_src.py
# --------------------------------------------------------------------------------
# def src(obj, *, pager=None):
#     """
#     Returns the source code of a given object using `less`.
#     Handles functions, classes, class instances, methods, and built-in functions.
#     """
#     if pager is None:
#         pager = _less_pager
#
#     # If obj is an instance of a class, get the class of the instance.
#     if (
#         not inspect.isclass(obj)
#         and not inspect.isfunction(obj)
#         and not inspect.ismethod(obj)
#     ):
#         obj = obj.__class__
#
#     try:
#         source_code = inspect.getsource(obj)
#         pager(source_code)
#     except OSError as e:
#         print(f"Cannot retrieve source code: {e}")
#     except TypeError as e:
#         print(f"TypeError: {e}")
#     except Exception as e:
#         print(f"Error: {e}")

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_src.py
# --------------------------------------------------------------------------------
