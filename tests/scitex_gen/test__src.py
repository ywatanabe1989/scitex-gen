#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-03 02:55:33 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/gen/test__src.py

"""Test suite for scitex_gen._src module."""

import pytest

pytest.importorskip("torch")
import builtins
import contextlib
import subprocess as _real_subprocess

from scitex_gen import _src as _src_mod
from scitex_gen import src


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


class _FakeProcess:
    """Minimal stand-in for a subprocess.Popen instance."""

    def __init__(self, returncode=0):
        self.returncode = returncode
        self.communicate_calls = []

    def communicate(self, input=None):
        self.communicate_calls.append({"input": input})
        return (None, None)


class _FakeInspect:
    """Stand-in for inspect module exposing the symbols _src.py uses."""

    def __init__(self, getsource, real):
        self.getsource = getsource
        self._real = real

    def isclass(self, obj):
        return self._real.isclass(obj)

    def isfunction(self, obj):
        return self._real.isfunction(obj)

    def ismethod(self, obj):
        return self._real.ismethod(obj)


class _FakeSubprocess:
    """Stand-in for subprocess module exposing the symbols _src.py uses."""

    PIPE = _real_subprocess.PIPE

    def __init__(self, popen):
        self.Popen = popen


# Test fixtures
def sample_function():
    """A test function for source code retrieval."""
    # Arrange
    # Act
    # Assert
    return 42


class TestClass:
    """A test class for source code retrieval."""

    def method(self):
        """A test method."""
        return "test"


class TestSrc:
    """Test cases for the src function."""

    def test_src_with_function(self):
        """Test src with a regular function."""
        expected_source = "def sample_function():\n    return 42\n"
        fake_getsource = _Recorder(return_value=expected_source)
        fake_process = _FakeProcess(returncode=0)
        fake_popen = _Recorder(return_value=fake_process)

        fake_inspect = _FakeInspect(fake_getsource, _src_mod.inspect)
        fake_subprocess = _FakeSubprocess(fake_popen)

        with _swap_attr(_src_mod, "inspect", fake_inspect), _swap_attr(
            _src_mod, "subprocess", fake_subprocess
        ):
            src(sample_function)

        assert fake_getsource.call_count == 1
        assert fake_getsource.last_args == (sample_function,)
        assert fake_popen.call_count == 1
        assert fake_popen.last_args == (["less"],)
        assert fake_popen.last_kwargs == {
            "stdin": _real_subprocess.PIPE,
            "encoding": "utf8",
        }
        assert fake_process.communicate_calls == [{"input": expected_source}]

    def test_src_with_class(self):
        """Test src with a class."""
        expected_source = (
            'class TestClass:\n    def method(self):\n        return "test"\n'
        )
        fake_getsource = _Recorder(return_value=expected_source)
        fake_process = _FakeProcess(returncode=0)
        fake_popen = _Recorder(return_value=fake_process)

        fake_inspect = _FakeInspect(fake_getsource, _src_mod.inspect)
        fake_subprocess = _FakeSubprocess(fake_popen)

        with _swap_attr(_src_mod, "inspect", fake_inspect), _swap_attr(
            _src_mod, "subprocess", fake_subprocess
        ):
            src(TestClass)

        assert fake_getsource.call_count == 1
        assert fake_getsource.last_args == (TestClass,)
        assert fake_process.communicate_calls == [{"input": expected_source}]

    def test_src_with_instance(self):
        """Test src with a class instance."""
        instance = TestClass()
        expected_source = (
            'class TestClass:\n    def method(self):\n        return "test"\n'
        )
        fake_getsource = _Recorder(return_value=expected_source)
        fake_process = _FakeProcess(returncode=0)
        fake_popen = _Recorder(return_value=fake_process)

        fake_inspect = _FakeInspect(fake_getsource, _src_mod.inspect)
        fake_subprocess = _FakeSubprocess(fake_popen)

        with _swap_attr(_src_mod, "inspect", fake_inspect), _swap_attr(
            _src_mod, "subprocess", fake_subprocess
        ):
            src(instance)

        # Verify - should get source of the class, not instance
        assert fake_getsource.call_count == 1
        assert fake_getsource.last_args == (TestClass,)
        assert fake_process.communicate_calls == [{"input": expected_source}]

    def test_src_with_method(self):
        """Test src with a method."""
        method = TestClass.method
        expected_source = '    def method(self):\n        return "test"\n'
        fake_getsource = _Recorder(return_value=expected_source)
        fake_process = _FakeProcess(returncode=0)
        fake_popen = _Recorder(return_value=fake_process)

        fake_inspect = _FakeInspect(fake_getsource, _src_mod.inspect)
        fake_subprocess = _FakeSubprocess(fake_popen)

        with _swap_attr(_src_mod, "inspect", fake_inspect), _swap_attr(
            _src_mod, "subprocess", fake_subprocess
        ):
            src(method)

        assert fake_getsource.call_count == 1
        assert fake_getsource.last_args == (method,)
        assert fake_process.communicate_calls == [{"input": expected_source}]

    def test_src_with_builtin_function(self):
        """Test src with a built-in function."""
        fake_getsource = _Recorder(
            side_effect=OSError("could not get source code")
        )
        fake_print = _Recorder()

        fake_inspect = _FakeInspect(fake_getsource, _src_mod.inspect)

        with _swap_attr(_src_mod, "inspect", fake_inspect), _swap_attr(
            builtins, "print", fake_print
        ):
            src(print)

        assert fake_print.call_count >= 1
        error_msg = fake_print.last_args[0]
        assert "Cannot retrieve source code:" in error_msg

    def test_src_with_process_error(self):
        """Test src when less process returns non-zero exit code."""
        fake_getsource = _Recorder(return_value="def test():\n    pass\n")
        fake_process = _FakeProcess(returncode=1)  # Non-zero exit code
        fake_popen = _Recorder(return_value=fake_process)
        fake_print = _Recorder()

        fake_inspect = _FakeInspect(fake_getsource, _src_mod.inspect)
        fake_subprocess = _FakeSubprocess(fake_popen)

        with _swap_attr(_src_mod, "inspect", fake_inspect), _swap_attr(
            _src_mod, "subprocess", fake_subprocess
        ), _swap_attr(builtins, "print", fake_print):
            src(sample_function)

        # Verify error message was printed
        assert fake_print.call_count >= 1
        assert fake_print.calls[-1] == (
            ("Process exited with return code 1",),
            {},
        )

    def test_src_with_subprocess_error(self):
        """Test src when subprocess.Popen raises an error."""
        fake_getsource = _Recorder(return_value="def test():\n    pass\n")
        fake_popen = _Recorder(side_effect=OSError("less command not found"))
        fake_print = _Recorder()

        fake_inspect = _FakeInspect(fake_getsource, _src_mod.inspect)
        fake_subprocess = _FakeSubprocess(fake_popen)

        with _swap_attr(_src_mod, "inspect", fake_inspect), _swap_attr(
            _src_mod, "subprocess", fake_subprocess
        ), _swap_attr(builtins, "print", fake_print):
            src(sample_function)

        assert fake_print.call_count >= 1
        error_msg = fake_print.last_args[0]
        assert "Cannot retrieve source code:" in error_msg


class TestSrcEdgeCases:
    """Test edge cases for the src function."""

    def test_src_with_type_error(self):
        """Test src when inspect.getsource raises TypeError."""
        fake_getsource = _Recorder(
            side_effect=TypeError(
                "Object is not a module, class, method, function, etc."
            )
        )
        fake_print = _Recorder()

        fake_inspect = _FakeInspect(fake_getsource, _src_mod.inspect)

        with _swap_attr(_src_mod, "inspect", fake_inspect), _swap_attr(
            builtins, "print", fake_print
        ):
            src(sample_function)

        assert fake_print.call_count >= 1
        error_msg = fake_print.last_args[0]
        assert "TypeError:" in error_msg

    def test_src_with_unexpected_error(self):
        """Test src with unexpected error."""
        fake_getsource = _Recorder(
            side_effect=RuntimeError("Unexpected error")
        )
        fake_print = _Recorder()

        fake_inspect = _FakeInspect(fake_getsource, _src_mod.inspect)

        with _swap_attr(_src_mod, "inspect", fake_inspect), _swap_attr(
            builtins, "print", fake_print
        ):
            src(sample_function)

        assert fake_print.call_count >= 1
        error_msg = fake_print.last_args[0]
        assert "Error:" in error_msg

    def test_src_with_lambda(self):
        """Test src with a lambda function."""
        test_lambda = lambda x: x * 2
        expected_source = "test_lambda = lambda x: x * 2\n"
        fake_getsource = _Recorder(return_value=expected_source)
        fake_process = _FakeProcess(returncode=0)
        fake_popen = _Recorder(return_value=fake_process)

        fake_inspect = _FakeInspect(fake_getsource, _src_mod.inspect)
        fake_subprocess = _FakeSubprocess(fake_popen)

        with _swap_attr(_src_mod, "inspect", fake_inspect), _swap_attr(
            _src_mod, "subprocess", fake_subprocess
        ):
            src(test_lambda)

        assert fake_getsource.call_count == 1
        assert fake_process.communicate_calls == [{"input": expected_source}]

    def test_src_with_nested_class(self):
        """Test src with a nested class."""

        class OuterClass:
            class InnerClass:
                def inner_method(self):
                    return "inner"

        expected_source = '    class InnerClass:\n        def inner_method(self):\n            return "inner"\n'
        fake_getsource = _Recorder(return_value=expected_source)
        fake_process = _FakeProcess(returncode=0)
        fake_popen = _Recorder(return_value=fake_process)

        fake_inspect = _FakeInspect(fake_getsource, _src_mod.inspect)
        fake_subprocess = _FakeSubprocess(fake_popen)

        with _swap_attr(_src_mod, "inspect", fake_inspect), _swap_attr(
            _src_mod, "subprocess", fake_subprocess
        ):
            src(OuterClass.InnerClass)

        assert fake_getsource.call_count == 1
        assert fake_getsource.last_args == (OuterClass.InnerClass,)


class TestSrcIntegration:
    """Integration tests for src function."""

    def test_src_with_actual_source_retrieval(self):
        """Test src with actual source code retrieval."""
        fake_process = _FakeProcess(returncode=0)
        fake_popen = _Recorder(return_value=fake_process)
        fake_subprocess = _FakeSubprocess(fake_popen)

        # Define a function with known source
        def sample_function(x, y):
            """Sample function for testing."""
            return x + y

        with _swap_attr(_src_mod, "subprocess", fake_subprocess):
            src(sample_function)

        # Verify the actual source was passed to less
        assert fake_popen.call_count == 1
        assert len(fake_process.communicate_calls) == 1

        # The source should contain the function definition
        passed_source = fake_process.communicate_calls[0]["input"]
        assert "def sample_function" in passed_source
        assert "return x + y" in passed_source

    def test_src_preserves_formatting(self):
        """Test that src preserves source code formatting."""
        source_with_formatting = '''def formatted_function():
    """Docstring."""
    # Comment
    if True:
        return 42
    else:
        return 0
'''
        fake_getsource = _Recorder(return_value=source_with_formatting)
        fake_process = _FakeProcess(returncode=0)
        fake_popen = _Recorder(return_value=fake_process)

        fake_inspect = _FakeInspect(fake_getsource, _src_mod.inspect)
        fake_subprocess = _FakeSubprocess(fake_popen)

        with _swap_attr(_src_mod, "inspect", fake_inspect), _swap_attr(
            _src_mod, "subprocess", fake_subprocess
        ):
            src(sample_function)

        passed_source = fake_process.communicate_calls[0]["input"]
        assert passed_source == source_with_formatting


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_src.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-06-13 22:44:28 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/SciTeX-Code/src/scitex/gen/_src.py
# # ----------------------------------------
# import os
#
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# #!./env/bin/python3
#
# import inspect
# import subprocess
#
#
# def src(obj):
#     """
#     Returns the source code of a given object using `less`.
#     Handles functions, classes, class instances, methods, and built-in functions.
#     """
#     # If obj is an instance of a class, get the class of the instance.
#     if (
#         not inspect.isclass(obj)
#         and not inspect.isfunction(obj)
#         and not inspect.ismethod(obj)
#     ):
#         obj = obj.__class__
#
#     try:
#         # Attempt to retrieve the source code
#         source_code = inspect.getsource(obj)
#
#         # Assuming scitex_gen.less is a placeholder for displaying text with `less`
#         # This part of the code is commented out as it seems to be a placeholder
#         # scitex_gen.less(source_code)
#
#         # Open a subprocess to use `less` for displaying the source code
#         process = subprocess.Popen(["less"], stdin=subprocess.PIPE, encoding="utf8")
#         process.communicate(input=source_code)
#         if process.returncode != 0:
#             print(f"Process exited with return code {process.returncode}")
#     except OSError as e:
#         # Handle cases where the source code cannot be retrieved (e.g., built-in functions)
#         print(f"Cannot retrieve source code: {e}")
#     except TypeError as e:
#         # Handle cases where the object type is not supported
#         print(f"TypeError: {e}")
#     except Exception as e:
#         # Handle any other unexpected errors
#         print(f"Error: {e}")
#
#
# # def src(obj):
# #     """
# #     Returns the source code of a given object using `less`.
# #     Handles functions, classes, class instances, and methods.
# #     """
# #     # If obj is an instance of a class, get the class of the instance.
# #     if (
# #         not inspect.isclass(obj)
# #         and not inspect.isfunction(obj)
# #         and not inspect.ismethod(obj)
# #     ):
# #         obj = obj.__class__
#
# #     try:
# #         # Attempt to retrieve the source code
# #         source_code = inspect.getsource(obj)
# #         scitex_gen.less(source_code)
#
# #         # # Open a subprocess to use `less` for displaying the source code
# #         # process = subprocess.Popen(
# #         #     ["less"], stdin=subprocess.PIPE, encoding="utf8"
# #         # )
# #         # process.communicate(input=source_code)
# #         if process.returncode != 0:
# #             print(f"Process exited with return code {process.returncode}")
# #     except TypeError as e:
# #         # Handle cases where the object type is not supported
# #         print(f"TypeError: {e}")
# #     except Exception as e:
# #         # Handle any other unexpected errors
# #         print(f"Error: {e}")
#
# # (YOUR AWESOME CODE)
#
# if __name__ == "__main__":
#     import sys
#
#     import matplotlib.pyplot as plt
#
#     # Start
#     CONFIG, sys.stdout, sys.stderr, plt, CC = scitex.session.start(
#         sys, plt, verbose=False
#     )
#     import sys
#
#     # (YOUR AWESOME CODE)
#     # Close
#     scitex.session.close(CONFIG, verbose=False, notify=False)
#
# """
# /ssh:ywatanabe@444:/home/ywatanabe/proj/entrance/scitex/gen/_def.py
# """
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_src.py
# --------------------------------------------------------------------------------
