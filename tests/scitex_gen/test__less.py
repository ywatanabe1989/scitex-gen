#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-03 02:55:22 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/gen/test__less.py

"""Test suite for scitex_gen._less module."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("IPython")
import contextlib
import os
import tempfile

import IPython

from scitex_gen import less


@contextlib.contextmanager
def _swap_attr(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


@contextlib.contextmanager
def _set_env(**kw):
    import os
    saved = {k: os.environ.get(k) for k in kw}
    os.environ.update({k: str(v) for k, v in kw.items()})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _FakeTempFile:
    """Fake NamedTemporaryFile context manager."""

    def __init__(self, name):
        self.name = name
        self.writes = []
        self.write_call_count = 0

    def write(self, data):
        self.writes.append(data)
        self.write_call_count += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


class _FakeTempFileFactory:
    """Factory that returns a _FakeTempFile and records calls."""

    def __init__(self, name):
        self._name = name
        self.file = _FakeTempFile(name)
        self.calls = []  # list of (args, kwargs)

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.file


class _FakeIPython:
    """Fake IPython shell - records system() calls."""

    def __init__(self, system_side_effect=None):
        self.system_calls = []
        self._system_side_effect = system_side_effect

    def system(self, cmd):
        self.system_calls.append(cmd)
        if self._system_side_effect is not None:
            raise self._system_side_effect


class _Recorder:
    """Records calls (used for os.remove)."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class TestLess:
    """Test cases for the less function."""

    def test_less_basic_functionality(self):
        """Test that less properly displays output through IPython system command."""
        # Arrange
        tempfile_factory = _FakeTempFileFactory("/tmp/test_file.txt")
        remove_recorder = _Recorder()
        fake_ipython = _FakeIPython()

        with _swap_attr(tempfile, "NamedTemporaryFile", tempfile_factory), \
             _swap_attr(os, "remove", remove_recorder), \
             _swap_attr(IPython, "get_ipython", lambda: fake_ipython):
            # Act
            test_output = "Hello, World!"
            less(test_output)

        # Assert
        assert tempfile_factory.calls == [((), {"delete": False, "mode": "w+t"})]
        assert tempfile_factory.file.writes == [test_output]
        assert fake_ipython.system_calls == [f"less {tempfile_factory.file.name}"]
        assert remove_recorder.calls == [((tempfile_factory.file.name,), {})]

    def test_less_with_multiline_output(self):
        """Test less with multi-line text output."""
        # Arrange
        tempfile_factory = _FakeTempFileFactory("/tmp/test_multiline.txt")
        remove_recorder = _Recorder()
        fake_ipython = _FakeIPython()

        with _swap_attr(tempfile, "NamedTemporaryFile", tempfile_factory), \
             _swap_attr(os, "remove", remove_recorder), \
             _swap_attr(IPython, "get_ipython", lambda: fake_ipython):
            # Act
            test_output = """Line 1
Line 2
Line 3
This is a longer line with more content
Last line"""
            less(test_output)

        # Assert
        assert tempfile_factory.file.writes == [test_output]
        assert fake_ipython.system_calls == [f"less {tempfile_factory.file.name}"]
        assert remove_recorder.calls == [((tempfile_factory.file.name,), {})]

    def test_less_with_special_characters(self):
        """Test less with special characters and unicode."""
        # Arrange
        tempfile_factory = _FakeTempFileFactory("/tmp/test_special.txt")
        remove_recorder = _Recorder()
        fake_ipython = _FakeIPython()

        with _swap_attr(tempfile, "NamedTemporaryFile", tempfile_factory), \
             _swap_attr(os, "remove", remove_recorder), \
             _swap_attr(IPython, "get_ipython", lambda: fake_ipython):
            # Act
            test_output = "Special chars: @#$%^&*() Unicode: 你好世界 Émojis: 🚀💻"
            less(test_output)

        # Assert
        assert tempfile_factory.file.writes == [test_output]
        assert len(fake_ipython.system_calls) == 1

    def test_less_temp_file_cleanup(self):
        """Test that temporary file is created and cleaned up properly."""
        # Arrange
        tempfile_factory = _FakeTempFileFactory("/tmp/test_cleanup.txt")
        remove_recorder = _Recorder()
        fake_ipython = _FakeIPython()

        with _swap_attr(tempfile, "NamedTemporaryFile", tempfile_factory), \
             _swap_attr(os, "remove", remove_recorder), \
             _swap_attr(IPython, "get_ipython", lambda: fake_ipython):
            # Act
            less("Test output")

        # Assert
        assert remove_recorder.calls == [((tempfile_factory.file.name,), {})]

    def test_less_error_no_cleanup(self):
        """Test that cleanup is NOT called when system command fails."""
        # Arrange
        tempfile_factory = _FakeTempFileFactory("/tmp/test_no_cleanup.txt")
        remove_recorder = _Recorder()
        fake_ipython = _FakeIPython(
            system_side_effect=Exception("System command failed")
        )

        with _swap_attr(tempfile, "NamedTemporaryFile", tempfile_factory), \
             _swap_attr(os, "remove", remove_recorder), \
             _swap_attr(IPython, "get_ipython", lambda: fake_ipython):
            # Act / Assert
            with pytest.raises(Exception, match="System command failed"):
                less("Test output")

        # Assert cleanup NOT called
        assert remove_recorder.calls == []

    def test_less_error_handling(self):
        """Test error handling when IPython is not available."""
        # Arrange - get_ipython returns None
        with _swap_attr(IPython, "get_ipython", lambda: None):
            # Act / Assert
            with pytest.raises(AttributeError):
                less("Test output")


class TestLessIPythonIntegration:
    """Test cases for IPython-specific functionality."""

    def test_less_in_ipython_environment(self):
        """Test less when running in actual IPython environment."""
        # Arrange
        tempfile_factory = _FakeTempFileFactory("/tmp/ipython_test.txt")
        remove_recorder = _Recorder()
        fake_ipython = _FakeIPython()

        with _swap_attr(tempfile, "NamedTemporaryFile", tempfile_factory), \
             _swap_attr(os, "remove", remove_recorder), \
             _swap_attr(IPython, "get_ipython", lambda: fake_ipython):
            # Act
            less("IPython test content")

        # Assert
        assert len(fake_ipython.system_calls) == 1

    def test_less_system_command_execution(self):
        """Test that the system command is called correctly."""
        # Arrange
        test_filename = "/tmp/less_test_123.txt"
        tempfile_factory = _FakeTempFileFactory(test_filename)
        remove_recorder = _Recorder()
        fake_ipython = _FakeIPython()

        with _swap_attr(tempfile, "NamedTemporaryFile", tempfile_factory), \
             _swap_attr(os, "remove", remove_recorder), \
             _swap_attr(IPython, "get_ipython", lambda: fake_ipython):
            # Act
            less("Command test")

        # Assert
        expected_command = f"less {test_filename}"
        assert fake_ipython.system_calls == [expected_command]


class TestLessEdgeCases:
    """Test edge cases for the less function."""

    def test_less_empty_output(self):
        """Test less with empty string output."""
        # Arrange
        tempfile_factory = _FakeTempFileFactory("/tmp/empty.txt")
        remove_recorder = _Recorder()
        fake_ipython = _FakeIPython()

        with _swap_attr(tempfile, "NamedTemporaryFile", tempfile_factory), \
             _swap_attr(os, "remove", remove_recorder), \
             _swap_attr(IPython, "get_ipython", lambda: fake_ipython):
            # Act
            less("")

        # Assert
        assert tempfile_factory.file.writes == [""]
        assert len(fake_ipython.system_calls) == 1
        assert len(remove_recorder.calls) == 1

    def test_less_very_large_output(self):
        """Test less with very large output."""
        # Arrange
        tempfile_factory = _FakeTempFileFactory("/tmp/large.txt")
        remove_recorder = _Recorder()
        fake_ipython = _FakeIPython()

        with _swap_attr(tempfile, "NamedTemporaryFile", tempfile_factory), \
             _swap_attr(os, "remove", remove_recorder), \
             _swap_attr(IPython, "get_ipython", lambda: fake_ipython):
            # Act
            large_output = "x" * 10000 + "\n" + "y" * 10000
            less(large_output)

        # Assert
        assert tempfile_factory.file.writes == [large_output]
        assert len(fake_ipython.system_calls) == 1
        assert len(remove_recorder.calls) == 1


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
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_less.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-03 02:11:18 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/gen/_less.py
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-21 12:05:35"
# # Author: Yusuke Watanabe (ywatanabe@scitex.ai)
#
# """
# This script does XYZ.
# """
#
# import sys
#
# import matplotlib.pyplot as plt
# import scitex
#
# # Imports
#
# # # Config
# # CONFIG = scitex_gen.load_configs()
#
#
# # Functions
# def less(output):
#     """
#     Print the given output using `less` in an IPython or IPdb session.
#     """
#     import os
#     import tempfile
#
#     from IPython import get_ipython
#
#     # Create a temporary file to hold the output
#     with tempfile.NamedTemporaryFile(delete=False, mode="w+t") as tmpfile:
#         # Write the output to the temporary file
#         tmpfile.write(output)
#         tmpfile_name = tmpfile.name
#
#     # Use IPython's system command access to pipe the content of the temporary file to `less`
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
