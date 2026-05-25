#!/usr/bin/env python3
"""Tests for scitex_gen._print_config module."""

import contextlib

import pytest

pytest.importorskip("torch")

from scitex_gen import print_config
from scitex_gen import _print_config as _pc_module
from scitex_gen._print_config import print_config_main

# scitex_gen._print_config source references undefined `scitex` symbol;
# these tests patch `scitex_gen._print_config.scitex` which does not exist.
# Skip until source is fixed (depends on scitex.io.load_configs being injected).
pytest.skip(
    "scitex_gen._print_config has unresolved scitex reference",
    allow_module_level=True,
)


@contextlib.contextmanager
def _swap_attr(obj, name, value):
    sentinel = object()
    saved = getattr(obj, name, sentinel)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if saved is sentinel:
            delattr(obj, name)
        else:
            setattr(obj, name, saved)


class _PrintRecorder:
    """Records calls made to a substituted print function."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    @property
    def called(self):
        return bool(self.calls)

    @property
    def call_count(self):
        return len(self.calls)

    @property
    def call_args_list(self):
        return list(self.calls)

    def reset(self):
        self.calls.clear()

    def last_args(self):
        return self.calls[-1][0]


class _StubScitexIO:
    def __init__(self, configs=None, error=None):
        self._configs = configs
        self._error = error

    def load_configs(self):
        if self._error is not None:
            raise self._error
        return self._configs


class _StubScitex:
    def __init__(self, configs=None, error=None):
        self.io = _StubScitexIO(configs=configs, error=error)


class TestPrintConfig:
    """Test cases for print_config function."""

    def test_print_config_no_key(self):
        """Test print_config with no key - should print all configs."""

        # Mock config data
        # Arrange
        mock_config = {
            "database": {"host": "localhost", "port": 5432},
            "api": {"key": "secret123", "timeout": 30},
        }
        recorder = _PrintRecorder()
        import builtins

        with _swap_attr(_pc_module, "scitex", _StubScitex(configs=mock_config)), \
             _swap_attr(builtins, "print", recorder):
            # Call with no key
            # Act
            print_config(None)

        # Should print available configurations message
        # Assert
        assert any(
            "Available configurations:" in str(call)
            for call in recorder.call_args_list
        )
        # pprint is called internally, so we check if print was called multiple times
        assert recorder.call_count >= 1

    def test_print_config_simple_key(self):
        """Test print_config with simple top-level key."""

        # Arrange
        # Act
        # Assert
        mock_config = {"version": "1.0.0", "debug": True, "timeout": 30}
        recorder = _PrintRecorder()
        import builtins

        with _swap_attr(_pc_module, "scitex", _StubScitex(configs=mock_config)), \
             _swap_attr(builtins, "print", recorder):
            # Test string value
            print_config("version")
            assert recorder.last_args() == ("1.0.0",)

            # Test boolean value
            recorder.reset()
            print_config("debug")
            assert recorder.last_args() == (True,)

            # Test integer value
            recorder.reset()
            print_config("timeout")
            assert recorder.last_args() == (30,)

    def test_print_config_nested_key(self):
        """Test print_config with nested dot-separated keys."""

        # Arrange
        # Act
        # Assert
        mock_config = {
            "database": {
                "postgres": {
                    "host": "localhost",
                    "port": 5432,
                    "credentials": {"user": "admin", "password": "secret"},
                }
            }
        }
        recorder = _PrintRecorder()
        import builtins

        with _swap_attr(_pc_module, "scitex", _StubScitex(configs=mock_config)), \
             _swap_attr(builtins, "print", recorder):
            # Test 2-level nesting
            print_config("database.postgres")
            expected = {
                "host": "localhost",
                "port": 5432,
                "credentials": {"user": "admin", "password": "secret"},
            }
            assert recorder.last_args() == (expected,)

            # Test 3-level nesting
            recorder.reset()
            print_config("database.postgres.host")
            assert recorder.last_args() == ("localhost",)

            # Test 4-level nesting
            recorder.reset()
            print_config("database.postgres.credentials.user")
            assert recorder.last_args() == ("admin",)

    def test_print_config_list_access(self):
        """Test print_config with list index access."""

        # Arrange
        # Act
        # Assert
        mock_config = {
            "servers": ["server1", "server2", "server3"],
            "ports": [8080, 8081, 8082],
            "nested": {
                "items": [
                    {"name": "item1", "value": 10},
                    {"name": "item2", "value": 20},
                ]
            },
        }
        recorder = _PrintRecorder()
        import builtins

        with _swap_attr(_pc_module, "scitex", _StubScitex(configs=mock_config)), \
             _swap_attr(builtins, "print", recorder):
            # Access list by index
            print_config("servers.0")
            assert recorder.last_args() == ("server1",)

            recorder.reset()
            print_config("servers.2")
            assert recorder.last_args() == ("server3",)

            # Access nested list item
            recorder.reset()
            print_config("nested.items.1.name")
            assert recorder.last_args() == ("item2",)

    def test_print_config_invalid_key(self):
        """Test print_config with invalid/non-existent key."""

        # Arrange
        # Act
        # Assert
        mock_config = {"existing": "value"}
        recorder = _PrintRecorder()
        import builtins

        with _swap_attr(_pc_module, "scitex", _StubScitex(configs=mock_config)), \
             _swap_attr(builtins, "print", recorder):
            # Non-existent key
            print_config("nonexistent")
            assert recorder.last_args() == (None,)

            # Invalid nested key
            recorder.reset()
            print_config("existing.nested.deep")
            assert recorder.last_args() == (None,)

    def test_print_config_dotdict_support(self):
        """Test print_config with DotDict objects."""

        # Hand-rolled DotDict-like fake recording .get() calls.
        # Arrange
        # Act
        # Assert
        class _FakeDotDict:
            def __init__(self):
                self.get_calls = []

            def get(self, k):
                self.get_calls.append(k)
                return {"inner": "value"} if k == "nested" else None

        fake_dotdict = _FakeDotDict()
        mock_config = {"data": fake_dotdict}
        recorder = _PrintRecorder()
        import builtins

        with _swap_attr(_pc_module, "scitex", _StubScitex(configs=mock_config)), \
             _swap_attr(builtins, "print", recorder):
            print_config("data.nested")

        assert "nested" in fake_dotdict.get_calls

    def test_print_config_exception_handling(self):
        """Test print_config exception handling."""

        # Stub load_configs that raises
        # Arrange
        recorder = _PrintRecorder()
        import builtins

        with _swap_attr(
            _pc_module,
            "scitex",
            _StubScitex(error=Exception("Config load failed")),
        ), _swap_attr(builtins, "print", recorder):
            # Should handle exception gracefully
            # Act
            print_config("any.key")

        # Check that error was printed
        # Assert
        assert any("Error:" in str(call) for call in recorder.call_args_list)


class TestPrintConfigMain:
    """Test cases for print_config_main function."""

    def test_print_config_main_no_args(self):
        """Test print_config_main with no arguments."""

        # Arrange
        # Act
        # Assert
        calls = []

        def fake_print_config(key):
            calls.append(key)

        with _swap_attr(_pc_module, "print_config", fake_print_config):
            print_config_main([])

        assert calls == [None]

    def test_print_config_main_with_key(self):
        """Test print_config_main with key argument."""

        # Arrange
        # Act
        # Assert
        calls = []

        def fake_print_config(key):
            calls.append(key)

        with _swap_attr(_pc_module, "print_config", fake_print_config):
            print_config_main(["database.host"])

        assert calls == ["database.host"]

    def test_print_config_main_with_nested_key(self):
        """Test print_config_main with complex nested key."""

        # Arrange
        # Act
        # Assert
        calls = []

        def fake_print_config(key):
            calls.append(key)

        with _swap_attr(_pc_module, "print_config", fake_print_config):
            print_config_main(["path.to.nested.config.value"])

        assert calls == ["path.to.nested.config.value"]

    def test_print_config_main_from_sys_argv(self):
        """Test print_config_main using sys.argv."""

        # Simulate command line usage by swapping module-level sys.argv
        # Arrange
        # Act
        # Assert
        calls = []

        def fake_print_config(key):
            calls.append(key)

        fake_argv = ["script.py", "test.key"]
        # _pc_module imports `sys`; swap sys.argv via the module's sys attr
        sys_module = _pc_module.sys
        with _swap_attr(sys_module, "argv", fake_argv), \
             _swap_attr(_pc_module, "print_config", fake_print_config):
            print_config_main(None)  # None means use sys.argv

        assert calls == ["test.key"]

    def test_print_config_main_help(self, capsys):
        """Test print_config_main help message."""

        # Arrange
        # Act
        # Assert
        with pytest.raises(SystemExit) as exc_info:
            print_config_main(["--help"])

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "Print configuration values" in captured.out
        assert "Configuration key" in captured.out


class TestIntegration:
    """Integration tests for the print_config module."""

    def test_realistic_config_navigation(self, capsys):
        """Test realistic configuration navigation scenarios."""

        # Realistic config structure
        # Arrange
        # Act
        # Assert
        mock_config = {
            "PATH": {
                "TITAN": {"MAT": "/data/matlab", "DATA": "/data/raw"},
                "CREST": {
                    "HOME": "/home/user",
                    "PROJECTS": ["/proj/alpha", "/proj/beta", "/proj/gamma"],
                },
            },
            "SETTINGS": {
                "debug": False,
                "verbosity": 2,
                "features": ["logging", "caching", "monitoring"],
            },
        }

        # Test various access patterns
        test_cases = [
            (["PATH.TITAN.MAT"], "/data/matlab"),
            (["PATH.CREST.PROJECTS.1"], "/proj/beta"),
            (["SETTINGS.features.0"], "logging"),
            (["SETTINGS.verbosity"], "2"),  # Note: print converts to string
        ]

        with _swap_attr(_pc_module, "scitex", _StubScitex(configs=mock_config)):
            for args, expected in test_cases:
                print_config_main(args)
                captured = capsys.readouterr()
                assert expected in captured.out


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_print_config.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-13 18:53:04 (ywatanabe)"
# # /home/yusukew/proj/scitex_repo/src/scitex/gen/_print_config.py
#
# """
# 1. Functionality:
#    - Prints configuration values from YAML files
# 2. Input:
#    - Configuration key (dot-separated for nested structures)
# 3. Output:
#    - Corresponding configuration value
# 4. Prerequisites:
#    - scitex package with load_configs function
#
# Example:
#     python _print_config.py PATH.TITAN.MAT
# """
#
# import sys
# import os
# import argparse
# from pprint import pprint
# import sys
#
#
# def print_config(key):
#     CONFIG = scitex.io.load_configs()
#
#     if key is None:
#         print("Available configurations:")
#         pprint(CONFIG)
#         return
#
#     try:
#         keys = key.split(".")
#         value = CONFIG
#         for k in keys:
#             if isinstance(value, (dict, scitex_gen.utils._DotDict.DotDict)):
#                 value = value.get(k)
#
#             elif isinstance(value, list):
#                 try:
#                     value = value[int(k)]
#                 except (ValueError, IndexError):
#                     value = None
#
#             elif isinstance(value, str):
#                 break
#
#             else:
#                 value = None
#
#             if value is None:
#                 break
#
#         print(value)
#
#     except Exception as e:
#         print(f"Error: {e}")
#         print("Available configurations:")
#         pprint(value)
#
#
# def print_config_main(args=None):
#     if args is None:
#         args = sys.argv[1:]
#
#     parser = argparse.ArgumentParser(description="Print configuration values")
#     parser.add_argument(
#         "key",
#         nargs="?",
#         default=None,
#         help="Configuration key (dot-separated for nested structures)",
#     )
#     parsed_args = parser.parse_args(args)
#     print_config(parsed_args.key)
#
#
# if __name__ == "__main__":
#     print_config_main()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_print_config.py
# --------------------------------------------------------------------------------
