#!/usr/bin/env python3
"""Tests for scitex_gen._print_config module.

`print_config` loads a configuration mapping and prints the value addressed by
a dot-separated key. The loader is injected (no mocks): tests pass a real dict
via the `config_loader` keyword and observe the real stdout with `capsys`.
"""

import sys

import pytest

pytest.importorskip("torch")

from scitex_gen import print_config
from scitex_gen._print_config import print_config_main


@pytest.fixture
def restore_sys_argv():
    saved = sys.argv[:]
    try:
        yield
    finally:
        sys.argv = saved


@pytest.fixture
def realistic_config():
    return {
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


# --- print_config ---------------------------------------------------------


def test_print_config_with_no_key_prints_header(capsys):
    # Arrange
    config = {"database": {"host": "localhost"}}
    # Act
    print_config(None, config_loader=lambda: config)
    # Assert
    assert "Available configurations:" in capsys.readouterr().out


def test_print_config_prints_string_value(capsys):
    # Arrange
    # Act
    print_config("version", config_loader=lambda: {"version": "1.0.0"})
    # Assert
    assert capsys.readouterr().out == "1.0.0\n"


def test_print_config_prints_boolean_value(capsys):
    # Arrange
    # Act
    print_config("debug", config_loader=lambda: {"debug": True})
    # Assert
    assert capsys.readouterr().out == "True\n"


def test_print_config_prints_integer_value(capsys):
    # Arrange
    # Act
    print_config("timeout", config_loader=lambda: {"timeout": 30})
    # Assert
    assert capsys.readouterr().out == "30\n"


def test_print_config_navigates_to_nested_scalar(capsys):
    # Arrange
    config = {"database": {"postgres": {"host": "localhost", "port": 5432}}}
    # Act
    print_config("database.postgres.host", config_loader=lambda: config)
    # Assert
    assert capsys.readouterr().out == "localhost\n"


def test_print_config_navigates_to_deeply_nested_scalar(capsys):
    # Arrange
    config = {"database": {"postgres": {"credentials": {"user": "admin"}}}}
    # Act
    print_config("database.postgres.credentials.user", config_loader=lambda: config)
    # Assert
    assert capsys.readouterr().out == "admin\n"


def test_print_config_accesses_first_list_item(capsys):
    # Arrange
    config = {"servers": ["server1", "server2", "server3"]}
    # Act
    print_config("servers.0", config_loader=lambda: config)
    # Assert
    assert capsys.readouterr().out == "server1\n"


def test_print_config_accesses_last_list_item(capsys):
    # Arrange
    config = {"servers": ["server1", "server2", "server3"]}
    # Act
    print_config("servers.2", config_loader=lambda: config)
    # Assert
    assert capsys.readouterr().out == "server3\n"


def test_print_config_accesses_nested_list_item_field(capsys):
    # Arrange
    config = {"nested": {"items": [{"name": "item1"}, {"name": "item2"}]}}
    # Act
    print_config("nested.items.1.name", config_loader=lambda: config)
    # Assert
    assert capsys.readouterr().out == "item2\n"


def test_print_config_prints_none_for_missing_key(capsys):
    # Arrange
    # Act
    print_config("nonexistent", config_loader=lambda: {"existing": "value"})
    # Assert
    assert capsys.readouterr().out == "None\n"


def test_print_config_stops_descending_at_string_value(capsys):
    # Arrange
    # Act
    print_config("existing.nested.deep", config_loader=lambda: {"existing": "value"})
    # Assert
    assert capsys.readouterr().out == "value\n"


# --- print_config_main ----------------------------------------------------


def test_main_with_no_args_prints_header(capsys):
    # Arrange
    # Act
    print_config_main([], config_loader=lambda: {"a": 1})
    # Assert
    assert "Available configurations:" in capsys.readouterr().out


def test_main_with_simple_key_prints_value(capsys):
    # Arrange
    # Act
    print_config_main(["version"], config_loader=lambda: {"version": "1.0.0"})
    # Assert
    assert capsys.readouterr().out == "1.0.0\n"


def test_main_with_nested_key_prints_value(capsys):
    # Arrange
    config = {"database": {"host": "localhost"}}
    # Act
    print_config_main(["database.host"], config_loader=lambda: config)
    # Assert
    assert capsys.readouterr().out == "localhost\n"


def test_main_reads_key_from_sys_argv(capsys, restore_sys_argv):
    # Arrange
    sys.argv = ["script.py", "version"]
    # Act
    print_config_main(None, config_loader=lambda: {"version": "1.0.0"})
    # Assert
    assert capsys.readouterr().out == "1.0.0\n"


def test_main_help_exits_with_code_zero():
    # Arrange
    code = None
    # Act
    try:
        print_config_main(["--help"])
    except SystemExit as exc:
        code = exc.code
    # Assert
    assert code == 0


def test_main_help_output_mentions_description(capsys):
    # Arrange
    # Act
    try:
        print_config_main(["--help"])
    except SystemExit:
        pass
    # Assert
    assert "Print configuration values" in capsys.readouterr().out


def test_main_help_output_mentions_key_argument(capsys):
    # Arrange
    # Act
    try:
        print_config_main(["--help"])
    except SystemExit:
        pass
    # Assert
    assert "Configuration key" in capsys.readouterr().out


@pytest.mark.parametrize(
    "key, expected",
    [
        ("PATH.TITAN.MAT", "/data/matlab"),
        ("PATH.CREST.PROJECTS.1", "/proj/beta"),
        ("SETTINGS.features.0", "logging"),
        ("SETTINGS.verbosity", "2"),
    ],
)
def test_main_navigates_realistic_config(realistic_config, key, expected, capsys):
    # Arrange
    # Act
    print_config_main([key], config_loader=lambda: realistic_config)
    # Assert
    assert expected in capsys.readouterr().out


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_print_config.py
# --------------------------------------------------------------------------------
# def print_config(key, *, config_loader=None):
#     if config_loader is None:
#         import scitex
#
#         config_loader = scitex.io.load_configs
#     CONFIG = config_loader()
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
#             if isinstance(value, dict):
#                 value = value.get(k)
#             elif isinstance(value, list):
#                 try:
#                     value = value[int(k)]
#                 except (ValueError, IndexError):
#                     value = None
#             elif isinstance(value, str):
#                 break
#             else:
#                 value = None
#             if value is None:
#                 break
#         print(value)
#     except Exception as e:
#         print(f"Error: {e}")
#         print("Available configurations:")
#         pprint(value)

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_print_config.py
# --------------------------------------------------------------------------------
