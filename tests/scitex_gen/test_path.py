"""Namespace tests for scitex_gen vs scitex_path.

The gen/path.py file is currently a placeholder for future path-generation
utilities. The actual path manipulation lives in scitex_path. These tests
verify the namespace separation rather than the placeholder file itself
(testing an empty module's emptiness was green-bar theater — removed per
PA-307 doctrine).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_scitex_gen_imports_without_error():
    # Arrange
    # Arrange
    module_name = "scitex_gen"

    # Act
    # Act
    import scitex_gen as imported

    # Assert
    # Assert
    assert imported.__name__ == module_name


def test_scitex_gen_does_not_shadow_os_path_functions():
    # Arrange
    # Arrange
    shadowed_names = {
        "abspath",
        "dirname",
        "basename",
        "exists",
        "isfile",
        "isdir",
        "join",
        "splitext",
    }

    # Act
    import scitex_gen

    # Act
    gen_attrs = set(dir(scitex_gen))

    # Assert
    # Assert
    assert not gen_attrs & shadowed_names


def test_scitex_path_is_separate_module_from_scitex_gen():
    # Arrange
    # Arrange
    pytest.importorskip("scitex_path")
    import scitex_path  # noqa: F401  (presence check via importorskip)

    import scitex_gen

    # Act
    # Act
    gen_dir = Path(scitex_gen.__file__).parent

    # Assert
    # Assert
    assert gen_dir.name == "scitex_gen"


def test_scitex_path_directory_matches_its_package_name():
    # Arrange
    # Arrange
    pytest.importorskip("scitex_path")
    import scitex_path

    # Act
    # Act
    path_dir = Path(scitex_path.__file__).parent

    # Assert
    # Assert
    assert path_dir.name == "scitex_path"
