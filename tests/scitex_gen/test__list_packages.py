#!/usr/bin/env python3
# Time-stamp: "2025-05-31 21:30:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/gen/test__list_packages.py

"""Tests for list_packages function.

`list_packages` discovers installed distributions and introspects each one.
Both collaborators are injected via keyword arguments (`distributions_fn`,
`inspect_fn`) — no mocks. Tests pass hand-rolled fakes: `FakeDistribution`
objects and a `RecordingInspector` that records calls and returns queued
DataFrames (or raises queued errors).
"""

import inspect
import sys

import pandas as pd
import pytest

pytest.importorskip("torch")

from scitex_gen import list_packages, main

_EXPECTED_INSPECT_KWARGS = {
    "docstring": False,
    "print_output": False,
    "columns": ["Name"],
    "root_only": True,
    "max_depth": 1,
    "skip_depwarnings": True,
}


class FakeDistribution:
    """Stand-in for an importlib.metadata Distribution (only `.name` is used)."""

    def __init__(self, name):
        self.name = name


class RecordingInspector:
    """Records each call and returns queued results (or raises queued errors)."""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def __call__(self, package_name, **kwargs):
        self.calls.append((package_name, kwargs))
        if isinstance(self._results, list):
            result = self._results[len(self.calls) - 1]
        else:
            result = self._results
        if isinstance(result, Exception):
            raise result
        return result

    @property
    def inspected_packages(self):
        return [name for name, _ in self.calls]


def _distributions_of(*names):
    dists = [FakeDistribution(name) for name in names]
    return lambda: dists


@pytest.fixture
def restore_recursion_limit():
    original = sys.getrecursionlimit()
    try:
        yield
    finally:
        sys.setrecursionlimit(original)


# --- basic functionality --------------------------------------------------


def test_list_packages_returns_dataframe():
    # Arrange
    inspector = RecordingInspector(pd.DataFrame({"Name": ["numpy.array"]}))
    # Act
    result = list_packages(
        distributions_fn=_distributions_of("numpy"), inspect_fn=inspector
    )
    # Assert
    assert isinstance(result, pd.DataFrame)


def test_list_packages_result_has_name_column():
    # Arrange
    inspector = RecordingInspector(pd.DataFrame({"Name": ["numpy.array"]}))
    # Act
    result = list_packages(
        distributions_fn=_distributions_of("numpy"), inspect_fn=inspector
    )
    # Assert
    assert "Name" in result.columns


def test_list_packages_inspects_each_distribution():
    # Arrange
    inspector = RecordingInspector(pd.DataFrame({"Name": ["x"]}))
    # Act
    list_packages(
        distributions_fn=_distributions_of("numpy", "pandas", "scipy"),
        inspect_fn=inspector,
    )
    # Assert
    assert len(inspector.calls) == 3


# --- skip-pattern filtering -----------------------------------------------


@pytest.fixture
def skip_pattern_inspector():
    inspector = RecordingInspector(pd.DataFrame({"Name": ["test.module"]}))
    list_packages(
        distributions_fn=_distributions_of(
            "numpy", "nvidia-cuda-runtime", "pillow", "pandas"
        ),
        inspect_fn=inspector,
    )
    return inspector


def test_skip_patterns_inspect_only_safe_packages(skip_pattern_inspector):
    # Arrange
    # Act
    # Assert
    assert len(skip_pattern_inspector.calls) == 2


@pytest.mark.parametrize("expected_package", ["numpy", "pandas"])
def test_skip_patterns_include_safe_packages(skip_pattern_inspector, expected_package):
    # Arrange
    # Act
    # Assert
    assert expected_package in skip_pattern_inspector.inspected_packages


@pytest.mark.parametrize("excluded_package", ["nvidia_cuda_runtime", "pillow"])
def test_skip_patterns_exclude_problematic_packages(
    skip_pattern_inspector, excluded_package
):
    # Arrange
    # Act
    # Assert
    assert excluded_package not in skip_pattern_inspector.inspected_packages


# --- safelist prioritization ----------------------------------------------


@pytest.fixture
def prioritization_inspector():
    inspector = RecordingInspector(pd.DataFrame({"Name": ["test.module"]}))
    list_packages(
        distributions_fn=_distributions_of(
            "unknown-package", "numpy", "another-unknown", "pandas"
        ),
        inspect_fn=inspector,
    )
    return inspector


def test_safelist_prioritizes_numpy_before_unknown(prioritization_inspector):
    # Arrange
    inspected = prioritization_inspector.inspected_packages
    # Act
    # Assert
    assert inspected.index("numpy") < inspected.index("unknown_package")


def test_safelist_prioritizes_pandas_before_unknown(prioritization_inspector):
    # Arrange
    inspected = prioritization_inspector.inspected_packages
    # Act
    # Assert
    assert inspected.index("pandas") < inspected.index("unknown_package")


# --- error handling -------------------------------------------------------


def test_skip_errors_true_returns_dataframe():
    # Arrange
    inspector = RecordingInspector(
        [Exception("Import error"), pd.DataFrame({"Name": ["pandas.DataFrame"]})]
    )
    # Act
    result = list_packages(
        distributions_fn=_distributions_of("numpy", "pandas"),
        inspect_fn=inspector,
        skip_errors=True,
    )
    # Assert
    assert isinstance(result, pd.DataFrame)


def test_skip_errors_true_keeps_successful_results():
    # Arrange
    inspector = RecordingInspector(
        [Exception("Import error"), pd.DataFrame({"Name": ["pandas.DataFrame"]})]
    )
    # Act
    result = list_packages(
        distributions_fn=_distributions_of("numpy", "pandas"),
        inspect_fn=inspector,
        skip_errors=True,
    )
    # Assert
    assert result["Name"].tolist() == ["pandas.DataFrame"]


def test_skip_errors_false_reraises_error():
    # Arrange
    inspector = RecordingInspector(Exception("Import error"))
    ctx = pytest.raises(Exception, match="Import error")
    # Act
    # Assert
    with ctx:
        list_packages(
            distributions_fn=_distributions_of("numpy"),
            inspect_fn=inspector,
            skip_errors=False,
        )


def test_verbose_prints_error_message(capsys):
    # Arrange
    inspector = RecordingInspector(Exception("Test error"))
    # Act
    list_packages(
        distributions_fn=_distributions_of("numpy"),
        inspect_fn=inspector,
        verbose=True,
        skip_errors=True,
    )
    # Assert
    assert "Error processing numpy: Test error" in capsys.readouterr().out


# --- empty / no-package handling ------------------------------------------


def test_empty_results_return_dataframe_with_name_column():
    # Arrange
    inspector = RecordingInspector(pd.DataFrame())
    # Act
    result = list_packages(
        distributions_fn=_distributions_of("numpy"), inspect_fn=inspector
    )
    # Assert
    assert "Name" in result.columns


def test_empty_results_return_zero_rows():
    # Arrange
    inspector = RecordingInspector(pd.DataFrame())
    # Act
    result = list_packages(
        distributions_fn=_distributions_of("numpy"), inspect_fn=inspector
    )
    # Assert
    assert len(result) == 0


def test_no_packages_found_returns_empty_dataframe():
    # Arrange
    inspector = RecordingInspector(pd.DataFrame({"Name": ["x"]}))
    # Act
    result = list_packages(distributions_fn=lambda: [], inspect_fn=inspector)
    # Assert
    assert len(result) == 0


def test_no_packages_found_skips_inspection():
    # Arrange
    inspector = RecordingInspector(pd.DataFrame({"Name": ["x"]}))
    # Act
    list_packages(distributions_fn=lambda: [], inspect_fn=inspector)
    # Assert
    assert inspector.calls == []


# --- deduplication and sorting --------------------------------------------


def test_duplicate_modules_are_removed():
    # Arrange
    inspector = RecordingInspector(
        [
            pd.DataFrame({"Name": ["shared.module", "numpy.array"]}),
            pd.DataFrame({"Name": ["shared.module", "pandas.DataFrame"]}),
        ]
    )
    # Act
    result = list_packages(
        distributions_fn=_distributions_of("numpy", "pandas"), inspect_fn=inspector
    )
    # Assert
    assert len(result) == 3


def test_results_are_sorted_by_name():
    # Arrange
    inspector = RecordingInspector(
        pd.DataFrame({"Name": ["zzz.module", "aaa.module", "mmm.module"]})
    )
    # Act
    result = list_packages(
        distributions_fn=_distributions_of("numpy"), inspect_fn=inspector
    )
    # Assert
    assert result["Name"].tolist() == ["aaa.module", "mmm.module", "zzz.module"]


# --- parameter forwarding -------------------------------------------------


def test_max_depth_is_forwarded_to_inspector():
    # Arrange
    inspector = RecordingInspector(pd.DataFrame({"Name": ["numpy.array"]}))
    # Act
    list_packages(
        max_depth=3, distributions_fn=_distributions_of("numpy"), inspect_fn=inspector
    )
    # Assert
    assert inspector.calls[0][1]["max_depth"] == 3


def test_root_only_is_forwarded_to_inspector():
    # Arrange
    inspector = RecordingInspector(pd.DataFrame({"Name": ["numpy.array"]}))
    # Act
    list_packages(
        root_only=False,
        distributions_fn=_distributions_of("numpy"),
        inspect_fn=inspector,
    )
    # Assert
    assert inspector.calls[0][1]["root_only"] is False


def test_inspector_called_with_expected_default_kwargs():
    # Arrange
    inspector = RecordingInspector(pd.DataFrame({"Name": ["numpy.array"]}))
    # Act
    list_packages(
        distributions_fn=_distributions_of("numpy"), inspect_fn=inspector
    )
    # Assert
    assert inspector.calls[0] == ("numpy", _EXPECTED_INSPECT_KWARGS)


def test_hyphenated_package_name_converted_to_underscore():
    # Arrange
    inspector = RecordingInspector(pd.DataFrame({"Name": ["sklearn.test"]}))
    # Act
    list_packages(
        distributions_fn=_distributions_of("scikit-learn"), inspect_fn=inspector
    )
    # Assert
    assert inspector.inspected_packages == ["scikit_learn"]


def test_list_packages_raises_recursion_limit(restore_recursion_limit):
    # Arrange
    inspector = RecordingInspector(pd.DataFrame({"Name": ["x"]}))
    # Act
    list_packages(distributions_fn=lambda: [], inspect_fn=inspector)
    # Assert
    assert sys.getrecursionlimit() == 10_000


# --- main -----------------------------------------------------------------


def test_main_is_callable():
    # Arrange
    # Act
    # Assert
    assert callable(main)


def test_main_has_no_required_parameters():
    # Arrange
    sig = inspect.signature(main)
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


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_list_packages.py
# --------------------------------------------------------------------------------
# def list_packages(
#     max_depth=1, root_only=True, skip_errors=True, verbose=False,
#     *, distributions_fn=None, inspect_fn=None,
# ) -> pd.DataFrame:
#     """Lists all installed packages and their modules."""
#     if distributions_fn is None:
#         distributions_fn = distributions
#     if inspect_fn is None:
#         inspect_fn = inspect_module
#     sys.setrecursionlimit(10_000)
#     ...
#     installed_packages = [
#         dist.name.replace("-", "_")
#         for dist in distributions_fn()
#         if not any(pat in dist.name.lower() for pat in skip_patterns)
#     ]
#     ...
#     for package_name in installed_packages:
#         try:
#             df = inspect_fn(package_name, docstring=False, print_output=False,
#                             columns=["Name"], root_only=root_only,
#                             max_depth=max_depth, skip_depwarnings=True)
#             if not df.empty:
#                 all_dfs.append(df)
#         except Exception as err:
#             if verbose:
#                 print(f"Error processing {package_name}: {err}")
#             if not skip_errors:
#                 raise
#     if not all_dfs:
#         return pd.DataFrame(columns=["Name"])
#     combined_df = pd.concat(all_dfs, ignore_index=True)
#     return combined_df.drop_duplicates().sort_values("Name")

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_list_packages.py
# --------------------------------------------------------------------------------
