#!/usr/bin/env python3
# Timestamp: "2025-05-31 19:45:00 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/scitex_repo/tests/scitex/gen/test__to_even.py

import pytest

pytest.importorskip("torch")
import decimal
import inspect

import numpy as np

from scitex_gen import to_even, to_odd


@pytest.mark.parametrize(
    "value, expected",
    [(1, 0), (3, 2), (5, 4), (7, 6), (99, 98), (1001, 1000)],
)
def test_to_even_rounds_odd_integers_down(value, expected):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "value, expected",
    [(0, 0), (2, 2), (4, 4), (6, 6), (100, 100), (1000, 1000)],
)
def test_to_even_keeps_even_integers_unchanged(value, expected):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "value, expected",
    [(-1, -2), (-2, -2), (-3, -4), (-4, -4), (-99, -100)],
)
def test_to_even_rounds_negative_integers_down(value, expected):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "value, expected",
    [(3.7, 2), (4.9, 4), (5.1, 4), (6.0, 6), (2.3, 2), (1.9, 0)],
)
def test_to_even_floors_positive_floats_to_even(value, expected):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "value, expected",
    [(-1.5, -2), (-2.3, -4), (-3.7, -4), (-4.1, -6)],
)
def test_to_even_floors_negative_floats_to_even(value, expected):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0, 0),
        (0.1, 0),
        (-0.1, -2),
        (0.0001, 0),
        (0.9999, 0),
        (-0.0001, -2),
        (-0.9999, -2),
        (1e-10, 0),
        (-1e-10, -2),
    ],
)
def test_to_even_handles_near_zero_values(value, expected):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (1000000, 1000000),
        (1000001, 1000000),
        (999999, 999998),
        (1234567, 1234566),
        (10**18 + 1, 10**18),
        (10**18, 10**18),
        (2**31 - 1, 2**31 - 2),
        (2**31, 2**31),
        (2**63 - 1, 2**63 - 2),
    ],
)
def test_to_even_handles_large_integer_values(value, expected):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (-(10**18) - 1, -(10**18) - 2),
        (-(10**18), -(10**18)),
        (-(2**31) + 1, -(2**31)),
        (-(2**31), -(2**31)),
        (-(2**63) + 1, -(2**63)),
    ],
)
def test_to_even_handles_large_negative_values(value, expected):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.int8(5), 4),
        (np.int16(7), 6),
        (np.int32(9), 8),
        (np.int64(11), 10),
        (np.uint8(13), 12),
        (np.uint16(15), 14),
        (np.uint32(17), 16),
        (np.uint64(19), 18),
    ],
)
def test_to_even_handles_numpy_integer_types(value, expected):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "value, expected",
    [(np.float16(3.5), 2), (np.float32(5.7), 4), (np.float64(7.9), 6)],
)
def test_to_even_handles_numpy_float_types(value, expected):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.array(5), 4),
        (np.array(6), 6),
        (np.array([7])[0], 6),
        (np.array([8])[0], 8),
    ],
)
def test_to_even_handles_numpy_scalar_arrays(value, expected):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert result == expected


@pytest.mark.parametrize("value, expected", [(True, 0), (False, 0)])
def test_to_even_treats_booleans_as_zero(value, expected):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (decimal.Decimal("5.5"), 4),
        (decimal.Decimal("6.0"), 6),
        (decimal.Decimal("-3.7"), -4),
        (decimal.Decimal("7.5"), 6),
    ],
)
def test_to_even_handles_decimal_values(value, expected):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "value, exception",
    [
        (float("nan"), ValueError),
        (float("inf"), OverflowError),
        (float("-inf"), OverflowError),
        ("hello", TypeError),
        ("5", TypeError),
        ("5.5", TypeError),
        ([1, 2, 3], TypeError),
        ({1: 2}, TypeError),
        (None, TypeError),
        (complex(5, 0), TypeError),
    ],
)
def test_to_even_raises_for_invalid_inputs(value, exception):
    # Arrange
    ctx = pytest.raises(exception)
    # Act
    # Assert
    with ctx:
        to_even(value)


@pytest.mark.parametrize("value", [-100, -3.5, 0, 1, 2, 3, 5, 10, 7.9, 99])
def test_to_even_returns_even_result(value):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert result % 2 == 0


@pytest.mark.parametrize("value", [1, 2, 3, 4, 5, -1, -2, -3, 3.5, -3.5, 7.9, -7.9])
def test_to_even_result_within_one_of_input(value):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert abs(int(value) - result) <= 1


@pytest.mark.parametrize("value", [1, 2, 3, 4, 5, -1, -2, -3, 3.5, -3.5])
def test_to_even_is_idempotent_on_results(value):
    # Arrange
    once = to_even(value)
    # Act
    twice = to_even(once)
    # Assert
    assert twice == once


def test_to_even_preserves_monotonic_order():
    # Arrange
    values = [-10, -5, -1, 0, 1, 5, 10]
    # Act
    results = [to_even(v) for v in values]
    # Assert
    assert all(results[i] <= results[i + 1] for i in range(len(results) - 1))


@pytest.mark.parametrize(
    "value", [1, 2, 3.5, 4.5, -1, -2.5, np.float32(5.5), np.int64(6), decimal.Decimal("7.5")]
)
def test_to_even_returns_integer_type(value):
    # Arrange
    # Act
    result = to_even(value)
    # Assert
    assert isinstance(result, int) and not isinstance(result, bool)


@pytest.mark.parametrize(
    "value", [0, 1, 2, 3, 4, 5, -1, -2, -3, -4, 3.5, -3.5, 7.9, -7.9]
)
def test_to_even_follows_floor_minus_parity_formula(value):
    # Arrange
    expected = int(value) - (int(value) % 2)
    # Act
    result = to_even(value)
    # Assert
    assert result == expected


def test_to_even_handles_custom_int_object():
    # Arrange
    class CustomNumber:
        def __init__(self, value):
            self.value = value

        def __int__(self):
            return int(self.value)

    # Act
    result = to_even(CustomNumber(5))
    # Assert
    assert result == 4


def test_to_even_handles_custom_even_object():
    # Arrange
    class CustomNumber:
        def __init__(self, value):
            self.value = value

        def __int__(self):
            return int(self.value)

    # Act
    result = to_even(CustomNumber(6))
    # Assert
    assert result == 6


def test_to_even_raises_for_object_without_int():
    # Arrange
    class BadObject:
        pass

    ctx = pytest.raises(TypeError)
    # Act
    # Assert
    with ctx:
        to_even(BadObject())


def test_to_even_is_callable_function():
    # Arrange
    # Act
    # Assert
    assert callable(to_even)


def test_to_even_accepts_single_parameter():
    # Arrange
    sig = inspect.signature(to_even)
    # Act
    params = list(sig.parameters.keys())
    # Assert
    assert len(params) == 1


def test_to_even_parameter_is_named_n():
    # Arrange
    sig = inspect.signature(to_even)
    # Act
    params = list(sig.parameters.keys())
    # Assert
    assert params[0] == "n"


def test_to_even_has_docstring_present():
    # Arrange
    # Act
    docstring = to_even.__doc__
    # Assert
    assert docstring is not None


def test_to_even_docstring_describes_purpose():
    # Arrange
    # Act
    docstring = to_even.__doc__
    # Assert
    assert "Convert a number to the nearest even number" in docstring


def test_to_even_attribute_and_name_are_identical():
    # Arrange
    import scitex_gen
    from scitex_gen import to_even as to_even_name

    # Act
    to_even_attr = scitex_gen.to_even
    # Assert
    assert to_even_name is to_even_attr


@pytest.mark.parametrize("value", [1, 2, 3, 4, 5, 6])
def test_to_even_complements_to_odd_by_one(value):
    # Arrange
    even_result = to_even(value)
    # Act
    odd_result = to_odd(value)
    # Assert
    assert abs(even_result - odd_result) == 1


def test_to_even_batch_results_match_input_length():
    # Arrange
    values = list(range(1000))
    # Act
    results = [to_even(v) for v in values]
    # Assert
    assert len(results) == len(values)


def test_to_even_batch_results_are_all_even():
    # Arrange
    values = list(range(1000))
    # Act
    results = [to_even(v) for v in values]
    # Assert
    assert all(r % 2 == 0 for r in results)


def test_to_even_is_stateless_across_threads():
    # Arrange
    import threading

    results = []

    def worker(value):
        results.append((value, to_even(value)))

    threads = [threading.Thread(target=worker, args=(v,)) for v in range(100)]
    # Act
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Assert
    assert all(result == int(val) - (int(val) % 2) for val, result in results)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_to_even.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-25 23:40:12 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/gen/_to_even.py
#
# THIS_FILE = "/home/ywatanabe/proj/scitex_repo/src/scitex/gen/_to_even.py"
#
#
# def to_even(n):
#     """Convert a number to the nearest even number less than or equal to itself.
#
#     Parameters
#     ----------
#     n : int or float
#         The input number to be converted.
#
#     Returns
#     -------
#     int
#         The nearest even number less than or equal to the input.
#
#     Example
#     -------
#     >>> to_even(5)
#     4
#     >>> to_even(6)
#     6
#     >>> to_even(3.7)
#     2
#     >>> to_even(-2.3)
#     -4
#     >>> to_even(-0.1)
#     -2
#     """
#     import math
#
#     # Handle integers directly to avoid float conversion issues with large numbers
#     # Note: bool is a subclass of int, so we need to exclude it
#     if isinstance(n, int) and not isinstance(n, bool):
#         if n % 2 == 0:
#             return int(n)  # Ensure we return int, not bool
#         else:
#             return int(n - 1)  # Ensure we return int, not bool
#
#     # Handle special float values
#     if isinstance(n, float):
#         if math.isnan(n):
#             raise ValueError("Cannot convert NaN to even")
#         if math.isinf(n):
#             raise OverflowError("Cannot convert infinity to even")
#         # Python can actually convert sys.float_info.max to int, so we don't need this check
#         # Only infinity truly can't be converted
#
#     # Try to handle custom objects with __int__ (but not float types)
#     if hasattr(n, "__int__") and not isinstance(n, (float, bool)):
#         try:
#             n_int = int(n)
#             if n_int % 2 == 0:
#                 return int(n_int)
#             else:
#                 return int(n_int - 1)
#         except:
#             pass
#
#     # Check for string type explicitly - raise TypeError
#     if isinstance(n, str):
#         raise TypeError(f"must be real number, not {type(n).__name__}")
#
#     # Convert to float for all other cases
#     try:
#         n_float = float(n)
#     except (TypeError, ValueError):
#         raise TypeError(f"must be real number, not {type(n).__name__}")
#
#     # Use floor for float values
#     floored = int(math.floor(n_float))
#
#     # If odd, subtract 1 to get the next lower even number
#     if floored % 2 != 0:
#         return int(floored - 1)  # Ensure we return int, not bool
#     return int(floored)  # Ensure we return int, not bool
#
#
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_to_even.py
# --------------------------------------------------------------------------------
