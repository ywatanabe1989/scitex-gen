#!/usr/bin/env python3
# Timestamp: "2025-05-31 19:45:00 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/scitex_repo/tests/scitex/gen/test__to_even.py

import pytest

pytest.importorskip("torch")
import contextlib
import decimal
import math
import sys
import warnings
from typing import Union

import numpy as np

from scitex_gen import to_even


@contextlib.contextmanager
def _swap_attr(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


class TestToEvenBasicFunctionality:
    """Test basic functionality of the to_even function."""

    def test_odd_integers_to_even_1_0_split_1(self):
        """Test conversion of odd integers to even."""
        # Arrange
        # Act
        # Assert
        assert to_even(1) == 0

    def test_odd_integers_to_even_1_0_split_2(self):
        """Test conversion of odd integers to even."""
        # Arrange
        to_even(1) == 0
        # Act
        # Assert
        assert to_even(3) == 2

    def test_odd_integers_to_even_1_0_split_3(self):
        """Test conversion of odd integers to even."""
        # Arrange
        to_even(1) == 0
        to_even(3) == 2
        # Act
        # Assert
        assert to_even(5) == 4

    def test_odd_integers_to_even_1_0_split_4(self):
        """Test conversion of odd integers to even."""
        # Arrange
        to_even(1) == 0
        to_even(3) == 2
        to_even(5) == 4
        # Act
        # Assert
        assert to_even(7) == 6

    def test_odd_integers_to_even_1_0_split_5(self):
        """Test conversion of odd integers to even."""
        # Arrange
        to_even(1) == 0
        to_even(3) == 2
        to_even(5) == 4
        to_even(7) == 6
        # Act
        # Assert
        assert to_even(99) == 98

    def test_odd_integers_to_even_1_0_split_6(self):
        """Test conversion of odd integers to even."""
        # Arrange
        to_even(1) == 0
        to_even(3) == 2
        to_even(5) == 4
        to_even(7) == 6
        to_even(99) == 98
        # Act
        # Assert
        assert to_even(1001) == 1000

    def test_even_integers_to_even_0_0_split_1(self):
        """Test that even integers remain unchanged."""
        # Arrange
        # Act
        # Assert
        assert to_even(0) == 0

    def test_even_integers_to_even_0_0_split_2(self):
        """Test that even integers remain unchanged."""
        # Arrange
        to_even(0) == 0
        # Act
        # Assert
        assert to_even(2) == 2

    def test_even_integers_to_even_0_0_split_3(self):
        """Test that even integers remain unchanged."""
        # Arrange
        to_even(0) == 0
        to_even(2) == 2
        # Act
        # Assert
        assert to_even(4) == 4

    def test_even_integers_to_even_0_0_split_4(self):
        """Test that even integers remain unchanged."""
        # Arrange
        to_even(0) == 0
        to_even(2) == 2
        to_even(4) == 4
        # Act
        # Assert
        assert to_even(6) == 6

    def test_even_integers_to_even_0_0_split_5(self):
        """Test that even integers remain unchanged."""
        # Arrange
        to_even(0) == 0
        to_even(2) == 2
        to_even(4) == 4
        to_even(6) == 6
        # Act
        # Assert
        assert to_even(100) == 100

    def test_even_integers_to_even_0_0_split_6(self):
        """Test that even integers remain unchanged."""
        # Arrange
        to_even(0) == 0
        to_even(2) == 2
        to_even(4) == 4
        to_even(6) == 6
        to_even(100) == 100
        # Act
        # Assert
        assert to_even(1000) == 1000

    def test_negative_integers_to_even_1_2_split_1(self):
        """Test conversion of negative integers."""
        # Arrange
        # Act
        # Assert
        assert to_even(-1) == -2

    def test_negative_integers_to_even_1_2_split_2(self):
        """Test conversion of negative integers."""
        # Arrange
        to_even(-1) == -2
        # Act
        # Assert
        assert to_even(-2) == -2

    def test_negative_integers_to_even_1_2_split_3(self):
        """Test conversion of negative integers."""
        # Arrange
        to_even(-1) == -2
        to_even(-2) == -2
        # Act
        # Assert
        assert to_even(-3) == -4

    def test_negative_integers_to_even_1_2_split_4(self):
        """Test conversion of negative integers."""
        # Arrange
        to_even(-1) == -2
        to_even(-2) == -2
        to_even(-3) == -4
        # Act
        # Assert
        assert to_even(-4) == -4

    def test_negative_integers_to_even_1_2_split_5(self):
        """Test conversion of negative integers."""
        # Arrange
        to_even(-1) == -2
        to_even(-2) == -2
        to_even(-3) == -4
        to_even(-4) == -4
        # Act
        # Assert
        assert to_even(-99) == -100

    def test_floats_to_even_3_7_2_split_1(self):
        """Test conversion of float values."""
        # Arrange
        # Act
        # Assert
        assert to_even(3.7) == 2

    def test_floats_to_even_3_7_2_split_2(self):
        """Test conversion of float values."""
        # Arrange
        to_even(3.7) == 2
        # Act
        # Assert
        assert to_even(4.9) == 4

    def test_floats_to_even_3_7_2_split_3(self):
        """Test conversion of float values."""
        # Arrange
        to_even(3.7) == 2
        to_even(4.9) == 4
        # Act
        # Assert
        assert to_even(5.1) == 4

    def test_floats_to_even_3_7_2_split_4(self):
        """Test conversion of float values."""
        # Arrange
        to_even(3.7) == 2
        to_even(4.9) == 4
        to_even(5.1) == 4
        # Act
        # Assert
        assert to_even(6.0) == 6

    def test_floats_to_even_3_7_2_split_5(self):
        """Test conversion of float values."""
        # Arrange
        to_even(3.7) == 2
        to_even(4.9) == 4
        to_even(5.1) == 4
        to_even(6.0) == 6
        # Act
        # Assert
        assert to_even(2.3) == 2

    def test_floats_to_even_3_7_2_split_6(self):
        """Test conversion of float values."""
        # Arrange
        to_even(3.7) == 2
        to_even(4.9) == 4
        to_even(5.1) == 4
        to_even(6.0) == 6
        to_even(2.3) == 2
        # Act
        # Assert
        assert to_even(1.9) == 0

    def test_negative_floats_to_even_1_5_2_split_1(self):
        """Test conversion of negative float values."""
        # Arrange
        # Act
        # Assert
        assert to_even(-1.5) == -2

    def test_negative_floats_to_even_1_5_2_split_2(self):
        """Test conversion of negative float values."""
        # Arrange
        to_even(-1.5) == -2
        # Act
        # Assert
        assert to_even(-2.3) == -4

    def test_negative_floats_to_even_1_5_2_split_3(self):
        """Test conversion of negative float values."""
        # Arrange
        to_even(-1.5) == -2
        to_even(-2.3) == -4
        # Act
        # Assert
        assert to_even(-3.7) == -4

    def test_negative_floats_to_even_1_5_2_split_4(self):
        """Test conversion of negative float values."""
        # Arrange
        to_even(-1.5) == -2
        to_even(-2.3) == -4
        to_even(-3.7) == -4
        # Act
        # Assert
        assert to_even(-4.1) == -6

    def test_edge_cases_to_even_0_0_split_1(self):
        """Test edge cases and special values."""
        # Arrange
        # Act
        # Assert
        assert to_even(0) == 0

    def test_edge_cases_to_even_0_0_split_2(self):
        """Test edge cases and special values."""
        # Arrange
        to_even(0) == 0
        # Act
        # Assert
        assert to_even(0.0) == 0

    def test_edge_cases_to_even_0_0_split_3(self):
        """Test edge cases and special values."""
        # Arrange
        to_even(0) == 0
        to_even(0.0) == 0
        # Act
        # Assert
        assert to_even(0.1) == 0

    def test_edge_cases_to_even_0_0_split_4(self):
        """Test edge cases and special values."""
        # Arrange
        to_even(0) == 0
        to_even(0.0) == 0
        to_even(0.1) == 0
        # Act
        # Assert
        assert to_even(-0.1) == -2

    def test_large_numbers_to_even_1000000_1000000_split_1(self):
        """Test with large numbers."""
        # Arrange
        # Act
        # Assert
        assert to_even(1000000) == 1000000

    def test_large_numbers_to_even_1000000_1000000_split_2(self):
        """Test with large numbers."""
        # Arrange
        to_even(1000000) == 1000000
        # Act
        # Assert
        assert to_even(1000001) == 1000000

    def test_large_numbers_to_even_1000000_1000000_split_3(self):
        """Test with large numbers."""
        # Arrange
        to_even(1000000) == 1000000
        to_even(1000001) == 1000000
        # Act
        # Assert
        assert to_even(999999) == 999998

    def test_large_numbers_to_even_1000000_1000000_split_4(self):
        """Test with large numbers."""
        # Arrange
        to_even(1000000) == 1000000
        to_even(1000001) == 1000000
        to_even(999999) == 999998
        # Act
        # Assert
        assert to_even(1234567) == 1234566

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            (1, 0),
            (2, 2),
            (3.5, 2),
            (4.5, 4),
            (-1, -2),
            (-2, -2),
            (0, 0),
        ],
    )
    def test_parametrized_to_even_input_val_expected(self, input_val, expected):
        """Parametrized test for various inputs."""
        # Arrange
        # Act
        # Assert
        assert to_even(input_val) == expected


class TestToEvenNumericTypes:
    """Test to_even with various numeric types."""

    def test_numpy_integers_to_even_np_int8_5_4_split_1(self):
        """Test with numpy integer types."""
        # Arrange
        # Act
        # Assert
        assert to_even(np.int8(5)) == 4

    def test_numpy_integers_to_even_np_int8_5_4_split_2(self):
        """Test with numpy integer types."""
        # Arrange
        to_even(np.int8(5)) == 4
        # Act
        # Assert
        assert to_even(np.int16(7)) == 6

    def test_numpy_integers_to_even_np_int8_5_4_split_3(self):
        """Test with numpy integer types."""
        # Arrange
        to_even(np.int8(5)) == 4
        to_even(np.int16(7)) == 6
        # Act
        # Assert
        assert to_even(np.int32(9)) == 8

    def test_numpy_integers_to_even_np_int8_5_4_split_4(self):
        """Test with numpy integer types."""
        # Arrange
        to_even(np.int8(5)) == 4
        to_even(np.int16(7)) == 6
        to_even(np.int32(9)) == 8
        # Act
        # Assert
        assert to_even(np.int64(11)) == 10

    def test_numpy_integers_to_even_np_int8_5_4_split_5(self):
        """Test with numpy integer types."""
        # Arrange
        to_even(np.int8(5)) == 4
        to_even(np.int16(7)) == 6
        to_even(np.int32(9)) == 8
        to_even(np.int64(11)) == 10
        # Act
        # Assert
        assert to_even(np.uint8(13)) == 12

    def test_numpy_integers_to_even_np_int8_5_4_split_6(self):
        """Test with numpy integer types."""
        # Arrange
        to_even(np.int8(5)) == 4
        to_even(np.int16(7)) == 6
        to_even(np.int32(9)) == 8
        to_even(np.int64(11)) == 10
        to_even(np.uint8(13)) == 12
        # Act
        # Assert
        assert to_even(np.uint16(15)) == 14

    def test_numpy_integers_to_even_np_int8_5_4_split_7(self):
        """Test with numpy integer types."""
        # Arrange
        to_even(np.int8(5)) == 4
        to_even(np.int16(7)) == 6
        to_even(np.int32(9)) == 8
        to_even(np.int64(11)) == 10
        to_even(np.uint8(13)) == 12
        to_even(np.uint16(15)) == 14
        # Act
        # Assert
        assert to_even(np.uint32(17)) == 16

    def test_numpy_integers_to_even_np_int8_5_4_split_8(self):
        """Test with numpy integer types."""
        # Arrange
        to_even(np.int8(5)) == 4
        to_even(np.int16(7)) == 6
        to_even(np.int32(9)) == 8
        to_even(np.int64(11)) == 10
        to_even(np.uint8(13)) == 12
        to_even(np.uint16(15)) == 14
        to_even(np.uint32(17)) == 16
        # Act
        # Assert
        assert to_even(np.uint64(19)) == 18

    def test_numpy_floats_to_even_np_float16_3_5_2_split_1(self):
        """Test with numpy float types."""
        # Arrange
        # Act
        # Assert
        assert to_even(np.float16(3.5)) == 2

    def test_numpy_floats_to_even_np_float16_3_5_2_split_2(self):
        """Test with numpy float types."""
        # Arrange
        to_even(np.float16(3.5)) == 2
        # Act
        # Assert
        assert to_even(np.float32(5.7)) == 4

    def test_numpy_floats_to_even_np_float16_3_5_2_split_3(self):
        """Test with numpy float types."""
        # Arrange
        to_even(np.float16(3.5)) == 2
        to_even(np.float32(5.7)) == 4
        # Act
        # Assert
        assert to_even(np.float64(7.9)) == 6

    def test_numpy_floats_to_even_np_float16_3_5_2_split_4(self):
        """Test with numpy float types."""
        # Arrange
        to_even(np.float16(3.5)) == 2
        to_even(np.float32(5.7)) == 4
        to_even(np.float64(7.9)) == 6
        # Act
        # Assert
        if hasattr(np, 'float128'):
            assert to_even(np.float128(9.1)) == 8

    def test_numpy_arrays_to_even_np_array_5_4_split_1(self):
        """Test with numpy arrays (should work on scalar elements)."""
        # Arrange
        # Act
        # Assert
        assert to_even(np.array(5)) == 4

    def test_numpy_arrays_to_even_np_array_5_4_split_2(self):
        """Test with numpy arrays (should work on scalar elements)."""
        # Arrange
        to_even(np.array(5)) == 4
        # Act
        # Assert
        assert to_even(np.array(6)) == 6

    def test_numpy_arrays_to_even_np_array_5_4_split_3(self):
        """Test with numpy arrays (should work on scalar elements)."""
        # Arrange
        to_even(np.array(5)) == 4
        to_even(np.array(6)) == 6
        # Act
        # Assert
        assert to_even(np.array([7])[0]) == 6

    def test_numpy_arrays_to_even_np_array_5_4_split_4(self):
        """Test with numpy arrays (should work on scalar elements)."""
        # Arrange
        to_even(np.array(5)) == 4
        to_even(np.array(6)) == 6
        to_even(np.array([7])[0]) == 6
        # Act
        # Assert
        assert to_even(np.array([8])[0]) == 8

    def test_python_numeric_types_split_1(self):
        """Test with various Python numeric types."""
        # Arrange
        # Act
        # Assert
        with pytest.raises(TypeError):
            to_even(complex(5, 0))

    def test_python_numeric_types_split_2(self):
        """Test with various Python numeric types."""
        # Arrange
        try:
            to_even(complex(5, 0))
        except Exception:
            pass
        # Act
        # Assert
        assert to_even(True) == 0

    def test_python_numeric_types_split_3(self):
        """Test with various Python numeric types."""
        # Arrange
        try:
            to_even(complex(5, 0))
        except Exception:
            pass
        to_even(True) == 0
        # Act
        # Assert
        assert to_even(False) == 0

    def test_python_numeric_types_split_4(self):
        """Test with various Python numeric types."""
        # Arrange
        try:
            to_even(complex(5, 0))
        except Exception:
            pass
        to_even(True) == 0
        to_even(False) == 0
        # Act
        # Assert
        assert to_even(decimal.Decimal('5.5')) == 4

    def test_python_numeric_types_split_5(self):
        """Test with various Python numeric types."""
        # Arrange
        try:
            to_even(complex(5, 0))
        except Exception:
            pass
        to_even(True) == 0
        to_even(False) == 0
        to_even(decimal.Decimal('5.5')) == 4
        # Act
        # Assert
        assert to_even(decimal.Decimal('6.0')) == 6

    def test_python_numeric_types_split_6(self):
        """Test with various Python numeric types."""
        # Arrange
        try:
            to_even(complex(5, 0))
        except Exception:
            pass
        to_even(True) == 0
        to_even(False) == 0
        to_even(decimal.Decimal('5.5')) == 4
        to_even(decimal.Decimal('6.0')) == 6
        # Act
        # Assert
        assert to_even(decimal.Decimal('-3.7')) == -4


class TestToEvenEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    def test_very_large_numbers_split_1(self):
        """Test with very large numbers."""
        # Arrange
        # Act
        # Assert
        assert to_even(10 ** 18 + 1) == 10 ** 18

    def test_very_large_numbers_split_2(self):
        """Test with very large numbers."""
        # Arrange
        to_even(10 ** 18 + 1) == 10 ** 18
        # Act
        # Assert
        assert to_even(10 ** 18) == 10 ** 18

    def test_very_large_numbers_split_3(self):
        """Test with very large numbers."""
        # Arrange
        to_even(10 ** 18 + 1) == 10 ** 18
        to_even(10 ** 18) == 10 ** 18
        # Act
        # Assert
        assert to_even(2 ** 31 - 1) == 2 ** 31 - 2

    def test_very_large_numbers_split_4(self):
        """Test with very large numbers."""
        # Arrange
        to_even(10 ** 18 + 1) == 10 ** 18
        to_even(10 ** 18) == 10 ** 18
        to_even(2 ** 31 - 1) == 2 ** 31 - 2
        # Act
        # Assert
        assert to_even(2 ** 31) == 2 ** 31

    def test_very_large_numbers_split_5(self):
        """Test with very large numbers."""
        # Arrange
        to_even(10 ** 18 + 1) == 10 ** 18
        to_even(10 ** 18) == 10 ** 18
        to_even(2 ** 31 - 1) == 2 ** 31 - 2
        to_even(2 ** 31) == 2 ** 31
        # Act
        # Assert
        assert to_even(2 ** 63 - 1) == 2 ** 63 - 2

    def test_very_small_negative_numbers_split_1(self):
        """Test with very small negative numbers."""
        # Arrange
        # Act
        # Assert
        assert to_even(-10 ** 18 - 1) == -10 ** 18 - 2

    def test_very_small_negative_numbers_split_2(self):
        """Test with very small negative numbers."""
        # Arrange
        to_even(-10 ** 18 - 1) == -10 ** 18 - 2
        # Act
        # Assert
        assert to_even(-10 ** 18) == -10 ** 18

    def test_very_small_negative_numbers_split_3(self):
        """Test with very small negative numbers."""
        # Arrange
        to_even(-10 ** 18 - 1) == -10 ** 18 - 2
        to_even(-10 ** 18) == -10 ** 18
        # Act
        # Assert
        assert to_even(-2 ** 31 + 1) == -2 ** 31

    def test_very_small_negative_numbers_split_4(self):
        """Test with very small negative numbers."""
        # Arrange
        to_even(-10 ** 18 - 1) == -10 ** 18 - 2
        to_even(-10 ** 18) == -10 ** 18
        to_even(-2 ** 31 + 1) == -2 ** 31
        # Act
        # Assert
        assert to_even(-2 ** 31) == -2 ** 31

    def test_very_small_negative_numbers_split_5(self):
        """Test with very small negative numbers."""
        # Arrange
        to_even(-10 ** 18 - 1) == -10 ** 18 - 2
        to_even(-10 ** 18) == -10 ** 18
        to_even(-2 ** 31 + 1) == -2 ** 31
        to_even(-2 ** 31) == -2 ** 31
        # Act
        # Assert
        assert to_even(-2 ** 63 + 1) == -2 ** 63

    def test_near_zero_values_split_1(self):
        """Test values very close to zero."""
        # Arrange
        # Act
        # Assert
        assert to_even(0.0001) == 0

    def test_near_zero_values_split_2(self):
        """Test values very close to zero."""
        # Arrange
        to_even(0.0001) == 0
        # Act
        # Assert
        assert to_even(0.9999) == 0

    def test_near_zero_values_split_3(self):
        """Test values very close to zero."""
        # Arrange
        to_even(0.0001) == 0
        to_even(0.9999) == 0
        # Act
        # Assert
        assert to_even(-0.0001) == -2

    def test_near_zero_values_split_4(self):
        """Test values very close to zero."""
        # Arrange
        to_even(0.0001) == 0
        to_even(0.9999) == 0
        to_even(-0.0001) == -2
        # Act
        # Assert
        assert to_even(-0.9999) == -2

    def test_near_zero_values_split_5(self):
        """Test values very close to zero."""
        # Arrange
        to_even(0.0001) == 0
        to_even(0.9999) == 0
        to_even(-0.0001) == -2
        to_even(-0.9999) == -2
        # Act
        # Assert
        assert to_even(1e-10) == 0

    def test_near_zero_values_split_6(self):
        """Test values very close to zero."""
        # Arrange
        to_even(0.0001) == 0
        to_even(0.9999) == 0
        to_even(-0.0001) == -2
        to_even(-0.9999) == -2
        to_even(1e-10) == 0
        # Act
        # Assert
        assert to_even(-1e-10) == -2

    def test_infinity_and_nan_split_1(self):
        """Test with infinity and NaN values."""
        # Arrange
        # Act
        # Assert
        with pytest.raises(OverflowError):
            to_even(float('inf'))

    def test_infinity_and_nan_split_2(self):
        """Test with infinity and NaN values."""
        # Arrange
        try:
            to_even(float('inf'))
        except Exception:
            pass
        # Act
        # Assert
        with pytest.raises(OverflowError):
            to_even(float('-inf'))

    def test_infinity_and_nan_split_3(self):
        """Test with infinity and NaN values."""
        # Arrange
        try:
            to_even(float('inf'))
        except Exception:
            pass
        try:
            to_even(float('-inf'))
        except Exception:
            pass
        # Act
        # Assert
        with pytest.raises(ValueError):
            to_even(float('nan'))

    def test_special_float_values_split_1(self):
        """Test special floating point values."""
        # Arrange
        # Act
        # Assert
        assert to_even(sys.float_info.min) == 0

    def test_special_float_values_split_2(self):
        """Test special floating point values."""
        # Arrange
        to_even(sys.float_info.min) == 0
        # Act
        # Assert
        assert to_even(-sys.float_info.min) == -2

    def test_special_float_values_split_3(self):
        """Test special floating point values."""
        # Arrange
        to_even(sys.float_info.min) == 0
        to_even(-sys.float_info.min) == -2
        result = to_even(sys.float_info.max)
        # Act
        # Assert
        assert isinstance(result, int)

    def test_special_float_values_split_4(self):
        """Test special floating point values."""
        # Arrange
        to_even(sys.float_info.min) == 0
        to_even(-sys.float_info.min) == -2
        result = to_even(sys.float_info.max)
        isinstance(result, int)
        # Act
        # Assert
        assert result % 2 == 0

    def test_special_float_values_split_5(self):
        """Test special floating point values."""
        # Arrange
        to_even(sys.float_info.min) == 0
        to_even(-sys.float_info.min) == -2
        result = to_even(sys.float_info.max)
        isinstance(result, int)
        result % 2 == 0
        # Act
        # Assert
        with pytest.raises(OverflowError):
            to_even(float('inf'))


class TestToEvenMathematicalProperties:
    """Test mathematical properties of the to_even function."""

    def test_idempotence_smoke_case(self):
        """Test that applying to_even twice gives the same result."""
        # Arrange
        # Act
        # Assert
        values = [1, 2, 3, 4, 5, -1, -2, -3, 3.5, -3.5]
        for val in values:
            result1 = to_even(val)
            result2 = to_even(result1)
            assert result1 == result2

    def test_monotonicity_all_results_i_results_i_1_for_i_in_range_len_resul(self):
        """Test that to_even preserves order (is monotonic)."""
        # Arrange
        values = [-10, -5, -1, 0, 1, 5, 10]
        # Act
        results = [to_even(v) for v in values]
        # Assert
        assert all(results[i] <= results[i + 1] for i in range(len(results) - 1))

    def test_distance_property_smoke_case(self):
        """Test that result is at most 1 unit away from input."""
        # Arrange
        # Act
        # Assert
        values = [1, 2, 3, 4, 5, -1, -2, -3, 3.5, -3.5, 7.9, -7.9]
        for val in values:
            result = to_even(val)
            assert abs(int(val) - result) <= 1

    def test_parity_property_smoke_case(self):
        """Test that result is always even."""
        # Arrange
        # Act
        # Assert
        values = list(range(-100, 101)) + [x / 10 for x in range(-100, 101)]
        for val in values:
            result = to_even(val)
            assert result % 2 == 0

    def test_floor_relationship_smoke_case(self):
        """Test relationship with floor function."""
        # Arrange
        values = [3.2, 3.7, 4.0, 4.5, -3.2, -3.7, -4.0, -4.5]
        # Act
        observed = [(to_even(val), math.floor(val)) for val in values]
        expected = [
            (floor_val if floor_val % 2 == 0 else floor_val - 1, floor_val)
            for _, floor_val in observed
        ]
        # Assert
        assert observed == expected


class TestToEvenErrorHandling:
    """Test error handling and invalid inputs."""

    def test_non_numeric_types_split_1(self):
        """Test with non-numeric types."""
        # Arrange
        # Act
        # Assert
        with pytest.raises((TypeError, ValueError)):
            to_even('hello')

    def test_non_numeric_types_split_2(self):
        """Test with non-numeric types."""
        # Arrange
        try:
            to_even('hello')
        except Exception:
            pass
        # Act
        # Assert
        with pytest.raises(TypeError):
            to_even([1, 2, 3])

    def test_non_numeric_types_split_3(self):
        """Test with non-numeric types."""
        # Arrange
        try:
            to_even('hello')
        except Exception:
            pass
        try:
            to_even([1, 2, 3])
        except Exception:
            pass
        # Act
        # Assert
        with pytest.raises(TypeError):
            to_even({1: 2})

    def test_non_numeric_types_split_4(self):
        """Test with non-numeric types."""
        # Arrange
        try:
            to_even('hello')
        except Exception:
            pass
        try:
            to_even([1, 2, 3])
        except Exception:
            pass
        try:
            to_even({1: 2})
        except Exception:
            pass
        # Act
        # Assert
        with pytest.raises(TypeError):
            to_even(None)

    def test_string_numbers_raises_typeerror_split_1(self):
        """Test with string representations of numbers."""
        # Arrange
        # Act
        # Assert
        with pytest.raises(TypeError):
            to_even('5')

    def test_string_numbers_raises_typeerror_split_2(self):
        """Test with string representations of numbers."""
        # Arrange
        try:
            to_even('5')
        except Exception:
            pass
        # Act
        # Assert
        with pytest.raises(TypeError):
            to_even('5.5')

    def test_custom_objects_to_even_customnumber_5_4_split_1(self):
        """Test with custom objects."""
        # Arrange

        class CustomNumber:

            def __init__(self, value):
                self.value = value

            def __int__(self):
                return int(self.value)
        # Act
        # Assert
        assert to_even(CustomNumber(5)) == 4

    def test_custom_objects_to_even_customnumber_5_4_split_2(self):
        """Test with custom objects."""
        # Arrange

        class CustomNumber:

            def __init__(self, value):
                self.value = value

            def __int__(self):
                return int(self.value)
        to_even(CustomNumber(5)) == 4
        # Act
        # Assert
        assert to_even(CustomNumber(6)) == 6

    def test_custom_objects_to_even_customnumber_5_4_split_3(self):
        """Test with custom objects."""
        # Arrange

        class CustomNumber:

            def __init__(self, value):
                self.value = value

            def __int__(self):
                return int(self.value)
        to_even(CustomNumber(5)) == 4
        to_even(CustomNumber(6)) == 6

        class BadObject:
            pass
        # Act
        # Assert
        with pytest.raises(TypeError):
            to_even(BadObject())


class TestToEvenPerformance:
    """Test performance characteristics of to_even."""

    def test_performance_consistency_max_time_min_time_3_0(self):
        """Test that performance is consistent across input values."""
        # Arrange
        import time

        # Test with different magnitudes
        test_values = [1, 100, 10000, 1000000]
        times = []

        for val in test_values:
            start = time.perf_counter()
            for _ in range(10000):
                to_even(val)
            end = time.perf_counter()
            times.append(end - start)

        # Check that times don't vary by more than 3x (relaxed due to system load variance)
        max_time = max(times)
        # Act
        min_time = min(times)
        # Assert
        assert (
            max_time < min_time * 3.0
        ), f"Performance varied too much: {min_time:.4f}s - {max_time:.4f}s"

    def test_batch_operations_len_results_len_values_split_1(self):
        """Test performance with batch operations."""
        # Arrange
        values = list(range(1000))
        import time
        start = time.perf_counter()
        results = [to_even(v) for v in values]
        end = time.perf_counter()
        # Act
        # Assert
        assert len(results) == len(values)

    def test_batch_operations_len_results_len_values_split_2(self):
        """Test performance with batch operations."""
        # Arrange
        values = list(range(1000))
        import time
        start = time.perf_counter()
        results = [to_even(v) for v in values]
        end = time.perf_counter()
        len(results) == len(values)
        # Act
        # Assert
        assert all((r % 2 == 0 for r in results))

    def test_batch_operations_len_results_len_values_split_3(self):
        """Test performance with batch operations."""
        # Arrange
        values = list(range(1000))
        import time
        start = time.perf_counter()
        results = [to_even(v) for v in values]
        end = time.perf_counter()
        len(results) == len(values)
        all((r % 2 == 0 for r in results))
        # Act
        # Assert
        assert end - start < 0.1


class TestToEvenAlgorithmVerification:
    """Verify the algorithm implementation."""

    def test_algorithm_formula_smoke_case(self):
        """Test that the algorithm follows the formula: int(n) - (int(n) % 2)."""
        # Arrange
        # Act
        # Assert
        test_values = [0, 1, 2, 3, 4, 5, -1, -2, -3, -4, 3.5, -3.5, 7.9, -7.9]

        for val in test_values:
            expected = int(val) - (int(val) % 2)
            assert to_even(val) == expected

    def test_modulo_behavior_n_1_2_1_split_1(self):
        """Test understanding of modulo with negative numbers."""
        # Arrange
        # Act
        # Assert
        assert -1 % 2 == 1

    def test_modulo_behavior_n_1_2_1_split_2(self):
        """Test understanding of modulo with negative numbers."""
        # Arrange
        -1 % 2 == 1
        # Act
        # Assert
        assert -3 % 2 == 1

    def test_modulo_behavior_n_1_2_1_split_3(self):
        """Test understanding of modulo with negative numbers."""
        # Arrange
        -1 % 2 == 1
        -3 % 2 == 1
        # Act
        # Assert
        assert -2 % 2 == 0

    def test_modulo_behavior_n_1_2_1_split_4(self):
        """Test understanding of modulo with negative numbers."""
        # Arrange
        -1 % 2 == 1
        -3 % 2 == 1
        -2 % 2 == 0
        # Act
        # Assert
        assert -4 % 2 == 0

    def test_modulo_behavior_n_1_2_1_split_5(self):
        """Test understanding of modulo with negative numbers."""
        # Arrange
        -1 % 2 == 1
        -3 % 2 == 1
        -2 % 2 == 0
        -4 % 2 == 0
        # Act
        # Assert
        assert to_even(-1) == -1 - 1

    def test_modulo_behavior_n_1_2_1_split_6(self):
        """Test understanding of modulo with negative numbers."""
        # Arrange
        -1 % 2 == 1
        -3 % 2 == 1
        -2 % 2 == 0
        -4 % 2 == 0
        to_even(-1) == -1 - 1
        # Act
        # Assert
        assert to_even(-3) == -3 - 1

    def test_modulo_behavior_n_1_2_1_split_7(self):
        """Test understanding of modulo with negative numbers."""
        # Arrange
        -1 % 2 == 1
        -3 % 2 == 1
        -2 % 2 == 0
        -4 % 2 == 0
        to_even(-1) == -1 - 1
        to_even(-3) == -3 - 1
        # Act
        # Assert
        assert to_even(-2) == -2 - 0

    def test_modulo_behavior_n_1_2_1_split_8(self):
        """Test understanding of modulo with negative numbers."""
        # Arrange
        -1 % 2 == 1
        -3 % 2 == 1
        -2 % 2 == 0
        -4 % 2 == 0
        to_even(-1) == -1 - 1
        to_even(-3) == -3 - 1
        to_even(-2) == -2 - 0
        # Act
        # Assert
        assert to_even(-4) == -4 - 0


class TestToEvenDocumentation:
    """Test documentation and examples."""

    def test_docstring_examples_to_even_5_4_split_1(self):
        """Test examples from the docstring."""
        # Arrange
        # Act
        # Assert
        assert to_even(5) == 4

    def test_docstring_examples_to_even_5_4_split_2(self):
        """Test examples from the docstring."""
        # Arrange
        to_even(5) == 4
        # Act
        # Assert
        assert to_even(6) == 6

    def test_docstring_examples_to_even_5_4_split_3(self):
        """Test examples from the docstring."""
        # Arrange
        to_even(5) == 4
        to_even(6) == 6
        # Act
        # Assert
        assert to_even(3.7) == 2

    def test_function_signature_callable_to_even_split_1(self):
        """Test function signature and attributes."""
        # Arrange
        import inspect
        # Act
        # Assert
        assert callable(to_even)

    def test_function_signature_callable_to_even_split_2(self):
        """Test function signature and attributes."""
        # Arrange
        import inspect
        callable(to_even)
        sig = inspect.signature(to_even)
        params = list(sig.parameters.keys())
        # Act
        # Assert
        assert len(params) == 1

    def test_function_signature_callable_to_even_split_3(self):
        """Test function signature and attributes."""
        # Arrange
        import inspect
        callable(to_even)
        sig = inspect.signature(to_even)
        params = list(sig.parameters.keys())
        len(params) == 1
        # Act
        # Assert
        assert params[0] == 'n'

    def test_function_signature_callable_to_even_split_4(self):
        """Test function signature and attributes."""
        # Arrange
        import inspect
        callable(to_even)
        sig = inspect.signature(to_even)
        params = list(sig.parameters.keys())
        len(params) == 1
        params[0] == 'n'
        # Act
        # Assert
        assert to_even.__doc__ is not None

    def test_function_signature_callable_to_even_split_5(self):
        """Test function signature and attributes."""
        # Arrange
        import inspect
        callable(to_even)
        sig = inspect.signature(to_even)
        params = list(sig.parameters.keys())
        len(params) == 1
        params[0] == 'n'
        to_even.__doc__ is not None
        # Act
        # Assert
        assert 'Convert a number to the nearest even number' in to_even.__doc__


class TestToEvenIntegration:
    """Test integration with other parts of the system."""

    def test_import_from_package_split_1(self):
        """Test different import methods."""
        # Arrange
        import scitex_gen
        from scitex_gen import to_even as to_even1
        to_even2 = scitex_gen.to_even
        # Act
        # Assert
        assert to_even1 is to_even2

    def test_import_from_package_split_2(self):
        """Test different import methods."""
        # Arrange
        import scitex_gen
        from scitex_gen import to_even as to_even1
        to_even2 = scitex_gen.to_even
        to_even1 is to_even2
        # Act
        # Assert
        assert to_even1(5) == 4

    def test_import_from_package_split_3(self):
        """Test different import methods."""
        # Arrange
        import scitex_gen
        from scitex_gen import to_even as to_even1
        to_even2 = scitex_gen.to_even
        to_even1 is to_even2
        to_even1(5) == 4
        # Act
        # Assert
        assert to_even2(5) == 4

    def test_with_other_gen_functions(self):
        """Test interaction with other gen module functions."""
        # Arrange
        try:
            from scitex_gen import to_odd
        except ImportError:
            # to_odd might not exist, that's okay
            return
        # Act
        results = []
        for val in [1, 2, 3, 4, 5, 6]:
            even_result = to_even(val)
            odd_result = to_odd(val)
            results.append((even_result % 2, odd_result % 2, abs(even_result - odd_result)))
        # Assert
        assert all(r == (0, 1, 1) for r in results)

    def test_type_consistency_smoke_case_split_1(self):
        """Test that return type is always int."""
        # Arrange
        test_inputs = [1, 2, 3.5, 4.5, -1, -2.5, np.float32(5.5), np.int64(6), True, False, decimal.Decimal('7.5')]
        # Act
        # Assert
        for inp in test_inputs:
            result = to_even(inp)
            assert isinstance(result, int)
            not isinstance(result, bool)

    def test_type_consistency_smoke_case_split_2(self):
        """Test that return type is always int."""
        # Arrange
        test_inputs = [1, 2, 3.5, 4.5, -1, -2.5, np.float32(5.5), np.int64(6), True, False, decimal.Decimal('7.5')]
        for inp in test_inputs:
            result = to_even(inp)
            isinstance(result, int)
            not isinstance(result, bool)
        # Act
        # Assert
        for inp in test_inputs:
            result = to_even(inp)
            isinstance(result, int)
            assert not isinstance(result, bool)


class TestToEvenRobustness:
    """Test robustness and edge cases."""

    def test_thread_safety_len_results_len_test_values_split_1(self):
        """Test basic thread safety (function should be stateless)."""
        # Arrange
        import threading
        results = []

        def worker(value):
            result = to_even(value)
            results.append((value, result))
        threads = []
        test_values = list(range(100))
        for val in test_values:
            t = threading.Thread(target=worker, args=(val,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        # Act
        # Assert
        assert len(results) == len(test_values)

    def test_thread_safety_len_results_len_test_values_split_2(self):
        """Test basic thread safety (function should be stateless)."""
        # Arrange
        import threading
        results = []

        def worker(value):
            result = to_even(value)
            results.append((value, result))
        threads = []
        test_values = list(range(100))
        for val in test_values:
            t = threading.Thread(target=worker, args=(val,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        len(results) == len(test_values)
        # Act
        # Assert
        for val, result in results:
            expected = int(val) - int(val) % 2
            assert result == expected

    def test_repeated_calls_smoke_case_split_1_split_1(self):
        """Test repeated calls with same and different values."""
        # Arrange
        # Act
        # Assert
        for _ in range(100):
            assert to_even(5) == 4
            to_even(6) == 6

    def test_repeated_calls_smoke_case_split_1_split_2(self):
        """Test repeated calls with same and different values."""
        # Arrange
        for _ in range(100):
            to_even(5) == 4
            to_even(6) == 6
        # Act
        # Assert
        for _ in range(100):
            to_even(5) == 4
            assert to_even(6) == 6

    def test_repeated_calls_smoke_case_split_2_split_1(self):
        """Test repeated calls with same and different values."""
        # Arrange
        for _ in range(100):
            to_even(5) == 4
            to_even(6) == 6
        # Act
        # Assert
        for i in range(100):
            result = to_even(i)
            assert result % 2 == 0
            result <= i
            i - result <= 1

    def test_repeated_calls_smoke_case_split_2_split_2(self):
        """Test repeated calls with same and different values."""
        # Arrange
        for _ in range(100):
            to_even(5) == 4
            to_even(6) == 6
        for i in range(100):
            result = to_even(i)
            result % 2 == 0
            result <= i
            i - result <= 1
        # Act
        # Assert
        for i in range(100):
            result = to_even(i)
            result % 2 == 0
            assert result <= i
            i - result <= 1

    def test_repeated_calls_smoke_case_split_2_split_3(self):
        """Test repeated calls with same and different values."""
        # Arrange
        for _ in range(100):
            to_even(5) == 4
            to_even(6) == 6
        for i in range(100):
            result = to_even(i)
            result % 2 == 0
            result <= i
            i - result <= 1
        for i in range(100):
            result = to_even(i)
            result % 2 == 0
            result <= i
            i - result <= 1
        # Act
        # Assert
        for i in range(100):
            result = to_even(i)
            result % 2 == 0
            result <= i
            assert i - result <= 1

    def test_memory_efficiency_final_objects_initial_objects_100(self):
        """Test that function doesn't leak memory."""
        # Arrange
        import gc

        # Get initial object count
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Call function many times
        for i in range(1000):
            to_even(i)

        # Check object count hasn't grown significantly
        gc.collect()
        # Act
        final_objects = len(gc.get_objects())

        # Allow some growth for test infrastructure
        # Assert
        assert final_objects - initial_objects < 100


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
