#!/usr/bin/env python3
# Time-stamp: "2025-05-31 22:20:00 (claude)"
# File: ./tests/scitex/gen/test__to_odd.py

"""
Comprehensive tests for scitex_gen._to_odd module.

This module tests:
- to_odd function with various numeric inputs
- Edge cases and special values
- Type handling
"""

import pytest

pytest.importorskip("torch")
import numpy as np

from scitex_gen import to_odd


class TestToOddBasic:
    """Test basic to_odd functionality."""

    def test_even_integers_to_odd_2_1(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(2) == 1

    def test_even_integers_to_odd_4_3(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(4) == 3

    def test_even_integers_to_odd_6_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(6) == 5

    def test_even_integers_to_odd_8_7(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(8) == 7

    def test_even_integers_to_odd_10_9(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(10) == 9

    def test_even_integers_to_odd_100_99(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(100) == 99


    def test_odd_integers_to_odd_1_1(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(1) == 1

    def test_odd_integers_to_odd_3_3(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(3) == 3

    def test_odd_integers_to_odd_5_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(5) == 5

    def test_odd_integers_to_odd_7_7(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(7) == 7

    def test_odd_integers_to_odd_9_9(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(9) == 9

    def test_odd_integers_to_odd_99_99(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(99) == 99


    def test_zero_to_odd_0_1(self):
        """Test conversion of zero."""

        # 0 is even, so should become -1
        # Arrange
        # Act
        # Assert
        assert to_odd(0) == -1

    def test_negative_even_to_odd_2_3(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-2) == -3

    def test_negative_even_to_odd_4_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-4) == -5

    def test_negative_even_to_odd_6_7(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-6) == -7

    def test_negative_even_to_odd_8_9(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-8) == -9


    def test_negative_odd_to_odd_1_1(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-1) == -1

    def test_negative_odd_to_odd_3_3(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-3) == -3

    def test_negative_odd_to_odd_5_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-5) == -5

    def test_negative_odd_to_odd_7_7(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-7) == -7

    def test_negative_odd_to_odd_9_9(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-9) == -9



class TestToOddFloats:
    """Test to_odd with floating point numbers."""

    def test_float_truncation_to_odd_5_1_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(5.1) == 5

    def test_float_truncation_to_odd_5_5_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(5.5) == 5

    def test_float_truncation_to_odd_5_9_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(5.9) == 5

    def test_float_truncation_to_odd_6_1_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(6.1) == 5

    def test_float_truncation_to_odd_6_5_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(6.5) == 5

    def test_float_truncation_to_odd_6_9_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(6.9) == 5


    def test_float_even_base_to_odd_4_1_3(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(4.1) == 3

    def test_float_even_base_to_odd_4_5_3(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(4.5) == 3

    def test_float_even_base_to_odd_4_9_3(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(4.9) == 3

    def test_float_even_base_to_odd_8_3_7(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(8.3) == 7


    def test_negative_floats_to_odd_5_1_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-5.1) == -5

    def test_negative_floats_to_odd_5_5_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-5.5) == -5

    def test_negative_floats_to_odd_5_9_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-5.9) == -5

    def test_negative_floats_to_odd_6_1_7(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-6.1) == -7

    def test_negative_floats_to_odd_6_5_7(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-6.5) == -7

    def test_negative_floats_to_odd_6_9_7(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-6.9) == -7


    def test_documentation_examples_to_odd_6_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(6) == 5

    def test_documentation_examples_to_odd_7_7(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(7) == 7

    def test_documentation_examples_to_odd_5_8_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(5.8) == 5



class TestToOddEdgeCases:
    """Test edge cases for to_odd function."""

    def test_large_numbers_to_odd_1000000_999999(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(1000000) == 999999

    def test_large_numbers_to_odd_1000001_1000001(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(1000001) == 1000001

    def test_large_numbers_to_odd_10_9_10_9_1(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(10**9) == 10**9 - 1

    def test_large_numbers_to_odd_10_9_1_10_9_1(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(10**9 + 1) == 10**9 + 1


    def test_special_floats_to_odd_0_1_1(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(0.1) == -1

    def test_special_floats_to_odd_0_5_1(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(0.5) == -1

    def test_special_floats_to_odd_0_9_1(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(0.9) == -1

    def test_special_floats_to_odd_0_1_1(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-0.1) == -1

    def test_special_floats_to_odd_0_5_1(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-0.5) == -1

    def test_special_floats_to_odd_0_9_1(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(-0.9) == -1


    def test_numpy_types_to_odd_np_int32_6_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(np.int32(6)) == 5

    def test_numpy_types_to_odd_np_int64_7_7(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(np.int64(7)) == 7

    def test_numpy_types_to_odd_np_int16_8_7(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(np.int16(8)) == 7

    def test_numpy_types_to_odd_np_float32_6_5_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(np.float32(6.5)) == 5

    def test_numpy_types_to_odd_np_float64_7_9_7(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(np.float64(7.9)) == 7


    def test_boolean_inputs_to_odd_true_1(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(True) == 1  # True -> 1 (already odd)

    def test_boolean_inputs_to_odd_false_1(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd(False) == -1  # False -> 0 -> -1



class TestToOddConsecutive:
    """Test to_odd with consecutive inputs."""

    def test_consecutive_integers_results_equals_expected(self):
        """Test pattern with consecutive integers."""

        # Arrange
        results = [to_odd(i) for i in range(10)]
        # Act
        expected = [-1, 1, 1, 3, 3, 5, 5, 7, 7, 9]
        # Assert
        assert results == expected

    def test_consecutive_negative_results_equals_expected(self):
        """Test pattern with consecutive negative integers."""

        # Arrange
        results = [to_odd(i) for i in range(-5, 6)]
        # Act
        expected = [-5, -5, -3, -3, -1, -1, 1, 1, 3, 3, 5]
        # Assert
        assert results == expected

    def test_sequence_properties_smoke_case(self):
        """Test mathematical properties of the conversion."""

        # Property: result is always odd
        # Arrange
        # Act
        # Assert
        for n in range(-20, 21):
            result = to_odd(n)
            assert result % 2 != 0, f"to_odd({n})={result} is not odd"

        # Property: result <= input
        for n in range(-20, 21):
            result = to_odd(n)
            assert result <= n, f"to_odd({n})={result} is greater than input"

        # Property: difference is at most 1
        for n in range(-20, 21):
            result = to_odd(n)
            assert n - result <= 1, f"to_odd({n})={result} differs by more than 1"


class TestToOddParameterized:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            (0, -1),
            (1, 1),
            (2, 1),
            (3, 3),
            (4, 3),
            (5, 5),
            (-1, -1),
            (-2, -3),
            (-3, -3),
            (-4, -5),
            (10.7, 9),
            (-10.7, -11),
            (0.5, -1),
            (-0.5, -1),
        ],
    )
    def test_various_inputs_to_odd_input_val_expected(self, input_val, expected):
        """Test to_odd with various inputs using parametrization."""

        # Arrange
        # Act
        # Assert
        assert to_odd(input_val) == expected

    @pytest.mark.parametrize("n", range(-100, 101, 10))
    def test_even_conversion_pattern(self, n):
        """Test that even numbers are converted correctly."""

        # Arrange
        # Act
        # Assert
        if n % 2 == 0:
            # Even numbers should become n-1
            assert to_odd(n) == n - 1
        else:
            # Odd numbers should stay the same
            assert to_odd(n) == n


class TestToOddTypeHandling:
    """Test type handling and conversions."""

    def test_string_numeric_to_odd_5_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd("5") == 5

    def test_string_numeric_to_odd_6_5(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd("6") == 5

    def test_string_numeric_to_odd_7_7(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        assert to_odd("7") == 7

    def test_string_numeric_raises_valueerror(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        with pytest.raises(ValueError):
            to_odd("5.5")


    def test_none_input_raises_typeerror(self):
        """Test behavior with None input."""

        # Arrange
        # Act
        # Assert
        with pytest.raises(TypeError):
            to_odd(None)

    def test_complex_numbers_raises_typeerror(self):
        """Test behavior with complex numbers."""

        # Complex numbers don't support int() conversion
        # Arrange
        # Act
        # Assert
        with pytest.raises(TypeError):
            to_odd(3 + 4j)

    def test_infinity_raises_valueerror_overflowerror(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        with pytest.raises((ValueError, OverflowError)):
            to_odd(float("inf"))

    def test_infinity_raises_valueerror_overflowerror(self):
        # Arrange
        # Act
        # Assert
        # Arrange
        # Act
        # Assert
        with pytest.raises((ValueError, OverflowError)):
            to_odd(float("-inf"))


    def test_nan_raises_valueerror(self):
        """Test behavior with NaN."""

        # NaN can't be converted to int
        # Arrange
        # Act
        # Assert
        with pytest.raises(ValueError):
            to_odd(float("nan"))


class TestToOddIntegration:
    """Integration tests for to_odd function."""

    def test_with_array_processing(self):
        """Test using to_odd with array processing."""

        # Process array of values
        # Arrange
        inputs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        results = [to_odd(x) for x in inputs]
        # Act
        expected = [1, 1, 3, 3, 5, 5, 7, 7, 9, 9]
        # Assert
        assert results == expected

    def test_with_numpy_vectorize(self):
        """Test using to_odd with numpy vectorize."""

        # Vectorize the function
        # Arrange
        vec_to_odd = np.vectorize(to_odd)

        # Test on array
        inputs = np.array([1, 2, 3, 4, 5, 6])
        results = vec_to_odd(inputs)
        # Act
        expected = np.array([1, 1, 3, 3, 5, 5])

        # Assert
        assert np.array_equal(results, expected)

    def test_use_case_kernel_sizes_all_s_2_0_for_s_in_odd_sizes(self):
        # Arrange
        # Arrange
        kernel_sizes = [3, 4, 5, 6, 7, 8, 9]
        # Act
        odd_sizes = [to_odd(k) for k in kernel_sizes]
        # Act
        # Assert
        # Assert
        assert all(s % 2 != 0 for s in odd_sizes)

    def test_use_case_kernel_sizes_odd_sizes_equals_n_3_3_5_5_7_7_9(self):
        # Arrange
        # Arrange
        kernel_sizes = [3, 4, 5, 6, 7, 8, 9]
        # Act
        odd_sizes = [to_odd(k) for k in kernel_sizes]
        # Act
        # Assert
        # Assert
        assert odd_sizes == [3, 3, 5, 5, 7, 7, 9]



if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_to_odd.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-25 23:40:22 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/gen/_to_odd.py
#
# THIS_FILE = "/home/ywatanabe/proj/scitex_repo/src/scitex/gen/_to_odd.py"
#
#
# def to_odd(n):
#     """Convert a number to the nearest odd number less than or equal to itself.
#
#     Parameters
#     ----------
#     n : int or float
#         The input number to be converted.
#
#     Returns
#     -------
#     int
#         The nearest odd number less than or equal to the input.
#
#     Example
#     -------
#     >>> to_odd(6)
#     5
#     >>> to_odd(7)
#     7
#     >>> to_odd(5.8)
#     5
#     """
#     return int(n) - ((int(n) + 1) % 2)
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_to_odd.py
# --------------------------------------------------------------------------------
