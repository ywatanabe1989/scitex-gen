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

    def test_odd_integers_to_even_1_0(self):
        """Test conversion of odd integers to even."""
        # Arrange
        # Act
        # Assert
        assert to_even(1) == 0
        assert to_even(3) == 2
        assert to_even(5) == 4
        assert to_even(7) == 6
        assert to_even(99) == 98
        assert to_even(1001) == 1000

    def test_even_integers_to_even_0_0(self):
        """Test that even integers remain unchanged."""
        # Arrange
        # Act
        # Assert
        assert to_even(0) == 0
        assert to_even(2) == 2
        assert to_even(4) == 4
        assert to_even(6) == 6
        assert to_even(100) == 100
        assert to_even(1000) == 1000

    def test_negative_integers_to_even_1_2(self):
        """Test conversion of negative integers."""
        # Arrange
        # Act
        # Assert
        assert to_even(-1) == -2
        assert to_even(-2) == -2
        assert to_even(-3) == -4
        assert to_even(-4) == -4
        assert to_even(-99) == -100

    def test_floats_to_even_3_7_2(self):
        """Test conversion of float values."""
        # Arrange
        # Act
        # Assert
        assert to_even(3.7) == 2
        assert to_even(4.9) == 4
        assert to_even(5.1) == 4
        assert to_even(6.0) == 6
        assert to_even(2.3) == 2
        assert to_even(1.9) == 0

    def test_negative_floats_to_even_1_5_2(self):
        """Test conversion of negative float values."""
        # Arrange
        # Act
        # Assert
        assert to_even(-1.5) == -2
        assert to_even(-2.3) == -4
        assert to_even(-3.7) == -4
        assert to_even(-4.1) == -6

    def test_edge_cases_to_even_0_0(self):
        """Test edge cases and special values."""
        # Arrange
        # Act
        # Assert
        assert to_even(0) == 0
        assert to_even(0.0) == 0
        assert to_even(0.1) == 0
        assert to_even(-0.1) == -2

    def test_large_numbers_to_even_1000000_1000000(self):
        """Test with large numbers."""
        # Arrange
        # Act
        # Assert
        assert to_even(1000000) == 1000000
        assert to_even(1000001) == 1000000
        assert to_even(999999) == 999998
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

    def test_numpy_integers_to_even_np_int8_5_4(self):
        """Test with numpy integer types."""
        # Arrange
        # Act
        # Assert
        assert to_even(np.int8(5)) == 4
        assert to_even(np.int16(7)) == 6
        assert to_even(np.int32(9)) == 8
        assert to_even(np.int64(11)) == 10
        assert to_even(np.uint8(13)) == 12
        assert to_even(np.uint16(15)) == 14
        assert to_even(np.uint32(17)) == 16
        assert to_even(np.uint64(19)) == 18

    def test_numpy_floats_to_even_np_float16_3_5_2(self):
        """Test with numpy float types."""
        # Arrange
        # Act
        # Assert
        assert to_even(np.float16(3.5)) == 2
        assert to_even(np.float32(5.7)) == 4
        assert to_even(np.float64(7.9)) == 6
        if hasattr(np, "float128"):
            assert to_even(np.float128(9.1)) == 8

    def test_numpy_arrays_to_even_np_array_5_4(self):
        """Test with numpy arrays (should work on scalar elements)."""
        # Single element arrays
        # Arrange
        # Act
        # Assert
        assert to_even(np.array(5)) == 4
        assert to_even(np.array(6)) == 6
        assert to_even(np.array([7])[0]) == 6
        assert to_even(np.array([8])[0]) == 8

    def test_python_numeric_types(self):
        """Test with various Python numeric types."""
        # Complex numbers should raise TypeError when converted to int
        # Arrange
        # Act
        # Assert
        with pytest.raises(TypeError):
            to_even(complex(5, 0))

        # Boolean values
        assert to_even(True) == 0  # True -> 1 -> 0
        assert to_even(False) == 0  # False -> 0 -> 0

        # Decimal
        assert to_even(decimal.Decimal("5.5")) == 4
        assert to_even(decimal.Decimal("6.0")) == 6
        assert to_even(decimal.Decimal("-3.7")) == -4


class TestToEvenEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    def test_very_large_numbers(self):
        """Test with very large numbers."""
        # Arrange
        # Act
        # Assert
        assert to_even(10**18 + 1) == 10**18
        assert to_even(10**18) == 10**18
        assert to_even(2**31 - 1) == 2**31 - 2  # Max 32-bit int - 1
        assert to_even(2**31) == 2**31
        assert to_even(2**63 - 1) == 2**63 - 2  # Max 64-bit int - 1

    def test_very_small_negative_numbers(self):
        """Test with very small negative numbers."""
        # Arrange
        # Act
        # Assert
        assert to_even(-(10**18) - 1) == -(10**18) - 2
        assert to_even(-(10**18)) == -(10**18)
        assert to_even(-(2**31) + 1) == -(2**31)
        assert to_even(-(2**31)) == -(2**31)
        assert to_even(-(2**63) + 1) == -(2**63)

    def test_near_zero_values(self):
        """Test values very close to zero."""
        # Arrange
        # Act
        # Assert
        assert to_even(0.0001) == 0
        assert to_even(0.9999) == 0
        assert to_even(-0.0001) == -2
        assert to_even(-0.9999) == -2
        assert to_even(1e-10) == 0
        assert to_even(-1e-10) == -2

    def test_infinity_and_nan(self):
        """Test with infinity and NaN values."""
        # Positive infinity
        # Arrange
        # Act
        # Assert
        with pytest.raises(OverflowError):
            to_even(float("inf"))

        # Negative infinity
        with pytest.raises(OverflowError):
            to_even(float("-inf"))

        # NaN
        with pytest.raises(ValueError):
            to_even(float("nan"))

    def test_special_float_values(self):
        """Test special floating point values."""
        # Subnormal numbers
        # Arrange
        # Act
        # Assert
        assert to_even(sys.float_info.min) == 0
        assert to_even(-sys.float_info.min) == -2

        # Maximum float - Python can actually convert this to int
        # The result should be a very large even integer
        result = to_even(sys.float_info.max)
        assert isinstance(result, int)
        assert result % 2 == 0  # Should be even
        # Test actual overflow with infinity
        with pytest.raises(OverflowError):
            to_even(float("inf"))


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
        # Act
        # Assert
        values = [3.2, 3.7, 4.0, 4.5, -3.2, -3.7, -4.0, -4.5]
        for val in values:
            result = to_even(val)
            floor_val = math.floor(val)
            if floor_val % 2 == 0:
                assert result == floor_val
            else:
                assert result == floor_val - 1


class TestToEvenErrorHandling:
    """Test error handling and invalid inputs."""

    def test_non_numeric_types(self):
        """Test with non-numeric types."""
        # Strings that can't be converted to numbers
        # Arrange
        # Act
        # Assert
        with pytest.raises((TypeError, ValueError)):
            to_even("hello")

        with pytest.raises(TypeError):
            to_even([1, 2, 3])

        with pytest.raises(TypeError):
            to_even({1: 2})

        with pytest.raises(TypeError):
            to_even(None)

    def test_string_numbers_raises_typeerror(self):
        """Test with string representations of numbers."""
        # The implementation uses math.floor() which raises TypeError for strings
        # Arrange
        # Act
        # Assert
        with pytest.raises(TypeError):
            to_even("5")

        with pytest.raises(TypeError):
            to_even("5.5")

    def test_custom_objects_to_even_customnumber_5_4(self):
        """Test with custom objects."""

        # Arrange
        # Act
        class CustomNumber:
            def __init__(self, value):
                self.value = value

            def __int__(self):
                return int(self.value)

        # Should work if object implements __int__
        # Assert
        assert to_even(CustomNumber(5)) == 4
        assert to_even(CustomNumber(6)) == 6

        # Object without __int__
        class BadObject:
            pass

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

    def test_batch_operations_len_results_len_values(self):
        """Test performance with batch operations."""
        # Create test data
        # Arrange
        values = list(range(1000))

        # Time the operation
        import time

        start = time.perf_counter()
        results = [to_even(v) for v in values]
        # Act
        end = time.perf_counter()

        # Verify results
        # Assert
        assert len(results) == len(values)
        assert all(r % 2 == 0 for r in results)

        # Performance should be reasonable (< 0.1 seconds for 1000 values)
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

    def test_modulo_behavior_n_1_2_1(self):
        """Test understanding of modulo with negative numbers."""
        # Python's modulo with negative numbers
        # Arrange
        # Act
        # Assert
        assert (-1) % 2 == 1  # Not -1
        assert (-3) % 2 == 1
        assert (-2) % 2 == 0
        assert (-4) % 2 == 0

        # Verify to_even handles this correctly
        assert to_even(-1) == -1 - 1  # -2
        assert to_even(-3) == -3 - 1  # -4
        assert to_even(-2) == -2 - 0  # -2
        assert to_even(-4) == -4 - 0  # -4


class TestToEvenDocumentation:
    """Test documentation and examples."""

    def test_docstring_examples_to_even_5_4(self):
        """Test examples from the docstring."""
        # From docstring:
        # >>> to_even(5)
        # 4
        # Arrange
        # Act
        # Assert
        assert to_even(5) == 4

        # >>> to_even(6)
        # 6
        assert to_even(6) == 6

        # >>> to_even(3.7)
        # 2
        assert to_even(3.7) == 2

    def test_function_signature_callable_to_even(self):
        """Test function signature and attributes."""
        # Arrange
        # Act
        import inspect

        # Check function exists and is callable
        # Assert
        assert callable(to_even)

        # Check signature
        sig = inspect.signature(to_even)
        params = list(sig.parameters.keys())
        assert len(params) == 1
        assert params[0] == "n"

        # Check docstring exists
        assert to_even.__doc__ is not None
        assert "Convert a number to the nearest even number" in to_even.__doc__


class TestToEvenIntegration:
    """Test integration with other parts of the system."""

    def test_import_from_package(self):
        """Test different import methods."""
        # Direct import
        # Import module then access
        # Arrange
        import scitex_gen
        from scitex_gen import to_even as to_even1

        # Act
        to_even2 = scitex_gen.to_even

        # Both should be the same function
        # Assert
        assert to_even1 is to_even2

        # Both should work correctly
        assert to_even1(5) == 4
        assert to_even2(5) == 4

    def test_with_other_gen_functions(self):
        """Test interaction with other gen module functions."""
        # Test with to_odd if it exists
        # Arrange
        # Act
        # Assert
        try:
            from scitex_gen import to_odd

            # to_even and to_odd should have complementary behavior
            for val in [1, 2, 3, 4, 5, 6]:
                even_result = to_even(val)
                assert even_result % 2 == 0
                # If to_odd exists, verify relationship
                odd_result = to_odd(val)
                assert odd_result % 2 == 1
                assert abs(even_result - odd_result) == 1
        except ImportError:
            # to_odd might not exist, that's okay
            pass

    def test_type_consistency_smoke_case(self):
        """Test that return type is always int."""
        # Arrange
        # Act
        # Assert
        test_inputs = [
            1,
            2,
            3.5,
            4.5,
            -1,
            -2.5,
            np.float32(5.5),
            np.int64(6),
            True,
            False,
            decimal.Decimal("7.5"),
        ]

        for inp in test_inputs:
            result = to_even(inp)
            assert isinstance(result, int)
            assert not isinstance(result, bool)  # bool is subclass of int


class TestToEvenRobustness:
    """Test robustness and edge cases."""

    def test_thread_safety_len_results_len_test_values(self):
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

        # Act
        for t in threads:
            t.join()

        # Verify all results are correct
        # Assert
        assert len(results) == len(test_values)
        for val, result in results:
            expected = int(val) - (int(val) % 2)
            assert result == expected

    def test_repeated_calls_smoke_case(self):
        """Test repeated calls with same and different values."""
        # Same value repeatedly
        # Arrange
        # Act
        # Assert
        for _ in range(100):
            assert to_even(5) == 4
            assert to_even(6) == 6

        # Different values
        for i in range(100):
            result = to_even(i)
            assert result % 2 == 0
            assert result <= i
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
