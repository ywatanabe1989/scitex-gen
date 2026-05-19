#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-31 21:00:00 (Claude)"
# File: /tests/scitex/gen/test__symlog.py

import pytest

pytest.importorskip("torch")
import numpy as np

from scitex_gen import symlog


class TestSymlog:
    """Test cases for symmetric log transformation function."""

    def test_symlog_zero_result_equals_n_0(self):
        """Test symlog with zero input."""
        # Arrange
        # Act
        result = symlog(0)
        # Assert
        assert result == 0

        # Array of zeros
        result_array = symlog(np.array([0, 0, 0]))
        np.testing.assert_array_equal(result_array, np.array([0, 0, 0]))

    def test_symlog_positive_values_np_isclose_result_expected(self):
        # Arrange
        # Arrange
        result = symlog(0.5, linthresh=1.0)
        # Act
        expected = np.log1p(0.5 / 1.0)
        # Act
        # Assert
        # Assert
        assert np.isclose(result, expected)

    def test_symlog_positive_values_np_isclose_result_expected(self):
        # Arrange
        # Arrange
        result = symlog(0.5, linthresh=1.0)
        # Act
        expected = np.log1p(0.5 / 1.0)
        # Assert
        assert np.isclose(result, expected)
        # Value equal to linthresh
        result = symlog(1.0, linthresh=1.0)
        expected = np.log1p(1.0)
        # Act
        # Assert
        assert np.isclose(result, expected)

    def test_symlog_positive_values_np_isclose_result_expected(self):
        # Arrange
        # Arrange
        result = symlog(0.5, linthresh=1.0)
        # Act
        expected = np.log1p(0.5 / 1.0)
        # Assert
        assert np.isclose(result, expected)
        # Value equal to linthresh
        result = symlog(1.0, linthresh=1.0)
        expected = np.log1p(1.0)
        assert np.isclose(result, expected)
        # Large value > linthresh
        result = symlog(10.0, linthresh=1.0)
        expected = np.log1p(10.0)
        # Act
        # Assert
        assert np.isclose(result, expected)


    def test_symlog_negative_values_np_isclose_result_expected(self):
        # Arrange
        # Arrange
        result = symlog(-0.5, linthresh=1.0)
        # Act
        expected = -np.log1p(0.5)
        # Act
        # Assert
        # Assert
        assert np.isclose(result, expected)

    def test_symlog_negative_values_np_isclose_result_expected(self):
        # Arrange
        # Arrange
        result = symlog(-0.5, linthresh=1.0)
        # Act
        expected = -np.log1p(0.5)
        # Assert
        assert np.isclose(result, expected)
        # Large negative value
        result = symlog(-10.0, linthresh=1.0)
        expected = -np.log1p(10.0)
        # Act
        # Assert
        assert np.isclose(result, expected)


    def test_symlog_symmetry_np_allclose_pos_results_neg_results(self):
        """Test that symlog is symmetric around zero."""
        # Arrange
        values = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        pos_results = symlog(values)
        # Act
        neg_results = symlog(-values)

        # Should be equal in magnitude but opposite in sign
        # Assert
        assert np.allclose(pos_results, -neg_results)

    def test_symlog_array_input_result_shape_equals_x_shape(self):
        # Arrange
        # Arrange
        x = np.array([-10, -1, -0.1, 0, 0.1, 1, 10])
        # Act
        result = symlog(x)
        # Act
        # Assert
        # Assert
        assert result.shape == x.shape

    def test_symlog_array_input_result_3_0(self):
        # Arrange
        # Arrange
        x = np.array([-10, -1, -0.1, 0, 0.1, 1, 10])
        # Act
        result = symlog(x)
        # Act
        # Assert
        # Assert
        assert result[3] == 0


    def test_symlog_different_linthresh(self):
        """Test symlog with different linear threshold values."""
        # Arrange
        x = np.array([0.5, 1.0, 2.0, 5.0])

        # Smaller linthresh makes transformation more aggressive
        result_small = symlog(x, linthresh=0.1)
        # Act
        result_large = symlog(x, linthresh=10.0)

        # With smaller linthresh, values should be larger (more log-like)
        # Assert
        assert np.all(result_small > result_large)

    def test_symlog_preserves_sign(self):
        """Test that symlog preserves the sign of input values."""
        # Arrange
        x = np.array([-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5])
        # Act
        result = symlog(x)

        # Signs should match
        # Assert
        assert np.array_equal(np.sign(x), np.sign(result))

    def test_symlog_monotonic_np_all_differences_0(self):
        """Test that symlog is monotonically increasing."""
        # Arrange
        x = np.linspace(-10, 10, 100)
        result = symlog(x)

        # Check that differences are all positive (monotonic increasing)
        # Act
        differences = np.diff(result)
        # Assert
        assert np.all(differences > 0)

    def test_symlog_edge_cases_result_pos_0(self):
        # Arrange
        # Arrange
        tiny = 1e-10
        result_pos = symlog(tiny)
        # Act
        result_neg = symlog(-tiny)
        # Act
        # Assert
        # Assert
        assert result_pos > 0

    def test_symlog_edge_cases_result_neg_0(self):
        # Arrange
        # Arrange
        tiny = 1e-10
        result_pos = symlog(tiny)
        # Act
        result_neg = symlog(-tiny)
        # Act
        # Assert
        # Assert
        assert result_neg < 0

    def test_symlog_edge_cases_np_isclose_result_pos_result_neg(self):
        # Arrange
        # Arrange
        tiny = 1e-10
        result_pos = symlog(tiny)
        # Act
        result_neg = symlog(-tiny)
        # Act
        # Assert
        # Assert
        assert np.isclose(result_pos, -result_neg)

    def test_symlog_edge_cases_result_pos_0(self):
        # Arrange
        # Arrange
        tiny = 1e-10
        result_pos = symlog(tiny)
        # Act
        result_neg = symlog(-tiny)
        # Assert
        assert result_pos > 0
        assert result_neg < 0
        assert np.isclose(result_pos, -result_neg)
        # Very large values
        large = 1e10
        result_pos = symlog(large)
        result_neg = symlog(-large)
        # Act
        # Assert
        assert result_pos > 0

    def test_symlog_edge_cases_result_neg_0(self):
        # Arrange
        # Arrange
        tiny = 1e-10
        result_pos = symlog(tiny)
        # Act
        result_neg = symlog(-tiny)
        # Assert
        assert result_pos > 0
        assert result_neg < 0
        assert np.isclose(result_pos, -result_neg)
        # Very large values
        large = 1e10
        result_pos = symlog(large)
        result_neg = symlog(-large)
        # Act
        # Assert
        assert result_neg < 0


    def test_symlog_nan_handling_not_np_isnan_result_0(self):
        # Arrange
        # Arrange
        x = np.array([1.0, np.nan, -1.0])
        # Act
        result = symlog(x)
        # Act
        # Assert
        # Assert
        assert not np.isnan(result[0])

    def test_symlog_nan_handling_np_isnan_result_1(self):
        # Arrange
        # Arrange
        x = np.array([1.0, np.nan, -1.0])
        # Act
        result = symlog(x)
        # Act
        # Assert
        # Assert
        assert np.isnan(result[1])

    def test_symlog_nan_handling_not_np_isnan_result_2(self):
        # Arrange
        # Arrange
        x = np.array([1.0, np.nan, -1.0])
        # Act
        result = symlog(x)
        # Act
        # Assert
        # Assert
        assert not np.isnan(result[2])


    def test_symlog_inf_handling_result_pos_inf_equals_np_inf(self):
        # Arrange
        # Arrange
        result_pos_inf = symlog(np.inf)
        # Act
        result_neg_inf = symlog(-np.inf)
        # Act
        # Assert
        # Assert
        assert result_pos_inf == np.inf

    def test_symlog_inf_handling_result_neg_inf_equals_np_inf(self):
        # Arrange
        # Arrange
        result_pos_inf = symlog(np.inf)
        # Act
        result_neg_inf = symlog(-np.inf)
        # Act
        # Assert
        # Assert
        assert result_neg_inf == -np.inf


    def test_symlog_linthresh_validation(self):
        """Test symlog with different linthresh edge cases."""
        # Arrange
        x = np.array([1.0, 2.0, 3.0])

        # Very small linthresh
        # Act
        result = symlog(x, linthresh=1e-10)
        # Assert
        assert np.all(np.isfinite(result))

        # Zero linthresh should work but might cause division issues
        with np.errstate(divide="ignore"):
            result = symlog(x, linthresh=0)
            # Should handle gracefully, likely returning inf
            assert np.all(result == np.inf)

    def test_symlog_dtype_preservation_result_f32_dtype_equals_np_float32(self):
        # Arrange
        # Arrange
        x_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # Act
        result_f32 = symlog(x_f32)
        # Act
        # Assert
        # Assert
        assert result_f32.dtype == np.float32

    def test_symlog_dtype_preservation_result_f64_dtype_equals_np_float64(self):
        # Arrange
        # Arrange
        x_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # Act
        result_f32 = symlog(x_f32)
        # Assert
        assert result_f32.dtype == np.float32
        # Float64 input
        x_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result_f64 = symlog(x_f64)
        # Act
        # Assert
        assert result_f64.dtype == np.float64


    @pytest.mark.parametrize(
        "x,linthresh,expected_sign",
        [
            (5.0, 1.0, 1),
            (-5.0, 1.0, -1),
            (0.0, 1.0, 0),
            (0.1, 0.1, 1),
            (-0.1, 0.1, -1),
        ],
    )
    def test_symlog_parametrized_np_sign_result_expected_sign(self, x, linthresh, expected_sign):
        """Parametrized test for various inputs."""
        # Arrange
        # Act
        result = symlog(x, linthresh)
        # Assert
        assert np.sign(result) == expected_sign

        # For non-zero values, check magnitude
        if x != 0:
            expected_magnitude = np.log1p(abs(x) / linthresh)
            assert np.isclose(abs(result), expected_magnitude)

    def test_symlog_scalar_vs_array(self):
        """Test that scalar and array inputs give consistent results."""
        # Arrange
        scalar_result = symlog(5.0, linthresh=2.0)
        # Act
        array_result = symlog(np.array([5.0]), linthresh=2.0)

        # Assert
        assert np.isclose(scalar_result, array_result[0])


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_symlog.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-07-06 07:16:38 (ywatanabe)"
# # ./src/scitex/gen/_symlog.py
#
# import numpy as np
#
#
# def symlog(x, linthresh=1.0):
#     """
#     Apply a symmetric log transformation to the input data.
#
#     Parameters
#     ----------
#     x : array-like
#         Input data to be transformed.
#     linthresh : float, optional
#         Range within which the transformation is linear. Defaults to 1.0.
#
#     Returns
#     -------
#     array-like
#         Symmetrically transformed data.
#     """
#     sign_x = np.sign(x)
#     abs_x = np.abs(x)
#     return sign_x * (np.log1p(abs_x / linthresh))

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_symlog.py
# --------------------------------------------------------------------------------
