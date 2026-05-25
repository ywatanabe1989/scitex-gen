#!/usr/bin/env python3
# Timestamp: "2025-05-31 19:50:00 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/.claude-worktree/scitex_repo/tests/scitex/gen/test__norm.py

import pytest

torch = pytest.importorskip("torch")
import numpy as np

from scitex_gen import clip_perc, to_01, to_nan01, to_nanz, to_z, unbias


class TestNormalizationFunctions:
    """Test cases for normalization functions in scitex_gen._norm."""

    @pytest.fixture
    def sample_tensor(self):
        """Create a sample tensor for testing."""
        return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    @pytest.fixture
    def tensor_with_nan(self):
        """Create a tensor with NaN values."""
        return torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])

    def test_to_z_basic_result_shape_equals_sample_tensor_shape(self, sample_tensor):
        # Arrange
        # Arrange
        # Act
        result = to_z(sample_tensor, dim=-1, device="cpu")
        # Act
        # Assert
        # Assert
        assert result.shape == sample_tensor.shape

    def test_to_z_basic_torch_allclose_result_mean_dim_1_torch_zeros_2_atol_1e_06(self, sample_tensor):
        # Arrange
        # Arrange
        # Act
        result = to_z(sample_tensor, dim=-1, device="cpu")
        # Act
        # Assert
        # Assert
        assert torch.allclose(result.mean(dim=-1), torch.zeros(2), atol=1e-6)

    def test_to_z_basic_torch_allclose_result_std_dim_1_torch_ones_2_atol_1e_06(self, sample_tensor):
        # Arrange
        # Arrange
        # Act
        result = to_z(sample_tensor, dim=-1, device="cpu")
        # Act
        # Assert
        # Assert
        assert torch.allclose(result.std(dim=-1), torch.ones(2), atol=1e-6)


    def test_to_z_different_dims_torch_allclose_result_dim0_mean_dim_0_torch_zeros_3_atol_1e_(self, sample_tensor):
        # Arrange
        # Arrange
        # Act
        result_dim0 = to_z(sample_tensor, dim=0, device="cpu")
        # Act
        # Assert
        # Assert
        assert torch.allclose(result_dim0.mean(dim=0), torch.zeros(3), atol=1e-6)

    def test_to_z_different_dims_torch_allclose_result_dim0_std_dim_0_torch_ones_3_atol_1e_06(self, sample_tensor):
        # Arrange
        # Arrange
        # Act
        result_dim0 = to_z(sample_tensor, dim=0, device="cpu")
        # Act
        # Assert
        # Assert
        assert torch.allclose(result_dim0.std(dim=0), torch.ones(3), atol=1e-6)

    def test_to_z_different_dims_torch_allclose_result_dim1_mean_dim_1_torch_zeros_2_atol_1e_(self, sample_tensor):
        # Arrange
        # Act
        result_dim1 = to_z(sample_tensor, dim=1, device="cpu")
        # Assert
        assert torch.allclose(result_dim1.mean(dim=1), torch.zeros(2), atol=1e-6)

    def test_to_z_different_dims_torch_allclose_result_dim1_std_dim_1_torch_ones_2_atol_1e_06(self, sample_tensor):
        # Arrange
        # Act
        result_dim1 = to_z(sample_tensor, dim=1, device="cpu")
        # Assert
        assert torch.allclose(result_dim1.std(dim=1), torch.ones(2), atol=1e-6)


    def test_to_nanz_with_nan_result_shape_equals_tensor_with_nan_shape(self, tensor_with_nan):
        # Arrange
        # Arrange
        # Act
        result = to_nanz(tensor_with_nan, dim=-1, device="cpu")
        # Act
        # Assert
        # Assert
        assert result.shape == tensor_with_nan.shape

    def test_to_nanz_with_nan_torch_allclose_result_0_0_torch_tensor_expected_atol_1e_06(self, tensor_with_nan):
        # Arrange
        # For first row [1.0, nan, 3.0]: mean=2, std=sqrt(2) (Bessel-corrected).
        # z = (x - mean) / std → (1-2)/sqrt(2) ≈ -0.7071, (3-2)/sqrt(2) ≈ 0.7071.
        expected = 1.0 / (2**0.5)
        # Act
        result = to_nanz(tensor_with_nan, dim=-1, device="cpu")
        # Assert
        assert torch.allclose(result[0, 0], torch.tensor(-expected), atol=1e-6)

    def test_to_nanz_with_nan_torch_allclose_result_0_2_torch_tensor_expected_atol_1e_06(self, tensor_with_nan):
        # Arrange
        # For first row [1.0, nan, 3.0]: mean=2, std=sqrt(2) (Bessel-corrected).
        # z = (x - mean) / std → (1-2)/sqrt(2) ≈ -0.7071, (3-2)/sqrt(2) ≈ 0.7071.
        expected = 1.0 / (2**0.5)
        # Act
        result = to_nanz(tensor_with_nan, dim=-1, device="cpu")
        # Assert
        assert torch.allclose(result[0, 2], torch.tensor(expected), atol=1e-6)

    def test_to_nanz_with_nan_torch_isnan_result_0_1(self, tensor_with_nan):
        # Arrange
        # Act
        result = to_nanz(tensor_with_nan, dim=-1, device="cpu")
        # Assert
        assert torch.isnan(result[0, 1])


    def test_to_01_basic_result_shape_equals_sample_tensor_shape(self, sample_tensor):
        # Arrange
        # Arrange
        # Act
        result = to_01(sample_tensor, dim=-1, device="cpu")
        # Act
        # Assert
        # Assert
        assert result.shape == sample_tensor.shape

    def test_to_01_basic_torch_allclose_result_min_dim_1_0_torch_zeros_2(self, sample_tensor):
        # Arrange
        # Arrange
        # Act
        result = to_01(sample_tensor, dim=-1, device="cpu")
        # Act
        # Assert
        # Assert
        assert torch.allclose(result.min(dim=-1)[0], torch.zeros(2))

    def test_to_01_basic_torch_allclose_result_max_dim_1_0_torch_ones_2(self, sample_tensor):
        # Arrange
        # Arrange
        # Act
        result = to_01(sample_tensor, dim=-1, device="cpu")
        # Act
        # Assert
        # Assert
        assert torch.allclose(result.max(dim=-1)[0], torch.ones(2))


    def test_to_01_edge_case(self):
        """Test min-max scaling with constant values."""
        # All values are the same
        # Arrange
        constant_tensor = torch.ones(2, 3) * 5.0
        # Act
        result = to_01(constant_tensor, dim=-1, device="cpu")
        # Result should handle division by zero gracefully
        # Assert
        assert result.shape == constant_tensor.shape

    @pytest.mark.skipif(
        not hasattr(torch, "nanmin"),
        reason="torch.nanmin not available in this PyTorch version",
    )
    def test_to_nan01_with_nan_result_shape_equals_tensor_with_nan_shape(self, tensor_with_nan):
        # Arrange
        # Arrange
        # Act
        result = to_nan01(tensor_with_nan, dim=-1, device="cpu")
        # Act
        # Assert
        # Assert
        assert result.shape == tensor_with_nan.shape

    @pytest.mark.skipif(
        not hasattr(torch, "nanmin"),
        reason="torch.nanmin not available in this PyTorch version",
    )
    def test_to_nan01_with_nan_torch_allclose_result_0_0_torch_tensor_0_0_atol_1e_06(self, tensor_with_nan):
        # Arrange
        # Arrange
        # Act
        result = to_nan01(tensor_with_nan, dim=-1, device="cpu")
        # Act
        # Assert
        # Assert
        assert torch.allclose(result[0, 0], torch.tensor(0.0), atol=1e-6)

    @pytest.mark.skipif(
        not hasattr(torch, "nanmin"),
        reason="torch.nanmin not available in this PyTorch version",
    )
    def test_to_nan01_with_nan_torch_allclose_result_0_2_torch_tensor_1_0_atol_1e_06(self, tensor_with_nan):
        # Arrange
        # Arrange
        # Act
        result = to_nan01(tensor_with_nan, dim=-1, device="cpu")
        # Act
        # Assert
        # Assert
        assert torch.allclose(result[0, 2], torch.tensor(1.0), atol=1e-6)

    @pytest.mark.skipif(
        not hasattr(torch, "nanmin"),
        reason="torch.nanmin not available in this PyTorch version",
    )
    def test_to_nan01_with_nan_torch_isnan_result_0_1(self, tensor_with_nan):
        # Arrange
        # Arrange
        # Act
        result = to_nan01(tensor_with_nan, dim=-1, device="cpu")
        # Act
        # Assert
        # Assert
        assert torch.isnan(result[0, 1])


    def test_unbias_mean_result_shape_equals_sample_tensor_shape(self, sample_tensor):
        # Arrange
        # Arrange
        # Act
        result = unbias(sample_tensor, dim=-1, fn="mean", device="cpu")
        # Act
        # Assert
        # Assert
        assert result.shape == sample_tensor.shape

    def test_unbias_mean_torch_allclose_result_mean_dim_1_torch_zeros_2_atol_1e_06(self, sample_tensor):
        # Arrange
        # Arrange
        # Act
        result = unbias(sample_tensor, dim=-1, fn="mean", device="cpu")
        # Act
        # Assert
        # Assert
        assert torch.allclose(result.mean(dim=-1), torch.zeros(2), atol=1e-6)


    def test_unbias_min_result_shape_equals_sample_tensor_shape(self, sample_tensor):
        # Arrange
        # Arrange
        # Act
        result = unbias(sample_tensor, dim=-1, fn="min", device="cpu")
        # Act
        # Assert
        # Assert
        assert result.shape == sample_tensor.shape

    def test_unbias_min_torch_allclose_result_min_dim_1_0_torch_zeros_2(self, sample_tensor):
        # Arrange
        # Arrange
        # Act
        result = unbias(sample_tensor, dim=-1, fn="min", device="cpu")
        # Act
        # Assert
        # Assert
        assert torch.allclose(result.min(dim=-1)[0], torch.zeros(2))


    def test_unbias_invalid_method(self, sample_tensor):
        """Test unbias with invalid method raises error."""
        # Arrange
        # Act
        # Assert
        with pytest.raises(ValueError, match="Unsupported unbiasing method"):
            unbias(sample_tensor, dim=-1, fn="invalid", device="cpu")

    def test_clip_perc_basic_result_shape_equals_tensor_with_outliers_shape(self, sample_tensor):
        # Arrange
        # Arrange
        tensor_with_outliers = torch.tensor(
            [[1.0, 2.0, 3.0, 100.0], [-100.0, 4.0, 5.0, 6.0]]
        )
        # Act
        result = clip_perc(
            tensor_with_outliers, lower_perc=25, upper_perc=75, dim=-1, device="cpu"
        )
        # Act
        # Assert
        # Assert
        assert result.shape == tensor_with_outliers.shape

    def test_clip_perc_basic_result_0_3_100_0(self, sample_tensor):
        # Arrange
        # Arrange
        tensor_with_outliers = torch.tensor(
            [[1.0, 2.0, 3.0, 100.0], [-100.0, 4.0, 5.0, 6.0]]
        )
        # Act
        result = clip_perc(
            tensor_with_outliers, lower_perc=25, upper_perc=75, dim=-1, device="cpu"
        )
        # Act
        # Assert
        # Assert
        assert result[0, 3] < 100.0  # Upper outlier clipped

    def test_clip_perc_basic_result_1_0_100_0(self, sample_tensor):
        # Arrange
        # Arrange
        tensor_with_outliers = torch.tensor(
            [[1.0, 2.0, 3.0, 100.0], [-100.0, 4.0, 5.0, 6.0]]
        )
        # Act
        result = clip_perc(
            tensor_with_outliers, lower_perc=25, upper_perc=75, dim=-1, device="cpu"
        )
        # Act
        # Assert
        # Assert
        assert result[1, 0] > -100.0  # Lower outlier clipped


    def test_numpy_array_input_result_is_np_ndarray(self):
        # Arrange
        # Arrange
        np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # to_z should handle numpy arrays via torch_fn decorator
        # torch_fn preserves input type: numpy in -> numpy out
        # Act
        result = to_z(np_array, dim=-1, device="cpu")
        # Act
        # Assert
        # Assert
        assert isinstance(result, np.ndarray)

    def test_numpy_array_input_np_allclose_result_mean_axis_1_np_zeros_2_atol_1e_06(self):
        # Arrange
        # Arrange
        np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # to_z should handle numpy arrays via torch_fn decorator
        # torch_fn preserves input type: numpy in -> numpy out
        # Act
        result = to_z(np_array, dim=-1, device="cpu")
        # Act
        # Assert
        # Assert
        assert np.allclose(result.mean(axis=-1), np.zeros(2), atol=1e-6)


    def test_axis_parameter_result_shape_equals_sample_tensor_shape(self, sample_tensor):
        # Arrange
        # Arrange
        # Act
        result = to_z(sample_tensor, axis=0, device="cpu")
        # Act
        # Assert
        # Assert
        assert result.shape == sample_tensor.shape

    def test_axis_parameter_torch_allclose_result_mean_dim_0_torch_zeros_3_atol_1e_06(self, sample_tensor):
        # Arrange
        # Arrange
        # Act
        result = to_z(sample_tensor, axis=0, device="cpu")
        # Act
        # Assert
        # Assert
        assert torch.allclose(result.mean(dim=0), torch.zeros(3), atol=1e-6)


    @pytest.mark.parametrize("func", [to_z, to_01, unbias])
    def test_functions_preserve_gradient(self, func):
        """Test that functions preserve gradient information."""
        # Arrange
        tensor = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        result = func(tensor, dim=-1, device="cpu")

        # Should be able to compute gradients
        loss = result.sum()
        # Act
        loss.backward()
        # Assert
        assert tensor.grad is not None


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_norm.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-19 01:09:55 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/gen/_norm.py
#
# THIS_FILE = "/home/ywatanabe/proj/scitex_repo/src/scitex/gen/_norm.py"
#
# import torch
#
# from scitex.decorators import torch_fn
# from scitex.torch import nanstd
#
#
# @torch_fn
# def to_z(x, axis=-1, dim=None, device="cuda"):
#     """Standardizes tensor to zero mean and unit variance along specified dimension.
#
#     Parameters
#     ----------
#     xx : torch.Tensor
#         Input tensor
#     dim : int, optional
#         Dimension along which to standardize (preferred)
#     axis : int, optional
#         Alternative to dim for numpy compatibility
#     device : str
#         Device to use for computation
#
#     Returns
#     -------
#     torch.Tensor
#         Z-scored tensor
#     """
#     return (x - x.mean(dim=dim, keepdim=True)) / x.std(dim=dim, keepdim=True)
#
#
# @torch_fn
# def to_nanz(x, axis=-1, dim=None, device="cuda"):
#     """Standardizes tensor handling NaN values along specified dimension.
#
#     Parameters
#     ----------
#     xx : torch.Tensor
#         Input tensor
#     dim : int, optional
#         Dimension along which to standardize (preferred)
#     axis : int, optional
#         Alternative to dim for numpy compatibility
#     device : str
#         Device to use for computation
#
#     Returns
#     -------
#     torch.Tensor
#         Z-scored tensor with NaN handling
#     """
#     nan_mean = torch.nanmean(x, dim=dim, keepdim=True)
#     nan_std = nanstd(x, dim=dim, keepdim=True)
#     return (x - nan_mean) / nan_std
#
#
# @torch_fn
# def to_01(x, axis=-1, dim=None, device="cuda"):
#     """Min-max scales tensor to [0, 1] range along specified dimension.
#
#     Parameters
#     ----------
#     xx : torch.Tensor
#         Input tensor
#     dim : int, optional
#         Dimension along which to scale (preferred)
#     axis : int, optional
#         Alternative to dim for numpy compatibility
#     device : str
#         Device to use for computation
#
#     Returns
#     -------
#     torch.Tensor
#         Min-max scaled tensor
#     """
#     # Use dim if provided, otherwise use axis
#     dimension = dim if dim is not None else axis
#
#     if dimension is None:
#         # Scale entire tensor
#         x_min = x.min()
#         x_max = x.max()
#     else:
#         # Scale along specified dimension
#         x_min = x.min(dim=dimension, keepdim=True)[0]
#         x_max = x.max(dim=dimension, keepdim=True)[0]
#
#     # Avoid division by zero
#     return (x - x_min) / (x_max - x_min + 1e-8)
#
#
# @torch_fn
# def to_nan01(x, axis=-1, dim=None, device="cuda"):
#     """Min-max scales tensor handling NaN values along specified dimension.
#
#     Parameters
#     ----------
#     xx : torch.Tensor
#         Input tensor
#     dim : int, optional
#         Dimension along which to scale (preferred)
#     axis : int, optional
#         Alternative to dim for numpy compatibility
#     device : str
#         Device to use for computation
#
#     Returns
#     -------
#     torch.Tensor
#         Min-max scaled tensor with NaN handling
#     """
#     # Use dim if provided, otherwise use axis
#     dimension = dim if dim is not None else axis
#
#     if dimension is None:
#         # Scale entire tensor
#         x_min = torch.nanmin(x)
#         x_max = torch.nanmax(x)
#     else:
#         # Scale along specified dimension
#         x_min = torch.nanmin(x, dim=dimension, keepdim=True)[0]
#         x_max = torch.nanmax(x, dim=dimension, keepdim=True)[0]
#
#     # Avoid division by zero
#     return (x - x_min) / (x_max - x_min + 1e-8)
#
#
# @torch_fn
# def unbias(x, axis=-1, dim=None, fn="mean", device="cuda"):
#     """Removes bias from tensor using specified method along dimension.
#
#     Parameters
#     ----------
#     xx : torch.Tensor
#         Input tensor
#     dim : int, optional
#         Dimension along which to unbias (preferred)
#     axis : int, optional
#         Alternative to dim for numpy compatibility
#     fn : str
#         Method to use for unbiasing ('mean' or 'min')
#     device : str
#         Device to use for computation
#
#     Returns
#     -------
#     torch.Tensor
#         Unbiased tensor
#     """
#     if fn == "mean":
#         return x - x.mean(dim=dim, keepdims=True)
#     if fn == "min":
#         return x - x.min(dim=dim, keepdims=True)[0]
#     raise ValueError(f"Unsupported unbiasing method: {fn}")
#
#
# @torch_fn
# def clip_perc(
#     x,
#     lower_perc=2.5,
#     upper_perc=97.5,
#     low=None,
#     high=None,
#     axis=-1,
#     dim=None,
#     device="cuda",
# ):
#     """Clips tensor values between specified percentiles along dimension.
#
#     Parameters
#     ----------
#     x : torch.Tensor
#         Input tensor
#     lower_perc : float
#         Lower percentile (0-100)
#     upper_perc : float
#         Upper percentile (0-100)
#     low : float, optional
#         Alternative name for lower_perc
#     high : float, optional
#         Alternative name for upper_perc
#     dim : int
#         Dimension along which to compute percentiles (preferred)
#     axis : int
#         Alternative to dim for numpy compatibility
#     device : str
#         Device to use for computation
#
#     Returns
#     -------
#     torch.Tensor
#         Clipped tensor
#     """
#     # Handle alternative parameter names
#     if low is not None:
#         lower_perc = low
#     if high is not None:
#         upper_perc = high
#
#     # Use dim if provided, otherwise use axis
#     dimension = dim if dim is not None else axis
#
#     lower = torch.quantile(x, lower_perc / 100, dim=dimension, keepdim=True)
#     upper = torch.quantile(x, upper_perc / 100, dim=dimension, keepdim=True)
#     return torch.clamp(x, min=lower, max=upper)
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_norm.py
# --------------------------------------------------------------------------------
