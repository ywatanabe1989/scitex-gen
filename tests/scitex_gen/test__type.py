#!/usr/bin/env python3
# Time-stamp: "2025-05-31 22:00:00 (claude)"
# File: ./tests/scitex/gen/test__type.py

"""
Comprehensive tests for scitex_gen._type module.

This module tests:
- var_info function for various data types
- ArrayLike type alias usage
- Edge cases and error handling
"""

import pytest

torch = pytest.importorskip("torch")
xr = pytest.importorskip("xarray")
from typing import get_args

import numpy as np
import pandas as pd

from scitex_gen import ArrayLike, var_info


class TestArrayLikeTypeAlias:
    """Test the ArrayLike type alias definition."""

    def test_arraylike_types_set_actual_types_set_expected_types(self):
        # Arrange
        # Arrange
        expected_types = (
            list,
            tuple,
            np.ndarray,
            pd.Series,
            pd.DataFrame,
            xr.DataArray,
            torch.Tensor,
        )
        # Act
        actual_types = get_args(ArrayLike)
        # Act
        # Assert
        # Assert
        assert set(actual_types) == set(expected_types)

    def test_arraylike_types_len_actual_types_is_7(self):
        # Arrange
        # Arrange
        expected_types = (
            list,
            tuple,
            np.ndarray,
            pd.Series,
            pd.DataFrame,
            xr.DataArray,
            torch.Tensor,
        )
        # Act
        actual_types = get_args(ArrayLike)
        # Act
        # Assert
        # Assert
        assert len(actual_types) == 7



class TestVarInfoBasicTypes:
    """Test var_info with basic Python types."""

    def test_int_result_type_int(self):
        # Arrange
        # Arrange
        # Act
        result = var_info(42)
        # Act
        # Assert
        # Assert
        assert result["type"] == "int"

    def test_int_length_not_in_result(self):
        # Arrange
        # Arrange
        # Act
        result = var_info(42)
        # Act
        # Assert
        # Assert
        assert "length" not in result

    def test_int_shape_not_in_result(self):
        # Arrange
        # Arrange
        # Act
        result = var_info(42)
        # Act
        # Assert
        # Assert
        assert "shape" not in result

    def test_int_dimensions_not_in_result(self):
        # Arrange
        # Arrange
        # Act
        result = var_info(42)
        # Act
        # Assert
        # Assert
        assert "dimensions" not in result


    def test_float_result_type_float(self):
        # Arrange
        # Arrange
        # Act
        result = var_info(3.14)
        # Act
        # Assert
        # Assert
        assert result["type"] == "float"

    def test_float_length_not_in_result(self):
        # Arrange
        # Arrange
        # Act
        result = var_info(3.14)
        # Act
        # Assert
        # Assert
        assert "length" not in result


    def test_string_result_type_str(self):
        # Arrange
        # Arrange
        # Act
        result = var_info("hello")
        # Act
        # Assert
        # Assert
        assert result["type"] == "str"

    def test_string_result_length_5(self):
        # Arrange
        # Arrange
        # Act
        result = var_info("hello")
        # Act
        # Assert
        # Assert
        assert result["length"] == 5

    def test_string_shape_not_in_result(self):
        # Arrange
        # Arrange
        # Act
        result = var_info("hello")
        # Act
        # Assert
        # Assert
        assert "shape" not in result


    def test_list_result_type_list(self):
        # Arrange
        # Arrange
        # Act
        result = var_info([1, 2, 3])
        # Act
        # Assert
        # Assert
        assert result["type"] == "list"

    def test_list_result_length_3(self):
        # Arrange
        # Arrange
        # Act
        result = var_info([1, 2, 3])
        # Act
        # Assert
        # Assert
        assert result["length"] == 3

    def test_list_shape_not_in_result(self):
        # Arrange
        # Arrange
        # Act
        result = var_info([1, 2, 3])
        # Act
        # Assert
        # Assert
        assert "shape" not in result

    def test_list_dimensions_not_in_result(self):
        # Arrange
        # Arrange
        # Act
        result = var_info([1, 2, 3])
        # Act
        # Assert
        # Assert
        assert "dimensions" not in result


    def test_tuple_result_type_tuple(self):
        # Arrange
        # Arrange
        # Act
        result = var_info((1, 2, 3))
        # Act
        # Assert
        # Assert
        assert result["type"] == "tuple"

    def test_tuple_result_length_3(self):
        # Arrange
        # Arrange
        # Act
        result = var_info((1, 2, 3))
        # Act
        # Assert
        # Assert
        assert result["length"] == 3

    def test_tuple_shape_not_in_result(self):
        # Arrange
        # Arrange
        # Act
        result = var_info((1, 2, 3))
        # Act
        # Assert
        # Assert
        assert "shape" not in result


    def test_dict_result_type_dict(self):
        # Arrange
        # Arrange
        # Act
        result = var_info({"a": 1, "b": 2})
        # Act
        # Assert
        # Assert
        assert result["type"] == "dict"

    def test_dict_result_length_2(self):
        # Arrange
        # Arrange
        # Act
        result = var_info({"a": 1, "b": 2})
        # Act
        # Assert
        # Assert
        assert result["length"] == 2

    def test_dict_shape_not_in_result(self):
        # Arrange
        # Arrange
        # Act
        result = var_info({"a": 1, "b": 2})
        # Act
        # Assert
        # Assert
        assert "shape" not in result


    def test_set_result_type_set(self):
        # Arrange
        # Arrange
        # Act
        result = var_info({1, 2, 3})
        # Act
        # Assert
        # Assert
        assert result["type"] == "set"

    def test_set_result_length_3(self):
        # Arrange
        # Arrange
        # Act
        result = var_info({1, 2, 3})
        # Act
        # Assert
        # Assert
        assert result["length"] == 3

    def test_set_shape_not_in_result(self):
        # Arrange
        # Arrange
        # Act
        result = var_info({1, 2, 3})
        # Act
        # Assert
        # Assert
        assert "shape" not in result


    def test_none_result_type_nonetype(self):
        # Arrange
        # Arrange
        # Act
        result = var_info(None)
        # Act
        # Assert
        # Assert
        assert result["type"] == "NoneType"

    def test_none_length_not_in_result(self):
        # Arrange
        # Arrange
        # Act
        result = var_info(None)
        # Act
        # Assert
        # Assert
        assert "length" not in result



class TestVarInfoNestedLists:
    """Test var_info with nested list structures."""

    def test_2d_list_result_type_list(self):
        # Arrange
        # Arrange
        data = [[1, 2, 3], [4, 5, 6]]
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "list"

    def test_2d_list_result_length_2(self):
        # Arrange
        # Arrange
        data = [[1, 2, 3], [4, 5, 6]]
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 2

    def test_2d_list_result_shape_2_3(self):
        # Arrange
        # Arrange
        data = [[1, 2, 3], [4, 5, 6]]
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == (2, 3)

    def test_2d_list_result_dimensions_2(self):
        # Arrange
        # Arrange
        data = [[1, 2, 3], [4, 5, 6]]
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 2


    def test_3d_list_result_type_list(self):
        # Arrange
        # Arrange
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "list"

    def test_3d_list_result_length_2(self):
        # Arrange
        # Arrange
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 2

    def test_3d_list_result_shape_2_2_2(self):
        # Arrange
        # Arrange
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == (2, 2, 2)

    def test_3d_list_result_dimensions_3(self):
        # Arrange
        # Arrange
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 3


    def test_jagged_list_result_type_list(self):
        # Arrange
        # Arrange
        data = [[1, 2], [3, 4, 5]]  # Different lengths
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "list"

    def test_jagged_list_result_length_2(self):
        # Arrange
        # Arrange
        data = [[1, 2], [3, 4, 5]]  # Different lengths
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 2

    def test_jagged_list_result_shape_2_2(self):
        # Arrange
        # Arrange
        data = [[1, 2], [3, 4, 5]]  # Different lengths
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == (2, 2)  # Uses first element's length

    def test_jagged_list_result_dimensions_2(self):
        # Arrange
        # Arrange
        data = [[1, 2], [3, 4, 5]]  # Different lengths
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 2


    def test_empty_nested_list_result_type_list(self):
        # Arrange
        # Arrange
        data = [[]]
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "list"

    def test_empty_nested_list_result_length_1(self):
        # Arrange
        # Arrange
        data = [[]]
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 1

    def test_empty_nested_list_result_shape_1_0(self):
        # Arrange
        # Arrange
        data = [[]]
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == (1, 0)

    def test_empty_nested_list_result_dimensions_2(self):
        # Arrange
        # Arrange
        data = [[]]
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 2



class TestVarInfoNumPy:
    """Test var_info with NumPy arrays."""

    def test_numpy_1d_result_type_ndarray(self):
        # Arrange
        # Arrange
        data = np.array([1, 2, 3, 4, 5])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "ndarray"

    def test_numpy_1d_result_length_5(self):
        # Arrange
        # Arrange
        data = np.array([1, 2, 3, 4, 5])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 5

    def test_numpy_1d_result_shape_5(self):
        # Arrange
        # Arrange
        data = np.array([1, 2, 3, 4, 5])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == (5,)

    def test_numpy_1d_result_dimensions_1(self):
        # Arrange
        # Arrange
        data = np.array([1, 2, 3, 4, 5])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 1


    def test_numpy_2d_result_type_ndarray(self):
        # Arrange
        # Arrange
        data = np.array([[1, 2], [3, 4], [5, 6]])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "ndarray"

    def test_numpy_2d_result_length_3(self):
        # Arrange
        # Arrange
        data = np.array([[1, 2], [3, 4], [5, 6]])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 3

    def test_numpy_2d_result_shape_3_2(self):
        # Arrange
        # Arrange
        data = np.array([[1, 2], [3, 4], [5, 6]])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == (3, 2)

    def test_numpy_2d_result_dimensions_2(self):
        # Arrange
        # Arrange
        data = np.array([[1, 2], [3, 4], [5, 6]])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 2


    def test_numpy_3d_result_type_ndarray(self):
        # Arrange
        # Arrange
        data = np.zeros((2, 3, 4))
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "ndarray"

    def test_numpy_3d_result_length_2(self):
        # Arrange
        # Arrange
        data = np.zeros((2, 3, 4))
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 2

    def test_numpy_3d_result_shape_2_3_4(self):
        # Arrange
        # Arrange
        data = np.zeros((2, 3, 4))
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == (2, 3, 4)

    def test_numpy_3d_result_dimensions_3(self):
        # Arrange
        # Arrange
        data = np.zeros((2, 3, 4))
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 3


    def test_numpy_scalar_result_type_float32(self):
        """Test var_info with numpy scalar."""
        # Note: numpy scalars (np.float32, np.int64, etc.) are not np.ndarray
        # so they don't get shape/dimensions in var_info
        # Arrange
        data = np.float32(3.14)
        # Act
        result = var_info(data)

        # Assert
        assert result["type"] == "float32"
        # Numpy scalars don't have shape attribute in the same way as arrays
        # var_info only adds shape for ndarray, DataFrame, Series, etc.


class TestVarInfoPandas:
    """Test var_info with Pandas objects."""

    def test_pandas_series_result_type_series(self):
        # Arrange
        # Arrange
        data = pd.Series([1, 2, 3, 4, 5])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "Series"

    def test_pandas_series_result_length_5(self):
        # Arrange
        # Arrange
        data = pd.Series([1, 2, 3, 4, 5])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 5

    def test_pandas_series_result_shape_5(self):
        # Arrange
        # Arrange
        data = pd.Series([1, 2, 3, 4, 5])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == (5,)

    def test_pandas_series_result_dimensions_1(self):
        # Arrange
        # Arrange
        data = pd.Series([1, 2, 3, 4, 5])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 1


    def test_pandas_dataframe_result_type_dataframe(self):
        # Arrange
        # Arrange
        data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "DataFrame"

    def test_pandas_dataframe_result_length_3(self):
        # Arrange
        # Arrange
        data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 3

    def test_pandas_dataframe_result_shape_3_2(self):
        # Arrange
        # Arrange
        data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == (3, 2)

    def test_pandas_dataframe_result_dimensions_2(self):
        # Arrange
        # Arrange
        data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 2


    def test_pandas_empty_dataframe_result_type_dataframe(self):
        # Arrange
        # Arrange
        data = pd.DataFrame()
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "DataFrame"

    def test_pandas_empty_dataframe_result_length_0(self):
        # Arrange
        # Arrange
        data = pd.DataFrame()
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 0

    def test_pandas_empty_dataframe_result_shape_0_0(self):
        # Arrange
        # Arrange
        data = pd.DataFrame()
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == (0, 0)

    def test_pandas_empty_dataframe_result_dimensions_2(self):
        # Arrange
        # Arrange
        data = pd.DataFrame()
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 2



class TestVarInfoPyTorch:
    """Test var_info with PyTorch tensors."""

    def test_torch_1d_result_type_tensor(self):
        # Arrange
        # Arrange
        data = torch.tensor([1, 2, 3, 4, 5])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "Tensor"

    def test_torch_1d_result_length_5(self):
        # Arrange
        # Arrange
        data = torch.tensor([1, 2, 3, 4, 5])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 5

    def test_torch_1d_result_shape_torch_size_5(self):
        # Arrange
        # Arrange
        data = torch.tensor([1, 2, 3, 4, 5])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == torch.Size([5])

    def test_torch_1d_result_dimensions_1(self):
        # Arrange
        # Arrange
        data = torch.tensor([1, 2, 3, 4, 5])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 1


    def test_torch_2d_result_type_tensor(self):
        # Arrange
        # Arrange
        data = torch.zeros(3, 4)
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "Tensor"

    def test_torch_2d_result_length_3(self):
        # Arrange
        # Arrange
        data = torch.zeros(3, 4)
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 3

    def test_torch_2d_result_shape_torch_size_3_4(self):
        # Arrange
        # Arrange
        data = torch.zeros(3, 4)
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == torch.Size([3, 4])

    def test_torch_2d_result_dimensions_2(self):
        # Arrange
        # Arrange
        data = torch.zeros(3, 4)
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 2


    def test_torch_4d_result_type_tensor(self):
        # Arrange
        # Arrange
        data = torch.randn(16, 3, 224, 224)  # Batch, channels, height, width
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "Tensor"

    def test_torch_4d_result_length_16(self):
        # Arrange
        # Arrange
        data = torch.randn(16, 3, 224, 224)  # Batch, channels, height, width
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 16

    def test_torch_4d_result_shape_torch_size_16_3_224_224(self):
        # Arrange
        # Arrange
        data = torch.randn(16, 3, 224, 224)  # Batch, channels, height, width
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == torch.Size([16, 3, 224, 224])

    def test_torch_4d_result_dimensions_4(self):
        # Arrange
        # Arrange
        data = torch.randn(16, 3, 224, 224)  # Batch, channels, height, width
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 4



class TestVarInfoXArray:
    """Test var_info with xarray objects."""

    def test_xarray_dataarray_result_type_dataarray(self):
        # Arrange
        # Arrange
        data = xr.DataArray(np.random.randn(2, 3, 4), dims=["x", "y", "z"])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "DataArray"

    def test_xarray_dataarray_result_length_2(self):
        # Arrange
        # Arrange
        data = xr.DataArray(np.random.randn(2, 3, 4), dims=["x", "y", "z"])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 2

    def test_xarray_dataarray_result_shape_2_3_4(self):
        # Arrange
        # Arrange
        data = xr.DataArray(np.random.randn(2, 3, 4), dims=["x", "y", "z"])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == (2, 3, 4)

    def test_xarray_dataarray_result_dimensions_3(self):
        # Arrange
        # Arrange
        data = xr.DataArray(np.random.randn(2, 3, 4), dims=["x", "y", "z"])
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 3


    def test_xarray_with_coords_result_type_dataarray(self):
        # Arrange
        # Arrange
        data = xr.DataArray(
            np.random.randn(3, 4),
            coords={"x": [1, 2, 3], "y": [10, 20, 30, 40]},
            dims=["x", "y"],
        )
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "DataArray"

    def test_xarray_with_coords_result_length_3(self):
        # Arrange
        # Arrange
        data = xr.DataArray(
            np.random.randn(3, 4),
            coords={"x": [1, 2, 3], "y": [10, 20, 30, 40]},
            dims=["x", "y"],
        )
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 3

    def test_xarray_with_coords_result_shape_3_4(self):
        # Arrange
        # Arrange
        data = xr.DataArray(
            np.random.randn(3, 4),
            coords={"x": [1, 2, 3], "y": [10, 20, 30, 40]},
            dims=["x", "y"],
        )
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == (3, 4)

    def test_xarray_with_coords_result_dimensions_2(self):
        # Arrange
        # Arrange
        data = xr.DataArray(
            np.random.randn(3, 4),
            coords={"x": [1, 2, 3], "y": [10, 20, 30, 40]},
            dims=["x", "y"],
        )
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["dimensions"] == 2



class TestVarInfoEdgeCases:
    """Test var_info with edge cases."""

    def test_empty_list_result_type_list(self):
        # Arrange
        # Arrange
        # Act
        result = var_info([])
        # Act
        # Assert
        # Assert
        assert result["type"] == "list"

    def test_empty_list_result_length_0(self):
        # Arrange
        # Arrange
        # Act
        result = var_info([])
        # Act
        # Assert
        # Assert
        assert result["length"] == 0

    def test_empty_list_shape_not_in_result(self):
        # Arrange
        # Arrange
        # Act
        result = var_info([])
        # Act
        # Assert
        # Assert
        assert "shape" not in result

    def test_empty_list_dimensions_not_in_result(self):
        # Arrange
        # Arrange
        # Act
        result = var_info([])
        # Act
        # Assert
        # Assert
        assert "dimensions" not in result


    def test_custom_object_result_type_customobject(self):
        # Arrange
        # Arrange
        class CustomObject:
            def __len__(self):
                return 42
        obj = CustomObject()
        # Act
        result = var_info(obj)
        # Act
        # Assert
        # Assert
        assert result["type"] == "CustomObject"

    def test_custom_object_result_length_42(self):
        # Arrange
        # Arrange
        class CustomObject:
            def __len__(self):
                return 42
        obj = CustomObject()
        # Act
        result = var_info(obj)
        # Act
        # Assert
        # Assert
        assert result["length"] == 42

    def test_custom_object_shape_not_in_result(self):
        # Arrange
        # Arrange
        class CustomObject:
            def __len__(self):
                return 42
        obj = CustomObject()
        # Act
        result = var_info(obj)
        # Act
        # Assert
        # Assert
        assert "shape" not in result


    def test_generator_result_type_generator(self):
        # Arrange
        # Arrange
        gen = (x for x in range(5))
        # Act
        result = var_info(gen)
        # Act
        # Assert
        # Assert
        assert result["type"] == "generator"

    def test_generator_length_not_in_result(self):
        # Arrange
        # Arrange
        gen = (x for x in range(5))
        # Act
        result = var_info(gen)
        # Act
        # Assert
        # Assert
        assert "length" not in result  # Generators don't have __len__

    def test_generator_shape_not_in_result(self):
        # Arrange
        # Arrange
        gen = (x for x in range(5))
        # Act
        result = var_info(gen)
        # Act
        # Assert
        # Assert
        assert "shape" not in result


    def test_mixed_nested_list_result_type_list(self):
        # Arrange
        # Arrange
        data = [[1, 2], "string", [3, 4]]  # Mixed types
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == "list"

    def test_mixed_nested_list_result_length_3(self):
        # Arrange
        # Arrange
        data = [[1, 2], "string", [3, 4]]  # Mixed types
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["length"] == 3

    def test_mixed_nested_list_shape_in_result(self):
        # Arrange
        # Arrange
        data = [[1, 2], "string", [3, 4]]  # Mixed types
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert "shape" in result

    def test_mixed_nested_list_result_shape_3_2(self):
        # Arrange
        # Arrange
        data = [[1, 2], "string", [3, 4]]  # Mixed types
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["shape"] == (3, 2)



class TestVarInfoIntegration:
    """Integration tests for var_info function."""

    def test_documentation_example_info_type_ndarray(self):
        # Arrange
        # Arrange
        data = np.array([[1, 2], [3, 4]])
        # Act
        info = var_info(data)
        # Act
        # Assert
        # Assert
        assert info["type"] == "ndarray"

    def test_documentation_example_info_length_2(self):
        # Arrange
        # Arrange
        data = np.array([[1, 2], [3, 4]])
        # Act
        info = var_info(data)
        # Act
        # Assert
        # Assert
        assert info["length"] == 2

    def test_documentation_example_info_shape_2_2(self):
        # Arrange
        # Arrange
        data = np.array([[1, 2], [3, 4]])
        # Act
        info = var_info(data)
        # Act
        # Assert
        # Assert
        assert info["shape"] == (2, 2)

    def test_documentation_example_info_dimensions_2(self):
        # Arrange
        # Arrange
        data = np.array([[1, 2], [3, 4]])
        # Act
        info = var_info(data)
        # Act
        # Assert
        # Assert
        assert info["dimensions"] == 2


    def test_multiple_calls_result1_type_list(self):
        # Arrange
        # Arrange
        result1 = var_info([1, 2, 3])
        result2 = var_info(np.array([1, 2, 3]))
        # Act
        result3 = var_info("string")
        # Act
        # Assert
        # Assert
        assert result1["type"] == "list"

    def test_multiple_calls_result2_type_ndarray(self):
        # Arrange
        # Arrange
        result1 = var_info([1, 2, 3])
        result2 = var_info(np.array([1, 2, 3]))
        # Act
        result3 = var_info("string")
        # Act
        # Assert
        # Assert
        assert result2["type"] == "ndarray"

    def test_multiple_calls_result3_type_str(self):
        # Arrange
        # Arrange
        result1 = var_info([1, 2, 3])
        result2 = var_info(np.array([1, 2, 3]))
        # Act
        result3 = var_info("string")
        # Act
        # Assert
        # Assert
        assert result3["type"] == "str"


    @pytest.mark.parametrize(
        "data,expected_type,has_shape",
        [
            ([1, 2, 3], "list", False),
            (np.array([1, 2, 3]), "ndarray", True),
            (pd.Series([1, 2, 3]), "Series", True),
            (torch.tensor([1, 2, 3]), "Tensor", True),
            ("test", "str", False),
            (42, "int", False),
        ],
    )
    def test_parametrized_types_result_type_expected_type(self, data, expected_type, has_shape):
        # Arrange
        # Arrange
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result["type"] == expected_type

    @pytest.mark.parametrize(
        "data,expected_type,has_shape",
        [
            ([1, 2, 3], "list", False),
            (np.array([1, 2, 3]), "ndarray", True),
            (pd.Series([1, 2, 3]), "Series", True),
            (torch.tensor([1, 2, 3]), "Tensor", True),
            ("test", "str", False),
            (42, "int", False),
        ],
    )
    def test_parametrized_types_shape_in_result_has_shape(self, data, expected_type, has_shape):
        # Arrange
        # Arrange
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert ("shape" in result) == has_shape



if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_type.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-17 12:45:50 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/gen/_type.py
#
# THIS_FILE = "/home/ywatanabe/proj/scitex_repo/src/scitex/gen/_type.py"
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-03 10:33:13 (ywatanabe)"
# # File: placeholder.py
#
# from typing import Any, Union
# import numpy as np
# import pandas as pd
# import torch
# import xarray as xr
#
# ArrayLike = Union[
#     list, tuple, np.ndarray, pd.Series, pd.DataFrame, xr.DataArray, torch.Tensor
# ]
#
#
# def var_info(variable: Any) -> dict:
#     """Returns type and structural information about a variable.
#
#     Example
#     -------
#     >>> data = np.array([[1, 2], [3, 4]])
#     >>> info = var_info(data)
#     >>> print(info)
#     {
#         'type': 'numpy.ndarray',
#         'length': 2,
#         'shape': (2, 2),
#         'dimensions': 2
#     }
#
#     Parameters
#     ----------
#     variable : Any
#         Variable to inspect.
#
#     Returns
#     -------
#     dict
#         Dictionary containing variable information.
#     """
#     info = {"type": type(variable).__name__}
#
#     # Length check
#     if hasattr(variable, "__len__"):
#         info["length"] = len(variable)
#
#     # Shape check for array-like objects
#     if isinstance(
#         variable, (np.ndarray, pd.DataFrame, pd.Series, xr.DataArray, torch.Tensor)
#     ):
#         info["shape"] = variable.shape
#         info["dimensions"] = len(variable.shape)
#
#     # Special handling for nested lists
#     elif isinstance(variable, list):
#         if variable and isinstance(variable[0], list):
#             depth = 1
#             current = variable
#             shape = [len(variable)]
#             while current and isinstance(current[0], list):
#                 shape.append(len(current[0]))
#                 current = current[0]
#                 depth += 1
#             info["shape"] = tuple(shape)
#             info["dimensions"] = depth
#
#     return info
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_type.py
# --------------------------------------------------------------------------------
