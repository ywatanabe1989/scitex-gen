import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
xr = pytest.importorskip("xarray")

from scitex_gen import var_info


class TestVarInfoBasicTypes:
    """Test var_info with basic Python types."""

    def test_integer_result_equals_type_int(self):
        """Test var_info with integer."""
        # Arrange
        # Act
        result = var_info(42)
        # Assert
        assert result == {"type": "int"}

    def test_float_result_equals_type_float(self):
        """Test var_info with float."""
        # Arrange
        # Act
        result = var_info(3.14)
        # Assert
        assert result == {"type": "float"}

    def test_string_result_equals_type_str_length_5(self):
        """Test var_info with string."""
        # Arrange
        # Act
        result = var_info("hello")
        # Assert
        assert result == {"type": "str", "length": 5}

    def test_boolean_result_equals_type_bool(self):
        """Test var_info with boolean."""
        # Arrange
        # Act
        result = var_info(True)
        # Assert
        assert result == {"type": "bool"}

    def test_none_result_equals_type_nonetype(self):
        """Test var_info with None."""
        # Arrange
        # Act
        result = var_info(None)
        # Assert
        assert result == {"type": "NoneType"}

    def test_dict_result_equals_type_dict_length_2(self):
        """Test var_info with dictionary."""
        # Arrange
        # Act
        result = var_info({"a": 1, "b": 2})
        # Assert
        assert result == {"type": "dict", "length": 2}

    def test_set_result_equals_type_set_length_3(self):
        """Test var_info with set."""
        # Arrange
        # Act
        result = var_info({1, 2, 3})
        # Assert
        assert result == {"type": "set", "length": 3}

    def test_tuple_result_equals_type_tuple_length_3(self):
        """Test var_info with tuple."""
        # Arrange
        # Act
        result = var_info((1, 2, 3))
        # Assert
        assert result == {"type": "tuple", "length": 3}


class TestVarInfoLists:
    """Test var_info with list structures."""

    def test_empty_list_result_equals_type_list_length_0(self):
        """Test var_info with empty list."""
        # Arrange
        # Act
        result = var_info([])
        # Assert
        assert result == {"type": "list", "length": 0}

    def test_flat_list_result_equals_type_list_length_4(self):
        """Test var_info with flat list."""
        # Arrange
        # Act
        result = var_info([1, 2, 3, 4])
        # Assert
        assert result == {"type": "list", "length": 4}

    def test_nested_list_2d(self):
        """Test var_info with 2D nested list."""
        # Arrange
        data = [[1, 2, 3], [4, 5, 6]]
        # Act
        result = var_info(data)
        # Assert
        assert result == {"type": "list", "length": 2, "shape": (2, 3), "dimensions": 2}

    def test_nested_list_3d(self):
        """Test var_info with 3D nested list."""
        # Arrange
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        # Act
        result = var_info(data)
        # Assert
        assert result == {
            "type": "list",
            "length": 2,
            "shape": (2, 2, 2),
            "dimensions": 3,
        }

    def test_irregular_nested_list(self):
        """Test var_info with irregular nested list."""
        # Only checks first element's shape
        # Arrange
        data = [[1, 2, 3], [4, 5]]  # Irregular
        # Act
        result = var_info(data)
        # Assert
        assert result == {
            "type": "list",
            "length": 2,
            "shape": (2, 3),  # Uses first element's length
            "dimensions": 2,
        }

    def test_mixed_type_list(self):
        """Test var_info with mixed type list."""
        # Arrange
        data = [1, "hello", 3.14, [1, 2]]
        # Act
        result = var_info(data)
        # Assert
        assert result == {"type": "list", "length": 4}


class TestVarInfoNumPy:
    """Test var_info with NumPy arrays."""

    def test_numpy_1d_result_equals_type_ndarray_length_(self):
        """Test var_info with 1D NumPy array."""
        # Arrange
        arr = np.array([1, 2, 3, 4])
        # Act
        result = var_info(arr)
        # Assert
        assert result == {
            "type": "ndarray",
            "length": 4,
            "shape": (4,),
            "dimensions": 1,
        }

    def test_numpy_2d_result_equals_type_ndarray_length_(self):
        """Test var_info with 2D NumPy array."""
        # Arrange
        arr = np.array([[1, 2], [3, 4]])
        # Act
        result = var_info(arr)
        # Assert
        assert result == {
            "type": "ndarray",
            "length": 2,
            "shape": (2, 2),
            "dimensions": 2,
        }

    def test_numpy_3d_result_equals_type_ndarray_length_(self):
        """Test var_info with 3D NumPy array."""
        # Arrange
        arr = np.zeros((2, 3, 4))
        # Act
        result = var_info(arr)
        # Assert
        assert result == {
            "type": "ndarray",
            "length": 2,
            "shape": (2, 3, 4),
            "dimensions": 3,
        }

    def test_numpy_scalar_result_type_int64(self):
        """Test var_info with NumPy scalar."""
        # Note: numpy scalars like np.int64 are not np.ndarray instances,
        # so var_info doesn't add shape/dimensions for them
        # Arrange
        scalar = np.int64(42)
        # Act
        result = var_info(scalar)
        # Assert
        assert result["type"] == "int64"
        # Scalars don't get shape in var_info since they're not ndarray


class TestVarInfoPandas:
    """Test var_info with Pandas objects."""

    def test_pandas_series_result_equals_type_series_length_4(self):
        """Test var_info with Pandas Series."""
        # Arrange
        series = pd.Series([1, 2, 3, 4])
        # Act
        result = var_info(series)
        # Assert
        assert result == {"type": "Series", "length": 4, "shape": (4,), "dimensions": 1}

    def test_pandas_dataframe_result_equals_type_dataframe_lengt(self):
        """Test var_info with Pandas DataFrame."""
        # Arrange
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # Act
        result = var_info(df)
        # Assert
        assert result == {
            "type": "DataFrame",
            "length": 3,  # Number of rows
            "shape": (3, 2),
            "dimensions": 2,
        }

    def test_empty_dataframe_result_equals_type_dataframe_lengt(self):
        """Test var_info with empty DataFrame."""
        # Arrange
        df = pd.DataFrame()
        # Act
        result = var_info(df)
        # Assert
        assert result == {
            "type": "DataFrame",
            "length": 0,
            "shape": (0, 0),
            "dimensions": 2,
        }


class TestVarInfoXArray:
    """Test var_info with xarray objects."""

    def test_xarray_dataarray_result_equals_type_dataarray_lengt(self):
        """Test var_info with xarray DataArray."""
        # Arrange
        data = xr.DataArray(
            np.random.randn(2, 3),
            dims=["x", "y"],
            coords={"x": [1, 2], "y": [10, 20, 30]},
        )
        # Act
        result = var_info(data)
        # Assert
        assert result == {
            "type": "DataArray",
            "length": 2,
            "shape": (2, 3),
            "dimensions": 2,
        }

    def test_xarray_1d_result_equals_type_dataarray_lengt(self):
        """Test var_info with 1D xarray."""
        # Arrange
        data = xr.DataArray([1, 2, 3, 4], dims=["time"])
        # Act
        result = var_info(data)
        # Assert
        assert result == {
            "type": "DataArray",
            "length": 4,
            "shape": (4,),
            "dimensions": 1,
        }


class TestVarInfoTorch:
    """Test var_info with PyTorch tensors."""

    def test_torch_1d_result_equals_type_tensor_length_4(self):
        """Test var_info with 1D torch tensor."""
        # Arrange
        tensor = torch.tensor([1, 2, 3, 4])
        # Act
        result = var_info(tensor)
        # Assert
        assert result == {
            "type": "Tensor",
            "length": 4,
            "shape": torch.Size([4]),
            "dimensions": 1,
        }

    def test_torch_2d_result_equals_type_tensor_length_3(self):
        """Test var_info with 2D torch tensor."""
        # Arrange
        tensor = torch.zeros(3, 4)
        # Act
        result = var_info(tensor)
        # Assert
        assert result == {
            "type": "Tensor",
            "length": 3,
            "shape": torch.Size([3, 4]),
            "dimensions": 2,
        }

    def test_torch_4d_result_equals_type_tensor_length_1(self):
        """Test var_info with 4D torch tensor (common in CNNs)."""
        # Arrange
        tensor = torch.randn(16, 3, 224, 224)  # batch, channels, height, width
        # Act
        result = var_info(tensor)
        # Assert
        assert result == {
            "type": "Tensor",
            "length": 16,
            "shape": torch.Size([16, 3, 224, 224]),
            "dimensions": 4,
        }


class TestVarInfoEdgeCases:
    """Test var_info with edge cases."""

    def test_custom_class_result_equals_type_myclass(self):
        """Test var_info with custom class."""

        # Arrange
        class MyClass:
            pass

        obj = MyClass()
        # Act
        result = var_info(obj)
        # Assert
        assert result == {"type": "MyClass"}

    def test_function_result_equals_type_function(self):
        """Test var_info with function."""

        # Arrange
        def my_func():
            pass

        # Act
        result = var_info(my_func)
        # Assert
        assert result == {"type": "function"}

    def test_lambda_result_equals_type_function(self):
        """Test var_info with lambda."""
        # Arrange
        f = lambda x: x + 1
        # Act
        result = var_info(f)
        # Assert
        assert result == {"type": "function"}

    def test_generator_result_type_generator(self):
        # Arrange
        # Arrange
        gen = (x for x in range(10))
        # Act
        result = var_info(gen)
        # Act
        # Assert
        # Assert
        assert result["type"] == "generator"

    def test_generator_length_not_in_result(self):
        # Arrange
        # Arrange
        gen = (x for x in range(10))
        # Act
        result = var_info(gen)
        # Act
        # Assert
        # Assert
        assert "length" not in result


    def test_bytes_result_equals_type_bytes_length_5(self):
        """Test var_info with bytes."""
        # Arrange
        data = b"hello"
        # Act
        result = var_info(data)
        # Assert
        assert result == {"type": "bytes", "length": 5}

    def test_range_result_equals_type_range_length_10(self):
        """Test var_info with range object."""
        # Arrange
        r = range(10)
        # Act
        result = var_info(r)
        # Assert
        assert result == {"type": "range", "length": 10}


class TestVarInfoIntegration:
    """Integration tests for var_info."""

    def test_docstring_example_info_equals_type_ndarray_length_(self):
        """Test the example from the docstring."""
        # Arrange
        data = np.array([[1, 2], [3, 4]])
        # Act
        info = var_info(data)
        # Assert
        assert info == {
            "type": "ndarray",
            "length": 2,
            "shape": (2, 2),
            "dimensions": 2,
        }

    def test_complex_nested_structure_result_equals_type_dict_length_3(self):
        # Arrange
        # Arrange
        data = {
            "arrays": [np.array([1, 2, 3]), np.array([[1, 2], [3, 4]])],
            "tensors": torch.tensor([1.0, 2.0, 3.0]),
            "df": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        }
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert result == {"type": "dict", "length": 3}

    def test_complex_nested_structure_var_info_data_arrays_0_shape_3(self):
        # Arrange
        # Arrange
        data = {
            "arrays": [np.array([1, 2, 3]), np.array([[1, 2], [3, 4]])],
            "tensors": torch.tensor([1.0, 2.0, 3.0]),
            "df": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        }
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert var_info(data["arrays"][0])["shape"] == (3,)

    def test_complex_nested_structure_var_info_data_tensors_shape_torch_size_3(self):
        # Arrange
        # Arrange
        data = {
            "arrays": [np.array([1, 2, 3]), np.array([[1, 2], [3, 4]])],
            "tensors": torch.tensor([1.0, 2.0, 3.0]),
            "df": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        }
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert var_info(data["tensors"])["shape"] == torch.Size([3])

    def test_complex_nested_structure_var_info_data_df_shape_2_2(self):
        # Arrange
        # Arrange
        data = {
            "arrays": [np.array([1, 2, 3]), np.array([[1, 2], [3, 4]])],
            "tensors": torch.tensor([1.0, 2.0, 3.0]),
            "df": pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
        }
        # Act
        result = var_info(data)
        # Act
        # Assert
        # Assert
        assert var_info(data["df"])["shape"] == (2, 2)



# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_var_info.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-25 00:35:31 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/gen/_var_info.py
#
# THIS_FILE = "/home/ywatanabe/proj/scitex_repo/src/scitex/gen/_var_info.py"
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
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_var_info.py
# --------------------------------------------------------------------------------
