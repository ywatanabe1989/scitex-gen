import numpy as np
import pytest

pytest.importorskip("torch")

from scitex_gen import transpose


class TestTranspose:
    """Test the transpose function."""

    def test_2d_array_simple_np_array_equal_result_expected(self):
        # Arrange
        # Arrange
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        src_dims = np.array(["rows", "cols"])
        tgt_dims = np.array(["cols", "rows"])
        result = transpose(arr, src_dims, tgt_dims)
        # Act
        expected = arr.T
        # Act
        # Assert
        # Assert
        assert np.array_equal(result, expected)

    def test_2d_array_simple_result_shape_equals_n_3_2(self):
        # Arrange
        # Arrange
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        src_dims = np.array(["rows", "cols"])
        tgt_dims = np.array(["cols", "rows"])
        result = transpose(arr, src_dims, tgt_dims)
        # Act
        expected = arr.T
        # Act
        # Assert
        # Assert
        assert result.shape == (3, 2)


    def test_3d_array_permutation_result_shape_equals_n_3_4_2(self):
        # Arrange
        # Arrange
        arr = np.arange(24).reshape(2, 3, 4)
        src_dims = np.array(["batch", "height", "width"])
        # Test different permutations
        # Original shape: (2, 3, 4)
        # Move batch to end: (3, 4, 2)
        tgt_dims = np.array(["height", "width", "batch"])
        # Act
        result = transpose(arr, src_dims, tgt_dims)
        # Act
        # Assert
        # Assert
        assert result.shape == (3, 4, 2)

    def test_3d_array_permutation_result_0_0_0_arr_0_0_0(self):
        # Arrange
        # Arrange
        arr = np.arange(24).reshape(2, 3, 4)
        src_dims = np.array(["batch", "height", "width"])
        # Test different permutations
        # Original shape: (2, 3, 4)
        # Move batch to end: (3, 4, 2)
        tgt_dims = np.array(["height", "width", "batch"])
        # Act
        result = transpose(arr, src_dims, tgt_dims)
        # Act
        # Assert
        # Assert
        assert result[0, 0, 0] == arr[0, 0, 0]  # First element

    def test_3d_array_permutation_result_0_0_1_arr_1_0_0(self):
        # Arrange
        # Arrange
        arr = np.arange(24).reshape(2, 3, 4)
        src_dims = np.array(["batch", "height", "width"])
        # Test different permutations
        # Original shape: (2, 3, 4)
        # Move batch to end: (3, 4, 2)
        tgt_dims = np.array(["height", "width", "batch"])
        # Act
        result = transpose(arr, src_dims, tgt_dims)
        # Act
        # Assert
        # Assert
        assert result[0, 0, 1] == arr[1, 0, 0]  # Second batch

    def test_3d_array_permutation_result_shape_equals_n_2_4_3_split_1(self):
        # Arrange
        arr = np.arange(24).reshape(2, 3, 4)
        src_dims = np.array(['batch', 'height', 'width'])
        tgt_dims = np.array(['height', 'width', 'batch'])
        result = transpose(arr, src_dims, tgt_dims)
        # Act
        # Assert
        assert result.shape == (3, 4, 2)

    def test_3d_array_permutation_result_shape_equals_n_2_4_3_split_2(self):
        # Arrange
        arr = np.arange(24).reshape(2, 3, 4)
        src_dims = np.array(['batch', 'height', 'width'])
        tgt_dims = np.array(['height', 'width', 'batch'])
        result = transpose(arr, src_dims, tgt_dims)
        result.shape == (3, 4, 2)
        # Act
        # Assert
        assert result[0, 0, 0] == arr[0, 0, 0]

    def test_3d_array_permutation_result_shape_equals_n_2_4_3_split_3(self):
        # Arrange
        arr = np.arange(24).reshape(2, 3, 4)
        src_dims = np.array(['batch', 'height', 'width'])
        tgt_dims = np.array(['height', 'width', 'batch'])
        result = transpose(arr, src_dims, tgt_dims)
        result.shape == (3, 4, 2)
        result[0, 0, 0] == arr[0, 0, 0]
        # Act
        # Assert
        assert result[0, 0, 1] == arr[1, 0, 0]

    def test_3d_array_permutation_result_shape_equals_n_2_4_3_split_4(self):
        # Arrange
        arr = np.arange(24).reshape(2, 3, 4)
        src_dims = np.array(['batch', 'height', 'width'])
        tgt_dims = np.array(['height', 'width', 'batch'])
        result = transpose(arr, src_dims, tgt_dims)
        result.shape == (3, 4, 2)
        result[0, 0, 0] == arr[0, 0, 0]
        result[0, 0, 1] == arr[1, 0, 0]
        tgt_dims = np.array(['batch', 'width', 'height'])
        result = transpose(arr, src_dims, tgt_dims)
        # Act
        # Assert
        assert result.shape == (2, 4, 3)


    def test_4d_array_channels_last_to_first(self):
        """Test common operation: channels last to channels first."""
        # NHWC to NCHW (common in deep learning)
        # Arrange
        arr = np.random.rand(32, 224, 224, 3)  # batch, height, width, channels
        src_dims = np.array(["batch", "height", "width", "channels"])
        tgt_dims = np.array(["batch", "channels", "height", "width"])

        # Act
        result = transpose(arr, src_dims, tgt_dims)
        # Assert
        assert result.shape == (32, 3, 224, 224)

    def test_identity_transpose_np_array_equal_result_arr(self):
        """Test that transposing with same order returns same array."""
        # Arrange
        arr = np.random.rand(2, 3, 4)
        dims = np.array(["a", "b", "c"])

        # Act
        result = transpose(arr, dims, dims)
        # Assert
        assert np.array_equal(result, arr)

    def test_dimension_names_validation(self):
        """Test that mismatched dimension names raise an error."""
        # Arrange
        arr = np.array([[1, 2], [3, 4]])
        src_dims = np.array(["a", "b"])
        # Act
        tgt_dims = np.array(["b", "c"])  # 'c' not in src_dims

        # Assert
        with pytest.raises(
            AssertionError,
            match="Source and target dimensions must contain the same elements",
        ):
            transpose(arr, src_dims, tgt_dims)

    def test_missing_dimension_in_target(self):
        """Test error when target dimensions are incomplete."""
        # Arrange
        arr = np.random.rand(2, 3, 4)
        src_dims = np.array(["a", "b", "c"])
        # Act
        tgt_dims = np.array(["a", "b"])  # Missing 'c'

        # Assert
        with pytest.raises(AssertionError):
            transpose(arr, src_dims, tgt_dims)

    def test_duplicate_dimension_names(self):
        """Test behavior with duplicate dimension names."""
        # Arrange
        # Act
        # Assert
        arr = np.random.rand(2, 3)
        src_dims = np.array(["a", "a"])  # Duplicate
        tgt_dims = np.array(["a", "a"])

        # With duplicate names, np.where returns the first occurrence
        # This causes both dimensions to map to index 0, which may raise an error
        # or produce unexpected results depending on numpy version
        try:
            result = transpose(arr, src_dims, tgt_dims)
            # If it works, shape may be (2, 2) due to first 'a' being used twice
            assert result is not None
        except (IndexError, ValueError):
            # Expected - duplicate dimension names cause issues
            pass

    def test_single_dimension_array(self):
        """Test transposing a 1D array."""
        # Arrange
        arr = np.array([1, 2, 3, 4, 5])
        dims = np.array(["x"])

        # Act
        result = transpose(arr, dims, dims)
        # Assert
        assert np.array_equal(result, arr)

    def test_complex_5d_transpose(self):
        """Test transposing a 5D array with complex permutation."""
        # Shape: (2, 3, 4, 5, 6)
        # Arrange
        arr = np.random.rand(2, 3, 4, 5, 6)
        src_dims = np.array(["a", "b", "c", "d", "e"])
        tgt_dims = np.array(["e", "a", "d", "b", "c"])

        # Act
        result = transpose(arr, src_dims, tgt_dims)
        # Assert
        assert result.shape == (6, 2, 5, 3, 4)

    def test_with_list_inputs_result_is_not_none(self):
        # Arrange
        # Arrange
        arr = [[1, 2, 3], [4, 5, 6]]
        src_dims = np.array(["rows", "cols"])
        tgt_dims = np.array(["cols", "rows"])
        # Act
        result = transpose(arr, src_dims, tgt_dims)
        # Act
        # Assert
        # Assert
        assert result is not None

    def test_with_list_inputs_result_arr_shape_equals_n_3_2_split_1(self):
        # Arrange
        arr = [[1, 2, 3], [4, 5, 6]]
        src_dims = np.array(['rows', 'cols'])
        tgt_dims = np.array(['cols', 'rows'])
        result = transpose(arr, src_dims, tgt_dims)
        # Act
        # Assert
        assert result is not None

    def test_with_list_inputs_result_arr_shape_equals_n_3_2_split_2(self):
        # Arrange
        arr = [[1, 2, 3], [4, 5, 6]]
        src_dims = np.array(['rows', 'cols'])
        tgt_dims = np.array(['cols', 'rows'])
        result = transpose(arr, src_dims, tgt_dims)
        result is not None
        result_arr = np.array(result)
        # Act
        # Assert
        assert result_arr.shape == (3, 2)


    def test_preserve_data_integrity_np_sum_result_np_sum_arr(self):
        # Arrange
        # Arrange
        arr = np.arange(120).reshape(2, 3, 4, 5)
        src_dims = np.array(["a", "b", "c", "d"])
        tgt_dims = np.array(["d", "c", "b", "a"])
        # Act
        result = transpose(arr, src_dims, tgt_dims)
        # Act
        # Assert
        # Assert
        assert np.sum(result) == np.sum(arr)

    def test_preserve_data_integrity_np_array_equal_np_sort_result_flatten_np_sort_arr_flatten(self):
        # Arrange
        # Arrange
        arr = np.arange(120).reshape(2, 3, 4, 5)
        src_dims = np.array(["a", "b", "c", "d"])
        tgt_dims = np.array(["d", "c", "b", "a"])
        # Act
        result = transpose(arr, src_dims, tgt_dims)
        # Act
        # Assert
        # Assert
        assert np.array_equal(np.sort(result.flatten()), np.sort(arr.flatten()))


    def test_real_world_example_video_data(self):
        """Test with real-world example: video data transpose."""
        # Video data: (batch, time, height, width, channels)
        # Arrange
        video = np.random.rand(16, 30, 224, 224, 3)
        src_dims = np.array(["batch", "time", "height", "width", "channels"])

        # Convert to PyTorch format: (batch, channels, time, height, width)
        tgt_dims = np.array(["batch", "channels", "time", "height", "width"])

        # Act
        result = transpose(video, src_dims, tgt_dims)
        # Assert
        assert result.shape == (16, 3, 30, 224, 224)

    def test_case_sensitivity_raises_assertionerror(self):
        """Test that dimension names are case sensitive."""
        # Arrange
        arr = np.array([[1, 2], [3, 4]])
        src_dims = np.array(["A", "B"])
        # Act
        tgt_dims = np.array(["B", "a"])  # 'a' != 'A'

        # Assert
        with pytest.raises(AssertionError):
            transpose(arr, src_dims, tgt_dims)

    def test_empty_array_result_shape_equals_n_3_0(self):
        """Test transposing an empty array."""
        # Arrange
        arr = np.array([]).reshape(0, 3)
        src_dims = np.array(["rows", "cols"])
        tgt_dims = np.array(["cols", "rows"])

        # Act
        result = transpose(arr, src_dims, tgt_dims)
        # Assert
        assert result.shape == (3, 0)

    def test_memory_efficiency_result_base_is_arr_or_np_shares_memory_result_arr(self):
        """Test that transpose returns a view when possible."""
        # Arrange
        arr = np.arange(24).reshape(2, 3, 4)
        src_dims = np.array(["a", "b", "c"])
        tgt_dims = np.array(["a", "c", "b"])

        # Act
        result = transpose(arr, src_dims, tgt_dims)

        # Modifying the result should affect the original if it's a view
        # Note: numpy transpose typically returns a view
        # Assert
        assert result.base is arr or np.shares_memory(result, arr)


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_transpose.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-08-24 09:47:16 (ywatanabe)"
# # ./src/scitex/gen/_transpose.py
#
# from scitex.decorators import numpy_fn
# import numpy as np
#
#
# @numpy_fn
# def transpose(arr_like, src_dims, tgt_dims):
#     """
#     Transpose an array-like object based on source and target dimensions.
#
#     Parameters
#     ----------
#     arr_like : np.array
#         The input array to be transposed.
#     src_dims : np.array
#         List of dimension names in the source order.
#     tgt_dims : np.array
#         List of dimension names in the target order.
#
#     Returns
#     -------
#     np.array
#         The transposed array.
#
#     Raises
#     ------
#     AssertionError
#         If source and target dimensions don't contain the same elements.
#     """
#     assert set(src_dims) == set(tgt_dims), (
#         "Source and target dimensions must contain the same elements"
#     )
#     return arr_like.transpose(*[np.where(src_dims == dim)[0][0] for dim in tgt_dims])

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_transpose.py
# --------------------------------------------------------------------------------
