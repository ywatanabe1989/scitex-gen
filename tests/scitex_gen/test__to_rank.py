import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scitex_gen import to_rank


class TestToRankBasic:
    """Test basic functionality of to_rank."""

    def test_simple_ascending_torch_allclose_ranks_expected(self):
        """Test ranking of simple ascending values."""
        # Arrange
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        ranks = to_rank(tensor)
        # Act
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # Assert
        assert torch.allclose(ranks, expected)

    def test_simple_descending_torch_allclose_ranks_expected(self):
        """Test ranking of simple descending values."""
        # Arrange
        tensor = torch.tensor([4.0, 3.0, 2.0, 1.0])
        ranks = to_rank(tensor)
        # Act
        expected = torch.tensor([4.0, 3.0, 2.0, 1.0])
        # Assert
        assert torch.allclose(ranks, expected)

    def test_unsorted_values_torch_allclose_ranks_expected(self):
        """Test ranking of unsorted values."""
        # Arrange
        tensor = torch.tensor([3.0, 1.0, 4.0, 2.0])
        ranks = to_rank(tensor)
        # Act
        expected = torch.tensor([3.0, 1.0, 4.0, 2.0])
        # Assert
        assert torch.allclose(ranks, expected)

    def test_single_value_torch_allclose_ranks_expected(self):
        """Test ranking of single value."""
        # Arrange
        tensor = torch.tensor([42.0])
        ranks = to_rank(tensor)
        # Act
        expected = torch.tensor([1.0])
        # Assert
        assert torch.allclose(ranks, expected)

    def test_empty_tensor_ranks_shape_equals_torch_size_0(self):
        """Test ranking of empty tensor."""
        # Arrange
        tensor = torch.tensor([])
        # Act
        ranks = to_rank(tensor)
        # Assert
        assert ranks.shape == torch.Size([0])


class TestToRankWithTies:
    """Test to_rank with tied values."""

    def test_two_tied_values(self):
        """Test ranking with two tied values."""
        # Arrange
        tensor = torch.tensor([1.0, 2.0, 2.0, 3.0])
        ranks = to_rank(tensor, method="average")
        # Ranks should be [1, 2.5, 2.5, 4]
        # Act
        expected = torch.tensor([1.0, 2.5, 2.5, 4.0])
        # Assert
        assert torch.allclose(ranks, expected)

    def test_all_tied_values(self):
        """Test ranking when all values are tied."""
        # Arrange
        tensor = torch.tensor([5.0, 5.0, 5.0, 5.0])
        ranks = to_rank(tensor, method="average")
        # All should have average rank (1+2+3+4)/4 = 2.5
        # Act
        expected = torch.tensor([2.5, 2.5, 2.5, 2.5])
        # Assert
        assert torch.allclose(ranks, expected)

    def test_multiple_tie_groups(self):
        """Test ranking with multiple groups of tied values."""
        # Arrange
        tensor = torch.tensor([1.0, 1.0, 3.0, 3.0, 3.0, 6.0])
        ranks = to_rank(tensor, method="average")
        # Ranks: [1.5, 1.5, 4, 4, 4, 6]
        # Act
        expected = torch.tensor([1.5, 1.5, 4.0, 4.0, 4.0, 6.0])
        # Assert
        assert torch.allclose(ranks, expected)

    def test_ties_at_beginning(self):
        """Test ties at the beginning of sorted values."""
        # Arrange
        tensor = torch.tensor([2.0, 2.0, 3.0, 4.0])
        ranks = to_rank(tensor, method="average")
        # Act
        expected = torch.tensor([1.5, 1.5, 3.0, 4.0])
        # Assert
        assert torch.allclose(ranks, expected)

    def test_ties_at_end(self):
        """Test ties at the end of sorted values."""
        # Arrange
        tensor = torch.tensor([1.0, 2.0, 3.0, 3.0])
        ranks = to_rank(tensor, method="average")
        # Act
        expected = torch.tensor([1.0, 2.0, 3.5, 3.5])
        # Assert
        assert torch.allclose(ranks, expected)


class TestToRankDataTypes:
    """Test to_rank with different data types."""

    def test_integer_tensor_torch_allclose_ranks_expected(self):
        """Test ranking of integer tensor."""
        # Arrange
        tensor = torch.tensor([3, 1, 4, 1, 5])
        ranks = to_rank(tensor)
        # Sorted: [1, 1, 3, 4, 5]
        # Ranks: 1 appears twice so both get average (1+2)/2=1.5
        # Expected: pos0(3)->3, pos1(1)->1.5, pos2(4)->4, pos3(1)->1.5, pos4(5)->5
        # Act
        expected = torch.tensor([3.0, 1.5, 4.0, 1.5, 5.0])
        # Assert
        assert torch.allclose(ranks, expected)

    def test_double_precision_ranks_dtype_equals_torch_float32(self):
        # Arrange
        # Arrange
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        # Act
        ranks = to_rank(tensor)
        # Act
        # Assert
        # Assert
        assert ranks.dtype == torch.float32  # Method "average" converts to float

    def test_double_precision_torch_allclose_ranks_expected(self):
        # Arrange
        # Arrange
        tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        # Act
        ranks = to_rank(tensor)
        # Assert
        assert ranks.dtype == torch.float32  # Method "average" converts to float
        expected = torch.tensor([1.0, 2.0, 3.0])
        # Act
        # Assert
        assert torch.allclose(ranks, expected)


    def test_negative_values_torch_allclose_ranks_expected(self):
        """Test ranking with negative values."""
        # Arrange
        tensor = torch.tensor([-3.0, -1.0, -2.0, 0.0, 1.0])
        ranks = to_rank(tensor)
        # Act
        expected = torch.tensor([1.0, 3.0, 2.0, 4.0, 5.0])
        # Assert
        assert torch.allclose(ranks, expected)

    def test_mixed_signs_torch_allclose_ranks_expected(self):
        """Test ranking with mixed positive and negative values."""
        # Arrange
        tensor = torch.tensor([3.0, -2.0, 0.0, -5.0, 1.0])
        ranks = to_rank(tensor)
        # Sorted: [-5, -2, 0, 1, 3] -> original positions get ranks
        # Act
        expected = torch.tensor([5.0, 2.0, 3.0, 1.0, 4.0])
        # Assert
        assert torch.allclose(ranks, expected)


class TestToRankWithDecorator:
    """Test that torch_fn decorator works correctly.

    Note: torch_fn preserves input types, so list->list, numpy->numpy.
    We convert results to tensor or use numpy for comparison.
    """

    def test_list_input_np_allclose_ranks_expected(self):
        """Test that list input works via torch_fn decorator."""
        # Arrange
        data = [3, 1, 4, 1, 5]
        ranks = to_rank(data)
        # Sorted: [1, 1, 3, 4, 5]
        # Ranks: 1 appears twice, average (1+2)/2=1.5
        # pos0(3)->3, pos1(1)->1.5, pos2(4)->4, pos3(1)->1.5, pos4(5)->5
        # Act
        expected = [3.0, 1.5, 4.0, 1.5, 5.0]
        # torch_fn returns same type as input (list)
        # Assert
        assert np.allclose(ranks, expected)

    def test_numpy_input_ranks_is_np_ndarray(self):
        # Arrange
        # Arrange
        data = np.array([2.0, 1.0, 3.0, 1.0])
        ranks = to_rank(data)
        # Act
        expected = np.array([3.0, 1.5, 4.0, 1.5])
        # Act
        # Assert
        # Assert
        assert isinstance(ranks, np.ndarray)

    def test_numpy_input_np_allclose_ranks_expected(self):
        # Arrange
        # Arrange
        data = np.array([2.0, 1.0, 3.0, 1.0])
        ranks = to_rank(data)
        # Act
        expected = np.array([3.0, 1.5, 4.0, 1.5])
        # Act
        # Assert
        # Assert
        assert np.allclose(ranks, expected)


    def test_mixed_input_types_np_allclose_ranks1_1_0_2_0_3_0(self):
        # Arrange
        # Arrange
        # Act
        ranks1 = to_rank((1, 2, 3))
        # Act
        # Assert
        # Assert
        assert np.allclose(ranks1, [1.0, 2.0, 3.0])

    def test_mixed_input_types_np_allclose_ranks2_4_0_2_0_3_0_1_0(self):
        # Arrange
        # Arrange
        # Act
        ranks1 = to_rank((1, 2, 3))
        # Assert
        assert np.allclose(ranks1, [1.0, 2.0, 3.0])
        # Nested list
        ranks2 = to_rank([4, 2, 3, 1])
        # Act
        # Assert
        assert np.allclose(ranks2, [4.0, 2.0, 3.0, 1.0])



class TestToRankEdgeCases:
    """Test edge cases for to_rank."""

    def test_large_tensor_torch_allclose_sorted_ranks_expected_ranks(self):
        """Test ranking of large tensor."""
        # Create tensor with known pattern
        # Arrange
        size = 1000
        tensor = torch.arange(size, dtype=torch.float32)
        torch.manual_seed(42)
        shuffled = tensor[torch.randperm(size)]

        ranks = to_rank(shuffled)

        # Verify all ranks from 1 to size are present
        sorted_ranks, _ = torch.sort(ranks)
        # Act
        expected_ranks = torch.arange(1, size + 1, dtype=torch.float32)
        # Assert
        assert torch.allclose(sorted_ranks, expected_ranks)

    def test_very_close_values_ranks_0_ranks_1_ranks_2_ranks_3(self):
        # Arrange
        # Arrange
        tensor = torch.tensor([1.0, 1.0 + 1e-10, 1.0 + 2e-10, 2.0])
        # Act
        ranks = to_rank(tensor)
        # Act
        # Assert
        # Assert
        assert ranks[0] <= ranks[1] <= ranks[2] <= ranks[3]

    def test_very_close_values_ranks_3_4_0(self):
        # Arrange
        # Arrange
        tensor = torch.tensor([1.0, 1.0 + 1e-10, 1.0 + 2e-10, 2.0])
        # Act
        ranks = to_rank(tensor)
        # Act
        # Assert
        # Assert
        assert ranks[3] == 4.0  # Last should definitely be rank 4


    def test_inf_values_torch_allclose_ranks_expected(self):
        """Test ranking with infinity values."""
        # Arrange
        tensor = torch.tensor([1.0, float("inf"), -float("inf"), 2.0])
        ranks = to_rank(tensor)
        # -inf should be rank 1, inf should be rank 4
        # Act
        expected = torch.tensor([2.0, 4.0, 1.0, 3.0])
        # Assert
        assert torch.allclose(ranks, expected)

    def test_nan_values_ranks_shape_equals_tensor_shape(self):
        """Test behavior with NaN values."""
        # Arrange
        tensor = torch.tensor([1.0, float("nan"), 2.0, 3.0])
        # Act
        ranks = to_rank(tensor)
        # NaN behavior might be undefined, but shouldn't crash
        # Assert
        assert ranks.shape == tensor.shape


class TestToRankRealWorld:
    """Test to_rank with real-world scenarios."""

    def test_percentile_ranking_torch_all_percentiles_0(self):
        # Arrange
        # Arrange
        scores = torch.tensor([85.0, 92.0, 78.0, 92.0, 88.0, 95.0, 82.0])
        ranks = to_rank(scores)
        # Convert to percentiles
        n = len(scores)
        # Act
        percentiles = (ranks - 0.5) / n * 100
        # Act
        # Assert
        # Assert
        assert torch.all(percentiles >= 0)

    def test_percentile_ranking_torch_all_percentiles_100(self):
        # Arrange
        # Arrange
        scores = torch.tensor([85.0, 92.0, 78.0, 92.0, 88.0, 95.0, 82.0])
        ranks = to_rank(scores)
        # Convert to percentiles
        n = len(scores)
        # Act
        percentiles = (ranks - 0.5) / n * 100
        # Act
        # Assert
        # Assert
        assert torch.all(percentiles <= 100)


    def test_competition_ranking_torch_allclose_ranks_expected(self):
        """Test ranking for competition scores (higher is better)."""
        # Arrange
        scores = torch.tensor([100.0, 95.0, 95.0, 90.0, 85.0])
        ranks = to_rank(-scores)  # Negate for descending order

        # In competition: 100 is 1st, both 95s tied for 2nd, 90 is 4th, 85 is 5th
        # Act
        expected = torch.tensor([1.0, 2.5, 2.5, 4.0, 5.0])
        # Assert
        assert torch.allclose(ranks, expected)

    def test_statistical_ranking_outlier_ranks_0_10(self):
        # Arrange
        # Arrange
        torch.manual_seed(42)
        normal_data = torch.randn(100) * 10 + 50
        outliers = torch.tensor([5.0, 95.0, 96.0, 97.0])
        data = torch.cat([normal_data, outliers])
        ranks = to_rank(data)
        # Check that outliers get extreme ranks
        # Act
        outlier_ranks = ranks[-4:]
        # Act
        # Assert
        # Assert
        assert outlier_ranks[0] < 10  # 5.0 should have very low rank

    def test_statistical_ranking_torch_all_outlier_ranks_1_95(self):
        # Arrange
        # Arrange
        torch.manual_seed(42)
        normal_data = torch.randn(100) * 10 + 50
        outliers = torch.tensor([5.0, 95.0, 96.0, 97.0])
        data = torch.cat([normal_data, outliers])
        ranks = to_rank(data)
        # Check that outliers get extreme ranks
        # Act
        outlier_ranks = ranks[-4:]
        # Act
        # Assert
        # Assert
        assert torch.all(outlier_ranks[1:] > 95)  # High values get high ranks



# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_to_rank.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 13:05:47 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/gen/_to_rank.py
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-08-29 22:10:06 (ywatanabe)"
# # ./src/scitex/gen/data_processing/_to_rank.py
#
# import torch
#
# # from .._converters import
# from scitex.decorators import torch_fn
#
#
# @torch_fn
# def to_rank(tensor, method="average"):
#     sorted_tensor, indices = torch.sort(tensor)
#     ranks = torch.empty_like(tensor)
#     ranks[indices] = (
#         torch.arange(len(tensor), dtype=tensor.dtype, device=tensor.device) + 1
#     )
#
#     if method == "average":
#         ranks = ranks.float()
#         ties = torch.nonzero(sorted_tensor[1:] == sorted_tensor[:-1])
#         for i in range(len(ties)):
#             start = ties[i]
#             end = start + 1
#             while (
#                 end < len(sorted_tensor) and sorted_tensor[end] == sorted_tensor[start]
#             ):
#                 end += 1
#             ranks[indices[start:end]] = ranks[indices[start:end]].mean()
#
#     return ranks
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_to_rank.py
# --------------------------------------------------------------------------------
