#!/usr/bin/env python3
# Time-stamp: "2025-05-31 22:00:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/gen/test__TimeStamper.py

"""Tests for TimeStamper class."""

import time
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("torch")

from scitex_gen import TimeStamper


class TestTimeStamper:
    """Test cases for TimeStamper class."""

    def test_initialization_ts_id_equals_n_1(self):
        """Test TimeStamper initialization."""
        # Arrange
        # Act
        ts = TimeStamper()

        # Assert
        assert ts.id == -1
        assert ts._is_simple is True
        assert isinstance(ts.start_time, float)
        assert isinstance(ts._df_record, pd.DataFrame)
        assert list(ts._df_record.columns) == [
            "timestamp",
            "elapsed_since_start",
            "elapsed_since_prev",
            "comment",
            "formatted_text",
        ]

    def test_initialization_detailed_ts_is_simple_is_false(self):
        """Test TimeStamper initialization with is_simple=False."""
        # Arrange
        # Act
        ts = TimeStamper(is_simple=False)

        # Assert
        assert ts._is_simple is False

    def test_call_basic_ts_id_equals_n_0(self):
        """Test basic timestamp creation."""
        # Arrange
        ts = TimeStamper()

        # Act
        result = ts("Test comment")

        # Assert
        assert ts.id == 0
        assert isinstance(result, str)
        assert "ID:0" in result
        assert "Test comment" in result
        assert len(ts._df_record) == 1

    def test_call_multiple_timestamps(self):
        """Test multiple timestamp creation."""
        # Arrange
        ts = TimeStamper()

        ts("First")
        ts("Second")
        # Act
        ts("Third")

        # Assert
        assert ts.id == 2
        assert len(ts._df_record) == 3
        assert ts._df_record.loc[0, "comment"] == "First"
        assert ts._df_record.loc[1, "comment"] == "Second"
        assert ts._df_record.loc[2, "comment"] == "Third"

    @patch("time.time")
    def test_elapsed_time_tracking(self, mock_time):
        """Test elapsed time tracking."""
        # Mock time progression
        # Arrange
        mock_time.side_effect = [0.0, 0.0, 1.0, 3.0]  # start, start, +1s, +3s

        ts = TimeStamper()
        ts("Start")
        # Act
        ts("One second")

        # Check elapsed times
        # First call has 0 elapsed time
        # Assert
        assert ts._df_record.loc[0, "elapsed_since_start"] == 0.0
        assert ts._df_record.loc[0, "elapsed_since_prev"] == 0.0
        # Second call has 1s from start and 1s from prev
        assert ts._df_record.loc[1, "elapsed_since_start"] == 1.0
        assert ts._df_record.loc[1, "elapsed_since_prev"] == 1.0

    def test_formatted_output_simple(self):
        """Test simple formatted output."""
        # Arrange
        ts = TimeStamper(is_simple=True)

        # Act
        result = ts("Test")

        # Simple format: "ID:0 | HH:MM:SS Test | "
        # Assert
        assert result.startswith("ID:0 | ")
        assert "Test" in result
        assert result.endswith(" | ")

    def test_formatted_output_detailed(self):
        """Test detailed formatted output."""
        # Arrange
        ts = TimeStamper(is_simple=False)

        # Act
        result = ts("Test")

        # Detailed format includes "total" and "prev"
        # Assert
        assert "Time (id:0):" in result
        assert "total" in result
        assert "prev" in result
        assert "[hh:mm:ss]:" in result
        assert "Test" in result
        assert result.endswith("\n")

    @patch("builtins.print")
    def test_verbose_output_not_mock_print_called(self, mock_print):
        """Test verbose output."""
        # Arrange
        ts = TimeStamper()

        # Test with verbose=False (default)
        # Act
        ts("Silent")
        # Assert
        assert not mock_print.called

        # Test with verbose=True
        result = ts("Verbose", verbose=True)
        mock_print.assert_called_once_with(result)

    def test_record_property_record_is_pd_dataframe(self):
        """Test record property returns correct columns."""
        # Arrange
        ts = TimeStamper()
        ts("Test1")
        ts("Test2")

        # Act
        record = ts.record

        # Assert
        assert isinstance(record, pd.DataFrame)
        assert list(record.columns) == [
            "timestamp",
            "elapsed_since_start",
            "elapsed_since_prev",
            "comment",
        ]
        assert "formatted_text" not in record.columns
        assert len(record) == 2

    def test_delta_basic_smoke_case(self):
        """Test delta calculation between timestamps."""
        # Arrange
        # Act
        # Assert
        with patch("time.time") as mock_time:
            mock_time.side_effect = [0.0, 0.0, 1.0, 3.0, 6.0]

            ts = TimeStamper()
            ts("T0")  # time=0.0
            ts("T1")  # time=1.0
            ts("T2")  # time=3.0

            # Delta between T1 and T0
            delta = ts.delta(1, 0)
            assert delta == 1.0  # 1.0 - 0.0

            # Delta between T2 and T1
            delta = ts.delta(2, 1)
            assert delta == 2.0  # 3.0 - 1.0

    def test_delta_negative_indices(self):
        """Test delta with negative indices."""
        # Arrange
        ts = TimeStamper()
        ts("T0")
        ts("T1")
        ts("T2")

        # -1 should refer to last timestamp (id=2)
        # -2 should refer to second-to-last (id=1)
        # Act
        delta = ts.delta(-1, -2)

        # Should be same as delta(2, 1)
        # Assert
        assert delta == ts.delta(2, 1)

    def test_delta_invalid_ids(self):
        """Test delta with invalid IDs."""
        # Arrange
        ts = TimeStamper()
        # Act
        ts("T0")

        # Test with non-existent ID
        # Assert
        with pytest.raises(ValueError, match="Invalid timestamp ID"):
            ts.delta(0, 5)

        # Test with another non-existent ID
        with pytest.raises(ValueError, match="Invalid timestamp ID"):
            ts.delta(10, 0)

    @patch("time.gmtime")
    def test_time_formatting_n_01_23_45_in_result(self, mock_gmtime):
        """Test time formatting."""
        # Mock gmtime to return predictable result
        # Use a proper struct_time object
        # Arrange
        import time as time_module

        mock_gmtime.return_value = time_module.struct_time(
            (1970, 1, 1, 1, 23, 45, 3, 1, 0)
        )

        ts = TimeStamper()
        # Act
        result = ts("Test")

        # Should format as HH:MM:SS
        # Assert
        assert "01:23:45" in result

    def test_continuous_operation_ts_df_record_loc_0_elapsed_since_start_0(self):
        """Test continuous operation with sleep."""
        # Arrange
        ts = TimeStamper()

        ts("Start")
        time.sleep(0.1)  # Small sleep
        # Act
        ts("After sleep")

        # Row 0 is "Start" - elapsed times should be nearly 0
        # Assert
        assert ts._df_record.loc[0, "elapsed_since_start"] >= 0
        # Row 1 is "After sleep" - should have elapsed >= 0.1s
        assert ts._df_record.loc[1, "elapsed_since_start"] >= 0.1
        assert ts._df_record.loc[1, "elapsed_since_prev"] >= 0.1

    def test_empty_comment_ts_id_equals_n_0(self):
        """Test timestamp with empty comment."""
        # Arrange
        ts = TimeStamper()

        # Act
        result = ts()  # No comment provided

        # Assert
        assert ts.id == 0
        assert isinstance(result, str)
        assert ts._df_record.loc[0, "comment"] == ""

    def test_dataframe_structure_len_df_is_3(self):
        """Test DataFrame structure after multiple operations."""
        # Arrange
        ts = TimeStamper()

        ts("First", verbose=False)
        ts("Second", verbose=True)
        ts("Third")

        # Act
        df = ts._df_record

        # Check structure
        # Assert
        assert len(df) == 3
        assert df.index.tolist() == [0, 1, 2]

        # Check data types
        assert df["timestamp"].dtype == float
        assert df["elapsed_since_start"].dtype == float
        assert df["elapsed_since_prev"].dtype == float
        # Accept either object or StringDtype (pandas >= 2.1 infers string)
        import pandas as pd

        for col in ("comment", "formatted_text"):
            assert df[col].dtype == object or pd.api.types.is_string_dtype(df[col])

    def test_prev_time_update(self):
        """Test that _prev time is updated correctly."""
        # Arrange
        ts = TimeStamper()

        # Act
        initial_prev = ts._prev
        # Assert
        assert initial_prev == ts.start_time

        ts("First")
        assert ts._prev != initial_prev

        prev_after_first = ts._prev
        ts("Second")
        assert ts._prev != prev_after_first

    def test_thread_safety_consideration(self):
        """Test basic thread safety considerations."""
        # Note: The class is not thread-safe, but test basic operation
        # Arrange
        ts = TimeStamper()

        # Simulate rapid calls
        # Act
        for i in range(10):
            ts(f"Call {i}")

        # Check all were recorded
        # Assert
        assert len(ts._df_record) == 10
        assert ts.id == 9


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_TimeStamper.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "ywatanabe (2024-11-07 16:06:50)"
# # File: ./scitex_repo/src/scitex/gen/_TimeStamper.py
#
# import time
# from typing import Union, Optional
# import pandas as pd
#
#
# class TimeStamper:
#     """
#     Functionality:
#         * Generates timestamps with comments and tracks elapsed time
#         * Records timestamps in a DataFrame for analysis
#         * Calculates time differences between timestamps
#     Input:
#         * Comments for each timestamp
#         * Format preference (simple or detailed)
#     Output:
#         * Formatted timestamp strings
#         * DataFrame with timestamp records
#         * Time differences between specified timestamps
#     Prerequisites:
#         * pandas
#     """
#
#     def __init__(self, is_simple: bool = True) -> None:
#         self.id: int = -1
#         self.start_time: float = time.time()
#         self._is_simple: bool = is_simple
#         self._prev: float = self.start_time
#         self._df_record: pd.DataFrame = pd.DataFrame(
#             columns=[
#                 "timestamp",
#                 "elapsed_since_start",
#                 "elapsed_since_prev",
#                 "comment",
#                 "formatted_text",
#             ]
#         )
#
#     def __call__(self, comment: str = "", verbose: bool = False) -> str:
#         now: float = time.time()
#         from_start: float = now - self.start_time
#         from_prev: float = now - self._prev
#
#         formatted_from_start: str = time.strftime("%H:%M:%S", time.gmtime(from_start))
#         formatted_from_prev: str = time.strftime("%H:%M:%S", time.gmtime(from_prev))
#
#         self.id += 1
#         self._prev = now
#
#         text: str = (
#             f"ID:{self.id} | {formatted_from_start} {comment} | "
#             if self._is_simple
#             else f"Time (id:{self.id}): total {formatted_from_start}, prev {formatted_from_prev} [hh:mm:ss]: {comment}\n"
#         )
#
#         self._df_record.loc[self.id] = [
#             now,
#             from_start,
#             from_prev,
#             comment,
#             text,
#         ]
#
#         if verbose:
#             print(text)
#         return text
#
#     @property
#     def record(self) -> pd.DataFrame:
#         """Returns the record DataFrame without the formatted_text column."""
#         return self._df_record[
#             [
#                 "timestamp",
#                 "elapsed_since_start",
#                 "elapsed_since_prev",
#                 "comment",
#             ]
#         ]
#
#     def delta(self, id1: int, id2: int) -> float:
#         """Calculates time difference between two timestamps.
#
#         Parameters
#         ----------
#         id1 : int
#             First timestamp ID
#         id2 : int
#             Second timestamp ID
#
#         Returns
#         -------
#         float
#             Time difference in seconds
#
#         Raises
#         ------
#         ValueError
#             If IDs don't exist in records
#         """
#         if id1 < 0:
#             id1 = len(self._df_record) + id1
#         if id2 < 0:
#             id2 = len(self._df_record) + id2
#
#         if not all(idx in self._df_record.index for idx in [id1, id2]):
#             raise ValueError("Invalid timestamp ID(s)")
#
#         return (
#             self._df_record.loc[id1, "timestamp"]
#             - self._df_record.loc[id2, "timestamp"]
#         )
#
#
# if __name__ == "__main__":
#     ts = TimeStamper(is_simple=True)
#     ts("Starting process")
#     time.sleep(1)
#     ts("One second later")
#     time.sleep(2)
#     ts("Two seconds later")
#
#
# # EOF
#
# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# # # Time-stamp: "ywatanabe (2024-11-07 16:06:50)"
# # # File: ./scitex_repo/src/scitex/gen/_TimeStamper.py
#
# # import time
# # import pandas as pd
#
#
# # class TimeStamper:
# #     """
# #     A class for generating timestamps with optional comments, tracking both the time since object creation and since the last call.
# #     """
#
# #     def __init__(self, is_simple=True):
# #         self.id = -1
# #         self.start_time = time.time()
# #         self._is_simple = is_simple
# #         self._prev = self.start_time
# #         self._df_record = pd.DataFrame(
# #             columns=[
# #                 "timestamp",
# #                 "elapsed_since_start",
# #                 "elapsed_since_prev",
# #                 "comment",
# #                 "formatted_text",
# #             ]
# #         )
#
# #     def __call__(self, comment="", verbose=False):
# #         now = time.time()
# #         from_start = now - self.start_time
# #         from_prev = now - self._prev
# #         formatted_from_start = time.strftime(
# #             "%H:%M:%S", time.gmtime(from_start)
# #         )
# #         formatted_from_prev = time.strftime("%H:%M:%S", time.gmtime(from_prev))
# #         self.id += 1
# #         self._prev = now
# #         text = (
# #             f"ID:{self.id} | {formatted_from_start} {comment} | "
# #             if self._is_simple
# #             else f"Time (id:{self.id}): total {formatted_from_start}, prev {formatted_from_prev} [hh:mm:ss]: {comment}\n"
# #         )
#
# #         # Update DataFrame directly
# #         self._df_record.loc[self.id] = [
# #             now,
# #             from_start,
# #             from_prev,
# #             comment,
# #             text,
# #         ]
#
# #         if verbose:
# #             print(text)
# #         return text
#
# #     @property
# #     def record(self):
# #         return self._df_record[
# #             [
# #                 "timestamp",
# #                 "elapsed_since_start",
# #                 "elapsed_since_prev",
# #                 "comment",
# #             ]
# #         ]
#
# #     def delta(self, id1, id2):
# #         """
# #         Calculate the difference in seconds between two timestamps identified by their IDs.
#
# #         Parameters:
# #             id1 (int): The ID of the first timestamp.
# #             id2 (int): The ID of the second timestamp.
#
# #         Returns:
# #             float: The difference in seconds between the two timestamps.
#
# #         Raises:
# #             ValueError: If either id1 or id2 is not in the DataFrame index.
# #         """
# #         # Adjust for negative indices, similar to negative list indexing in Python
# #         if id1 < 0:
# #             id1 = len(self._df_record) + id1
# #         if id2 < 0:
# #             id2 = len(self._df_record) + id2
#
# #         # Check if both IDs exist in the DataFrame
# #         if (
# #             id1 not in self._df_record.index
# #             or id2 not in self._df_record.index
# #         ):
# #             raise ValueError(
# #                 "One or both of the IDs do not exist in the record."
# #             )
#
# #         # Compute the difference in timestamps
# #         time_diff = (
# #             self._df_record.loc[id1, "timestamp"]
# #             - self._df_record.loc[id2, "timestamp"]
# #         )
# #         return time_diff
#
#
# # if __name__ == "__main__":
# #     ts = TimeStamper(is_simple=True)
# #     ts("Starting process")
# #     time.sleep(1)
# #     ts("One second later")
# #     time.sleep(2)
# #     ts("Two seconds later")
#
#
# # # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_TimeStamper.py
# --------------------------------------------------------------------------------
