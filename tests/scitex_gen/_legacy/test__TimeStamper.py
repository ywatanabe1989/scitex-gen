#!/usr/bin/env python3
# Time-stamp: "2025-05-31 22:00:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/gen/test__TimeStamper.py

"""Tests for TimeStamper class."""

import contextlib
import time

import pandas as pd
import pytest

pytest.importorskip("torch")

from scitex_gen import TimeStamper


@contextlib.contextmanager
def _swap_attr(obj, name, value):
    saved = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, saved)


class TestTimeStamper:
    """Test cases for TimeStamper class."""

    def test_initialization_ts_id_equals_n_1_split_1(self):
        """Test TimeStamper initialization."""
        # Arrange
        ts = TimeStamper()
        # Act
        # Assert
        assert ts.id == -1

    def test_initialization_ts_id_equals_n_1_split_2(self):
        """Test TimeStamper initialization."""
        # Arrange
        ts = TimeStamper()
        ts.id == -1
        # Act
        # Assert
        assert ts._is_simple is True

    def test_initialization_ts_id_equals_n_1_split_3(self):
        """Test TimeStamper initialization."""
        # Arrange
        ts = TimeStamper()
        ts.id == -1
        ts._is_simple is True
        # Act
        # Assert
        assert isinstance(ts.start_time, float)

    def test_initialization_ts_id_equals_n_1_split_4(self):
        """Test TimeStamper initialization."""
        # Arrange
        ts = TimeStamper()
        ts.id == -1
        ts._is_simple is True
        isinstance(ts.start_time, float)
        # Act
        # Assert
        assert isinstance(ts._df_record, pd.DataFrame)

    def test_initialization_ts_id_equals_n_1_split_5(self):
        """Test TimeStamper initialization."""
        # Arrange
        ts = TimeStamper()
        ts.id == -1
        ts._is_simple is True
        isinstance(ts.start_time, float)
        isinstance(ts._df_record, pd.DataFrame)
        # Act
        # Assert
        assert list(ts._df_record.columns) == ['timestamp', 'elapsed_since_start', 'elapsed_since_prev', 'comment', 'formatted_text']

    def test_initialization_detailed_ts_is_simple_is_false(self):
        """Test TimeStamper initialization with is_simple=False."""
        # Arrange
        # Act
        ts = TimeStamper(is_simple=False)

        # Assert
        assert ts._is_simple is False

    def test_call_basic_ts_id_equals_n_0_split_1(self):
        """Test basic timestamp creation."""
        # Arrange
        ts = TimeStamper()
        result = ts('Test comment')
        # Act
        # Assert
        assert ts.id == 0

    def test_call_basic_ts_id_equals_n_0_split_2(self):
        """Test basic timestamp creation."""
        # Arrange
        ts = TimeStamper()
        result = ts('Test comment')
        ts.id == 0
        # Act
        # Assert
        assert isinstance(result, str)

    def test_call_basic_ts_id_equals_n_0_split_3(self):
        """Test basic timestamp creation."""
        # Arrange
        ts = TimeStamper()
        result = ts('Test comment')
        ts.id == 0
        isinstance(result, str)
        # Act
        # Assert
        assert 'ID:0' in result

    def test_call_basic_ts_id_equals_n_0_split_4(self):
        """Test basic timestamp creation."""
        # Arrange
        ts = TimeStamper()
        result = ts('Test comment')
        ts.id == 0
        isinstance(result, str)
        'ID:0' in result
        # Act
        # Assert
        assert 'Test comment' in result

    def test_call_basic_ts_id_equals_n_0_split_5(self):
        """Test basic timestamp creation."""
        # Arrange
        ts = TimeStamper()
        result = ts('Test comment')
        ts.id == 0
        isinstance(result, str)
        'ID:0' in result
        'Test comment' in result
        # Act
        # Assert
        assert len(ts._df_record) == 1

    def test_call_multiple_timestamps_split_1(self):
        """Test multiple timestamp creation."""
        # Arrange
        ts = TimeStamper()
        ts('First')
        ts('Second')
        ts('Third')
        # Act
        # Assert
        assert ts.id == 2

    def test_call_multiple_timestamps_split_2(self):
        """Test multiple timestamp creation."""
        # Arrange
        ts = TimeStamper()
        ts('First')
        ts('Second')
        ts('Third')
        ts.id == 2
        # Act
        # Assert
        assert len(ts._df_record) == 3

    def test_call_multiple_timestamps_split_3(self):
        """Test multiple timestamp creation."""
        # Arrange
        ts = TimeStamper()
        ts('First')
        ts('Second')
        ts('Third')
        ts.id == 2
        len(ts._df_record) == 3
        # Act
        # Assert
        assert ts._df_record.loc[0, 'comment'] == 'First'

    def test_call_multiple_timestamps_split_4(self):
        """Test multiple timestamp creation."""
        # Arrange
        ts = TimeStamper()
        ts('First')
        ts('Second')
        ts('Third')
        ts.id == 2
        len(ts._df_record) == 3
        ts._df_record.loc[0, 'comment'] == 'First'
        # Act
        # Assert
        assert ts._df_record.loc[1, 'comment'] == 'Second'

    def test_call_multiple_timestamps_split_5(self):
        """Test multiple timestamp creation."""
        # Arrange
        ts = TimeStamper()
        ts('First')
        ts('Second')
        ts('Third')
        ts.id == 2
        len(ts._df_record) == 3
        ts._df_record.loc[0, 'comment'] == 'First'
        ts._df_record.loc[1, 'comment'] == 'Second'
        # Act
        # Assert
        assert ts._df_record.loc[2, 'comment'] == 'Third'

    def test_elapsed_time_tracking_split_1(self):
        """Test elapsed time tracking."""
        # Arrange
        values = iter([0.0, 0.0, 1.0, 3.0])

        def fake_time():
            return next(values)
        with _swap_attr(time, 'time', fake_time):
            ts = TimeStamper()
            ts('Start')
            ts('One second')
        # Act
        # Assert
        assert ts._df_record.loc[0, 'elapsed_since_start'] == 0.0

    def test_elapsed_time_tracking_split_2(self):
        """Test elapsed time tracking."""
        # Arrange
        values = iter([0.0, 0.0, 1.0, 3.0])

        def fake_time():
            return next(values)
        with _swap_attr(time, 'time', fake_time):
            ts = TimeStamper()
            ts('Start')
            ts('One second')
        ts._df_record.loc[0, 'elapsed_since_start'] == 0.0
        # Act
        # Assert
        assert ts._df_record.loc[0, 'elapsed_since_prev'] == 0.0

    def test_elapsed_time_tracking_split_3(self):
        """Test elapsed time tracking."""
        # Arrange
        values = iter([0.0, 0.0, 1.0, 3.0])

        def fake_time():
            return next(values)
        with _swap_attr(time, 'time', fake_time):
            ts = TimeStamper()
            ts('Start')
            ts('One second')
        ts._df_record.loc[0, 'elapsed_since_start'] == 0.0
        ts._df_record.loc[0, 'elapsed_since_prev'] == 0.0
        # Act
        # Assert
        assert ts._df_record.loc[1, 'elapsed_since_start'] == 1.0

    def test_elapsed_time_tracking_split_4(self):
        """Test elapsed time tracking."""
        # Arrange
        values = iter([0.0, 0.0, 1.0, 3.0])

        def fake_time():
            return next(values)
        with _swap_attr(time, 'time', fake_time):
            ts = TimeStamper()
            ts('Start')
            ts('One second')
        ts._df_record.loc[0, 'elapsed_since_start'] == 0.0
        ts._df_record.loc[0, 'elapsed_since_prev'] == 0.0
        ts._df_record.loc[1, 'elapsed_since_start'] == 1.0
        # Act
        # Assert
        assert ts._df_record.loc[1, 'elapsed_since_prev'] == 1.0

    def test_formatted_output_simple_split_1(self):
        """Test simple formatted output."""
        # Arrange
        ts = TimeStamper(is_simple=True)
        result = ts('Test')
        # Act
        # Assert
        assert result.startswith('ID:0 | ')

    def test_formatted_output_simple_split_2(self):
        """Test simple formatted output."""
        # Arrange
        ts = TimeStamper(is_simple=True)
        result = ts('Test')
        result.startswith('ID:0 | ')
        # Act
        # Assert
        assert 'Test' in result

    def test_formatted_output_simple_split_3(self):
        """Test simple formatted output."""
        # Arrange
        ts = TimeStamper(is_simple=True)
        result = ts('Test')
        result.startswith('ID:0 | ')
        'Test' in result
        # Act
        # Assert
        assert result.endswith(' | ')

    def test_formatted_output_detailed_split_1(self):
        """Test detailed formatted output."""
        # Arrange
        ts = TimeStamper(is_simple=False)
        result = ts('Test')
        # Act
        # Assert
        assert 'Time (id:0):' in result

    def test_formatted_output_detailed_split_2(self):
        """Test detailed formatted output."""
        # Arrange
        ts = TimeStamper(is_simple=False)
        result = ts('Test')
        'Time (id:0):' in result
        # Act
        # Assert
        assert 'total' in result

    def test_formatted_output_detailed_split_3(self):
        """Test detailed formatted output."""
        # Arrange
        ts = TimeStamper(is_simple=False)
        result = ts('Test')
        'Time (id:0):' in result
        'total' in result
        # Act
        # Assert
        assert 'prev' in result

    def test_formatted_output_detailed_split_4(self):
        """Test detailed formatted output."""
        # Arrange
        ts = TimeStamper(is_simple=False)
        result = ts('Test')
        'Time (id:0):' in result
        'total' in result
        'prev' in result
        # Act
        # Assert
        assert '[hh:mm:ss]:' in result

    def test_formatted_output_detailed_split_5(self):
        """Test detailed formatted output."""
        # Arrange
        ts = TimeStamper(is_simple=False)
        result = ts('Test')
        'Time (id:0):' in result
        'total' in result
        'prev' in result
        '[hh:mm:ss]:' in result
        # Act
        # Assert
        assert 'Test' in result

    def test_formatted_output_detailed_split_6(self):
        """Test detailed formatted output."""
        # Arrange
        ts = TimeStamper(is_simple=False)
        result = ts('Test')
        'Time (id:0):' in result
        'total' in result
        'prev' in result
        '[hh:mm:ss]:' in result
        'Test' in result
        # Act
        # Assert
        assert result.endswith('\n')

    def test_verbose_output_silent_call_does_not_print(self):
        """Verbose=False (default): nothing printed."""
        # Arrange
        ts = TimeStamper()
        import builtins
        calls = []

        def fake_print(*args, **kwargs):
            calls.append((args, kwargs))
        # Act
        with _swap_attr(builtins, 'print', fake_print):
            ts('Silent')
        # Assert
        assert calls == []

    def test_verbose_output_verbose_call_prints_result(self):
        """Verbose=True: prints the formatted result."""
        # Arrange
        ts = TimeStamper()
        import builtins
        calls = []

        def fake_print(*args, **kwargs):
            calls.append((args, kwargs))
        # Act
        with _swap_attr(builtins, 'print', fake_print):
            result = ts('Verbose', verbose=True)
        # Assert
        assert calls == [((result,), {})]

    def test_record_property_record_is_pd_dataframe_split_1(self):
        """Test record property returns correct columns."""
        # Arrange
        ts = TimeStamper()
        ts('Test1')
        ts('Test2')
        record = ts.record
        # Act
        # Assert
        assert isinstance(record, pd.DataFrame)

    def test_record_property_record_is_pd_dataframe_split_2(self):
        """Test record property returns correct columns."""
        # Arrange
        ts = TimeStamper()
        ts('Test1')
        ts('Test2')
        record = ts.record
        isinstance(record, pd.DataFrame)
        # Act
        # Assert
        assert list(record.columns) == ['timestamp', 'elapsed_since_start', 'elapsed_since_prev', 'comment']

    def test_record_property_record_is_pd_dataframe_split_3(self):
        """Test record property returns correct columns."""
        # Arrange
        ts = TimeStamper()
        ts('Test1')
        ts('Test2')
        record = ts.record
        isinstance(record, pd.DataFrame)
        list(record.columns) == ['timestamp', 'elapsed_since_start', 'elapsed_since_prev', 'comment']
        # Act
        # Assert
        assert 'formatted_text' not in record.columns

    def test_record_property_record_is_pd_dataframe_split_4(self):
        """Test record property returns correct columns."""
        # Arrange
        ts = TimeStamper()
        ts('Test1')
        ts('Test2')
        record = ts.record
        isinstance(record, pd.DataFrame)
        list(record.columns) == ['timestamp', 'elapsed_since_start', 'elapsed_since_prev', 'comment']
        'formatted_text' not in record.columns
        # Act
        # Assert
        assert len(record) == 2

    def test_delta_basic_smoke_case(self):
        """Test delta calculation between consecutive timestamps."""
        # Arrange
        values = iter([0.0, 0.0, 1.0, 3.0, 6.0])

        def fake_time():
            return next(values)
        # Act
        with _swap_attr(time, 'time', fake_time):
            ts = TimeStamper()
            ts('T0')
            ts('T1')
            ts('T2')
            deltas = (ts.delta(1, 0), ts.delta(2, 1))
        # Assert
        assert deltas == (1.0, 2.0)

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

    def test_delta_invalid_ids_split_1(self):
        """Test delta with invalid IDs."""
        # Arrange
        ts = TimeStamper()
        ts('T0')
        # Act
        # Assert
        with pytest.raises(ValueError, match='Invalid timestamp ID'):
            ts.delta(0, 5)

    def test_delta_invalid_ids_split_2(self):
        """Test delta with invalid IDs."""
        # Arrange
        ts = TimeStamper()
        ts('T0')
        try:
            ts.delta(0, 5)
        except Exception:
            pass
        # Act
        # Assert
        with pytest.raises(ValueError, match='Invalid timestamp ID'):
            ts.delta(10, 0)

    def test_time_formatting_n_01_23_45_in_result(self):
        """Test time formatting."""
        # Mock gmtime to return predictable result
        # Use a proper struct_time object
        # Arrange
        import time as time_module

        fixed_struct = time_module.struct_time((1970, 1, 1, 1, 23, 45, 3, 1, 0))

        def fake_gmtime(*_args, **_kwargs):
            return fixed_struct

        with _swap_attr(time_module, "gmtime", fake_gmtime):
            ts = TimeStamper()
            # Act
            result = ts("Test")

        # Should format as HH:MM:SS
        # Assert
        assert "01:23:45" in result

    def test_continuous_operation_ts_df_record_loc_0_elapsed_since_start_0_split_1(self):
        """Test continuous operation with sleep."""
        # Arrange
        ts = TimeStamper()
        ts('Start')
        time.sleep(0.1)
        ts('After sleep')
        # Act
        # Assert
        assert ts._df_record.loc[0, 'elapsed_since_start'] >= 0

    def test_continuous_operation_ts_df_record_loc_0_elapsed_since_start_0_split_2(self):
        """Test continuous operation with sleep."""
        # Arrange
        ts = TimeStamper()
        ts('Start')
        time.sleep(0.1)
        ts('After sleep')
        ts._df_record.loc[0, 'elapsed_since_start'] >= 0
        # Act
        # Assert
        assert ts._df_record.loc[1, 'elapsed_since_start'] >= 0.1

    def test_continuous_operation_ts_df_record_loc_0_elapsed_since_start_0_split_3(self):
        """Test continuous operation with sleep."""
        # Arrange
        ts = TimeStamper()
        ts('Start')
        time.sleep(0.1)
        ts('After sleep')
        ts._df_record.loc[0, 'elapsed_since_start'] >= 0
        ts._df_record.loc[1, 'elapsed_since_start'] >= 0.1
        # Act
        # Assert
        assert ts._df_record.loc[1, 'elapsed_since_prev'] >= 0.1

    def test_empty_comment_ts_id_equals_n_0_split_1(self):
        """Test timestamp with empty comment."""
        # Arrange
        ts = TimeStamper()
        result = ts()
        # Act
        # Assert
        assert ts.id == 0

    def test_empty_comment_ts_id_equals_n_0_split_2(self):
        """Test timestamp with empty comment."""
        # Arrange
        ts = TimeStamper()
        result = ts()
        ts.id == 0
        # Act
        # Assert
        assert isinstance(result, str)

    def test_empty_comment_ts_id_equals_n_0_split_3(self):
        """Test timestamp with empty comment."""
        # Arrange
        ts = TimeStamper()
        result = ts()
        ts.id == 0
        isinstance(result, str)
        # Act
        # Assert
        assert ts._df_record.loc[0, 'comment'] == ''

    def test_dataframe_structure_len_df_is_3_split_1(self):
        """Test DataFrame structure after multiple operations."""
        # Arrange
        ts = TimeStamper()
        ts('First', verbose=False)
        ts('Second', verbose=True)
        ts('Third')
        df = ts._df_record
        # Act
        # Assert
        assert len(df) == 3

    def test_dataframe_structure_len_df_is_3_split_2(self):
        """Test DataFrame structure after multiple operations."""
        # Arrange
        ts = TimeStamper()
        ts('First', verbose=False)
        ts('Second', verbose=True)
        ts('Third')
        df = ts._df_record
        len(df) == 3
        # Act
        # Assert
        assert df.index.tolist() == [0, 1, 2]

    def test_dataframe_structure_len_df_is_3_split_3(self):
        """Test DataFrame structure after multiple operations."""
        # Arrange
        ts = TimeStamper()
        ts('First', verbose=False)
        ts('Second', verbose=True)
        ts('Third')
        df = ts._df_record
        len(df) == 3
        df.index.tolist() == [0, 1, 2]
        # Act
        # Assert
        assert df['timestamp'].dtype == float

    def test_dataframe_structure_len_df_is_3_split_4(self):
        """Test DataFrame structure after multiple operations."""
        # Arrange
        ts = TimeStamper()
        ts('First', verbose=False)
        ts('Second', verbose=True)
        ts('Third')
        df = ts._df_record
        len(df) == 3
        df.index.tolist() == [0, 1, 2]
        df['timestamp'].dtype == float
        # Act
        # Assert
        assert df['elapsed_since_start'].dtype == float

    def test_dataframe_structure_len_df_is_3_split_5(self):
        """Test DataFrame structure after multiple operations."""
        # Arrange
        ts = TimeStamper()
        ts('First', verbose=False)
        ts('Second', verbose=True)
        ts('Third')
        df = ts._df_record
        len(df) == 3
        df.index.tolist() == [0, 1, 2]
        df['timestamp'].dtype == float
        df['elapsed_since_start'].dtype == float
        # Act
        # Assert
        assert df['elapsed_since_prev'].dtype == float

    def test_dataframe_structure_len_df_is_3_split_6(self):
        """Test DataFrame structure after multiple operations."""
        # Arrange
        ts = TimeStamper()
        ts('First', verbose=False)
        ts('Second', verbose=True)
        ts('Third')
        df = ts._df_record
        len(df) == 3
        df.index.tolist() == [0, 1, 2]
        df['timestamp'].dtype == float
        df['elapsed_since_start'].dtype == float
        df['elapsed_since_prev'].dtype == float
        import pandas as pd
        # Act
        # Assert
        for col in ('comment', 'formatted_text'):
            assert df[col].dtype == object or pd.api.types.is_string_dtype(df[col])

    def test_prev_time_update_split_1(self):
        """Test that _prev time is updated correctly."""
        # Arrange
        ts = TimeStamper()
        initial_prev = ts._prev
        # Act
        # Assert
        assert initial_prev == ts.start_time

    def test_prev_time_update_split_2(self):
        """Test that _prev time is updated correctly."""
        # Arrange
        ts = TimeStamper()
        initial_prev = ts._prev
        initial_prev == ts.start_time
        ts('First')
        # Act
        # Assert
        assert ts._prev != initial_prev

    def test_prev_time_update_split_3(self):
        """Test that _prev time is updated correctly."""
        # Arrange
        ts = TimeStamper()
        initial_prev = ts._prev
        initial_prev == ts.start_time
        ts('First')
        ts._prev != initial_prev
        prev_after_first = ts._prev
        ts('Second')
        # Act
        # Assert
        assert ts._prev != prev_after_first

    def test_thread_safety_consideration_split_1(self):
        """Test basic thread safety considerations."""
        # Arrange
        ts = TimeStamper()
        for i in range(10):
            ts(f'Call {i}')
        # Act
        # Assert
        assert len(ts._df_record) == 10

    def test_thread_safety_consideration_split_2(self):
        """Test basic thread safety considerations."""
        # Arrange
        ts = TimeStamper()
        for i in range(10):
            ts(f'Call {i}')
        len(ts._df_record) == 10
        # Act
        # Assert
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
