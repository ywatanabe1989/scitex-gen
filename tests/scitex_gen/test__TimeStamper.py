#!/usr/bin/env python3
# Time-stamp: "2025-05-31 22:00:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/gen/test__TimeStamper.py

"""Tests for TimeStamper class.

Time is injected via the ``clock`` keyword (a deterministic fake callable),
so elapsed-time and delta behavior is exercised against the real DataFrame
bookkeeping without mocking the ``time`` module. Verbose printing is observed
with the real ``capsys`` fixture.
"""

import time

import pandas as pd
import pytest

pytest.importorskip("torch")

from scitex_gen import TimeStamper

_RECORD_COLUMNS = [
    "timestamp",
    "elapsed_since_start",
    "elapsed_since_prev",
    "comment",
    "formatted_text",
]


class _FakeClock:
    """Deterministic clock returning successive values from a fixed sequence."""

    def __init__(self, times):
        self._times = list(times)
        self._index = 0

    def __call__(self):
        value = self._times[self._index]
        self._index += 1
        return value


@pytest.fixture
def stamper():
    return TimeStamper()


@pytest.fixture
def stamper_with_two_records():
    ts = TimeStamper()
    ts("Test1")
    ts("Test2")
    return ts


@pytest.fixture
def stamper_with_three_records():
    ts = TimeStamper()
    ts("T0")
    ts("T1")
    ts("T2")
    return ts


@pytest.fixture
def stamper_elapsed():
    ts = TimeStamper(clock=_FakeClock([0.0, 0.0, 1.0]))
    ts("Start")
    ts("One second")
    return ts


@pytest.fixture
def stamper_for_delta():
    ts = TimeStamper(clock=_FakeClock([0.0, 0.0, 1.0, 3.0]))
    ts("T0")
    ts("T1")
    ts("T2")
    return ts


@pytest.fixture
def stamper_after_sleep():
    ts = TimeStamper()
    ts("Start")
    time.sleep(0.1)
    ts("After sleep")
    return ts


# --- Initialization -------------------------------------------------------


def test_initialization_sets_id_to_minus_one(stamper):
    # Arrange
    # Act
    # Assert
    assert stamper.id == -1


def test_initialization_defaults_to_simple_mode(stamper):
    # Arrange
    # Act
    # Assert
    assert stamper._is_simple is True


def test_initialization_sets_float_start_time(stamper):
    # Arrange
    # Act
    # Assert
    assert isinstance(stamper.start_time, float)


def test_initialization_creates_empty_dataframe(stamper):
    # Arrange
    # Act
    # Assert
    assert isinstance(stamper._df_record, pd.DataFrame)


def test_initialization_dataframe_has_expected_columns(stamper):
    # Arrange
    # Act
    columns = list(stamper._df_record.columns)
    # Assert
    assert columns == _RECORD_COLUMNS


def test_initialization_detailed_disables_simple_mode():
    # Arrange
    # Act
    ts = TimeStamper(is_simple=False)
    # Assert
    assert ts._is_simple is False


# --- Basic call -----------------------------------------------------------


def test_call_increments_id_to_zero(stamper):
    # Arrange
    # Act
    stamper("Test comment")
    # Assert
    assert stamper.id == 0


def test_call_returns_string_result(stamper):
    # Arrange
    # Act
    result = stamper("Test comment")
    # Assert
    assert isinstance(result, str)


def test_call_result_contains_id_label(stamper):
    # Arrange
    # Act
    result = stamper("Test comment")
    # Assert
    assert "ID:0" in result


def test_call_result_contains_comment_text(stamper):
    # Arrange
    # Act
    result = stamper("Test comment")
    # Assert
    assert "Test comment" in result


def test_call_appends_one_record_row(stamper):
    # Arrange
    # Act
    stamper("Test comment")
    # Assert
    assert len(stamper._df_record) == 1


# --- Multiple calls -------------------------------------------------------


def test_multiple_calls_increment_id_to_two(stamper_with_three_records):
    # Arrange
    # Act
    # Assert
    assert stamper_with_three_records.id == 2


def test_multiple_calls_record_each_row(stamper_with_three_records):
    # Arrange
    # Act
    # Assert
    assert len(stamper_with_three_records._df_record) == 3


def test_multiple_calls_store_comments_in_order(stamper_with_three_records):
    # Arrange
    # Act
    comments = stamper_with_three_records._df_record["comment"].tolist()
    # Assert
    assert comments == ["T0", "T1", "T2"]


# --- Elapsed-time tracking (deterministic clock) --------------------------


def test_first_record_elapsed_since_start_is_zero(stamper_elapsed):
    # Arrange
    # Act
    # Assert
    assert stamper_elapsed._df_record.loc[0, "elapsed_since_start"] == 0.0


def test_first_record_elapsed_since_prev_is_zero(stamper_elapsed):
    # Arrange
    # Act
    # Assert
    assert stamper_elapsed._df_record.loc[0, "elapsed_since_prev"] == 0.0


def test_second_record_elapsed_since_start_is_one(stamper_elapsed):
    # Arrange
    # Act
    # Assert
    assert stamper_elapsed._df_record.loc[1, "elapsed_since_start"] == 1.0


def test_second_record_elapsed_since_prev_is_one(stamper_elapsed):
    # Arrange
    # Act
    # Assert
    assert stamper_elapsed._df_record.loc[1, "elapsed_since_prev"] == 1.0


# --- Formatted output -----------------------------------------------------


def test_simple_format_starts_with_id_label():
    # Arrange
    ts = TimeStamper(is_simple=True)
    # Act
    result = ts("Test")
    # Assert
    assert result.startswith("ID:0 | ")


def test_simple_format_contains_comment():
    # Arrange
    ts = TimeStamper(is_simple=True)
    # Act
    result = ts("Test")
    # Assert
    assert "Test" in result


def test_simple_format_ends_with_separator():
    # Arrange
    ts = TimeStamper(is_simple=True)
    # Act
    result = ts("Test")
    # Assert
    assert result.endswith(" | ")


def test_detailed_format_contains_id_label():
    # Arrange
    ts = TimeStamper(is_simple=False)
    # Act
    result = ts("Test")
    # Assert
    assert "Time (id:0):" in result


def test_detailed_format_contains_total_label():
    # Arrange
    ts = TimeStamper(is_simple=False)
    # Act
    result = ts("Test")
    # Assert
    assert "total" in result


def test_detailed_format_contains_prev_label():
    # Arrange
    ts = TimeStamper(is_simple=False)
    # Act
    result = ts("Test")
    # Assert
    assert "prev" in result


def test_detailed_format_contains_unit_label():
    # Arrange
    ts = TimeStamper(is_simple=False)
    # Act
    result = ts("Test")
    # Assert
    assert "[hh:mm:ss]:" in result


def test_detailed_format_contains_comment():
    # Arrange
    ts = TimeStamper(is_simple=False)
    # Act
    result = ts("Test")
    # Assert
    assert "Test" in result


def test_detailed_format_ends_with_newline():
    # Arrange
    ts = TimeStamper(is_simple=False)
    # Act
    result = ts("Test")
    # Assert
    assert result.endswith("\n")


# --- Verbose printing (real stdout capture) -------------------------------


def test_verbose_false_prints_nothing(stamper, capsys):
    # Arrange
    # Act
    stamper("Silent")
    # Assert
    assert capsys.readouterr().out == ""


def test_verbose_true_prints_text(stamper, capsys):
    # Arrange
    # Act
    result = stamper("Verbose", verbose=True)
    # Assert
    assert capsys.readouterr().out == result + "\n"


# --- record property ------------------------------------------------------


def test_record_property_returns_dataframe(stamper_with_two_records):
    # Arrange
    # Act
    record = stamper_with_two_records.record
    # Assert
    assert isinstance(record, pd.DataFrame)


def test_record_property_has_expected_columns(stamper_with_two_records):
    # Arrange
    # Act
    columns = list(stamper_with_two_records.record.columns)
    # Assert
    assert columns == _RECORD_COLUMNS[:-1]


def test_record_property_excludes_formatted_text(stamper_with_two_records):
    # Arrange
    # Act
    columns = list(stamper_with_two_records.record.columns)
    # Assert
    assert "formatted_text" not in columns


def test_record_property_length_matches_calls(stamper_with_two_records):
    # Arrange
    # Act
    record = stamper_with_two_records.record
    # Assert
    assert len(record) == 2


# --- delta ----------------------------------------------------------------


def test_delta_between_consecutive_stamps_is_one(stamper_for_delta):
    # Arrange
    # Act
    delta = stamper_for_delta.delta(1, 0)
    # Assert
    assert delta == 1.0


def test_delta_between_later_stamps_is_two(stamper_for_delta):
    # Arrange
    # Act
    delta = stamper_for_delta.delta(2, 1)
    # Assert
    assert delta == 2.0


def test_delta_negative_indices_match_positive(stamper_with_three_records):
    # Arrange
    # Act
    negative_delta = stamper_with_three_records.delta(-1, -2)
    # Assert
    assert negative_delta == stamper_with_three_records.delta(2, 1)


def test_delta_raises_for_high_invalid_id(stamper):
    # Arrange
    stamper("T0")
    ctx = pytest.raises(ValueError, match="Invalid timestamp ID")
    # Act
    # Assert
    with ctx:
        stamper.delta(0, 5)


def test_delta_raises_for_other_invalid_id(stamper):
    # Arrange
    stamper("T0")
    ctx = pytest.raises(ValueError, match="Invalid timestamp ID")
    # Act
    # Assert
    with ctx:
        stamper.delta(10, 0)


# --- Time formatting (deterministic clock) --------------------------------


def test_elapsed_time_formats_as_hh_mm_ss():
    # 5025 seconds == 01:23:45
    # Arrange
    ts = TimeStamper(clock=_FakeClock([0.0, 5025.0]))
    # Act
    result = ts("Test")
    # Assert
    assert "01:23:45" in result


# --- Continuous operation with real sleep ---------------------------------


def test_first_record_elapsed_is_nonnegative(stamper_after_sleep):
    # Arrange
    # Act
    # Assert
    assert stamper_after_sleep._df_record.loc[0, "elapsed_since_start"] >= 0


def test_second_record_elapsed_since_start_after_sleep(stamper_after_sleep):
    # Arrange
    # Act
    # Assert
    assert stamper_after_sleep._df_record.loc[1, "elapsed_since_start"] >= 0.1


def test_second_record_elapsed_since_prev_after_sleep(stamper_after_sleep):
    # Arrange
    # Act
    # Assert
    assert stamper_after_sleep._df_record.loc[1, "elapsed_since_prev"] >= 0.1


# --- Empty comment --------------------------------------------------------


def test_empty_comment_increments_id(stamper):
    # Arrange
    # Act
    stamper()
    # Assert
    assert stamper.id == 0


def test_empty_comment_returns_string(stamper):
    # Arrange
    # Act
    result = stamper()
    # Assert
    assert isinstance(result, str)


def test_empty_comment_stored_as_empty_string(stamper):
    # Arrange
    # Act
    stamper()
    # Assert
    assert stamper._df_record.loc[0, "comment"] == ""


# --- DataFrame structure --------------------------------------------------


def test_dataframe_length_matches_calls(stamper_with_three_records):
    # Arrange
    # Act
    # Assert
    assert len(stamper_with_three_records._df_record) == 3


def test_dataframe_index_is_sequential(stamper_with_three_records):
    # Arrange
    # Act
    index = stamper_with_three_records._df_record.index.tolist()
    # Assert
    assert index == [0, 1, 2]


def test_dataframe_timestamp_column_is_float(stamper_with_three_records):
    # Arrange
    # Act
    dtype = stamper_with_three_records._df_record["timestamp"].dtype
    # Assert
    assert dtype == float


def test_dataframe_elapsed_since_start_is_float(stamper_with_three_records):
    # Arrange
    # Act
    dtype = stamper_with_three_records._df_record["elapsed_since_start"].dtype
    # Assert
    assert dtype == float


def test_dataframe_elapsed_since_prev_is_float(stamper_with_three_records):
    # Arrange
    # Act
    dtype = stamper_with_three_records._df_record["elapsed_since_prev"].dtype
    # Assert
    assert dtype == float


def test_dataframe_text_columns_are_string_like(stamper_with_three_records):
    # Arrange
    df = stamper_with_three_records._df_record
    # Act
    string_like = [
        df[col].dtype == object or pd.api.types.is_string_dtype(df[col])
        for col in ("comment", "formatted_text")
    ]
    # Assert
    assert all(string_like)


# --- _prev tracking -------------------------------------------------------


def test_initial_prev_equals_start_time(stamper):
    # Arrange
    # Act
    # Assert
    assert stamper._prev == stamper.start_time


def test_prev_updates_after_first_call():
    # Arrange
    ts = TimeStamper(clock=_FakeClock([0.0, 1.0, 2.0, 3.0]))
    initial_prev = ts._prev
    # Act
    ts("First")
    # Assert
    assert ts._prev != initial_prev


def test_prev_updates_after_second_call():
    # Arrange
    ts = TimeStamper(clock=_FakeClock([0.0, 1.0, 2.0, 3.0]))
    ts("First")
    prev_after_first = ts._prev
    # Act
    ts("Second")
    # Assert
    assert ts._prev != prev_after_first


# --- Rapid sequential calls -----------------------------------------------


def test_rapid_calls_record_all_rows(stamper):
    # Arrange
    # Act
    for i in range(10):
        stamper(f"Call {i}")
    # Assert
    assert len(stamper._df_record) == 10


def test_rapid_calls_final_id_is_nine(stamper):
    # Arrange
    # Act
    for i in range(10):
        stamper(f"Call {i}")
    # Assert
    assert stamper.id == 9


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
#         """Calculates time difference between two timestamps."""
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
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_TimeStamper.py
# --------------------------------------------------------------------------------
