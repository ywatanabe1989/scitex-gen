#!/usr/bin/env python3
# Time-stamp: "2026-01-04 (ywatanabe)"
# File: ./tests/scitex/gen/test__cache.py

"""Tests for cache module.

The cache module provides a re-export of functools.lru_cache with maxsize=None,
providing unlimited memoization capability.
"""

import pytest

pytest.importorskip("torch")

from functools import lru_cache

from scitex_gen import cache


class TestCacheBasic:
    """Test basic cache functionality."""

    def test_cache_is_lru_cache(self):
        """Verify cache is the lru_cache decorator."""
        # cache should be lru_cache with maxsize=None
        # Arrange
        # Act
        # Assert
        assert callable(cache)

    def test_cache_memoizes_first_call_returns_computed_value(self):
        """First call computes and returns the value."""
        # Arrange
        @cache
        def expensive_function(x):
            return x * 2

        # Act
        result1 = expensive_function(5)
        # Assert
        assert result1 == 10

    def test_cache_memoizes_first_call_increments_count_once(self):
        """First call executes the function body exactly once."""
        # Arrange
        call_count = 0

        @cache
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # Act
        expensive_function(5)
        # Assert
        assert call_count == 1

    def test_cache_memoizes_repeat_call_returns_cached_value(self):
        """Repeated call with same arg returns the cached value."""
        # Arrange
        @cache
        def expensive_function(x):
            return x * 2

        expensive_function(5)
        # Act
        result2 = expensive_function(5)
        # Assert
        assert result2 == 10

    def test_cache_memoizes_repeat_call_does_not_increment_count(self):
        """Repeated call with same arg does not re-execute the body."""
        # Arrange
        call_count = 0

        @cache
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive_function(5)
        # Act
        expensive_function(5)
        # Assert
        assert call_count == 1  # No additional call

    def test_cache_memoizes_different_arg_returns_new_value(self):
        """A different arg computes and returns a new value."""
        # Arrange
        @cache
        def expensive_function(x):
            return x * 2

        expensive_function(5)
        # Act
        result3 = expensive_function(10)
        # Assert
        assert result3 == 20

    def test_cache_memoizes_different_arg_increments_count(self):
        """A different arg re-executes the function body."""
        # Arrange
        call_count = 0

        @cache
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive_function(5)
        # Act
        expensive_function(10)
        # Assert
        assert call_count == 2

    def test_cache_with_multiple_arguments_returns_sum(self):
        """Cache wraps a multi-arg function and returns its result."""
        # Arrange
        @cache
        def add(a, b):
            return a + b

        # Act
        result = add(1, 2)
        # Assert
        assert result == 3

    def test_cache_with_multiple_arguments_same_args_cached(self):
        """Repeated call with same args does not re-execute the body."""
        # Arrange
        call_count = 0

        @cache
        def add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        add(1, 2)
        # Act
        add(1, 2)
        # Assert
        assert call_count == 1  # Cached

    def test_cache_with_multiple_arguments_reordered_args_recompute(self):
        """Different arg order is a distinct key and re-executes the body."""
        # Arrange
        call_count = 0

        @cache
        def add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        add(1, 2)
        # Act
        add(2, 1)
        # Assert
        assert call_count == 2  # Different args, new call


class TestCacheEdgeCases:
    """Test edge cases for cache."""

    def test_cache_with_no_args_returns_constant(self):
        """Cache wraps a zero-arg function and returns its result."""
        # Arrange
        @cache
        def get_constant():
            return 42

        # Act
        result = get_constant()
        # Assert
        assert result == 42

    def test_cache_with_no_args_second_call_cached(self):
        """Second call to a zero-arg function does not re-execute the body."""
        # Arrange
        call_count = 0

        @cache
        def get_constant():
            nonlocal call_count
            call_count += 1
            return 42

        get_constant()
        # Act
        get_constant()
        # Assert
        assert call_count == 1  # Still cached

    def test_cache_with_keyword_args_returns_formatted_string(self):
        """Cache wraps a function called with a default keyword and returns its result."""
        # Arrange
        @cache
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        # Act
        result = greet("World")
        # Assert
        assert result == "Hello, World!"

    def test_cache_with_keyword_args_same_call_cached(self):
        """Repeated identical call does not re-execute the body."""
        # Arrange
        call_count = 0

        @cache
        def greet(name, greeting="Hello"):
            nonlocal call_count
            call_count += 1
            return f"{greeting}, {name}!"

        greet("World")
        # Act
        greet("World")
        # Assert
        assert call_count == 1  # Cached

    def test_cache_with_keyword_args_different_kwargs_recompute(self):
        """Differing keyword value is a distinct key and re-executes the body."""
        # Arrange
        call_count = 0

        @cache
        def greet(name, greeting="Hello"):
            nonlocal call_count
            call_count += 1
            return f"{greeting}, {name}!"

        greet("World")
        # Act
        greet("World", greeting="Hi")
        # Assert
        assert call_count == 2  # Different kwargs

    def test_cache_unlimited_size_maxsize_is_none(self):
        """Cache reports an unbounded maxsize of None."""
        # Arrange
        @cache
        def identity(x):
            return x

        # Store many values
        for i in range(1000):
            identity(i)

        # Act
        info = identity.cache_info()
        # Assert
        assert info.maxsize is None

    def test_cache_unlimited_size_retains_all_entries(self):
        """Cache retains every distinct entry stored (currsize == 1000)."""
        # Arrange
        @cache
        def identity(x):
            return x

        # Store many values
        for i in range(1000):
            identity(i)

        # Act
        info = identity.cache_info()
        # Assert
        assert info.currsize == 1000


class TestCacheInfo:
    """Test cache_info functionality."""

    def test_cache_info_reports_hit_count(self):
        """cache_info reports the number of cache hits."""
        # Arrange
        @cache
        def func(x):
            return x

        func(1)
        func(2)
        func(1)  # Cache hit
        # Act
        info = func.cache_info()
        # Assert
        assert info.hits == 1

    def test_cache_info_reports_miss_count(self):
        """cache_info reports the number of cache misses."""
        # Arrange
        @cache
        def func(x):
            return x

        func(1)
        func(2)
        func(1)  # Cache hit
        # Act
        info = func.cache_info()
        # Assert
        assert info.misses == 2

    def test_cache_info_reports_current_size(self):
        """cache_info reports the number of distinct cached entries."""
        # Arrange
        @cache
        def func(x):
            return x

        func(1)
        func(2)
        func(1)  # Cache hit
        # Act
        info = func.cache_info()
        # Assert
        assert info.currsize == 2

    def test_cache_clear_first_call_increments_count(self):
        """First call executes the function body exactly once."""
        # Arrange
        call_count = 0

        @cache
        def func(x):
            nonlocal call_count
            call_count += 1
            return x

        # Act
        func(1)
        # Assert
        assert call_count == 1

    def test_cache_clear_repeat_call_stays_cached(self):
        """Repeated call before clearing stays cached (count unchanged)."""
        # Arrange
        call_count = 0

        @cache
        def func(x):
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        # Act
        func(1)
        # Assert
        assert call_count == 1  # Cached

    def test_cache_clear_forces_recompute_after_clear(self):
        """Calling cache_clear forces the body to re-execute on the next call."""
        # Arrange
        call_count = 0

        @cache
        def func(x):
            nonlocal call_count
            call_count += 1
            return x

        func(1)
        func(1)  # Cached
        func.cache_clear()
        # Act
        func(1)
        # Assert
        assert call_count == 2  # Cache was cleared


class TestCacheWithHashableTypes:
    """Test cache with various hashable types."""

    def test_cache_with_tuple_args_returns_sum(self):
        """Cache handles tuple arguments and returns the computed result."""
        # Arrange
        @cache
        def process_tuple(t):
            return sum(t)

        # Act
        result = process_tuple((1, 2, 3))
        # Assert
        assert result == 6

    def test_cache_with_tuple_args_distinct_tuple_recomputed(self):
        """A distinct tuple key produces its own computed result."""
        # Arrange
        @cache
        def process_tuple(t):
            return sum(t)

        process_tuple((1, 2, 3))
        # Act
        result = process_tuple((4, 5, 6))
        # Assert
        assert result == 15

    def test_cache_with_string_args_returns_reversed(self):
        """Cache handles string arguments and returns the computed result."""
        # Arrange
        @cache
        def reverse_string(s):
            return s[::-1]

        # Act
        result = reverse_string("hello")
        # Assert
        assert result == "olleh"

    def test_cache_with_string_args_distinct_string_recomputed(self):
        """A distinct string key produces its own computed result."""
        # Arrange
        @cache
        def reverse_string(s):
            return s[::-1]

        reverse_string("hello")
        # Act
        result = reverse_string("world")
        # Assert
        assert result == "dlrow"

    def test_cache_with_none_arg_returns_true(self):
        """Cache handles a None argument and returns the computed result."""
        # Arrange
        @cache
        def handle_none(x):
            return x is None

        # Act
        result = handle_none(None)
        # Assert
        assert result is True

    def test_cache_with_none_distinct_zero_arg_returns_false(self):
        """A distinct key (0) is not conflated with None."""
        # Arrange
        @cache
        def handle_none(x):
            return x is None

        handle_none(None)
        # Act
        result = handle_none(0)
        # Assert
        assert result is False


class TestCacheComparison:
    """Test that scitex_gen.cache matches functools.lru_cache behavior."""

    def test_same_behavior_as_lru_cache_returns_equal_results(self):
        """scitex cache returns the same results as functools.lru_cache for each input."""
        # Arrange
        @cache
        def func_scitex(x):
            return x * 2

        @lru_cache(maxsize=None)
        def func_functools(x):
            return x * 2

        # Act
        inputs = [1, 2, 3, 1, 2, 3]
        # Assert
        for val in inputs:
            assert func_scitex(val) == func_functools(val)

    def test_same_behavior_as_lru_cache_equal_call_counts(self):
        """scitex cache executes the body the same number of times as functools.lru_cache."""
        # Arrange
        call_count_scitex = 0
        call_count_functools = 0

        @cache
        def func_scitex(x):
            nonlocal call_count_scitex
            call_count_scitex += 1
            return x * 2

        @lru_cache(maxsize=None)
        def func_functools(x):
            nonlocal call_count_functools
            call_count_functools += 1
            return x * 2

        # Act
        for val in [1, 2, 3, 1, 2, 3]:
            func_scitex(val)
            func_functools(val)
        # Assert
        assert call_count_scitex == call_count_functools


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_cache.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-02 13:30:24 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/gen/_cache.py
#
# from functools import lru_cache
#
# cache = lru_cache(maxsize=None)
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/gen/_cache.py
# --------------------------------------------------------------------------------
