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

    def test_cache_memoizes_function(self):
        """Test that cache properly memoizes function calls."""
        # Arrange
        call_count = 0

        @cache
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call computes
        # Act
        result1 = expensive_function(5)
        # Assert
        assert result1 == 10
        assert call_count == 1

        # Second call with same arg returns cached value
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # No additional call

        # Different arg computes again
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2

    def test_cache_with_multiple_arguments(self):
        """Test cache with functions that take multiple arguments."""
        # Arrange
        call_count = 0

        @cache
        # Act
        def add(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        # Assert
        assert add(1, 2) == 3
        assert call_count == 1

        assert add(1, 2) == 3
        assert call_count == 1  # Cached

        assert add(2, 1) == 3
        assert call_count == 2  # Different args, new call


class TestCacheEdgeCases:
    """Test edge cases for cache."""

    def test_cache_with_no_args(self):
        """Test cache with functions that take no arguments."""
        # Arrange
        call_count = 0

        @cache
        # Act
        def get_constant():
            nonlocal call_count
            call_count += 1
            return 42

        # Assert
        assert get_constant() == 42
        assert call_count == 1

        assert get_constant() == 42
        assert call_count == 1  # Still cached

    def test_cache_with_keyword_args(self):
        """Test cache with keyword arguments."""
        # Arrange
        call_count = 0

        @cache
        # Act
        def greet(name, greeting="Hello"):
            nonlocal call_count
            call_count += 1
            return f"{greeting}, {name}!"

        # Assert
        assert greet("World") == "Hello, World!"
        assert call_count == 1

        assert greet("World") == "Hello, World!"
        assert call_count == 1  # Cached

        assert greet("World", greeting="Hi") == "Hi, World!"
        assert call_count == 2  # Different kwargs

    def test_cache_unlimited_size(self):
        """Test that cache has unlimited size (maxsize=None)."""

        @cache
        # Arrange
        def identity(x):
            return x

        # Store many values
        for i in range(1000):
            identity(i)

        # All should still be cached
        # Act
        info = identity.cache_info()
        # Assert
        assert info.maxsize is None
        assert info.currsize == 1000


class TestCacheInfo:
    """Test cache_info functionality."""

    def test_cache_info_available(self):
        """Test that cache_info is available on decorated functions."""

        @cache
        # Arrange
        def func(x):
            return x

        func(1)
        func(2)
        func(1)  # Cache hit

        # Act
        info = func.cache_info()
        # Assert
        assert info.hits == 1
        assert info.misses == 2
        assert info.currsize == 2

    def test_cache_clear_call_count_equals_n_1(self):
        """Test that cache_clear works correctly."""
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

        func(1)
        assert call_count == 1  # Cached

        func.cache_clear()

        func(1)
        assert call_count == 2  # Cache was cleared


class TestCacheWithHashableTypes:
    """Test cache with various hashable types."""

    def test_cache_with_tuple_args(self):
        """Test cache with tuple arguments."""

        @cache
        # Arrange
        # Act
        def process_tuple(t):
            return sum(t)

        # Assert
        assert process_tuple((1, 2, 3)) == 6
        assert process_tuple((1, 2, 3)) == 6  # Cached
        assert process_tuple((4, 5, 6)) == 15

    def test_cache_with_string_args(self):
        """Test cache with string arguments."""

        @cache
        # Arrange
        # Act
        def reverse_string(s):
            return s[::-1]

        # Assert
        assert reverse_string("hello") == "olleh"
        assert reverse_string("hello") == "olleh"  # Cached
        assert reverse_string("world") == "dlrow"

    def test_cache_with_none(self):
        """Test cache with None argument."""

        @cache
        # Arrange
        # Act
        def handle_none(x):
            return x is None

        # Assert
        assert handle_none(None) is True
        assert handle_none(None) is True  # Cached
        assert handle_none(0) is False


class TestCacheComparison:
    """Test that scitex_gen.cache matches functools.lru_cache behavior."""

    def test_same_behavior_as_lru_cache(self):
        """Verify cache behaves like lru_cache(maxsize=None)."""
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

        # Same inputs
        # Act
        for val in [1, 2, 3, 1, 2, 3]:
            result1 = func_scitex(val)
            result2 = func_functools(val)
            assert result1 == result2

        # Same call counts
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
