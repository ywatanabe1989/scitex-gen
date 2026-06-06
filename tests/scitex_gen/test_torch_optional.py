#!/usr/bin/env python3
"""Tests for the optional-torch infrastructure across ``_type``,
``_var_info``, and ``_numeric._norm``.

Background
----------
Before 2026-06-07, ``scitex-gen`` declared ``torch`` as a bare core
dependency on line 28 of pyproject.toml. Bare ``pip install scitex-gen``
transitively pulled torch + nvidia-cuda + cublas + cuda-toolkit + triton
+ sympy + networkx (~4GB) — surfaced by proj-paper-ripple-wm's 4.4GB SIF.
The fix removed torch from core deps and made every torch use point
optional, so functions that actually need torch raise a clear
``ImportError`` naming the ``scitex-gen[torch]`` extra.

These tests pin:
* The optional-torch wiring exists in each touched module
  (``_TORCH_AVAILABLE`` flag, ``_ARRAY_LIKE_ALL_TYPES`` tuple, the
  ``_require_torch`` helper).
* When torch IS installed the legacy behaviour is preserved (the
  ``ArrayLike`` union and the ``isinstance`` tuple include
  ``torch.Tensor``, the ``_norm`` functions still work).
* When torch is sabotaged out of ``sys.modules`` and the modules are
  freshly imported, the optional path raises clear ``ImportError`` with
  ``scitex-gen[torch]`` in the message.

The subprocess-based "no-torch" simulation uses ``sys.modules['torch']
= None`` plus ``importlib.reload`` so we exercise the production
``except ImportError`` branches even in an environment where torch
itself IS installed (CI typically has it).
"""

import importlib
import subprocess
import sys
import textwrap

import pytest


# =============================================================================
# Structure: the optional-torch infrastructure is wired up in source.
# =============================================================================
class TestOptionalTorchWiringExists:
    """The modules expose the optional-torch sentinel symbols."""

    def test_type_module_has_torch_available_flag(self):
        # Arrange
        from scitex_gen import _type
        # Act / Assert
        assert hasattr(_type, "_TORCH_AVAILABLE")

    def test_type_module_has_array_like_all_types_tuple(self):
        # Arrange
        from scitex_gen import _type
        # Act / Assert
        assert hasattr(_type, "_ARRAY_LIKE_ALL_TYPES")

    def test_var_info_module_has_torch_available_flag(self):
        # Arrange
        from scitex_gen import _var_info
        # Act / Assert
        assert hasattr(_var_info, "_TORCH_AVAILABLE")

    def test_var_info_module_has_array_like_all_types_tuple(self):
        # Arrange
        from scitex_gen import _var_info
        # Act / Assert
        assert hasattr(_var_info, "_ARRAY_LIKE_ALL_TYPES")

    def test_norm_module_has_require_torch_helper(self):
        # Arrange
        from scitex_gen._numeric import _norm
        # Act / Assert
        assert callable(getattr(_norm, "_require_torch", None))

    def test_norm_module_has_torch_available_flag(self):
        # Arrange
        from scitex_gen._numeric import _norm
        # Act / Assert
        assert hasattr(_norm, "_TORCH_AVAILABLE")


# =============================================================================
# Torch present: legacy behaviour preserved.
# =============================================================================
class TestTorchPresentPreservesLegacyBehaviour:
    """When torch IS installed (the typical CI environment), the
    optional wiring keeps the original union + isinstance tuple."""

    def test_type_array_like_includes_torch_tensor_when_present(self):
        # Arrange
        torch = pytest.importorskip("torch")
        from scitex_gen._type import _ARRAY_LIKE_ALL_TYPES
        # Act / Assert
        assert torch.Tensor in _ARRAY_LIKE_ALL_TYPES

    def test_var_info_array_like_includes_torch_tensor_when_present(self):
        # Arrange
        torch = pytest.importorskip("torch")
        from scitex_gen._var_info import _ARRAY_LIKE_ALL_TYPES
        # Act / Assert
        assert torch.Tensor in _ARRAY_LIKE_ALL_TYPES

    def test_norm_require_torch_noop_when_torch_available(self):
        # Arrange
        pytest.importorskip("torch")
        from scitex_gen._numeric._norm import _require_torch
        # Act — should NOT raise when torch is installed.
        _require_torch()
        # Assert — reaching here proves no exception raised.
        assert True


# =============================================================================
# Torch absent: the gates raise clear, actionable ImportError.
# =============================================================================
class TestTorchAbsentRaisesClearError:
    """Subprocess-based simulation: sabotage ``sys.modules['torch']`` to
    force the optional ``except ImportError`` branches in the production
    code, then assert each gated function raises with the right hint."""

    _SUBPROCESS_PROGRAM = textwrap.dedent(
        """
        import importlib, sys
        # Sabotage torch import BEFORE scitex_gen modules are imported.
        sys.modules["torch"] = None
        # Reload any pre-imported _norm so the optional branch executes.
        for mod in list(sys.modules):
            if mod.startswith("scitex_gen._numeric._norm") or mod.startswith(
                "scitex_gen._type"
            ) or mod.startswith("scitex_gen._var_info"):
                del sys.modules[mod]
        from scitex_gen._numeric import _norm
        try:
            _norm._require_torch()
        except ImportError as exc:
            print("RAISED:", exc)
        else:
            print("DID_NOT_RAISE")
        """
    )

    def test_require_torch_subprocess_raises_with_extra_hint(self):
        # Arrange / Act
        result = subprocess.run(
            [sys.executable, "-c", self._SUBPROCESS_PROGRAM],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Assert — combined stdout must show the ImportError naming the
        # ``scitex-gen[torch]`` extra. Exit code is incidental; we read
        # the print-line.
        out = result.stdout
        assert "RAISED:" in out
        assert "scitex-gen[torch]" in out

    def test_require_torch_subprocess_hint_uses_sys_executable(self):
        # Arrange / Act
        result = subprocess.run(
            [sys.executable, "-c", self._SUBPROCESS_PROGRAM],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Assert — venv-correct: ``{sys.executable} -m pip install`` so
        # users with an active venv don't hit a system pip.
        out = result.stdout
        assert "-m pip install" in out
