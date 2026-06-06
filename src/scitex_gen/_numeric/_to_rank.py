#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-02 13:05:47 (ywatanabe)"
# File: ./scitex_repo/src/scitex/gen/_to_rank.py
#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-08-29 22:10:06 (ywatanabe)"
# ./src/scitex/gen/data_processing/_to_rank.py

import sys

# torch is OPTIONAL — see scitex_gen pyproject.toml note. ``to_rank``
# is torch-native; calling it without the [torch] extra raises a clear
# ImportError naming the right install command.
try:
    import torch  # type: ignore[import-not-found]

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# from .._converters import
from scitex_decorators import torch_fn


def _require_torch() -> None:
    """Raise a clear ``ImportError`` naming ``scitex-gen[torch]`` when
    torch is not installed. Called by every torch-using function in
    this module before it touches ``torch.*``."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "scitex_gen._numeric._to_rank requires torch. Install "
            f"with: {sys.executable} -m pip install "
            "'scitex-gen[torch]'"
        )


@torch_fn
def to_rank(tensor, method="average"):
    _require_torch()
    sorted_tensor, indices = torch.sort(tensor)
    ranks = torch.empty_like(tensor)
    ranks[indices] = (
        torch.arange(len(tensor), dtype=tensor.dtype, device=tensor.device) + 1
    )

    if method == "average":
        ranks = ranks.float()
        ties = torch.nonzero(sorted_tensor[1:] == sorted_tensor[:-1])
        for i in range(len(ties)):
            start = ties[i]
            end = start + 1
            while (
                end < len(sorted_tensor) and sorted_tensor[end] == sorted_tensor[start]
            ):
                end += 1
            ranks[indices[start:end]] = ranks[indices[start:end]].mean()

    return ranks


# EOF
