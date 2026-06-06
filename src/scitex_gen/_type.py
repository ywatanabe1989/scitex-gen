#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-17 12:45:50 (ywatanabe)"
# File: ./scitex_repo/src/scitex/gen/_type.py

THIS_FILE = "/home/ywatanabe/proj/scitex_repo/src/scitex/gen/_type.py"

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-03 10:33:13 (ywatanabe)"
# File: placeholder.py

from typing import Any, Union

import numpy as np
import pandas as pd
import xarray as xr

# torch is OPTIONAL — bare ``pip install scitex-gen`` does NOT carry it
# (would otherwise pull ~4GB of nvidia-cuda + cublas + triton + sympy +
# networkx as transitive deps just for the ``torch.Tensor`` isinstance
# check below). Install ``scitex-gen[torch]`` to enable the torch path.
try:
    import torch as _torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is absent
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# Build the array-like type universe with or without torch.Tensor at the
# end; everything else is core-dep so it's always present. Keeping the
# list-vs-tuple/np/pd/xr prefix stable means downstream isinstance checks
# get a single source of truth.
_ARRAY_LIKE_BASE_TYPES = (
    np.ndarray,
    pd.Series,
    pd.DataFrame,
    xr.DataArray,
)
_ARRAY_LIKE_ALL_TYPES = (
    _ARRAY_LIKE_BASE_TYPES + ((_torch.Tensor,) if _TORCH_AVAILABLE else ())
)

if _TORCH_AVAILABLE:
    ArrayLike = Union[
        list,
        tuple,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        xr.DataArray,
        _torch.Tensor,
    ]
else:
    ArrayLike = Union[
        list,
        tuple,
        np.ndarray,
        pd.Series,
        pd.DataFrame,
        xr.DataArray,
    ]


def var_info(variable: Any) -> dict:
    """Returns type and structural information about a variable.

    Example
    -------
    >>> data = np.array([[1, 2], [3, 4]])
    >>> info = var_info(data)
    >>> print(info)
    {
        'type': 'numpy.ndarray',
        'length': 2,
        'shape': (2, 2),
        'dimensions': 2
    }

    Parameters
    ----------
    variable : Any
        Variable to inspect.

    Returns
    -------
    dict
        Dictionary containing variable information.
    """
    info = {"type": type(variable).__name__}

    # Length check
    if hasattr(variable, "__len__"):
        info["length"] = len(variable)

    # Shape check for array-like objects. ``_ARRAY_LIKE_ALL_TYPES``
    # includes ``torch.Tensor`` when the optional torch extra is
    # installed, and omits it otherwise — same semantics on a bare
    # install, just without the torch slice.
    if isinstance(variable, _ARRAY_LIKE_ALL_TYPES):
        info["shape"] = variable.shape
        info["dimensions"] = len(variable.shape)

    # Special handling for nested lists
    elif isinstance(variable, list):
        if variable and isinstance(variable[0], list):
            depth = 1
            current = variable
            shape = [len(variable)]
            while current and isinstance(current[0], list):
                shape.append(len(current[0]))
                current = current[0]
                depth += 1
            info["shape"] = tuple(shape)
            info["dimensions"] = depth

    return info


# EOF
