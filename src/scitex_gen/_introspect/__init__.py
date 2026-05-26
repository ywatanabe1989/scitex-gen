"""Introspection / shape-handling / metadata-extraction helpers.

Covers DimHandler (tensor-dim plumbing), list_packages, xml↔dict and
mat→npy converters, and small lookup utilities (cache, alternate_kwarg).
Public re-exports keep `from scitex_gen import cache` callers stable.
"""

from __future__ import annotations

from ._alternate_kwarg import alternate_kwarg
from ._cache import cache
from ._list_packages import list_packages, main
from ._mat2py import (
    dir2npy,
    keys2npa,
    mat2dict,
    mat2npa,
    mat2npy,
    public_keys,
    save_npa,
)
from ._xml2dict import XmlDictConfig, XmlListConfig, xml2dict

# Optional: DimHandler requires torch.
try:
    from ._DimHandler import DimHandler
except ImportError:
    DimHandler = None

__all__ = [
    "DimHandler",
    "XmlDictConfig",
    "XmlListConfig",
    "alternate_kwarg",
    "cache",
    "dir2npy",
    "keys2npa",
    "list_packages",
    "main",
    "mat2dict",
    "mat2npa",
    "mat2npy",
    "public_keys",
    "save_npa",
    "xml2dict",
]
