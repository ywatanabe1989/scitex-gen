#!/usr/bin/env python3
"""Scitex gen module.

NOTE: This module is being refactored. Many functions are being moved to
more appropriate locations. For backward compatibility, they are re-exported
here with deprecation warnings.

Recommended imports:
- ci -> scitex.stats.descriptive.ci
- check_host, is_host, verify_host -> scitex.os
- detect_environment, is_notebook, is_script -> scitex.context
- list_api -> scitex.introspect
- run_shellcommand, run_shellscript -> scitex.sh
- xml2dict, XmlDictConfig, XmlListConfig -> scitex.io
- title_case -> scitex.str
- symlink -> scitex.path
"""

from __future__ import annotations

import warnings


def _deprecation_warning(old_path, new_path):
    warnings.warn(
        f"{old_path} is deprecated, use {new_path} instead",
        DeprecationWarning,
        stacklevel=3,
    )


# ci -> scitex_stats.descriptive.ci (with re-export for backward compat)
from scitex_stats.descriptive import ci

# Optional: DimHandler requires torch
try:
    from ._introspect._DimHandler import DimHandler
except ImportError:
    DimHandler = None
# check_host moved to scitex.os (re-export for backward compatibility)
from scitex_os import check_host, is_host, verify_host

from ._introspect._alternate_kwarg import alternate_kwarg
from ._introspect._cache import cache
from ._legacy._deprecated_close import close as _deprecated_close
from ._legacy._deprecated_close import running2finished as _deprecated_running2finished

# _start.py moved to old/ directory - functionality now in scitex.session
# BACKWARD COMPATIBILITY: Import deprecated wrappers
from ._legacy._deprecated_start import start as _deprecated_start

# _close.py moved to old/ directory - functionality now in scitex.session
# Optional: _embed requires torch
try:
    from ._ipython._embed import embed
except ImportError:
    embed = None
# list_api moved to scitex.introspect (re-export for backward compatibility)
from scitex_introspect import list_api

from ._ipython._is_ipython import is_ipython, is_script
from ._ipython._less import less
from ._introspect._list_packages import list_packages, main
from ._introspect._mat2py import (
    dir2npy,
    keys2npa,
    mat2dict,
    mat2npa,
    mat2npy,
    public_keys,
    save_npa,
)

# Optional: _norm requires torch
try:
    from ._numeric._norm import clip_perc, to_01, to_nan01, to_nanz, to_z, unbias
except ImportError:
    clip_perc = None
    to_01 = None
    to_nan01 = None
    to_nanz = None
    to_z = None
    unbias = None
# shell functions moved to scitex.sh (re-export for backward compatibility)
from scitex_sh import run_shellcommand, run_shellscript

from ._ipython._paste import paste
from ._fs._print_config import print_config, print_config_main
from ._fs._src import src
from ._legacy._TimeStamper import TimeStamper

# Override the imported functions with deprecated wrappers
start = _deprecated_start
close = _deprecated_close
running2finished = _deprecated_running2finished

# environment detection moved to scitex.context (re-export for backward compatibility)
from scitex_context import (
    detect_environment,
    get_notebook_directory,
    get_notebook_info_simple,
    get_notebook_name,
    get_notebook_path,
    get_output_directory,
    is_notebook,
)

# title_case moved to scitex.str (re-export for backward compatibility)
from scitex_str import title_case

from ._fs._symlink import symlink
from ._numeric._symlog import symlog
from ._fs._title2path import title2path
from ._numeric._to_even import to_even
from ._numeric._to_odd import to_odd

# Optional: _to_rank requires torch
try:
    from ._numeric._to_rank import to_rank
except ImportError:
    to_rank = None
from ._numeric._transpose import transpose

# Optional: _type and _var_info require xarray (declared via the [torch] extra).
from scitex_dev import try_import_optional

ArrayLike = try_import_optional(
    "._type", "ArrayLike", extra="torch", pkg="scitex-gen", package=__name__
)
var_info = try_import_optional(
    "._type", "var_info", extra="torch", pkg="scitex-gen", package=__name__
)

# _var_info is the canonical home; if importable, it overrides the _type fallback.
_var_info_module = try_import_optional(
    "._var_info", extra="torch", pkg="scitex-gen", package=__name__
)
if _var_info_module is not None:
    ArrayLike = _var_info_module.ArrayLike
    var_info = _var_info_module.var_info
from ._wrap import wrap
from ._introspect._xml2dict import XmlDictConfig, XmlListConfig, xml2dict

# Import from misc module
from .misc import connect_nums, float_linspace

__all__ = [
    "ArrayLike",
    "ArrayLike",
    "DimHandler",
    "TimeStamper",
    "XmlDictConfig",
    "XmlListConfig",
    "alternate_kwarg",
    "cache",
    "check_host",
    "ci",
    "clip_perc",
    "close",
    "connect_nums",
    "dir2npy",
    "embed",
    "float_linspace",
    "list_api",
    "is_host",
    "is_ipython",
    "is_script",
    "keys2npa",
    "less",
    "list_packages",
    "mat2dict",
    "mat2npa",
    "mat2npy",
    "paste",
    "print_config",
    "print_config_main",
    "public_keys",
    "run_shellcommand",
    "run_shellscript",
    "running2finished",
    "save_npa",
    "src",
    "start",
    "symlink",
    "symlog",
    "title2path",
    "title_case",
    "to_01",
    "to_even",
    "to_nan01",
    "to_nanz",
    "to_odd",
    "to_rank",
    "to_z",
    "transpose",
    "unbias",
    "var_info",
    "var_info",
    "verify_host",
    "wrap",
    "xml2dict",
    "detect_environment",
    "get_output_directory",
    "is_notebook",
    "get_notebook_path",
    "get_notebook_name",
    "get_notebook_directory",
]
