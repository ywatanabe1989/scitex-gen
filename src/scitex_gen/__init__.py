#!/usr/bin/env python3
"""Scitex gen module — general-purpose utilities.

NOTE: This module is being refactored. Many functions are being moved to
more appropriate locations. For backward compatibility, they are re-exported
here.

Recommended imports:
- ci -> scitex.stats.descriptive.ci
- check_host, is_host, verify_host -> scitex.os
- detect_environment, is_notebook, is_script -> scitex.context
- list_api -> scitex.introspect
- run_shellcommand, run_shellscript -> scitex.sh
- xml2dict, XmlDictConfig, XmlListConfig -> scitex.io
- title_case -> scitex.str
- symlink -> scitex.path

Top-level imports are PEP 562 lazy — `import scitex_gen` is cheap and pulls
in neither torch nor scitex_stats. Public symbols load on first attribute
access.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        __version__ = _pkg_version("scitex-gen")
    except PackageNotFoundError:  # pragma: no cover - editable install w/o metadata
        __version__ = "0.0.0+local"
    del _pkg_version, PackageNotFoundError
except ImportError:  # pragma: no cover - Python <3.8 fallback
    __version__ = "0.0.0+local"


# ---------------------------------------------------------------------------
# PEP 562 lazy attribute map: public-name → module that defines it.
# A leading "." denotes a relative submodule; otherwise it is an ecosystem
# peer package. The public name equals the attribute name in that module.
# ---------------------------------------------------------------------------
_LAZY_ATTRS: dict[str, str] = {
    # --- Ecosystem peers (re-exported for backward compatibility) -----------
    "ci": "scitex_stats.descriptive",
    "check_host": "scitex_os",
    "is_host": "scitex_os",
    "verify_host": "scitex_os",
    "list_api": "scitex_introspect",
    "run_shellcommand": "scitex_sh",
    "run_shellscript": "scitex_sh",
    "title_case": "scitex_str",
    "detect_environment": "scitex_context",
    "get_notebook_directory": "scitex_context",
    "get_notebook_name": "scitex_context",
    "get_notebook_path": "scitex_context",
    "get_output_directory": "scitex_context",
    "is_notebook": "scitex_context",
    # --- Local submodules ---------------------------------------------------
    "alternate_kwarg": "._introspect._alternate_kwarg",
    "cache": "._introspect._cache",
    "list_packages": "._introspect._list_packages",
    "main": "._introspect._list_packages",
    "dir2npy": "._introspect._mat2py",
    "keys2npa": "._introspect._mat2py",
    "mat2dict": "._introspect._mat2py",
    "mat2npa": "._introspect._mat2py",
    "mat2npy": "._introspect._mat2py",
    "public_keys": "._introspect._mat2py",
    "save_npa": "._introspect._mat2py",
    "xml2dict": "._introspect._xml2dict",
    "XmlDictConfig": "._introspect._xml2dict",
    "XmlListConfig": "._introspect._xml2dict",
    "is_ipython": "._ipython._is_ipython",
    "is_script": "._ipython._is_ipython",
    "less": "._ipython._less",
    "paste": "._ipython._paste",
    "print_config": "._fs._print_config",
    "print_config_main": "._fs._print_config",
    "src": "._fs._src",
    "symlink": "._fs._symlink",
    "title2path": "._fs._title2path",
    "TimeStamper": "._legacy._TimeStamper",
    "start": "._legacy._deprecated_start",
    "close": "._legacy._deprecated_close",
    "running2finished": "._legacy._deprecated_close",
    "symlog": "._numeric._symlog",
    "transpose": "._numeric._transpose",
    "wrap": "._wrap",
    "connect_nums": ".misc",
    "float_linspace": ".misc",
}

# Optional public names whose backing module needs heavy/optional deps
# (torch, xarray); they resolve to ``None`` when those deps are absent,
# preserving the historical try/except behaviour.
# name: (module, attr_name_in_module)
_OPTIONAL_ATTRS: dict[str, tuple[str, str]] = {
    "DimHandler": ("._introspect._DimHandler", "DimHandler"),
    "embed": ("._ipython._embed", "embed"),
    "clip_perc": ("._numeric._norm", "clip_perc"),
    "to_01": ("._numeric._norm", "to_01"),
    "to_nan01": ("._numeric._norm", "to_nan01"),
    "to_nanz": ("._numeric._norm", "to_nanz"),
    "to_z": ("._numeric._norm", "to_z"),
    "unbias": ("._numeric._norm", "unbias"),
    "to_rank": ("._numeric._to_rank", "to_rank"),
    # _var_info is the canonical home; _type is the historical fallback.
    "ArrayLike": ("._var_info", "ArrayLike"),
    "var_info": ("._var_info", "var_info"),
}


def _load_lazy_attr(name: str):
    """Resolve a `_LAZY_ATTRS` name and cache it."""
    from importlib import import_module

    mod_name = _LAZY_ATTRS.get(name)
    if mod_name is None:
        return None
    package = __name__ if mod_name.startswith(".") else None
    mod = import_module(mod_name, package)
    attr = getattr(mod, name)
    globals()[name] = attr
    return attr


def _load_optional_attr(name: str):
    """Resolve an `_OPTIONAL_ATTRS` name and cache it (None on failure)."""
    from importlib import import_module

    spec = _OPTIONAL_ATTRS.get(name)
    if spec is None:
        return None
    mod_name, attr_name = spec
    try:
        mod = import_module(mod_name, __name__)
        attr = getattr(mod, attr_name, None)
    except ImportError:
        # _var_info needs torch/xarray; fall back to the lighter _type module.
        if name in {"ArrayLike", "var_info"}:
            try:
                mod = import_module("._type", __name__)
                attr = getattr(mod, name, None)
            except ImportError:
                attr = None
        else:
            attr = None
    globals()[name] = attr
    return attr


def __getattr__(name: str):
    """PEP 562 lazy-loader: import on first access, cache, return."""
    # Reference the dispatch tables directly here (via ``.get(name)``) rather
    # than delegating the lookup to a helper: the PA-102 auditor recognizes the
    # PEP 562 pattern only when ``_LAZY_ATTRS``/``_OPTIONAL_ATTRS`` are
    # subscripted/`.get`-ed inside ``__getattr__`` itself, and uses that to pull
    # the dynamically-exposed ``__all__`` names from the tables.
    if _LAZY_ATTRS.get(name) is not None:
        return _load_lazy_attr(name)
    if _OPTIONAL_ATTRS.get(name) is not None:
        return _load_optional_attr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(_LAZY_ATTRS) | set(_OPTIONAL_ATTRS) | set(globals()))


__all__ = [
    "__version__",
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
    "to_nan01",
    "to_nanz",
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
