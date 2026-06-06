#!/usr/bin/env python3
"""scitex-gen — retired.

This package used to host a grab-bag of "general-purpose" utilities. As of
v0.1.13 the surface has been fully redistributed across the SciTeX
ecosystem and `scitex_gen` no longer exports any public symbols. The
package is kept as an installable shell so that consumers can `pip
install scitex-gen` without errors while their import paths migrate.

Where each former symbol now lives
----------------------------------
| Old import path                          | New canonical home          |
|------------------------------------------|-----------------------------|
| symlog, to_rank, transpose               | scitex_math                 |
| to_z, to_01, to_nan01, to_nanz,          | scitex_math                 |
| unbias, clip_perc, connect_nums,         | scitex_math                 |
| float_linspace, to_even, to_odd          | scitex_math                 |
| DimHandler                               | scitex_nn                   |
| mat2dict, mat2npa, mat2npy, dir2npy,     | scitex_io                   |
| save_npa, keys2npa, public_keys          | scitex_io                   |
| xml2dict, XmlDictConfig, XmlListConfig   | scitex_io                   |
| print_config, print_config_main          | scitex_io                   |
| list_packages, main, src                 | scitex_introspect           |
| title2path                               | scitex_path                 |
| var_info                                 | scitex_types                |
| is_ipython, is_script                    | scitex_context              |
| embed, less, paste                       | scitex_context              |
| cache, wrap, alternate_kwarg             | scitex_decorators           |

Removed entirely (no replacement; were already deprecated):
- start, close, running2finished, TimeStamper

Removed entirely (the destination peer already had a better implementation;
no semantic-preserving port was warranted):
- symlink           (use scitex_path.symlink — richer API)
- ArrayLike         (use scitex_types.ArrayLike — lazy optional-dep aware)
- wrap              (use scitex_decorators.wrap — preserves _original_func)

Re-exports dropped (use the canonical home directly):
- ci, check_host, is_host, verify_host, list_api, run_shellcommand,
  run_shellscript, title_case, detect_environment, is_notebook,
  get_notebook_path, get_notebook_name, get_notebook_directory,
  get_output_directory.

See ADR-0001 for the architecture decision; the CHANGELOG entry for
v0.1.13 carries the per-symbol migration mapping.
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
# Retired surface. The dispatch tables stay (empty) so the PEP 562 lazy
# loader continues to exist as a stable shape for any future re-export.
# ---------------------------------------------------------------------------
_LAZY_ATTRS: dict[str, str] = {}
_OPTIONAL_ATTRS: dict[str, tuple[str, str]] = {}


def __getattr__(name: str):
    """All former public symbols are retired — raise ImportError on access."""
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r} — scitex_gen has been "
        "retired (v0.1.13). See the package docstring or the CHANGELOG for "
        "the new canonical import path."
    )


def __dir__() -> list[str]:
    return sorted(globals())


__all__ = [
    "__version__",
]
