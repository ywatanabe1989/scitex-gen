"""Filesystem and path-shaped helpers — symlink, title→path, src locator,
print_config.

Public re-exports keep `from scitex_gen import symlink, src` callers working.
"""

from __future__ import annotations

from ._print_config import print_config, print_config_main
from ._src import src
from ._symlink import symlink
from ._title2path import title2path

__all__ = [
    "print_config",
    "print_config_main",
    "src",
    "symlink",
    "title2path",
]
