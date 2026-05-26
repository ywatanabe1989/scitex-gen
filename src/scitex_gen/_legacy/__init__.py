"""Legacy / deprecated helpers retained for backward compatibility.

`close`/`running2finished` and `start` are shim wrappers that emit
`DeprecationWarning` and redirect to `scitex_session`. `TimeStamper` is
the standalone time-stamping helper kept here pending its eventual home
in a dedicated `scitex-time` package.

Re-exported from the top-level `scitex_gen.__init__` so existing
imports like `from scitex_gen import close, TimeStamper` keep working.
"""

from __future__ import annotations

from ._deprecated_close import close, running2finished
from ._deprecated_start import start
from ._TimeStamper import TimeStamper

__all__ = [
    "TimeStamper",
    "close",
    "running2finished",
    "start",
]
