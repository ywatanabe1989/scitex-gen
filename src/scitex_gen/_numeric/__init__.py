"""Numeric helpers — rounding/snapping, normalization, transpose, symlog.

Public re-exports keep the historical `from scitex_gen import to_even`
surface stable. Direct submodule imports are also supported, e.g.
`from scitex_gen._numeric._norm import to_z`.
"""

from __future__ import annotations

from ._to_even import to_even
from ._to_odd import to_odd
from ._transpose import transpose
from ._symlog import symlog

# Optional: _to_rank requires torch (pulled in via the [torch] extra).
try:
    from ._to_rank import to_rank
except ImportError:
    to_rank = None

# Optional: _norm requires torch.
try:
    from ._norm import clip_perc, to_01, to_nan01, to_nanz, to_z, unbias
except ImportError:
    clip_perc = None
    to_01 = None
    to_nan01 = None
    to_nanz = None
    to_z = None
    unbias = None

__all__ = [
    "clip_perc",
    "symlog",
    "to_01",
    "to_even",
    "to_nan01",
    "to_nanz",
    "to_odd",
    "to_rank",
    "to_z",
    "transpose",
    "unbias",
]
