"""IPython / interactive-session helpers — embed, paste, less, etc.

Public re-exports preserve `from scitex_gen import is_ipython` callers.
Notebook path detection lives here because it's REPL/Jupyter-shaped.
"""

from __future__ import annotations

from ._is_ipython import is_ipython, is_script
from ._less import less
from ._paste import paste

# Optional: _embed requires torch.
try:
    from ._embed import embed
except ImportError:
    embed = None

__all__ = [
    "embed",
    "is_ipython",
    "is_script",
    "less",
    "paste",
]
