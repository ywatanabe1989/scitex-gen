#!/usr/bin/env python3
# Timestamp: 2026-04-29
# File: src/scitex_gen/_title_case.py

"""Title case conversion — bundled in scitex_gen for version stability.

Delegates to the `titlecase` PyPI package (NYT Manual of Style) with a
small callback for technical acronyms common in the SciTeX domain.
Bundling here avoids a dependency on a specific scitex_str release.
"""

from __future__ import annotations

import re

from titlecase import set_small_word_list
from titlecase import titlecase as _titlecase

set_small_word_list(
    "a|an|and|as|at|but|by|en|for|if|in|nor|of|on|or|the|to|v\\.?|via|vs\\.?|with"
)

_ACRONYMS: set[str] = {
    "api",
    "url",
    "uri",
    "cpu",
    "gpu",
    "tpu",
    "ram",
    "io",
    "os",
    "vm",
    "ip",
    "tcp",
    "udp",
    "http",
    "https",
    "html",
    "css",
    "js",
    "ts",
    "json",
    "xml",
    "yaml",
    "csv",
    "pdf",
    "sql",
    "db",
    "rest",
    "ci",
    "cd",
    "cli",
    "ide",
    "ssh",
    "ssl",
    "tls",
    "id",
    "uuid",
    "ai",
    "ml",
    "dl",
    "nlp",
    "lstm",
    "gru",
    "cnn",
    "rnn",
    "gan",
    "vae",
    "eeg",
    "ecg",
    "mri",
    "fmri",
    "ct",
    "pet",
    "iqr",
    "rms",
    "snr",
    "fbi",
    "cia",
    "nsa",
    "nasa",
    "aws",
}

_NUMBERED_ACRONYM = re.compile(r"^\d+[a-zA-Z]{1,3}$")


def _acronym_callback(word: str, **_kwargs) -> str | None:
    lower = word.lower()
    if lower in _ACRONYMS:
        return lower.upper()
    if lower.endswith("s") and lower[:-1] in _ACRONYMS:
        return lower[:-1].upper() + "s"
    if _NUMBERED_ACRONYM.match(word):
        return word.upper()
    return None


def title_case(text: str) -> str:
    """Convert a string to title case (NYT Manual of Style + SciTeX acronyms).

    Examples
    --------
    >>> title_case("the cat and the dog")
    'The Cat and the Dog'
    >>> title_case("welcome to the world of ai and using CPUs for gaming")
    'Welcome to the World of AI and Using CPUs for Gaming'
    """
    return _titlecase(text, callback=_acronym_callback)


# EOF
