#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-03 02:11:18 (ywatanabe)"
# File: ./scitex_repo/src/scitex/gen/_less.py
#!./env/bin/python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-04-21 12:05:35"
# Author: Yusuke Watanabe (ywatanabe@scitex.ai)

"""
This script does XYZ.
"""

# Imports


# Functions
def less(output, *, get_ipython=None):
    """
    Print the given output using `less` in an IPython or IPdb session.

    Parameters
    ----------
    output : str
        Text to display through ``less``.
    get_ipython : callable, optional
        Zero-argument callable returning the active IPython shell. Defaults to
        ``IPython.get_ipython``; injectable so callers (and tests) can supply a
        real shell substitute without patching.
    """
    import os
    import tempfile

    if get_ipython is None:
        from IPython import get_ipython as _default_get_ipython

        get_ipython = _default_get_ipython

    # Create a temporary file to hold the output
    with tempfile.NamedTemporaryFile(delete=False, mode="w+t") as tmpfile:
        # Write the output to the temporary file
        tmpfile.write(output)
        tmpfile_name = tmpfile.name

    # Use IPython's system command access to pipe the content of the temporary file to `less`
    get_ipython().system(f"less {tmpfile_name}")

    # Clean up the temporary file
    os.remove(tmpfile_name)


# EOF
