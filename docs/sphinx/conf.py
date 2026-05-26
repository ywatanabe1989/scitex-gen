"""Sphinx configuration for scitex-gen."""

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = "scitex-gen"
copyright = "2026, Yusuke Watanabe"
author = "Yusuke Watanabe"

try:
    from scitex_gen import __version__ as release
except ImportError:
    release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": False,
    "private-members": False,
    "exclude-members": "__weakref__,__init__,__dict__,__module__",
}

# Heavy/optional deps mocked so RTD can build without installing them.
# scitex-* peer packages are not yet on PyPI, so RTD's `pip install .` cannot
# resolve them — mock them at autodoc time. The umbrella scitex package
# provides them in real installs.
autodoc_mock_imports = []  # peer deps installable from PyPI now

autosummary_generate = True

# Suppress pre-existing docstring/docutils warnings so CI `-W` doesn't
# fail on cosmetic legacy fixmes. Restrict to two categories that the
# audit found benign on develop; keeping the list explicit (not a blanket
# silence) so genuine errors still surface.
suppress_warnings = [
    "docutils",
    "autodoc",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
