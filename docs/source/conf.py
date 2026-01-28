from __future__ import annotations

import os
import sys
from datetime import date
from importlib.metadata import version as pkg_version

# Add project root + src/ to import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

project = "spectralmap"
author = "Your Name"
copyright = f"{date.today().year}, {author}"

# Project versioning
try:
    release = pkg_version("spectralmap")            # full version, e.g. "0.1.0"
except Exception:
    release = "0.1.0"

version = ".".join(release.split(".")[:2])          # short X.Y, e.g. "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST (optional Markdown support)
myst_enable_extensions = ["dollarmath", "colon_fence"]

autosummary_generate = True
autodoc_member_order = "bysource"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Cross-link to external docs (Sphinx 9: inventory must be None or string paths, not {})
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "starry": ("https://starry.readthedocs.io/en/latest/", None),
}