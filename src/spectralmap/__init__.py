"""Surface-mapping tools for rotational, eclipse, and Doppler observations.

The package exposes mode-specific modules for rotational phase curves,
secondary-eclipse light curves, Doppler imaging spectra, Bayesian linear
algebra, plotting, clustering, and small data utilities. Heavy optional
dependencies are imported lazily where possible so that ``import spectralmap``
remains usable in documentation and lightweight test environments.
"""

import logging
import warnings
from importlib.metadata import version as _version

# --- Suppress noisy dependency warnings ---
# 1. Theano MKL warning (logging)
#    "install mkl with `conda install mkl-service`: No module named 'mkl'"
try:
    logging.getLogger("theano.link.c.cmodule").setLevel(logging.ERROR)
except Exception:
    pass

# 2. PyMC3 "outdated" warning (logging)
#    "The version of PyMC you are using is very outdated..."
#    PyMC3 resets logger level on import, so we use a Filter instead.
try:
    class _PyMC3Filter(logging.Filter):
        def filter(self, record):
            return "outdated" not in record.getMessage()
    
    logging.getLogger("pymc3").addFilter(_PyMC3Filter())
except Exception:
    pass

# 3. pkg_resources deprecation warning from starry (warnings)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
# ------------------------------------------

from . import bayesian_linalg
from . import core

# Optional convenience imports: keep top-level package import resilient when
# optional plotting/ML dependencies are not installed.
try:
    from . import cluster
except ModuleNotFoundError:
    cluster = None

try:
    from . import plotting
except ModuleNotFoundError:
    plotting = None

__all__ = ["__version__", "bayesian_linalg", "core", "cluster", "plotting"]
__version__ = _version("spectralmap")
