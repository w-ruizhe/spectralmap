"""spectralmap: rotational + eclipse mapping toolkit (template).

A starting point for:
- rotational modulation mapping (phase curves, spot/cloud inference)
- eclipse / occultation mapping (ingress/egress constraints)
- wavelength-resolved ("spectral") mapping workflows
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

__all__ = ["__version__"]
__version__ = _version("spectralmap")
