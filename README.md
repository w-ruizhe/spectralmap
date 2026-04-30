# spectralmap

A toolkit for **spectroscopic rotational mapping** of brown dwarfs and exoplanets. This package provides tools to invert lightcurves into surface maps, cluster surface spectral features, and analyze their variations.

## Quick start

**Requirement:** Python 3.9 (strictly required due to `theano-pymc` dependencies).

```bash
git clone <https://github.com/w-ruizhe/spectralmap.git>
cd spectralmap

# Create a virtual environment (ensure python is 3.9)
conda create -n spectralmap
conda activate spectralmap

# Install for users
pip install .

# Or
pip install spectralmap

# Optional: editable install with dev & docs dependencies
# pip install -e ".[dev,docs]"

# Run tests in a controlled isolated environment (installs spectralmap first)
tox -e py39

# (Alternative, current env)
# pip install -e ".[dev]"
# pytest
```

Build docs locally:

```bash
cd docs
make html
open _build/html/index.html
```

## Features

- **`spectralmap.mapping`**: Invert lightcurves to recover surface maps using `starry`.
- **`spectralmap.mapping` (Doppler mode)**: Solve spectral time-series Doppler imaging problems via `starry.DopplerMap`.
- **`spectralmap.cluster`**: Identify distinct spectral regions on the recovered maps.
- **`spectralmap.bayesian_linalg`**: Tools for linear and Bayesian map solving.

## Documentation

The repository includes detailed tutorials and theory in `docs/source/tutorials`:
1. **Forward Modeling**: Simulating lightcurves from maps.
2. **Map Inversion**: Solving for surface maps from time-series data.
3. **Retrieval Workflow**: Extracting spectra from maps.
4. **Case Studies**: Full demos for **SIMP 0136** and **Luhman 16B**.
5. **Theory**: Mathematical framework and PCA on surface spectra.

---

MIT License.
