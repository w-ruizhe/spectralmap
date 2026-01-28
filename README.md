# spectralmap (template)

A starter repository for **rotational + eclipse/occultation surface mapping**, with optional
**wavelength-resolved (“spectral”) mapping**, plus a **Sphinx + Read the Docs** documentation site.

## Quick start

```bash
git clone <YOUR_REPO_URL>
cd spectralmap
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
pytest
```

Build docs locally:

```bash
cd docs
make html
open _build/html/index.html
```

## What’s included

- `src/spectralmap/` minimal Python package layout (`src/` style)
- `docs/` Sphinx project (reStructuredText + optional Markdown via MyST)
- `.readthedocs.yaml` configuration
- GitHub Actions workflow to build docs and run tests

---

MIT License.
