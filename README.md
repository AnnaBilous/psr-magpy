# psr-magpy

psr-magpy traces magnetic field lines in a pulsar magnetosphere. It runs that tracing as named experiments and series, including the batch products: an open-field-line map, a skymap, and PA.

## Installation

Python 3.9 or 3.10. Clone the repository and install from that checkout:

```bash
git clone https://github.com/AnnaBilous/psr-magpy.git
cd psr-magpy
pip install .
```

Developer install, including tests:

```bash
pip install -e ".[dev]"
```

`pip install -e ".[docs]"` installs Sphinx. Build the HTML with `make docs`.

`psr-magpy-setup` creates `fits/`, `plots/`, `logs/`, and `configs/` under a chosen output directory and configures logging. It is not required for `import psr_magpy`.

## Tests

From the repository root, after the developer install:

```bash
pytest
```

Coverage for `experiment/`, `workflow/`, `io/`, and `viz/` uses source paths:

```bash
pytest --cov=src/psr_magpy/experiment --cov=src/psr_magpy/workflow --cov=src/psr_magpy/io --cov=src/psr_magpy/viz
```

## Documentation

Narrative docs are in `docs/`. `make docs` writes HTML under `docs/_build/html`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

GPL-3.0-or-later. See `LICENSE`.
