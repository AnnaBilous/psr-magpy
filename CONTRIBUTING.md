# Contributing

## Docstrings

NumPy style. Section names are `Parameters`, `Returns`, and `Raises`, not `Args`. The longer description is `docs/docstring_style_guide.md`.

## Errors

Build one message, log it, then raise it:

```python
error_msg = "message"
logger.error(error_msg)
raise ValueError(error_msg)
```

## Tests

From the repository root, after `pip install -e ".[dev]"`:

```bash
pytest
```

## Docs

```bash
pip install -e ".[docs]"
make docs
```

## What not to add

Do not open a pull request that adds `src/development/`, data products, or `docs/_build/`.
