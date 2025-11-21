# psr-magpy
Playground for pulsar magnetic field line calculations

## Installation

## Demo Outputs

When running demos, outputs are saved to:
- Linux/Mac: `~/.psr_magpy/demo_outputs/`
- Windows: `C:\Users\<username>\.psr_magpy\demo_outputs\`

You can specify a custom output location with the `--output-dir` parameter.

## Code Style

### Docstrings

All code must use **NumPy-style** docstrings. See `docs/docstring_style_guide.md` for details.

**Key requirements:**
- Use `Parameters:` not `Args:`
- Include type information: `name : type`
- Add section underlines: `-----`
- Keep docstrings concise and focused on usage

**Example:**
```python
def calculate_field(r: np.ndarray) -> np.ndarray:
    """Calculate magnetic field at position.
    
    Parameters
    ----------
    r : np.ndarray
        Position vector (3-element array).
    
    Returns
    -------
    np.ndarray
        Magnetic field vector.
    """
```

## Documentation

Documentation is built automatically from docstrings using Sphinx.

**Build documentation:**
```bash
# Install documentation dependencies
pip install -e .[docs]

# Build HTML documentation
make docs

# Or use the build script
python tools/build_docs.py

# Generate API docs only
make docs-api

# Clean and rebuild
make docs-clean docs
```

The documentation will be generated in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser to view it.
