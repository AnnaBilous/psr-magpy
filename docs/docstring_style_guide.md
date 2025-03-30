# Docstring Style Guide for psr-magpy

This document outlines the docstring standards for the psr-magpy project.

## NumPy Docstring Format

All code in psr-magpy should use NumPy-style docstrings. This format was chosen for:

1. Clear separation of sections
2. Good support for scientific notation and units
3. Compatibility with Sphinx documentation generation

## Required Sections

### For All Objects

- **Short summary**: First line of the docstring, briefly describing purpose
- **Extended description**: More detailed explanation (when needed)

### For Functions and Methods

- **Parameters**: All parameters with types, descriptions, and units when applicable
- **Returns**: What the function returns, with type and description
- **Raises**: Exceptions that may be raised (when applicable)

### For Classes

- **Parameters**: Constructor parameters
- **Attributes**: Class attributes
- **Methods**: Brief description of key methods (optional, can be in class docstring)

### Optional Sections (When Relevant)

- **See Also**: Related functions or classes
- **Notes**: Implementation details, mathematical background
- **Examples**: Usage examples
- **References**: Citations to papers or books

## Units and Mathematical Notation

- Always include units in square brackets: `distance : float, Distance to pulsar [km]`
- Use LaTeX for equations with the `.. math::` directive
- Define coordinate systems and conventions clearly

## Examples

Examples should be runnable code that demonstrates usage. Include expected output as comments or in doctest format.

## Reference

See `docs/docstring_templates.py` for complete examples of properly formatted docstrings. 