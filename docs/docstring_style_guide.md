# Docstring Style Guide for psr-magpy

This document outlines the docstring standards for the psr-magpy project.

## Docstring Format

All code in psr-magpy should use **NumPy-style** docstrings that prioritize readability and practical usage. The format emphasizes:

1. **Brief, descriptive summaries** - One-line descriptions that capture the essence
2. **NumPy style sections** - Use `Parameters`, `Returns`, `Raises` (not `Args`)
3. **Type information** - Include types in docstrings for Sphinx compatibility
4. **Conciseness** - Remove redundant information already in type hints
5. **Consistency** - Follow NumPy docstring convention throughout

## Required Sections

### For All Objects

- **Short summary**: First line of the docstring, briefly describing purpose
- **Extended description**: More detailed explanation (when needed)

### For Functions and Methods

- **Parameters**: Document all parameters with type and description. Format: `name : type`
- **Returns**: What the function returns with type. Format: `type - Description`
- **Raises**: Exceptions that may be raised (when applicable)

### For Classes

- **Purpose**: What the class represents and its main functionality
- **Key methods**: Brief mention of important methods (optional)

### Optional Sections (When Relevant)

- **Examples**: Usage examples for complex functionality
- **Notes**: Implementation details, mathematical background
- **See Also**: Related functions or classes

## Docstring Patterns

### Simple Functions/Methods

```python
def colatitude(self) -> np.ndarray:
    """Get colatitude values in degrees."""
    # implementation
```

### Methods with Parameters

```python
def subset_grid(self, grid_indices: np.ndarray) -> "FootpointGrid":
    """Create subset of grid points.
    
    Parameters
    ----------
    grid_indices : np.ndarray
        Indices of grid points to include in subset.
    
    Returns
    -------
    FootpointGrid
        New grid containing only specified points.
    """
```

### Classes

```python
class SphericalGrid(FootpointGrid):
    """Regular spherical grid with uniform spacing in longitude and latitude."""
    
    def __init__(self, nlon: int, nlat: int, pole_lat: float = 0.0):
        """Initialize spherical grid.
        
        Parameters
        ----------
        nlon : int
            Number of longitude divisions.
        nlat : int
            Number of latitude divisions.
        pole_lat : float, optional
            Latitude of grid pole in degrees. Default is 0.0.
        """
```

### Complex Methods

```python
def set_coordinates(self, coordinates: Optional[CartesianRepresentation]) -> None:
    """Set new coordinates for grid points.
    
    Parameters
    ----------
    coordinates : Optional[CartesianRepresentation]
        New coordinates for grid points. Can be None.
    
    Raises
    ------
    ValueError
        If coordinates length doesn't match nline_ori.
    """
```

## Units and Mathematical Notation

- Include units in parameter descriptions when relevant: `pole_lat: Latitude of grid pole in degrees`
- Use clear, descriptive names rather than mathematical symbols in docstrings
- Reserve mathematical notation for comments or separate documentation

## Error Handling Documentation

Document exceptions that are raised:

```python
def validate_power_of_two(self):
    """Validate that nside is a power of 2."""
    if not (self.nside & (self.nside - 1) == 0):
        error_msg = f"nside must be a power of 2, got {self.nside}"
        logger.error(error_msg)
        raise ValueError(error_msg)
```

## Examples

Include examples for complex functionality:

```python
def copy(self) -> "FootpointGrid":
    """Create a copy of the grid.
    
    Examples:
        >>> grid = SphericalGrid(12, 6)
        >>> copied = grid.copy()
        >>> np.allclose(copied.footpoint_coordinates.xyz.value,
        ...             grid.footpoint_coordinates.xyz.value)
        True
    """
```

## Style Guidelines

1. **Be concise** - Prefer shorter, clearer descriptions over verbose documentation
2. **Focus on usage** - Explain what the code does, not how it does it
3. **Use type hints** - Let type hints carry parameter information
4. **Document exceptions** - Always document custom exceptions that may be raised
5. **Be consistent** - Follow established patterns in the codebase
6. **Include examples** - For complex functionality, provide usage examples

## Reference

See `docs/docstring_templates.py` for complete examples of properly formatted docstrings. 