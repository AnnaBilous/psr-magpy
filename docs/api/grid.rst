Grid Module
===========

The grid module provides different grid types for sampling points on a sphere, essential for magnetic field line tracing in pulsar magnetosphere modeling.

Overview
--------

The module defines an abstract base class ``FootpointGrid`` and four concrete implementations:

- **SphericalGrid**: Regular latitude/longitude grid with uniform spacing
- **HealpixGrid**: Uniform spherical sampling using HEALPix algorithm
- **RandomGrid**: Random uniform distribution of points on a sphere
- **CustomGrid**: User-defined coordinates

All grid types support coordinate conversion, subsetting, I/O operations, and visualization.

Grid Types
----------

SphericalGrid
~~~~~~~~~~~~~

Regular grid with uniform spacing in longitude and latitude.

.. code-block:: python

    from psr_magpy.grid.grid import SphericalGrid
    
    # Create a 12x6 spherical grid
    grid = SphericalGrid(nlon=12, nlat=6)
    print(f"Grid has {grid.nline} points")
    
    # Create rotated grid (pole at 30° latitude)
    rotated_grid = SphericalGrid(nlon=16, nlat=8, pole_lat=30.0)
    
    # Access coordinates
    lon = grid.longitude  # Longitude in degrees
    colat = grid.colatitude  # Colatitude in degrees
    
    # Get Cartesian coordinates
    xyz = grid.footpoint_coordinates.xyz.value

HealpixGrid
~~~~~~~~~~~

Uniform spherical sampling using HEALPix algorithm. Resolution parameter must be a power of 2.

.. code-block:: python

    from psr_magpy.grid.grid import HealpixGrid
    
    # Create HEALPix grid with nside=4 (192 points)
    grid = HealpixGrid(nside=4)
    print(f"Grid has {grid.nline} points")
    
    # Valid nside values: 1, 2, 4, 8, 16, 32, 64, 128, ...
    high_res_grid = HealpixGrid(nside=64)  # 49,152 points

RandomGrid
~~~~~~~~~~

Random uniform distribution of points on a sphere using Muller's method.

.. code-block:: python

    from psr_magpy.grid.grid import RandomGrid
    
    # Create random grid with 1000 points
    grid = RandomGrid(nline_ori=1000)
    print(f"Grid has {grid.nline} points")
    
    # Each call generates different random coordinates
    grid2 = RandomGrid(nline_ori=1000)  # Different coordinates

CustomGrid
~~~~~~~~~~

Grid with user-provided coordinates.

.. code-block:: python

    from psr_magpy.grid.grid import CustomGrid
    from astropy.coordinates import CartesianRepresentation
    import numpy as np
    
    # Create custom grid with specific coordinates
    npoints = 100
    coords = CartesianRepresentation(
        x=np.random.random(npoints),
        y=np.random.random(npoints),
        z=np.random.random(npoints)
    )
    grid = CustomGrid(nline_ori=npoints, coordinates=coords)
    
    # Create empty grid (NaN coordinates)
    empty_grid = CustomGrid(nline_ori=50)
    
    # Set coordinates later
    empty_grid.set_coordinates(coords)

Common Operations
----------------

Coordinate Access
~~~~~~~~~~~~~~~~

All grid types provide consistent coordinate access:

.. code-block:: python

    # Get spherical coordinates
    lon = grid.longitude      # Longitude in degrees [0, 360)
    colat = grid.colatitude   # Colatitude in degrees [0, 180]
    
    # Get Cartesian coordinates
    xyz = grid.footpoint_coordinates.xyz.value  # Shape (3, nline)
    
    # Convert to spherical representation
    from astropy.coordinates import PhysicsSphericalRepresentation
    spherical = grid.footpoint_coordinates.represent_as(PhysicsSphericalRepresentation)

Grid Subsetting
~~~~~~~~~~~~~~~

Create subsets of grid points:

.. code-block:: python

    # Create subset with first 10 points
    subset = grid.subset_grid(np.arange(10))
    print(f"Subset has {subset.nline} points")
    print(f"Subset is partial: {subset.partial}")
    
    # Create subset with specific indices
    indices = np.array([0, 5, 10, 15])
    subset = grid.subset_grid(indices)
    
    # Subset preserves original coordinates
    assert subset.footpoint_coordinates is not None

Grid Copying
~~~~~~~~~~~~

Create independent copies of grids:

.. code-block:: python

    # Copy preserves coordinates exactly
    copied = grid.copy()
    
    # Copies are independent
    copied.nline = 999
    assert grid.nline != copied.nline
    
    # But coordinates are identical
    np.testing.assert_allclose(
        copied.footpoint_coordinates.xyz.value,
        grid.footpoint_coordinates.xyz.value
    )

I/O Operations
--------------

Save and load grids to/from FITS files:

.. code-block:: python

    # Save grid to FITS file
    grid.save_to_fits("my_grid.fits")
    
    # Load grid from FITS file
    from psr_magpy.grid.grid import FootpointGrid
    loaded_grid = FootpointGrid.load_from_fits("my_grid.fits")
    
    # Loaded grid has same properties
    assert loaded_grid.grid_type == grid.grid_type
    assert loaded_grid.nline == grid.nline
    np.testing.assert_allclose(
        loaded_grid.footpoint_coordinates.xyz.value,
        grid.footpoint_coordinates.xyz.value
    )

Visualization
-------------

Plot grid coordinates:

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Create plot
    fig, ax = plt.subplots()
    grid.plot_footpoint_coordinates(ax, marker='o', color='red', s=10)
    
    # Customize plot
    ax.set_title("Grid Footpoints")
    plt.show()

Error Handling
--------------

The module provides comprehensive error handling:

.. code-block:: python

    # Invalid parameters raise ValueError
    try:
        HealpixGrid(nside=3)  # Not a power of 2
    except ValueError as e:
        print(f"Error: {e}")
    
    # Empty subset indices raise ValueError
    try:
        grid.subset_grid(np.array([]))
    except ValueError as e:
        print(f"Error: {e}")
    
    # Invalid indices raise ValueError
    try:
        grid.subset_grid(np.array([-1, 0, 1]))
    except ValueError as e:
        print(f"Error: {e}")

Performance Considerations
-------------------------

- **SphericalGrid**: Fastest for regular sampling, good for systematic studies
- **HealpixGrid**: Uniform coverage, efficient for large-scale surveys
- **RandomGrid**: Good for statistical studies, but coordinates change each time
- **CustomGrid**: Most flexible, but requires manual coordinate management

Memory usage scales linearly with number of points. For very large grids (>100k points), consider using subsets or streaming approaches.

Advanced Usage
--------------

Grid Type Conversion
~~~~~~~~~~~~~~~~~~~

Convert between grid types by extracting coordinates:

.. code-block:: python

    # Start with spherical grid
    spherical = SphericalGrid(nlon=12, nlat=6)
    
    # Convert to custom grid with same coordinates
    custom = CustomGrid(
        nline_ori=spherical.nline,
        coordinates=spherical.footpoint_coordinates
    )

Coordinate Validation
~~~~~~~~~~~~~~~~~~~~

Validate that coordinates are unit vectors:

.. code-block:: python

    # Check unit vectors
    xyz = grid.footpoint_coordinates.xyz.value
    r = np.sqrt(np.sum(xyz**2, axis=0))
    np.testing.assert_allclose(r, 1.0, rtol=1e-10)

Grid Comparison
~~~~~~~~~~~~~~~

Compare grids for equality:

.. code-block:: python

    def grids_equal(grid1, grid2, rtol=1e-10):
        """Compare two grids for equality."""
        if grid1.grid_type != grid2.grid_type:
            return False
        if grid1.nline != grid2.nline:
            return False
        return np.allclose(
            grid1.footpoint_coordinates.xyz.value,
            grid2.footpoint_coordinates.xyz.value,
            rtol=rtol
        )

API Reference
-------------

.. automodule:: psr_magpy.grid.grid
   :members:
   :undoc-members:
   :show-inheritance: 