Field Components Module
=======================

The field components module provides magnetic field implementations for neutron star magnetosphere modeling, including dipole and quadrupole field components with parameter validation and coordinate transformations.

Overview
--------

The module defines an abstract base class ``FieldComponent`` and concrete implementations for different multipole field types:

- **DipoleFieldComponent**: Standard dipole magnetic field
- **QuadrupoleFieldComponent**: Quadrupole magnetic field component

All field components support:
- Parameter validation using Pydantic models
- Coordinate transformations between spin and magnetic frames
- Field strength scaling and orientation control
- Offset positioning from the origin
- Comprehensive error handling and logging

Core Classes
------------

FieldComponentParams
~~~~~~~~~~~~~~~~~~~

Pydantic model for validated field component parameters.

.. code-block:: python

    from psr_magpy.field.components import FieldComponentParams, MultipoleType
    
    # Create parameters with validation
    params = FieldComponentParams(
        B_magnitude=1.0,
        moment_orientation_latitude_degree=30.0,
        moment_orientation_longitude_degree=45.0,
        offset_r_ns=[0.1, 0.0, 0.0],
        moment_type=MultipoleType.DIPOLE
    )
    
    # Invalid parameters raise validation errors
    try:
        invalid_params = FieldComponentParams(
            B_magnitude=-1.0,  # Must be non-negative
            moment_type=MultipoleType.DIPOLE
        )
    except ValueError as e:
        print(f"Validation error: {e}")

FieldComponent
~~~~~~~~~~~~~

Abstract base class for all magnetic field components.

.. code-block:: python

    from psr_magpy.field.components import FieldComponent
    
    # Cannot instantiate directly (abstract class)
    try:
        component = FieldComponent()
    except TypeError:
        print("FieldComponent is abstract")
    
    # Provides common interface for all field types
    # - get_field(position): Calculate field at given position
    # - get_moment_type(): Return multipole type
    # - Cached properties for orientation and transformations

Field Component Types
--------------------

DipoleFieldComponent
~~~~~~~~~~~~~~~~~~~

Standard dipole magnetic field implementation.

.. code-block:: python

    from psr_magpy.field.components import DipoleFieldComponent
    import numpy as np
    
    # Create dipole with default parameters (aligned with Z-axis)
    dipole = DipoleFieldComponent()
    
    # Create dipole with custom parameters
    dipole = DipoleFieldComponent(
        B_magnitude=2.0,                    # Field strength scaling
        moment_orientation_latitude_degree=30.0,   # Tilt angle
        moment_orientation_longitude_degree=45.0,  # Rotation angle
        offset_r_ns=[0.1, 0.0, 0.0]        # Offset from origin
    )
    
    # Calculate field at specific position
    position = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    field = dipole.get_field(position)
    print(f"Field at {position}: {field}")
    
    # Field follows 1/r³ falloff
    pos1 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    pos2 = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    field1 = dipole.get_field(pos1)
    field2 = dipole.get_field(pos2)
    ratio = np.linalg.norm(field2) / np.linalg.norm(field1)
    print(f"Field ratio (r=2/r=1): {ratio:.3f} (expected ~0.125)")

QuadrupoleFieldComponent
~~~~~~~~~~~~~~~~~~~~~~~~

Quadrupole magnetic field implementation.

.. code-block:: python

    from psr_magpy.field.components import QuadrupoleFieldComponent
    
    # Create quadrupole field
    quadrupole = QuadrupoleFieldComponent(
        B_magnitude=0.5,
        moment_orientation_latitude_degree=0.0
    )
    
    # Quadrupole field falls off faster than dipole
    position = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    field_quad = quadrupole.get_field(position)
    field_dip = dipole.get_field(position)
    
    print(f"Quadrupole field magnitude: {np.linalg.norm(field_quad):.3f}")
    print(f"Dipole field magnitude: {np.linalg.norm(field_dip):.3f}")

Common Operations
----------------

Field Calculation
~~~~~~~~~~~~~~~~

Calculate magnetic field at arbitrary positions:

.. code-block:: python

    import numpy as np
    
    # Single position
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    field = dipole.get_field(pos)
    print(f"Field: {field}")
    
    # Multiple positions (loop through)
    positions = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
    ], dtype=np.float32)
    
    fields = []
    for pos in positions:
        field = dipole.get_field(pos)
        fields.append(field)
    
    fields = np.array(fields)
    print(f"Field array shape: {fields.shape}")

Parameter Access
~~~~~~~~~~~~~~~

Access and modify component parameters:

.. code-block:: python

    # Access current parameters
    print(f"Field strength: {dipole.params.B_magnitude}")
    print(f"Orientation: {dipole.params.moment_orientation_latitude_degree}°")
    print(f"Offset: {dipole.params.offset_r_ns}")
    print(f"Type: {dipole.params.moment_type}")
    
    # Cached properties for efficient access
    orientation = dipole.moment_orientation
    offset = dipole.moment_offset_vector
    rot_matrix = dipole.rotate_magn_to_spin_frame

Coordinate Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~

Field components handle coordinate transformations automatically:

.. code-block:: python

    # Position in spin frame
    pos_spin = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    
    # Field calculation automatically transforms to magnetic frame
    field = dipole.get_field(pos_spin)
    
    # Manual transformation available
    pos_magn = dipole._transform_to_magnetic_frame(pos_spin)
    print(f"Spin frame: {pos_spin}")
    print(f"Magnetic frame: {pos_magn}")

Error Handling
--------------

Comprehensive error handling for invalid inputs:

.. code-block:: python

    # Invalid position (non-finite values)
    try:
        invalid_pos = np.array([np.nan, 1.0, 2.0], dtype=np.float32)
        field = dipole.get_field(invalid_pos)
    except ValueError as e:
        print(f"Error: {e}")
    
    # Invalid parameters
    try:
        invalid_dipole = DipoleFieldComponent(B_magnitude=-1.0)
    except Exception as e:
        print(f"Parameter error: {e}")
    
    # Position too close to origin (singularity handling)
    pos_close = np.array([0.0, 0.0, 1e-11], dtype=np.float32)
    field_close = dipole.get_field(pos_close)
    print(f"Field near origin: {field_close}")  # Returns zero field

Performance Considerations
-------------------------

Cached Properties
~~~~~~~~~~~~~~~~

Field components use cached properties for efficient repeated access:

.. code-block:: python

    # These calculations are cached after first access
    orientation = dipole.moment_orientation
    rot_matrix = dipole.rotate_magn_to_spin_frame
    
    # Subsequent accesses use cached values
    orientation2 = dipole.moment_orientation  # No recalculation
    rot_matrix2 = dipole.rotate_magn_to_spin_frame  # No recalculation

Vectorization
~~~~~~~~~~~~

For multiple positions, consider vectorizing field calculations:

.. code-block:: python

    # Efficient: Calculate fields for grid of positions
    y, z = np.meshgrid(np.linspace(-2, 2, 50), np.linspace(-2, 2, 50))
    x = np.zeros_like(y)
    
    # Reshape for vectorized calculation
    positions = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    positions = positions.astype(np.float32)
    
    # Calculate fields (this could be optimized further)
    fields = []
    for pos in positions:
        field = dipole.get_field(pos)
        fields.append(field)
    
    fields = np.array(fields).reshape(x.shape + (3,))

Integration Examples
-------------------

Combining Multiple Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create complex field configurations:

.. code-block:: python

    # Create dipole and quadrupole components
    dipole = DipoleFieldComponent(B_magnitude=1.0)
    quadrupole = QuadrupoleFieldComponent(B_magnitude=0.1)
    
    # Calculate combined field
    position = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    field_dip = dipole.get_field(position)
    field_quad = quadrupole.get_field(position)
    field_total = field_dip + field_quad
    
    print(f"Dipole contribution: {field_dip}")
    print(f"Quadrupole contribution: {field_quad}")
    print(f"Total field: {field_total}")

Field Line Tracing Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Field components integrate with field line tracing:

.. code-block:: python

    from psr_magpy.line.calculator import FieldLineCalculator
    from psr_magpy.grid.grid import SphericalGrid
    
    # Create field component
    dipole = DipoleFieldComponent(B_magnitude=1.0)
    
    # Create footpoint grid
    grid = SphericalGrid(nlon=12, nlat=6)
    
    # Create field line calculator
    calculator = FieldLineCalculator(
        field_components=[dipole],
        grid=grid
    )
    
    # Trace field lines
    field_lines = calculator.calculate_field_lines()

Visualization
-------------

Plot field vectors:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create grid for visualization
    y, z = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    x = np.zeros_like(y)
    
    # Calculate field vectors
    dipole = DipoleFieldComponent()
    fields = []
    positions = []
    
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            pos = np.array([x[i,j], y[i,j], z[i,j]], dtype=np.float32)
            field = dipole.get_field(pos)
            positions.append(pos)
            fields.append(field)
    
    positions = np.array(positions)
    fields = np.array(fields)
    
    # Plot field vectors
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot arrows colored by field strength
    magnitudes = np.linalg.norm(fields, axis=1)
    scatter = ax.quiver(
        positions[:, 1], positions[:, 2],
        fields[:, 1], fields[:, 2],
        magnitudes,
        cmap='viridis',
        scale=50
    )
    
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_title('Dipole Field Vectors (YZ plane)')
    plt.colorbar(scatter, label='Field Magnitude')
    plt.show()

API Reference
-------------

.. automodule:: psr_magpy.field.components
   :members:
   :undoc-members:
   :show-inheritance: 