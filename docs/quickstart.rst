Quickstart
==========

This guide will help you get started with psr-magpy quickly.

Basic Usage
----------

**Note: FieldLine objects serve dual roles:**
- **Configuration templates** (YAML) - define what to save
- **Data containers** (FITS) - store actual field line data

Import the package:

.. code-block:: python

   import psr_magpy as pm

Create a magnetic field:

.. code-block:: python

   from psr_magpy.field.multipole import MultipoleField
   
   # Create a dipole field
   field = MultipoleField(l_max=1)
   
   # Set the orientation
   field.set_orientation(alpha=0.5, beta=0.2)  # in radians

Create a grid for field line tracing:

.. code-block:: python

   from psr_magpy.grid.grid import SphericalGrid
   
   # Create a spherical grid with 1000 points
   grid = SphericalGrid(nline=1000)

Calculate field lines:

.. code-block:: python

   from psr_magpy.line.calculator import FieldLineCalculator
   from psr_magpy.line.field_line import FieldLine
   
   # Create a calculator
   calculator = FieldLineCalculator(field, grid)
   
   # Create a configuration template for field lines
   # This defines which properties to save during tracing
   template = FieldLine.template_all()  # Save all properties
   
   # Calculate field lines using the template
   field_lines = calculator.get_field_lines(template=template)

Visualize the results:

.. code-block:: python

   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D
   
   fig = plt.figure(figsize=(10, 8))
   ax = fig.add_subplot(111, projection='3d')
   
   for line in field_lines:
       ax.plot(line.x, line.y, line.z, 'b-', alpha=0.5)
   
   # Add a sphere for the star
   u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
   x = np.cos(u) * np.sin(v)
   y = np.sin(u) * np.sin(v)
   z = np.cos(v)
   ax.plot_surface(x, y, z, color='r', alpha=0.2)
   
   plt.show() 