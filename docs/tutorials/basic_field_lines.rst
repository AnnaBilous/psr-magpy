Basic Field Line Tracing
=======================

This tutorial will guide you through the process of tracing field lines in a simple dipole field.

Setup
-----

First, import the necessary modules:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D
   
   import psr_magpy as pm
   from psr_magpy.field.multipole import MultipoleField
   from psr_magpy.grid.grid import SphericalGrid
   from psr_magpy.line.calculator import FieldLineCalculator
   from psr_magpy.line.field_line import FieldLine

Creating a Field
---------------

We'll start by creating a simple dipole field:

.. code-block:: python

   # Create a dipole field
   field = MultipoleField(l_max=1)
   
   # Set the orientation (alpha is the inclination angle)
   field.set_orientation(alpha=np.radians(30), beta=0)  # 30 degrees inclination

Creating a Grid
--------------

Next, we'll create a grid of starting points for our field lines:

.. code-block:: python

   # Create a spherical grid with 100 points
   grid = SphericalGrid(nline=100)
   
   # You can also customize the grid parameters
   # grid = SphericalGrid(nline=100, rmin=1.0, rmax=1.0, tmin=0, tmax=np.pi, pmin=0, pmax=2*np.pi)

Calculating Field Lines
----------------------

Now we can calculate the field lines:

.. code-block:: python

   # Create a calculator
   calculator = FieldLineCalculator(field, grid)
   
   # Create a template for field lines
   template = FieldLine(save_coordinates=True, save_field=True)
   
   # Calculate field lines
   field_lines = calculator.get_field_lines(template=template)

Visualizing the Results
---------------------

Finally, let's visualize the field lines:

.. code-block:: python

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
   
   # Set axis limits
   ax.set_xlim(-5, 5)
   ax.set_ylim(-5, 5)
   ax.set_zlim(-5, 5)
   
   # Add labels
   ax.set_xlabel('X')
   ax.set_ylabel('Y')
   ax.set_zlabel('Z')
   ax.set_title('Dipole Field Lines')
   
   plt.show()

Advanced Options
--------------

You can customize the field line tracing process with various options:

.. code-block:: python

   from psr_magpy.line.calculator import TracingConfig
   
   # Create a tracing configuration
   config = TracingConfig(
       step_size=0.01,        # Integration step size
       max_steps=1000,        # Maximum number of steps
       rtol=1e-6,             # Relative tolerance
       atol=1e-6,             # Absolute tolerance
       direction='both',      # Trace in both directions
       stop_radius=50.0,      # Stop when reaching this radius
       save_results=True      # Save results to file
   )
   
   # Calculate field lines with the configuration
   field_lines = calculator.get_field_lines(template=template, config=config) 