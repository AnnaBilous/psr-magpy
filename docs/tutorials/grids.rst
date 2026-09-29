Start grids
===========

Field lines start on one of four grids. ``GridConfig`` holds exactly one of them. ``create_grid`` builds the grid object and ``nline`` is the number of starting points.

- ``SphericalGridConfig`` / ``SphericalGrid``: ``nlon``, ``nlat``, ``pole_lat``.
- ``HealpixGridConfig`` / ``HealpixGrid``: ``nside``, a power of two. ``nline`` is ``12 * nside**2``.
- ``RandomGridConfig`` / ``RandomGrid``: ``nline_ori`` draws that many directions. Save the grid with ``save_to_fits`` before an experiment writes batches.
- ``CustomGridConfig`` / ``CustomGrid``: you pass the coordinates. The same save rule applies.

.. literalinclude:: ../../demo/pipeline/grids.py
   :language: python

.. code-block:: bash

   python demo/pipeline/grids.py

The run prints ``SphericalGrid nline=6``, ``HealpixGrid nline=12``, ``RandomGrid nline=4``, ``CustomGrid nline=2``.
