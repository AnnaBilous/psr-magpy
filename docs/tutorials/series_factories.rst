Series factories
================

A factory holds a list where a plain config holds one value. ``expand_grid_configs`` turns ``GridFactoryConfig`` into one ``GridConfig`` per grid. The script puts four grids in one factory:

- ``SphericalGridFactoryConfig``
- ``HealpixGridFactoryConfig``
- ``RandomGridFactoryConfig``
- ``CustomGridFactoryConfig``

An integer ``nline_ori`` plus that many 3-vectors is one grid, as in the script (``nline_ori=2`` and two coordinates). A list of ``nline_ori`` values is one grid per entry.

``FieldFactoryConfig`` and ``FieldComponentFactoryConfig`` do the same for components. Two component groups and four grids are eight experiments. ``TracingFactoryConfig`` and ``TemplateFactoryConfig`` expand the same way; here each has one value.

``run_experiment_series_from_file`` and ``run_experiment_series_in_memory_from_file`` read a saved series YAML and run it. That YAML is the two-grid dipole series from :doc:`series_disk`.

.. literalinclude:: ../../demo/pipeline/series_factories.py
   :language: python

.. code-block:: bash

   python demo/pipeline/series_factories.py

The run prints ``expanded_grids=4``, ``experiments=8``, ``from_file=2/2``, ``from_file_memory=2/2``.
