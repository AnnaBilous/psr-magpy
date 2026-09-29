Disk series
===========

A series expands lists into experiments, writes one YAML file, and runs each experiment to disk.

``get_parameter_combinations`` builds the Cartesian product. A value that is not a list stays fixed. In the script, ``nlat: [3, 4]`` with ``nlon: 4`` is two grids, so two experiments.

``SphericalGridFactoryConfig`` is what the series actually expands. ``ExperimentSeriesConfig.save`` writes YAML. ``ExperimentSeriesConfig.load`` reads it back. ``run_experiment_series`` then traces the lines.

``DeletionStrategy.NONE`` leaves an existing series directory in place. ``OverwriteStrategy.ALL`` replaces files written by this run. Both live in ``psr_magpy.experiment.experiment_series_runner``.

Batches are written to ``<output>/dipole_series/lines``. Configs are under ``dipole_series/configs``. The post-processing page uses that same layout.

.. literalinclude:: ../../demo/pipeline/series_disk.py
   :language: python

``tutorial_layout.py`` in the same directory builds the series. Run from the repository root:

.. code-block:: bash

   python demo/pipeline/series_disk.py

The last line is ``success=2/2``. ``output=`` is a new temporary directory.
