Plots
=====

Plotting imports come from ``psr_magpy.viz``. This page uses ``plot_skymap`` and ``plot_PA_vs_phi``. A saved series is loaded with ``psr_magpy.viz.snapshot.load_snapshot_data``. Healpix projection is ``psr_magpy.viz.healpix``. Panel helpers are in ``psr_magpy.viz.pa_panels``. Pseudo-3D lines are in ``psr_magpy.viz.pseudo3d``.

The script below does not load a series. It plots small arrays and writes two PNG files.

.. literalinclude:: ../../demo/pipeline/plot_panels.py
   :language: python

.. code-block:: bash

   python demo/pipeline/plot_panels.py

``output=`` is the directory that contains ``skymap.png`` and ``pa.png``.
