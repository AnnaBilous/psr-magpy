Plots
=====

The script plots four arrays with ``plot_field_lines_2d``, ``plot_OFLR_map``, ``plot_skymap``, and ``plot_PA_vs_phi``. A saved series is loaded with ``load_snapshot_data`` on the :doc:`products` page. Healpix projection is ``psr_magpy.viz.healpix``. Panel helpers are in ``psr_magpy.viz.pa_panels``. Pseudo-3D lines are in ``psr_magpy.viz.pseudo3d``.

.. literalinclude:: ../../demo/pipeline/plot_panels.py
   :language: python

.. code-block:: bash

   python demo/pipeline/plot_panels.py

``output=`` is the directory that contains ``field_lines.png``, ``oflr.png``, ``skymap.png``, and ``pa.png``.
