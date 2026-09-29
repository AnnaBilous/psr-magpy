Edges, a fine subset, and a snapshot
====================================

:doc:`postprocess` writes the open-field-line map, the skymap, and the PA extract from a spherical series. This script starts lines on a HEALPix grid (``nside=1``) that also has a quadrupole, then calls:

- ``find_OFL_regions_from_batch_files``
- ``calculate_and_save_OFL_edges``
- ``calculate_and_save_OFL_regions``
- ``create_fine_subset_workflow``
- ``load_snapshot_data``

``load_snapshot_data`` reads the map, the skymap, and the PA extract back. The experiment must contain a dipole and a quadrupole, and its grid must be HEALPix.

.. literalinclude:: ../../demo/pipeline/products.py
   :language: python

.. code-block:: bash

   python demo/pipeline/products.py

The run prints ``regions_finite=4``, ``edges=1``, ``ofl_regions=1``, ``fine=healpix_fine``, ``snapshot_lines=4``.
