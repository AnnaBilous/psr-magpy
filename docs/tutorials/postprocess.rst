Open-field-line map, skymap, and PA
====================================

This script runs the same series as :doc:`series_disk`, then three calls, in this order:

1. ``psr_magpy.workflow.extract_ofl_map``
2. ``psr_magpy.workflow.make_skymap``
3. ``psr_magpy.workflow.extract_PA.make_extract_PA``

The series directory is ``<output>/dipole_series``. Line batches are in ``lines/``. ``R_LC`` is ``2`` stellar radii, inside the traced radius of this grid. The open-field-line maps go to ``data_products/OFLR_maps``. The skymap goes to ``data_products/skymaps``. The PA extract goes to ``data_products/extracted_PA``.

The open-field-line map stores ``R_LC`` at footpoint indices. The skymap counts samples by tangent colatitude and azimuth. Both can be HEALPix arrays of the same length. They are not the same map.

A footpoint is open when its maximum cylindrical radius is greater than ``R_LC``, or when it stays inside the cylinder and climbs above ``h_LC_coeff * R_LC``. Equality with ``R_LC`` is closed.

``make_extract_PA(..., save=True)`` writes the FITS file and returns an empty ``FieldLineSet``. Several ``R_LC`` values return a dict of those sets. The file is the result.

.. literalinclude:: ../../demo/pipeline/postprocess.py
   :language: python

.. code-block:: bash

   python demo/pipeline/postprocess.py

``ofl_open`` is the number of open footpoints in the first experiment. ``pa_files`` lists the PA FITS file.
