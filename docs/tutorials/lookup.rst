Lookup and unit conversions
===========================

``experiment_matches_thD_fQ`` is true when an experiment has a dipole at latitude ``thD`` degrees and a quadrupole of strength ``fQ``. ``infer_experiment_count`` reads the saved configs. ``find_experiment_numbers_by_thD_fQ`` returns the matching indices.

``get_R_LC`` converts a period in milliseconds to a light-cylinder radius in stellar radii. ``get_P_ms`` is the inverse. ``tangent_phi_to_observer_phi`` maps a tangent azimuth to observer phase. ``PA_RVM`` is the rotating-vector-model position angle.

.. literalinclude:: ../../demo/pipeline/lookup.py
   :language: python

.. code-block:: bash

   python demo/pipeline/lookup.py

The run prints ``match=True``, ``n_experiments=1``, ``numbers=[0]``, ``R_LC=4.77``, ``P_ms=1.0``, ``observer_phi=270.0``, ``PA=0.000``.
