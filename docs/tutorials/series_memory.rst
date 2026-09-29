In-memory series
================

Same two-grid series as :doc:`series_disk`. ``run_experiment_series_in_memory`` keeps each result on the ``SeriesResult`` and does not write FITS batches.

.. literalinclude:: ../../demo/pipeline/series_memory.py
   :language: python

.. code-block:: bash

   python demo/pipeline/series_memory.py

The last lines are ``success=2/2`` and ``kept=2``.
