Quickstart
==========

Read :doc:`concepts` first. The names on that page are the ones this script uses.

The script below builds one ``ExperimentConfig`` and calls ``run_experiment``. The grid is 4 by 3 starting points. Output is a new temporary directory, printed as ``output=``.

.. literalinclude:: ../demo/pipeline/pipeline_walkthrough.py
   :language: python

Run it from the repository root, after ``pip install -e ".[dev]"`` or ``pip install .``:

.. code-block:: bash

   python demo/pipeline/pipeline_walkthrough.py

A single line without an experiment is still available through ``FieldLineCalculator``. That calculator is not the front door.
