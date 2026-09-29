Installation
============

Prerequisites
-------------

* Python 3.9 or newer (``requires-python`` is ``>=3.9``).
* Declared dependencies, not optional extras: NumPy ``>=1.26.4,<2``, Astropy ``>=6,<7``, SciPy ``>=1.9.3``, Matplotlib ``>=3.5.0``, tqdm ``>=4.62.0``, pydantic ``>=2``, PyYAML ``>=6``, numba ``>=0.54.1``, CyRK ``>=0.16.0``, healpy ``>=1.16.0``, python-dotenv ``>=1``, and psutil ``>=5.9``.
* ``healpy`` needs the CFITSIO stack that its wheels normally carry. If no wheel exists for the platform, installation fails there.

Install from a clone
--------------------

.. code-block:: bash

   git clone https://github.com/AnnaBilous/psr-magpy.git
   cd psr-magpy
   pip install .

Developer install
-----------------

.. code-block:: bash

   pip install -e ".[dev]"

Then, from the repository root, run ``pytest``.

Coverage for ``experiment/``, ``workflow/``, ``io/``, and ``viz/`` uses source paths. The package form ``--cov=psr_magpy.experiment`` imports that package before ``conftest`` and Astropy raises ``LoggingError``.

.. code-block:: bash

   pytest --cov=src/psr_magpy/experiment --cov=src/psr_magpy/workflow --cov=src/psr_magpy/io --cov=src/psr_magpy/viz

``pip install -e ".[docs]"`` installs Sphinx. Building the docs is a later step.

``psr-magpy-setup`` creates ``fits/``, ``plots/``, ``logs/``, and ``configs/`` under a chosen output directory and configures logging. It is not required for ``import psr_magpy``.
