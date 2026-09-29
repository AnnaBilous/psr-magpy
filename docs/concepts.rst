Concepts
========

These are the names the package exports. Later pages use them in this order.

``FieldConfig`` and ``FieldComponentConfig``
    The magnetosphere model: which multipole components are present, and their strengths and orientations.

``GridConfig``
    Where the field lines start. A spherical start grid is ``SphericalGridConfig``. A HEALPix start grid is ``HealpixGridConfig``.

``FieldLineTemplate`` and ``FieldLine``
    ``FieldLineTemplate`` chooses which quantities to store. ``FieldLine`` is one traced line. They are different classes. ``FieldLine.template_all()`` is not the template constructor.

``ExperimentConfig`` and ``run_experiment``
    One run: a field, a grid, a tracing setup, and a template, executed by ``run_experiment``. ``run_experiment`` lives in ``psr_magpy.experiment.experiment_runner``.

``ExperimentSeriesConfig`` and ``run_experiment_series``
    Many runs from expanded parameters. ``run_experiment_series`` lives in ``psr_magpy.experiment.experiment_series_runner``. When a previous output is already on disk, the runner takes a ``DeletionStrategy`` or an ``OverwriteStrategy``.

Batches and products
    The I/O layer writes line batches as FITS. ``psr_magpy.workflow`` builds the open-field-line map, the skymap, and the PA extract from those batches afterwards.
