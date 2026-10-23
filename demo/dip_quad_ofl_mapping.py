"""Dipole plus quadrupole series for OFL region mapping.

Dipole colatitude runs from 0 to 90 deg (step 2). Quadrupole ``B_magnitude``
runs from 9 to 4 (step 0.1). Footpoints are HEALPix ``nside=128``. Lines are
integrated to 300 stellar radii with the Numba field and the CyRK solver, and
checked by reversing the integration. Saved columns are the OFL subset:
``footpoint_id``, ``r_cyl_max``, ``z_abs_max``, ``reversed_error``.
"""

import argparse
from typing import List

import numpy as np

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.experiment_series_runner import (
    DeletionStrategy,
    run_experiment_series,
)
from psr_magpy.experiment.factory_configs import (
    FieldComponentFactoryConfig,
    FieldFactoryConfig,
    GridFactoryConfig,
    TracingFactoryConfig,
)
from psr_magpy.experiment.grid_config import HealpixGridConfig
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__)

NSIDE = 128
MAX_INTEGRATION_RADIUS = 500.0
DIPOLE_B = 1.0
THD_START = 0.0
THD_STOP = 90.0
THD_STEP = 2.0
QUAD_B_START = 0.0
QUAD_B_STOP = 4.0
QUAD_B_STEP = 0.1


def inclusive_steps(start: float, stop: float, step: float) -> List[float]:
    """Return ``start`` .. ``stop`` inclusive.

    ``step`` is positive. If ``stop < start``, the sequence decreases.
    """
    if step <= 0.0:
        error_msg = f"step must be positive, got {step}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    direction = 1.0 if stop >= start else -1.0
    values = np.arange(start, stop + direction * step / 2.0, direction * step)
    return [float(value) for value in np.round(values, decimals=10)]


def build_series(output_dir: str, n_cores: int) -> ExperimentSeriesConfig:
    """Build the dipole-plus-quadrupole series.

    Parameters
    ----------
    output_dir
        Directory that will hold the series (configs, lines, logs).
    n_cores
        Worker count for line integration.
    """
    if n_cores < 1:
        error_msg = f"n_cores must be >= 1, got {n_cores}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    thd = inclusive_steps(THD_START, THD_STOP, THD_STEP)
    quad_b = inclusive_steps(QUAD_B_START, QUAD_B_STOP, QUAD_B_STEP)
    dipole = FieldComponentFactoryConfig(
        moment_type="dipole",
        B_magnitude=DIPOLE_B,
        moment_orientation_latitude_degree=thd,
        moment_orientation_longitude_degree=0.0,
    )
    quadrupole = FieldComponentFactoryConfig(
        moment_type="quadrupole",
        B_magnitude=quad_b,
        moment_orientation_latitude_degree=0.0,
        moment_orientation_longitude_degree=0.0,
    )
    return ExperimentSeriesConfig(
        name=f"healpix_nside_{NSIDE:04d}_dip_quad_ofl",
        output_dir=output_dir,
        description=(
            "Inclined dipole plus quadrupole on a HEALPix grid "
            f"(nside={NSIDE}) for OFL region mapping"
        ),
        field=FieldFactoryConfig(components=[[dipole, quadrupole]]),
        grid=GridFactoryConfig(grids=[HealpixGridConfig(nside=NSIDE)]),
        tracing=TracingFactoryConfig(
            max_integration_radius=MAX_INTEGRATION_RADIUS,
            save_results_to_disk=True,
            max_memory_mb=100.0,
            max_step=1.0,
            dt=0.05,
            verify_by_reversing=True,
            progress_bar=True,
            n_cores=n_cores,
            field_backend="numba",
            solver_backend="cyrk",
        ),
        template=FieldLineTemplate.template_OFL_subset(),
    )


def main() -> None:
    """Create the series on disk and trace every experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the series data will be saved",
    )
    parser.add_argument(
        "--n-cores",
        required=True,
        type=int,
        help="Number of cores used to trace field lines",
    )
    args = parser.parse_args()

    series = build_series(args.output_dir, args.n_cores)
    experiments = series.create_experiment_series(
        save_configs=True,
        delete_existing_configs=True,
    )
    print(f"Created {len(experiments)} experiments")
    print(f"output_dir={series.output_dir}")
    print(f"n_cores={args.n_cores}")

    series_result = run_experiment_series(
        series_config=series,
        save_individual_configs=True,
        save_grids=True,
        continue_on_failure=True,
        series_deletion_strategy=DeletionStrategy.EXCEPT_CONFIGS,
    )
    series_result.print_summary()


if __name__ == "__main__":
    main()
