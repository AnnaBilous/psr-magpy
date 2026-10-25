"""Per-footpoint maximum period for which a traced line is still open.

Reads successful experiments from ``<output_dir>/<series>/<series>_results_summary.yaml``.

``R_RADIAL_MAX`` is the spherical radius at which tracing stopped. Both
coordinates lie inside that sphere, so each is capped by it, and they limit
the light cylinder differently:

* cylindrical radius enters the open-line test directly, so its ceiling is
  ``R_RADIAL_MAX``;
* height enters as ``|z| > h_LC * R_LC``, so its ceiling is
  ``R_RADIAL_MAX / h_LC``.

The smallest period in the survey is the one whose light cylinder sits on the
star, ``R_LC = R_NS``. A line open only inside that cylinder is stored as NaN.
The map value is the larger period still open under those two limits.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import yaml
from astropy.io import fits

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.infrastructure.path_policy import (
    results_summary_filename,
    series_config_filename,
)
from psr_magpy.experiment.services.selection import parse_experiment_number
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.physical_conversions import get_P_ms

logger = get_logger(__name__)

DEFAULT_SERIES_NAME = "healpix_nside_0128_dip_quad_ofl"
R_NS = 1.0
R_RADIAL_MAX = 500.0
H_LC_COEFF = 3.0**1.5 / 2.0


def max_open_radius(r_cyl_max: np.ndarray, z_abs_max: np.ndarray) -> np.ndarray:
    """Least upper bound of ``R_LC`` for which the line is open.

    ``r_cyl_max`` and ``|z_abs_max|`` are capped by the radial tracing limit
    before they are turned into light-cylinder radii. The cylinder test uses
    the cylindrical radius as-is. The height test uses ``|z| / h_LC``.
    """
    r_cyl = np.minimum(np.asarray(r_cyl_max, dtype=np.float64), R_RADIAL_MAX)
    z_abs = np.minimum(np.abs(np.asarray(z_abs_max, dtype=np.float64)), R_RADIAL_MAX)
    return np.maximum(r_cyl, z_abs / H_LC_COEFF)


def max_open_period(r_cyl_max: np.ndarray, z_abs_max: np.ndarray) -> np.ndarray:
    """Maximum open period in milliseconds.

    Periods below ``P(R_LC = R_NS)`` are outside the survey. NaN when the line
    stays closed for every light cylinder at or outside the star.
    """
    radius = max_open_radius(r_cyl_max, z_abs_max)
    period = np.asarray(get_P_ms(radius), dtype=np.float64)
    return np.where(radius > R_NS, period, np.nan)


def completed_experiment_numbers(summary_path: Path) -> List[int]:
    """Return successful experiment indices from a series results summary."""
    if not summary_path.is_file():
        error_msg = f"Results summary not found: {summary_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    with summary_path.open() as handle:
        summary = yaml.safe_load(handle) or {}

    numbers: List[int] = []
    for entry in summary.get("experiment_results", []):
        if not entry.get("success"):
            continue
        number = parse_experiment_number(str(entry.get("name", "")))
        if number is None:
            error_msg = f"Cannot parse experiment number from {entry.get('name')!r}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        numbers.append(number)
    return sorted(set(numbers))


def write_max_p(output_path: Path, period_by_footpoint: np.ndarray) -> None:
    """Write one float64 period per footpoint as a FITS binary table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    period = np.asarray(period_by_footpoint, dtype=np.float64)
    columns = [
        fits.Column(name="MAX_P", format="D", array=period),
        fits.Column(
            name="GRID_INDICES",
            format="J",
            array=np.arange(period.size, dtype=np.int32),
        ),
    ]
    fits.BinTableHDU.from_columns(columns).writeto(output_path, overwrite=True)


def max_p_for_experiment(
    series: ExperimentSeriesConfig, experiment_number: int, output_dir: Path
) -> Path:
    """Load one experiment's lines and write its maximum-period map."""
    experiment = series.load_experiment_config(experiment_number)
    lines = experiment.load_combined_batch_data(
        FieldLineTemplate.template_OFL_subset(),
        str(series.get_lines_dir()),
    )
    if lines is None or lines.n_field_lines == 0:
        error_msg = f"No field lines for experiment {experiment_number}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    grid = experiment.grid.create_grid()
    if grid is None:
        error_msg = f"No grid for experiment {experiment_number}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    period = np.full(grid.nline_ori, np.nan, dtype=np.float64)
    footpoint_ids = np.asarray(lines.footpoint_id, dtype=int)
    values = max_open_period(lines.r_cyl_max, lines.z_abs_max)
    period[footpoint_ids] = values

    path = output_dir / f"max_P_{experiment_number:06d}.fits"
    write_max_p(path, period)
    logger.info(
        "Experiment %d: wrote %s (%d open footpoints)",
        experiment_number,
        path,
        int(np.sum(np.isfinite(period))),
    )
    return path


def main() -> None:
    """Write a maximum-period map for every successful experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Parent directory that holds the series (same as the mapping script)",
    )
    parser.add_argument(
        "--series-name",
        default=DEFAULT_SERIES_NAME,
        help="Series directory name",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    series_dir = output_dir / args.series_name
    summary_path = series_dir / results_summary_filename(args.series_name)
    numbers = completed_experiment_numbers(summary_path)
    print(f"successful_experiments={len(numbers)}")

    config_path = series_dir / "configs" / series_config_filename(args.series_name)
    if not config_path.is_file():
        error_msg = f"Series config not found: {config_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    series = ExperimentSeriesConfig.load(config_path)
    series.output_dir = str(output_dir)
    product_dir = series_dir / "data_products" / "max_P"

    for number in numbers:
        max_p_for_experiment(series, number, product_dir)

    print(f"output={product_dir}")


if __name__ == "__main__":
    main()
