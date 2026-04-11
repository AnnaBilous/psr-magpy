"""Find experiment indices from saved configs by dipole colatitude and quadrupole strength."""

from __future__ import annotations

import math
import re
from typing import List, Optional

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.field.components import MultipoleType
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)

_MATCH_RTOL = 1e-4
_MATCH_ATOL = 1e-4


def experiment_matches_thD_fQ(
    exp: ExperimentConfig,
    thD: float,
    fQ: float,
) -> bool:
    """True if the first dipole has colatitude ``thD`` and first quadrupole has ``B_magnitude`` ``fQ``.

    Floats are compared with ``math.isclose`` using module constants
    ``_MATCH_RTOL`` and ``_MATCH_ATOL``.
    """
    dip = next(
        (
            c
            for c in exp.field.components
            if c.moment_type == MultipoleType.DIPOLE.value
        ),
        None,
    )
    quad = next(
        (
            c
            for c in exp.field.components
            if c.moment_type == MultipoleType.QUADRUPOLE.value
        ),
        None,
    )
    if dip is None or quad is None:
        return False
    return math.isclose(
        float(dip.moment_orientation_latitude_degree),
        float(thD),
        rel_tol=_MATCH_RTOL,
        abs_tol=_MATCH_ATOL,
    ) and math.isclose(
        float(quad.B_magnitude),
        float(fQ),
        rel_tol=_MATCH_RTOL,
        abs_tol=_MATCH_ATOL,
    )


def infer_experiment_count(series: ExperimentSeriesConfig) -> int:
    """Infer how many experiments exist (indices ``0 .. N-1``) from summary or configs."""
    try:
        return series.get_total_experiments_from_summary()
    except (FileNotFoundError, KeyError):
        pass

    configs_dir = series.get_configs_dir()
    if not configs_dir.is_dir():
        error_msg = f"Configs directory does not exist: {configs_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    pattern = re.compile(rf"^{re.escape(series.name)}_(\d{{3}})(?:_subset)?\.yaml$")
    indices: List[int] = []
    for path in configs_dir.glob("*.yaml"):
        if path.name.endswith("_config.yaml"):
            continue
        m = pattern.match(path.name)
        if m:
            indices.append(int(m.group(1)))

    if not indices:
        error_msg = f"No experiment YAML files matched in {configs_dir}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return max(indices) + 1


def find_experiment_numbers_by_thD_fQ(
    series: ExperimentSeriesConfig,
    thD: float,
    fQ: float,
    *,
    n_experiments: Optional[int] = None,
) -> List[int]:
    """Return all experiment numbers whose YAML matches ``thD`` and ``fQ`` on dipole / quadrupole."""
    n = n_experiments if n_experiments is not None else infer_experiment_count(series)

    matches: List[int] = []
    for experiment_number in range(n):
        path = series.get_experiment_config_path(experiment_number)
        if not path.is_file():
            continue
        exp = ExperimentConfig.load(path)
        if experiment_matches_thD_fQ(exp, thD, fQ):
            matches.append(experiment_number)

    return matches
