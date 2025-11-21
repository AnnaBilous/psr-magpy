"""Workflow for OFL analysis from experiment series."""

from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyIOError

logger = get_logger(__name__, console_format_brief=True)


def analyze_ofl_from_series(
    series_config: ExperimentSeriesConfig,
    R_LC: float,
    h_LC_coeff: float = 3**1.5 / 2.0,
    experiment_numbers: Optional[List[int]] = None,
    template: Optional[FieldLineTemplate] = None,
) -> Dict[int, Dict[str, Any]]:
    """Analyze OFL footpoints from an experiment series.

    Loads OFL data from each experiment and extracts footpoint statistics.

    Parameters
    ----------
    series_config : ExperimentSeriesConfig
        Experiment series config.
    R_LC : float
        Light cylinder radius for OFL calculation.
    h_LC_coeff : float, optional
        Height coefficient for OFL calculation. Default is 3^1.5 / 2.0.
    experiment_numbers : Optional[List[int]], optional
        Optional list of experiment numbers to analyze. Default is None.
    template : Optional[FieldLineTemplate], optional
        Optional FieldLineTemplate. Defaults to template_OFL_subset.

    Returns
    -------
    Dict[int, Dict[str, Any]]
        Dictionary mapping experiment numbers to OFL statistics:
        {
            exp_num: {
                'ofl_count': int,
                'total_count': int,
                'ofl_fraction': float,
                'experiment_name': str
            }
        }
    """
    if template is None:
        template = FieldLineTemplate.template_OFL_subset()

    # Verify directories
    status = series_config.verify_directories(create_if_missing=False)
    if not status["lines"]["exists"]:
        raise PSRMagPyIOError("Lines directory must exist")

    # Get experiment numbers
    if experiment_numbers is None:
        experiments = series_config.create_experiment_series(save_configs=False)
        experiment_numbers = list(range(len(experiments)))

    results = {}
    lines_dir = series_config.get_lines_dir()

    for exp_num in experiment_numbers:
        try:
            experiment = series_config.load_experiment_config(exp_num)
            logger.debug(f"Analyzing OFL for experiment {exp_num}: {experiment.name}")

            # Load OFL data
            ofl_data = experiment.load_combined_batch_data(template, str(lines_dir))
            if ofl_data is None:
                logger.warning(f"No data found for experiment {exp_num}, skipping...")
                continue

            # Get OFL footpoints
            ofl_footpoints = ofl_data.get_OFL_footpoints(R_LC, h_LC_coeff=h_LC_coeff)
            total_count = len(ofl_data.footpoint_id) if hasattr(ofl_data, 'footpoint_id') else 0

            results[exp_num] = {
                "ofl_count": len(ofl_footpoints),
                "total_count": total_count,
                "ofl_fraction": len(ofl_footpoints) / total_count if total_count > 0 else 0.0,
                "experiment_name": experiment.name,
            }

            logger.info(
                f"Experiment {exp_num}: {len(ofl_footpoints)}/{total_count} OFL footpoints "
                f"({len(ofl_footpoints)/total_count*100:.1f}%)"
            )

        except Exception as e:
            error_msg = f"Failed to analyze experiment {exp_num}: {e}"
            logger.error(error_msg)
            continue

    return results

