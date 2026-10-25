"""Builder service for deterministic experiment-series expansion."""

from typing import Any, Dict, List, Optional

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_metadata import create_experiment_metadata
from psr_magpy.experiment.factory_configs import (
    expand_field_configs,
    expand_grid_configs,
    expand_template_configs,
    expand_tracing_configs,
)
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class SeriesBuilder:
    """Construct experiment lists from series-level configuration inputs."""

    def build_experiments(
        self,
        *,
        series_name: str,
        field: Any,
        grid: Any,
        tracing: Any,
        template: Any,
        description: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> List[ExperimentConfig]:
        """Build all experiment combinations in deterministic order."""
        # Expand factories to concrete configs first.
        field_configs = expand_field_configs(field)
        grid_configs = expand_grid_configs(grid)
        tracing_configs = expand_tracing_configs(tracing)
        template_configs = expand_template_configs(template)

        experiments: List[ExperimentConfig] = []
        experiment_count = -1

        for field_config in field_configs:
            for grid_config in grid_configs:
                for tracing_config in tracing_configs:
                    for template_config in template_configs:
                        experiment_count += 1

                        exp_metadata = create_experiment_metadata(
                            series_name,
                            metadata or {},
                            experiment_count,
                            field_config,
                            grid_config,
                            tracing_config,
                            template_config,
                        )
                        experiment = ExperimentConfig(
                            name=f"{series_name}_{experiment_count:06d}",
                            field=field_config,
                            grid=grid_config,
                            tracing=tracing_config,
                            template=template_config,
                            description=description,
                            metadata=exp_metadata,
                        )
                        experiments.append(experiment)

        logger.debug("Built %d experiments for series %s", len(experiments), series_name)
        return experiments
