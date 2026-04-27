"""Repository for experiment and series configuration persistence."""

from pathlib import Path
from typing import List, Optional

from psr_magpy.io import load_config, save_config

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.infrastructure.path_policy import (
    configs_dir,
    experiment_config_filename,
    series_config_filename,
    subset_experiment_config_filename,
)


class ConfigRepository:
    """Persistence adapter for series and experiment config YAML files."""

    def save_series_config(
        self,
        series_config: ExperimentSeriesConfig,
        *,
        overwrite: bool = True,
    ) -> Path:
        """Save series config and return written path."""
        cfg_dir = configs_dir(series_config.output_dir, series_config.name)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        path = cfg_dir / series_config_filename(series_config.name)
        save_config(series_config, path, overwrite=overwrite)
        return path

    def load_series_config(self, config_file: Path) -> ExperimentSeriesConfig:
        """Load series config from file."""
        return load_config(ExperimentSeriesConfig, config_file)

    def save_experiment_config(
        self,
        series_config: ExperimentSeriesConfig,
        experiment: ExperimentConfig,
        *,
        overwrite: bool = True,
        subset: bool = False,
    ) -> Path:
        """Save one experiment config and return written path."""
        cfg_dir = configs_dir(series_config.output_dir, series_config.name)
        cfg_dir.mkdir(parents=True, exist_ok=True)

        exp_number = self._parse_experiment_number(experiment.name)
        if exp_number is None:
            # Fallback to current name to preserve legacy behavior for unusual names.
            filename = f"{experiment.name}.yaml"
        elif subset:
            filename = subset_experiment_config_filename(series_config.name, exp_number)
        else:
            filename = experiment_config_filename(series_config.name, exp_number)

        path = cfg_dir / filename
        save_config(experiment, path, overwrite=overwrite)
        return path

    def load_experiment_config(
        self,
        series_config: ExperimentSeriesConfig,
        experiment_number: int,
        *,
        prefer_subset: bool = True,
    ) -> ExperimentConfig:
        """Load one experiment config by index."""
        cfg_dir = configs_dir(series_config.output_dir, series_config.name)
        regular = cfg_dir / experiment_config_filename(series_config.name, experiment_number)
        subset = cfg_dir / subset_experiment_config_filename(
            series_config.name, experiment_number
        )
        path = subset if (prefer_subset and subset.exists()) else regular
        return load_config(ExperimentConfig, path)

    def list_experiment_config_files(
        self,
        series_config: ExperimentSeriesConfig,
        *,
        prefer_subset: bool = False,
    ) -> List[Path]:
        """List experiment config files in canonical order."""
        cfg_dir = configs_dir(series_config.output_dir, series_config.name)
        if not cfg_dir.exists():
            return []

        all_files = [
            p for p in cfg_dir.glob("*.yaml") if p.name != series_config_filename(series_config.name)
        ]
        subset_files = sorted([p for p in all_files if p.name.endswith("_subset.yaml")])
        regular_files = sorted([p for p in all_files if not p.name.endswith("_subset.yaml")])

        if prefer_subset:
            return subset_files if subset_files else regular_files
        return regular_files if regular_files else subset_files

    @staticmethod
    def _parse_experiment_number(experiment_name: str) -> Optional[int]:
        """Parse experiment index from ``{series}_{idx}`` naming convention."""
        try:
            parts = experiment_name.rsplit("_", 1)
            if len(parts) == 2:
                return int(parts[1])
        except (ValueError, IndexError):
            return None
        return None
