"""Centralized filesystem path and filename policy for experiment artifacts."""

from pathlib import Path
from typing import Optional


CONFIGS_DIR = "configs"
GRIDS_DIR = "grids"
LINES_DIR = "lines"
LOGS_DIR = "logs"


def series_dir(output_dir: Optional[str], series_name: str) -> Path:
    """Return root directory for a series."""
    base = Path(output_dir) if output_dir is not None else Path(".")
    return base / series_name


def configs_dir(output_dir: Optional[str], series_name: str) -> Path:
    """Return configs directory path."""
    return series_dir(output_dir, series_name) / CONFIGS_DIR


def grids_dir(output_dir: Optional[str], series_name: str) -> Path:
    """Return grids directory path."""
    return series_dir(output_dir, series_name) / GRIDS_DIR


def lines_dir(output_dir: Optional[str], series_name: str) -> Path:
    """Return lines directory path."""
    return series_dir(output_dir, series_name) / LINES_DIR


def logs_dir(output_dir: Optional[str], series_name: str) -> Path:
    """Return logs directory path."""
    return series_dir(output_dir, series_name) / LOGS_DIR


def series_config_filename(series_name: str) -> str:
    """Return canonical series config filename."""
    return f"{series_name}_config.yaml"


def experiment_config_filename(series_name: str, experiment_number: int) -> str:
    """Return canonical per-experiment config filename."""
    return f"{series_name}_{experiment_number:03d}.yaml"


def subset_experiment_config_filename(series_name: str, experiment_number: int) -> str:
    """Return canonical subset experiment config filename."""
    return f"{series_name}_{experiment_number:03d}_subset.yaml"


def results_summary_filename(series_name: str) -> str:
    """Return canonical summary filename."""
    return f"{series_name}_results_summary.yaml"
