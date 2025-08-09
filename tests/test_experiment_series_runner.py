"""Tests for experiment series runner functionality."""

import pytest
from pathlib import Path
import shutil
import yaml
from datetime import datetime, timedelta
from collections import OrderedDict

from psr_magpy.experiment import ExperimentSeriesConfig, ExperimentConfig
from psr_magpy.experiment.experiment_series_runner import ExperimentSeriesRunner, run_experiment_series
from psr_magpy.experiment.factory_configs import (
    FieldFactoryConfig, GridFactoryConfig, TracingFactoryConfig,
    RandomGridFactoryConfig
)
from psr_magpy.experiment.experiment_config import (
    FieldConfig, GridConfig, TracingConfig,
    RandomGridConfig, CustomGridConfig
)
from psr_magpy.line import FieldLine

def create_test_series_config(tmp_path):
    """Helper to create a test series config."""
    output_dir = tmp_path / "test_series"
    output_dir.mkdir(parents=True)
    
    return ExperimentSeriesConfig(
        name="test_series",
        output_dir=str(output_dir),
        field=FieldFactoryConfig(
            components=[{
                "type": "dipole",
                "moment_type": "dipole",
                "moment_orientation_latitude_degree": [45.0, 90.0]
            }]
        ),
        grid=GridFactoryConfig(
            grids=[RandomGridFactoryConfig(nline_ori=10)]
        ),
        tracing=TracingFactoryConfig(),
        template=FieldLine.template_all()
    )

def test_run_specific_experiments(tmp_path):
    """Test running specific experiments by number."""
    series_config = create_test_series_config(tmp_path)
    experiments = series_config.create_experiment_series()
    
    # Run experiment 0
    result = run_experiment_series(
        series_config=series_config,
        experiment_numbers=[0]
    )
    
    assert result.total_experiments == 1
    assert len(result.experiment_results) == 1
    assert result.experiment_results[0].experiment_name.endswith("000")

def test_run_remaining_experiments(tmp_path):
    """Test running remaining experiments."""
    series_config = create_test_series_config(tmp_path)
    experiments = series_config.create_experiment_series()
    
    # Run first experiment
    run_experiment_series(
        series_config=series_config,
        experiment_numbers=[0]
    )
    
    # Run remaining - should be none left
    result = run_experiment_series(
        series_config=series_config,
        run_remaining=True
    )
    
    assert result.total_experiments == 0  # No experiments left

def test_overwrite_everything(tmp_path):
    """Test overwrite_everything option."""
    series_config = create_test_series_config(tmp_path)
    experiments = series_config.create_experiment_series()
    
    # Create some dummy files
    series_dir = Path(str(series_config.output_dir)) / series_config.name
    (series_dir / "grids").mkdir(parents=True, exist_ok=True)
    (series_dir / "lines").mkdir(parents=True, exist_ok=True)
    (series_dir / "test.fits").touch()
    (series_dir / f"{series_config.name}_results_summary.yaml").touch()
    
    # Run with overwrite_everything
    run_experiment_series(
        series_config=series_config,
        overwrite_everything=True
    )
    
    # Check directories are clean
    assert not (series_dir / "test.fits").exists()
    assert (series_dir / "grids").exists()
    assert (series_dir / "lines").exists()
    assert (series_dir / f"{series_config.name}_results_summary.yaml").exists()

def test_error_reporting_in_summary(tmp_path):
    """Test error reporting in summary YAML."""
    series_config = create_test_series_config(tmp_path)
    experiments = series_config.create_experiment_series()
    
    # Run experiments
    result = run_experiment_series(series_config=series_config)
    
    # Read summary YAML
    summary_path = Path(str(series_config.output_dir)) / series_config.name / f"{series_config.name}_results_summary.yaml"
    with open(summary_path) as f:
        summary = yaml.safe_load(f)
    
    # Check structure
    assert "experiment_results" in summary
    for exp_result in summary["experiment_results"]:
        assert "name" in exp_result
        assert "start_time" in exp_result
        assert "duration" in exp_result
        assert "success" in exp_result
        assert "error" in exp_result

def test_yaml_field_order(tmp_path):
    """Test YAML field order preservation."""
    series_config = create_test_series_config(tmp_path)
    experiments = series_config.create_experiment_series()
    
    # Run experiments
    run_experiment_series(series_config=series_config)
    
    # Read summary YAML preserving order
    summary_path = Path(str(series_config.output_dir)) / series_config.name / f"{series_config.name}_results_summary.yaml"
    with open(summary_path) as f:
        summary = yaml.safe_load(f)
    
    # Check order of fields in experiment results
    first_result = summary["experiment_results"][0]
    fields = list(first_result.keys())
    assert fields == ["name", "start_time", "duration", "success", "error", "batch_files"]

def test_duration_formatting(tmp_path):
    """Test duration formatting in summary."""
    series_config = create_test_series_config(tmp_path)
    experiments = series_config.create_experiment_series()
    
    # Run experiments
    result = run_experiment_series(series_config=series_config)
    
    # Read summary YAML
    summary_path = Path(str(series_config.output_dir)) / series_config.name / f"{series_config.name}_results_summary.yaml"
    with open(summary_path) as f:
        summary = yaml.safe_load(f)
    
    # Check duration format
    for exp_result in summary["experiment_results"]:
        duration = exp_result["duration"]
        assert any(unit in duration for unit in ["s", "m", "h", "d"])

def test_save_configs_during_series_creation(tmp_path):
    """Test saving configs during series creation."""
    series_config = create_test_series_config(tmp_path)
    
    # Create experiments with config saving
    experiments = series_config.create_experiment_series(save_configs=True)
    
    # Check configs directory
    configs_dir = Path(str(series_config.output_dir)) / series_config.name / "configs"
    assert configs_dir.exists()
    
    # Check series config
    assert (configs_dir / "test_series_config.yaml").exists()
    
    # Check individual configs
    for experiment in experiments:
        assert (configs_dir / f"{experiment.name}.yaml").exists()

def test_load_experiments_from_configs(tmp_path):
    """Test loading experiments from saved configs."""
    series_config = create_test_series_config(tmp_path)
    
    # Create and save experiments
    original_experiments = series_config.create_experiment_series(save_configs=True)
    assert len(original_experiments) > 0  # Ensure we have experiments to load
    
    # Load experiments
    loaded_experiments = series_config.load_experiments()
    
    assert len(loaded_experiments) == len(original_experiments)
    for orig, loaded in zip(original_experiments, loaded_experiments):
        assert orig.name == loaded.name
        assert orig.grid.model_dump() == loaded.grid.model_dump() 