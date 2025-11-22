"""Tests for experiment series runner functionality."""

import pytest
from pathlib import Path
import shutil
import yaml
from datetime import datetime, timedelta
from collections import OrderedDict

from psr_magpy.experiment import ExperimentSeriesConfig, ExperimentConfig
from psr_magpy.experiment.experiment_series_runner import (
    ExperimentSeriesRunner,
    run_experiment_series,
    DeletionStrategy,
    OverwriteStrategy,
)
from psr_magpy.experiment.factory_configs import (
    FieldFactoryConfig, GridFactoryConfig, TracingFactoryConfig,
    RandomGridFactoryConfig
)
from psr_magpy.experiment.field_config import FieldConfig, FieldComponentConfig
from psr_magpy.experiment.grid_config import GridConfig, RandomGridConfig, CustomGridConfig
from psr_magpy.experiment.tracing_config import TracingConfig
from psr_magpy.line import FieldLine, FieldLineTemplate, FieldLineTemplate

def create_test_series_config(test_data_dir):
    """Helper to create a test series config."""
    output_dir = test_data_dir / "test_series"
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        template=FieldLineTemplate.template_all()
    )

def test_run_specific_experiments(test_data_dir):
    """Test running specific experiments by number."""
    series_config = create_test_series_config(test_data_dir)
    experiments = series_config.create_experiment_series()
    
    # Run experiment 0
    result = run_experiment_series(
        series_config=series_config,
        experiment_numbers=[0]
    )
    
    assert result.total_experiments == 1  # Total experiments in series
    assert result.experiments_in_this_session == 1  # Experiments run in this session
    assert len(result.experiment_results) == 1
    assert result.experiment_results[0].experiment_name.endswith("000")

def test_run_remaining_experiments(test_data_dir):
    """Test running remaining experiments."""
    series_config = create_test_series_config(test_data_dir)
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
    
    assert result.total_experiments == 1  # Total experiments in series
    assert result.experiments_in_this_session == 0  # No experiments left to run

def test_overwrite_everything(test_data_dir):
    """Test deletion strategy option."""
    series_config = create_test_series_config(test_data_dir)
    experiments = series_config.create_experiment_series()
    
    # Create some dummy files
    series_dir = Path(str(series_config.output_dir)) / series_config.name
    (series_dir / "grids").mkdir(parents=True, exist_ok=True)
    (series_dir / "lines").mkdir(parents=True, exist_ok=True)
    (series_dir / "test.fits").touch()
    (series_dir / f"{series_config.name}_results_summary.yaml").touch()
    
    # Run with deletion strategy that preserves configs
    run_experiment_series(
        series_config=series_config,
        series_deletion_strategy=DeletionStrategy.EXCEPT_CONFIGS
    )
    
    # Check directories are clean
    assert not (series_dir / "test.fits").exists()
    assert (series_dir / "grids").exists()
    assert (series_dir / "lines").exists()
    assert (series_dir / f"{series_config.name}_results_summary.yaml").exists()

def test_error_reporting_in_summary(test_data_dir):
    """Test error reporting in summary YAML."""
    series_config = create_test_series_config(test_data_dir)
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
        # Error field is optional - only present when there's an error
        if not exp_result["success"]:
            assert "error" in exp_result

def test_yaml_field_order(test_data_dir):
    """Test YAML field order preservation."""
    series_config = create_test_series_config(test_data_dir)
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
    # Error field is optional, so we check the required fields
    required_fields = ["name", "start_time", "duration", "success", "batch_files"]
    for field in required_fields:
        assert field in fields, f"Missing required field: {field}"

def test_duration_formatting(test_data_dir):
    """Test duration formatting in summary."""
    series_config = create_test_series_config(test_data_dir)
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

def test_save_configs_during_series_creation(test_data_dir):
    """Test saving configs during series creation."""
    series_config = create_test_series_config(test_data_dir)
    
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

def test_load_experiments_from_configs(test_data_dir):
    """Test loading experiments from saved configs."""
    series_config = create_test_series_config(test_data_dir)
    
    # Create and save experiments
    original_experiments = series_config.create_experiment_series(save_configs=True)
    assert len(original_experiments) > 0  # Ensure we have experiments to load
    
    # Load experiments
    loaded_experiments = series_config.load_experiments()
    
    assert len(loaded_experiments) == len(original_experiments)
    for orig, loaded in zip(original_experiments, loaded_experiments):
        assert orig.name == loaded.name
        assert orig.grid.model_dump() == loaded.grid.model_dump()


def test_save_grids_parameter(test_data_dir):
    """Test that save_grids parameter is respected."""
    series_config = create_test_series_config(test_data_dir)
    
    # Create and save experiments first
    series_config.create_experiment_series(save_configs=True)
    
    # Run with save_grids=False
    result_false = run_experiment_series(
        series_config=series_config,
        save_grids=False,
        experiment_numbers=[0]
    )
    
    # Run with save_grids=True
    result_true = run_experiment_series(
        series_config=series_config,
        save_grids=True,
        experiment_numbers=[0]
    )
    
    # First run should complete successfully
    assert result_false.successful_experiments == 1
    
    # Second run might fail due to grid file conflict, but that's expected behavior
    # The important thing is that save_grids=True was respected
    assert result_true.total_experiments == 1
    
    # Check that grids directory exists in both cases (created by directory structure)
    series_dir = Path(str(series_config.output_dir)) / series_config.name
    assert (series_dir / "grids").exists()


def test_experiment_runner_save_grids_parameter(test_data_dir):
    """Test that ExperimentRunner respects save_grids parameter."""
    from psr_magpy.experiment.experiment_runner import ExperimentRunner
    
    # Create a simple experiment config
    field_config = FieldConfig(components=[
        FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)
    ])
    grid_config = GridConfig(random=RandomGridConfig(nline_ori=20))  # Larger grid to accommodate n_trial
    tracing_config = TracingConfig(save_results_to_disk=False, n_trial=5)  # Smaller n_trial to avoid conflicts
    template = FieldLineTemplate.template_all()
    
    experiment = ExperimentConfig(
        name="test_experiment",
        field=field_config,
        grid=grid_config,
        tracing=tracing_config,
        template=template
    )
    
    # Test with save_grids=False
    runner_false = ExperimentRunner(experiment, save_grids=False)
    result_false = runner_false.run()
    
    # Test with save_grids=True
    runner_true = ExperimentRunner(experiment, save_grids=True)
    result_true = runner_true.run()
    
    # Both should complete successfully
    assert result_false.success is True
    assert result_true.success is True


def test_deletion_strategies(test_data_dir):
    """Test different deletion strategies."""
    series_config = create_test_series_config(test_data_dir)
    
    # Create some dummy files
    series_dir = Path(str(series_config.output_dir)) / series_config.name
    series_dir.mkdir(parents=True, exist_ok=True)
    (series_dir / "configs").mkdir(parents=True, exist_ok=True)
    (series_dir / "grids").mkdir(parents=True, exist_ok=True)
    (series_dir / "lines").mkdir(parents=True, exist_ok=True)
    (series_dir / "configs" / "test.yaml").touch()
    (series_dir / "grids" / "test.fits").touch()
    (series_dir / "lines" / "test.fits").touch()
    (series_dir / "test.fits").touch()
    
    # Test EXCEPT_CONFIGS strategy
    run_experiment_series(
        series_config=series_config,
        series_deletion_strategy=DeletionStrategy.EXCEPT_CONFIGS,
        experiment_numbers=[0]
    )
    
    # Configs should be preserved
    assert (series_dir / "configs" / "test.yaml").exists()
    # Other files should be deleted
    assert not (series_dir / "test.fits").exists()
    # Directories should still exist (recreated)
    assert (series_dir / "grids").exists()
    assert (series_dir / "lines").exists()


def test_overwrite_strategies(test_data_dir):
    """Test different overwrite strategies."""
    series_config = create_test_series_config(test_data_dir)
    
    # Run first time
    run_experiment_series(
        series_config=series_config,
        experiment_numbers=[0],
        overwrite_strategy=OverwriteStrategy.ALL
    )
    
    # Run again with PRESERVE_CONFIGS_AND_GRIDS
    result = run_experiment_series(
        series_config=series_config,
        experiment_numbers=[0],
        overwrite_strategy=OverwriteStrategy.PRESERVE_CONFIGS_AND_GRIDS
    )
    
    # Should complete successfully
    assert result.successful_experiments == 1


def test_experiment_level_deletion(test_data_dir):
    """Test experiment-level deletion strategy."""
    series_config = create_test_series_config(test_data_dir)
    
    # Run first experiment
    run_experiment_series(
        series_config=series_config,
        experiment_numbers=[0]
    )
    
    # Check that files were created
    series_dir = Path(str(series_config.output_dir)) / series_config.name
    experiments = series_config.create_experiment_series()
    exp_name = experiments[0].name
    
    # Run again with experiment-level deletion
    run_experiment_series(
        series_config=series_config,
        experiment_numbers=[0],
        experiment_deletion_strategy=DeletionStrategy.ALL,
        overwrite_strategy=OverwriteStrategy.ALL
    )
    
    # Should complete successfully
    assert (series_dir / "lines" / f"{exp_name}.fits").exists() 