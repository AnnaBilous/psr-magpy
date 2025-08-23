#!/usr/bin/env python3
"""
Tests for in-memory experiment series runner functionality.
"""

import pytest
import tempfile
from pathlib import Path

from psr_magpy.experiment import (
    ExperimentSeriesConfig,
    FieldConfig,
    FieldComponentConfig,
    GridConfig,
    SphericalGridConfig,
    TracingConfig,
    run_experiment_series_in_memory,
)
from psr_magpy.line.field_line import FieldLine


class TestInMemoryExperimentSeriesRunner:
    """Test in-memory experiment series runner functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create simple field and grid for testing
        self.field_config = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)
        ])
        self.grid_config = GridConfig(spherical=SphericalGridConfig(nlon=6, nlat=4))
        self.tracing_config = TracingConfig()
        self.template = FieldLine.template_OFL_calc()
    
    def test_basic_in_memory_execution(self):
        """Test basic in-memory experiment series execution."""
        series_config = ExperimentSeriesConfig(
            name="test_series",
            field=self.field_config,
            grid=self.grid_config,
            tracing=self.tracing_config,
            template=self.template,
        )
        
        result = run_experiment_series_in_memory(
            series_config=series_config,
            keep_results=True,
            max_memory_mb=100.0,
            save_individual_configs=False,
            save_grids=False,
        )
        
        # Check basic result structure
        assert result.series_name == "test_series"
        assert result.total_experiments == 1
        assert result.successful_experiments == 1
        assert result.failed_experiments == 0
        assert len(result.experiment_results) == 1
        
        # Check that results are in memory
        experiment_result = result.experiment_results[0]
        assert experiment_result.success
        assert experiment_result.results is not None
        assert experiment_result.results.n_field_lines > 0
    
    def test_memory_limit_handling(self):
        """Test memory limit handling."""
        series_config = ExperimentSeriesConfig(
            name="test_series",
            field=self.field_config,
            grid=self.grid_config,
            tracing=self.tracing_config,
            template=self.template,
        )
        
        # Run with very low memory limit
        result = run_experiment_series_in_memory(
            series_config=series_config,
            keep_results=True,
            max_memory_mb=0.1,  # Very low limit
            save_individual_configs=False,
            save_grids=False,
        )
        
        # Should still complete successfully
        assert result.successful_experiments == 1
        assert result.failed_experiments == 0
    
    def test_keep_results_false(self):
        """Test behavior when keep_results=False."""
        series_config = ExperimentSeriesConfig(
            name="test_series",
            field=self.field_config,
            grid=self.grid_config,
            tracing=self.tracing_config,
            template=self.template,
        )
        
        result = run_experiment_series_in_memory(
            series_config=series_config,
            keep_results=False,
            max_memory_mb=100.0,
            save_individual_configs=False,
            save_grids=False,
        )
        
        # Results should be cleared from memory
        experiment_result = result.experiment_results[0]
        assert experiment_result.success
        assert experiment_result.results is None
    
    def test_experiment_filtering(self):
        """Test filtering experiments by number."""
        # Create series with multiple experiments using factory configs
        from psr_magpy.experiment import FieldFactoryConfig, FieldComponentFactoryConfig
        
        field_factory = FieldFactoryConfig(components=[
            FieldComponentFactoryConfig(moment_type="dipole", B_magnitude=[1.0, 2.0])
        ])
        
        series_config = ExperimentSeriesConfig(
            name="test_series",
            field=field_factory,
            grid=self.grid_config,
            tracing=self.tracing_config,
            template=self.template,
        )
        
        # Check how many experiments are created
        experiments = series_config.create_experiment_series()
        total_experiments = len(experiments)
        
        # Run only first experiment
        result = run_experiment_series_in_memory(
            series_config=series_config,
            experiment_numbers=[0],
            keep_results=True,
            max_memory_mb=100.0,
            save_individual_configs=False,
            save_grids=False,
        )
        
        assert result.total_experiments == total_experiments  # Total in series
        assert result.experiments_in_this_session == 1  # Actually run
        assert result.successful_experiments == 1
    
    def test_error_handling(self):
        """Test error handling in in-memory runner."""
        # Create invalid configuration to trigger error
        invalid_field_config = FieldConfig(components=[
            FieldComponentConfig(moment_type="invalid_type", B_magnitude=1.0)
        ])
        
        series_config = ExperimentSeriesConfig(
            name="test_series",
            field=invalid_field_config,
            grid=self.grid_config,
            tracing=self.tracing_config,
            template=self.template,
        )
        
        # Should handle error gracefully
        result = run_experiment_series_in_memory(
            series_config=series_config,
            keep_results=True,
            max_memory_mb=100.0,
            continue_on_failure=True,
            save_individual_configs=False,
            save_grids=False,
        )
        
        assert result.failed_experiments == 1
        assert result.successful_experiments == 0
    
    def test_memory_estimation(self):
        """Test that memory estimation works correctly."""
        series_config = ExperimentSeriesConfig(
            name="test_series",
            field=self.field_config,
            grid=self.grid_config,
            tracing=self.tracing_config,
            template=self.template,
        )
        
        result = run_experiment_series_in_memory(
            series_config=series_config,
            keep_results=True,
            max_memory_mb=100.0,
            save_individual_configs=False,
            save_grids=False,
        )
        
        # Check that memory estimation is available
        experiment_result = result.experiment_results[0]
        if experiment_result.results is not None:
            memory_size = experiment_result.results.estimate_size()
            assert memory_size > 0
            assert memory_size < 100 * 1024 * 1024  # Should be less than 100 MB
