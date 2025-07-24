"""Tests for experiment series functionality."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from psr_magpy.experiment import (
    ExperimentSeriesConfig, ExperimentSeriesRunner,
    FieldFactoryConfig, GridFactoryConfig, TemplateFactoryConfig,
    FieldConfig, GridConfig, TracingConfig,
    FieldComponentConfig, SphericalGridConfig,
    get_parameter_combinations, create_experiment_metadata, generate_experiment_name
)
from psr_magpy.line.field_line import FieldLine


class TestParameterExpansion:
    """Test parameter expansion utilities."""
    
    def test_get_parameter_combinations_single_value(self):
        """Test parameter combinations with single values."""
        params = {"a": 1, "b": 2, "c": 3}
        combinations = get_parameter_combinations(params)
        
        assert len(combinations) == 1
        assert combinations[0] == {"a": 1, "b": 2, "c": 3}
    
    def test_get_parameter_combinations_with_lists(self):
        """Test parameter combinations with list values."""
        params = {"a": 1, "b": [2, 3], "c": [4, 5]}
        combinations = get_parameter_combinations(params)
        
        assert len(combinations) == 4
        expected = [
            {"a": 1, "b": 2, "c": 4},
            {"a": 1, "b": 2, "c": 5},
            {"a": 1, "b": 3, "c": 4},
            {"a": 1, "b": 3, "c": 5}
        ]
        
        for combo in expected:
            assert combo in combinations
    
    def test_get_parameter_combinations_empty(self):
        """Test parameter combinations with empty input."""
        combinations = get_parameter_combinations({})
        assert combinations == [{}]


class TestExperimentMetadata:
    """Test experiment metadata utilities."""
    
    def test_generate_experiment_name(self):
        """Test experiment name generation."""
        name = generate_experiment_name("test_series", 1)
        assert name == "test_series_001"
        
        name = generate_experiment_name("test_series", 42)
        assert name == "test_series_042"
    
    def test_create_experiment_metadata(self):
        """Test experiment metadata creation."""
        field_config = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)
        ])
        grid_config = GridConfig(spherical=SphericalGridConfig(nlon=10, nlat=5))
        tracing_config = TracingConfig()
        template = FieldLine.template_all()
        
        metadata = create_experiment_metadata(
            "test_series", {}, 1, field_config, grid_config, tracing_config, template
        )
        
        assert metadata["experiment_number"] == 1
        assert metadata["series_name"] == "test_series"
        assert metadata["field_components"] == 1
        assert metadata["grid_type"] == "spherical"


class TestFactoryConfigs:
    """Test factory configuration classes."""
    
    def test_field_factory_config_single_component(self):
        """Test FieldFactoryConfig with single component."""
        component = FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)
        factory = FieldFactoryConfig(components=[component])
        
        assert len(factory.components) == 1
        assert factory.components[0] == component
    
    def test_field_factory_config_multiple_components(self):
        """Test FieldFactoryConfig with multiple components."""
        components = [
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0),
            FieldComponentConfig(moment_type="quadrupole", B_magnitude=0.5)
        ]
        factory = FieldFactoryConfig(components=[components])
        
        assert len(factory.components) == 1
        assert factory.components[0] == components
    
    def test_grid_factory_config(self):
        """Test GridFactoryConfig."""
        spherical_config = SphericalGridConfig(nlon=10, nlat=5)
        factory = GridFactoryConfig(grids=[spherical_config])
        
        assert len(factory.grids) == 1
        assert factory.grids[0] == spherical_config
    
    def test_template_factory_config(self):
        """Test TemplateFactoryConfig."""
        template1 = FieldLine.template_all()
        template2 = FieldLine()  # Use default template
        factory = TemplateFactoryConfig(templates=[template1, template2])
        
        assert len(factory.templates) == 2
        assert factory.templates[0] == template1
        assert factory.templates[1] == template2


class TestExperimentSeriesConfig:
    """Test ExperimentSeriesConfig class."""
    
    def test_create_simple_series(self):
        """Test creating a simple experiment series."""
        field_config = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)
        ])
        grid_config = GridConfig(spherical=SphericalGridConfig(nlon=10, nlat=5))
        tracing_config = TracingConfig()
        template = FieldLine.template_all()
        
        series_config = ExperimentSeriesConfig(
            name="test_series",
            field=field_config,
            grid=grid_config,
            tracing=tracing_config,
            template=template
        )
        
        experiments = series_config.create_experiment_series()
        
        assert len(experiments) == 1
        assert experiments[0].name == "test_series_001"
        assert experiments[0].field == field_config
        assert experiments[0].grid == grid_config
        assert experiments[0].tracing == tracing_config
        assert experiments[0].template == template
    
    def test_create_series_with_factory_configs(self):
        """Test creating series with factory configurations."""
        # Field factory with multiple components
        field_factory = FieldFactoryConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0),
            FieldComponentConfig(moment_type="quadrupole", B_magnitude=0.5)
        ])
        
        # Grid factory with multiple grids
        grid_factory = GridFactoryConfig(grids=[
            SphericalGridConfig(nlon=10, nlat=5),
            SphericalGridConfig(nlon=20, nlat=10)
        ])
        
        # Template factory with multiple templates
        template_factory = TemplateFactoryConfig(templates=[
            FieldLine.template_all(),
            FieldLine()  # Use default template instead of template_basic
        ])
        
        tracing_config = TracingConfig()
        
        series_config = ExperimentSeriesConfig(
            name="test_series",
            field=field_factory,
            grid=grid_factory,
            tracing=tracing_config,
            template=template_factory
        )
        
        experiments = series_config.create_experiment_series()
        
        # Should have 2 field configs × 2 grid configs × 1 tracing × 2 templates = 8 experiments
        assert len(experiments) == 8
        
        # Check experiment names
        experiment_names = [exp.name for exp in experiments]
        expected_names = [f"test_series_{i:03d}" for i in range(1, 9)]
        assert experiment_names == expected_names
    
    def test_save_config(self):
        """Test saving experiment series config."""
        field_config = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)
        ])
        grid_config = GridConfig(spherical=SphericalGridConfig(nlon=10, nlat=5))
        tracing_config = TracingConfig()
        template = FieldLine.template_all()
        
        config = ExperimentSeriesConfig(
            name="test_series",
            field=field_config,
            grid=grid_config,
            tracing=tracing_config,
            template=template,
            description="Test description",
            metadata={"test_key": "test_value"}
        )
        
        temp_path = tempfile.mktemp(suffix='.yaml')
        config.save(temp_path)
        
        # Check that file was created and contains expected content
        assert Path(temp_path).exists()
        
        with open(temp_path, 'r') as f_read:
            data = yaml.safe_load(f_read)
        
        # Clean up
        Path(temp_path).unlink()
        
        # Check that saved data contains expected fields
        assert data is not None
        assert data["name"] == "test_series"
        assert data["description"] == "Test description"
        assert data["metadata"]["test_key"] == "test_value"
        assert "field" in data
        assert "grid" in data
        assert "tracing" in data
        assert "template" in data


class TestExperimentSeriesRunner:
    """Test ExperimentSeriesRunner class."""
    
    def test_initialization(self):
        """Test ExperimentSeriesRunner initialization."""
        field_config = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)
        ])
        grid_config = GridConfig(spherical=SphericalGridConfig(nlon=10, nlat=5))
        tracing_config = TracingConfig()
        template = FieldLine.template_all()
        
        series_config = ExperimentSeriesConfig(
            name="test_series",
            field=field_config,
            grid=grid_config,
            tracing=tracing_config,
            template=template
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentSeriesRunner(
                series_config=series_config,
                output_directory=temp_dir,
                save_individual_configs=True,
                continue_on_failure=True
            )
            
            assert runner.series_config == series_config
            assert runner.output_directory == Path(temp_dir)
            assert runner.save_individual_configs is True
            assert runner.continue_on_failure is True
    
    @patch('psr_magpy.experiment.experiment_series_runner.run_experiment')
    def test_run_single_experiment(self, mock_run_experiment):
        """Test running a single experiment."""
        field_config = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)
        ])
        grid_config = GridConfig(spherical=SphericalGridConfig(nlon=10, nlat=5))
        tracing_config = TracingConfig()
        template = FieldLine.template_all()
        
        experiment = ExperimentSeriesConfig(
            name="test_experiment",
            field=field_config,
            grid=grid_config,
            tracing=tracing_config,
            template=template
        ).create_experiment_series()[0]
        
        # Mock successful experiment run
        mock_run_experiment.return_value = {"field_lines": []}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentSeriesRunner(
                series_config=ExperimentSeriesConfig(
                    name="test_series",
                    field=field_config,
                    grid=grid_config,
                    tracing=tracing_config,
                    template=template
                ),
                output_directory=Path(temp_dir)
            )
            
            results = runner._run_experiments_sequential([experiment])
            
            assert len(results) == 1
            assert results[0].experiment_name == "test_experiment_001"
            assert results[0].success is True
            assert results[0].duration > 0
            assert results[0].results == {"field_lines": []}
    
    @patch('psr_magpy.experiment.experiment_series_runner.run_experiment')
    def test_run_experiment_with_failure(self, mock_run_experiment):
        """Test running experiment that fails."""
        field_config = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)
        ])
        grid_config = GridConfig(spherical=SphericalGridConfig(nlon=10, nlat=5))
        tracing_config = TracingConfig()
        template = FieldLine.template_all()
        
        experiment = ExperimentSeriesConfig(
            name="test_experiment",
            field=field_config,
            grid=grid_config,
            tracing=tracing_config,
            template=template
        ).create_experiment_series()[0]
        
        # Mock failed experiment run
        mock_run_experiment.side_effect = ValueError("Test error")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentSeriesRunner(
                series_config=ExperimentSeriesConfig(
                    name="test_series",
                    field=field_config,
                    grid=grid_config,
                    tracing=tracing_config,
                    template=template
                ),
                output_directory=temp_dir,
                continue_on_failure=True
            )
            
            results = runner._run_experiments_sequential([experiment])
            
            assert len(results) == 1
            assert results[0].experiment_name == "test_experiment_001"
            assert results[0].success is False
            assert results[0].duration > 0
            assert results[0].error_message is not None
            assert "Test error" in results[0].error_message
    
    def test_save_individual_configs(self):
        """Test saving individual experiment configurations."""
        field_config = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)
        ])
        grid_config = GridConfig(spherical=SphericalGridConfig(nlon=10, nlat=5))
        tracing_config = TracingConfig()
        template = FieldLine.template_all()
        
        series_config = ExperimentSeriesConfig(
            name="test_series",
            field=field_config,
            grid=grid_config,
            tracing=tracing_config,
            template=template
        )
        
        experiments = series_config.create_experiment_series()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = ExperimentSeriesRunner(
                series_config=series_config,
                output_directory=temp_dir,
                save_individual_configs=True
            )
            
            runner._save_individual_configs(experiments)
            
            # Check that configs directory was created
            configs_dir = Path(temp_dir) / "configs"
            assert configs_dir.exists()
            
            # Check that individual config files were saved
            config_files = list(configs_dir.glob("*.yaml"))
            assert len(config_files) == 1
            assert config_files[0].name == "test_series_001.yaml"


class TestIntegration:
    """Integration tests for experiment series."""
    
    def test_full_series_workflow(self):
        """Test complete experiment series workflow."""
        # Create a simple series with factory configs
        field_factory = FieldFactoryConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0),
            FieldComponentConfig(moment_type="dipole", B_magnitude=2.0)
        ])
        
        grid_factory = GridFactoryConfig(grids=[
            SphericalGridConfig(nlon=5, nlat=3),
            SphericalGridConfig(nlon=10, nlat=5)
        ])
        
        template_factory = TemplateFactoryConfig(templates=[
            FieldLine()  # Use default template
        ])
        
        series_config = ExperimentSeriesConfig(
            name="integration_test",
            field=field_factory,
            grid=grid_factory,
            tracing=TracingConfig(),
            template=template_factory,
            description="Integration test series",
            metadata={"test_type": "integration"}
        )
        
        # Create experiments
        experiments = series_config.create_experiment_series()
        
        # Should have 2 field configs × 2 grid configs × 1 tracing × 1 template = 4 experiments
        assert len(experiments) == 4
        
        # Check experiment names
        expected_names = [f"integration_test_{i:03d}" for i in range(1, 5)]
        actual_names = [exp.name for exp in experiments]
        assert actual_names == expected_names
        
        # Check metadata
        for i, exp in enumerate(experiments, 1):
            assert exp.metadata["experiment_number"] == i
            assert exp.metadata["series_name"] == "integration_test"
            assert exp.metadata["test_type"] == "integration"
            assert exp.metadata["field_components"] == 1
            assert exp.metadata["grid_type"] == "spherical" 