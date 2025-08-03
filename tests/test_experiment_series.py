"""Tests for experiment series functionality."""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from psr_magpy.experiment import (
    ExperimentSeriesConfig, ExperimentSeriesRunner,
    FieldFactoryConfig, GridFactoryConfig, TracingFactoryConfig,
    FieldComponentFactoryConfig, SphericalGridFactoryConfig,
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
    
    def test_tracing_factory_config(self):
        """Test TracingFactoryConfig."""
        factory = TracingFactoryConfig(
            max_integration_radius=[2.0, 3.0],
            save_results=True,
            filename="test"
        )
        
        assert factory.max_integration_radius == [2.0, 3.0]
        assert factory.save_results is True
        assert factory.filename == "test"
    
    def test_field_component_factory_config(self):
        """Test FieldComponentFactoryConfig."""
        factory = FieldComponentFactoryConfig(
            moment_type="dipole",
            moment_orientation_latitude_degree=[0, 45],
            moment_orientation_longitude_degree=0.0,
            B_magnitude=1.0
        )
        
        assert factory.moment_type == "dipole"
        assert factory.moment_orientation_latitude_degree == [0, 45]
        assert factory.moment_orientation_longitude_degree == 0.0
        assert factory.B_magnitude == 1.0
    
    def test_spherical_grid_factory_config(self):
        """Test SphericalGridFactoryConfig."""
        factory = SphericalGridFactoryConfig(
            nlon=10,
            nlat=5,
            pole_lat=0.0
        )
        
        assert factory.nlon == 10
        assert factory.nlat == 5
        assert factory.pole_lat == 0.0


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
        
        # Tracing factory with multiple parameters
        tracing_factory = TracingFactoryConfig(
            max_integration_radius=[2.0, 3.0],
            save_results=True,
            filename="test"
        )
        
        template = FieldLine.template_all()
        
        series_config = ExperimentSeriesConfig(
            name="test_series",
            field=field_factory,
            grid=grid_factory,
            tracing=tracing_factory,
            template=template
        )
        
        experiments = series_config.create_experiment_series()
        
        # Should have 2 field configs × 2 grid configs × 2 tracing params × 1 template = 8 experiments
        assert len(experiments) == 8
        
        # Check experiment names
        experiment_names = [exp.name for exp in experiments]
        expected_names = [f"test_series_{i:03d}" for i in range(1, 9)]
        assert experiment_names == expected_names
    
    def test_create_directory_structure(self):
        """Test that directory structure is created correctly."""
        field_config = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)
        ])
        grid_config = GridConfig(spherical=SphericalGridConfig(nlon=10, nlat=5))
        tracing_config = TracingConfig()
        template = FieldLine.template_all()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            series_config = ExperimentSeriesConfig(
                name="test_series",
                field=field_config,
                grid=grid_config,
                tracing=tracing_config,
                template=template,
                output_dir=temp_dir
            )
            
            # Create experiments (this should create the directory structure)
            experiments = series_config.create_experiment_series()
            
            # Check that directories were created
            series_dir = Path(temp_dir) / "test_series"
            assert series_dir.exists()
            
            configs_dir = series_dir / "configs"
            assert configs_dir.exists()
            
            grids_dir = series_dir / "grids"
            assert grids_dir.exists()
            
            lines_dir = series_dir / "lines"
            assert lines_dir.exists()
            
            logs_dir = series_dir / "logs"
            assert logs_dir.exists()
    
    def test_save_and_load_config(self):
        """Test saving and loading experiment series config."""
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
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set output directory
            config.output_dir = temp_dir
            
            # Create directory structure first
            config.create_experiment_series()
            
            # Save config
            config.save()
            
            # Check that file was created
            config_file = Path(temp_dir) / "test_series" / "configs" / "test_series_config.yaml"
            assert config_file.exists()
            
            # Load config
            loaded_config = ExperimentSeriesConfig.load(temp_dir, "test_series")
            
            # Check that loaded config matches original
            assert loaded_config.name == config.name
            assert loaded_config.description == config.description
            assert loaded_config.metadata is not None
            assert config.metadata is not None
            assert loaded_config.metadata["test_key"] == config.metadata["test_key"]
    
    def test_print_experiment_summary(self):
        """Test printing experiment summary."""
        field_factory = FieldFactoryConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0),
            FieldComponentConfig(moment_type="dipole", B_magnitude=2.0)
        ])
        
        grid_factory = GridFactoryConfig(grids=[
            SphericalGridConfig(nlon=5, nlat=3),
            SphericalGridConfig(nlon=10, nlat=5)
        ])
        
        tracing_factory = TracingFactoryConfig(
            max_integration_radius=[2.0, 3.0],
            save_results=True,
            filename="test"
        )
        
        template = FieldLine.template_all()
        
        series_config = ExperimentSeriesConfig(
            name="test_series",
            field=field_factory,
            grid=grid_factory,
            tracing=tracing_factory,
            template=template,
            description="Test series"
        )
        
        # Test summary with all parameters
        summary_all = series_config.print_experiment_summary(show_changing_only=False)
        assert summary_all is None  # Method returns None, prints to stdout
        
        # Test summary with only changing parameters
        summary_changing = series_config.print_experiment_summary(show_changing_only=True)
        assert summary_changing is None  # Method returns None, prints to stdout


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
            # Create the directory structure that ExperimentSeriesRunner expects
            series_dir = Path(temp_dir) / "test_series"
            series_dir.mkdir(parents=True, exist_ok=True)
            
            runner = ExperimentSeriesRunner(
                series_config=series_config,
                output_directory=Path(temp_dir),
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
            series_config = ExperimentSeriesConfig(
                name="test_series",
                field=field_config,
                grid=grid_config,
                tracing=tracing_config,
                template=template,
                output_dir=temp_dir
            )
            
            # Create directory structure
            series_config.create_experiment_series()
            
            runner = ExperimentSeriesRunner(
                series_config=series_config,
                output_directory=Path(temp_dir)
            )
            
            results = runner._run_experiments([experiment])
            
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
            series_config = ExperimentSeriesConfig(
                name="test_series",
                field=field_config,
                grid=grid_config,
                tracing=tracing_config,
                template=template,
                output_dir=temp_dir
            )
            
            # Create directory structure
            series_config.create_experiment_series()
            
            runner = ExperimentSeriesRunner(
                series_config=series_config,
                output_directory=Path(temp_dir),
                continue_on_failure=True
            )
            
            results = runner._run_experiments([experiment])
            
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
            # Set output directory and create structure
            series_config.output_dir = temp_dir
            series_config.create_experiment_series()
            
            runner = ExperimentSeriesRunner(
                series_config=series_config,
                output_directory=Path(temp_dir),
                save_individual_configs=True
            )
            
            runner._save_individual_configs(experiments)
            
            # Check that configs directory was created in series subdirectory
            configs_dir = Path(temp_dir) / "test_series" / "configs"
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
        
        tracing_factory = TracingFactoryConfig(
            max_integration_radius=[2.0, 3.0],
            save_results=True,
            filename="test"
        )
        
        template = FieldLine.template_all()
        
        series_config = ExperimentSeriesConfig(
            name="integration_test",
            field=field_factory,
            grid=grid_factory,
            tracing=tracing_factory,
            template=template,
            description="Integration test series",
            metadata={"test_type": "integration"}
        )
        
        # Create experiments
        experiments = series_config.create_experiment_series()
        
        # Should have 2 field configs × 2 grid configs × 2 tracing params × 1 template = 8 experiments
        assert len(experiments) == 8
        
        # Check experiment names
        expected_names = [f"integration_test_{i:03d}" for i in range(1, 9)]
        actual_names = [exp.name for exp in experiments]
        assert actual_names == expected_names
        
        # Check metadata
        for i, exp in enumerate(experiments, 1):
            assert exp.metadata is not None
            assert exp.metadata["experiment_number"] == i
            assert exp.metadata["series_name"] == "integration_test"
            assert exp.metadata["test_type"] == "integration"
            assert exp.metadata["field_components"] == 1
            assert exp.metadata["grid_type"] == "spherical" 