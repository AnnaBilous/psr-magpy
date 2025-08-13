"""Tests for experiment series functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest

from psr_magpy.experiment.experiment_series import (
    ExperimentSeriesConfig,
    FieldConfig, GridConfig, TracingConfig,
    SphericalGridConfig,
    create_experiment_metadata, generate_experiment_name
)
from psr_magpy.experiment.experiment_series_runner import ExperimentSeriesRunner
from psr_magpy.experiment.parameter_expansion import get_parameter_combinations
from psr_magpy.experiment.factory_configs import (
    FieldFactoryConfig, GridFactoryConfig, TracingFactoryConfig,
    FieldComponentFactoryConfig, SphericalGridFactoryConfig
)
from psr_magpy.experiment.experiment_config import FieldComponentConfig
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.experiment.experiment_result import ExperimentResult


class TestParameterExpansion:
    """Test parameter expansion functionality."""

    def test_get_parameter_combinations_single_value(self):
        """Test parameter combinations with single values."""
        params = {"a": 1, "b": 2}
        combinations = get_parameter_combinations(params)
        assert len(combinations) == 1
        assert combinations[0] == params

    def test_get_parameter_combinations_with_lists(self):
        """Test parameter combinations with lists."""
        params = {"a": [1, 2], "b": [3, 4]}
        combinations = get_parameter_combinations(params)
        assert len(combinations) == 4
        assert {"a": 1, "b": 3} in combinations
        assert {"a": 1, "b": 4} in combinations
        assert {"a": 2, "b": 3} in combinations
        assert {"a": 2, "b": 4} in combinations

    def test_get_parameter_combinations_empty(self):
        """Test parameter combinations with empty input."""
        params = {}
        combinations = get_parameter_combinations(params)
        assert len(combinations) == 1
        assert combinations[0] == params


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
        assert metadata["field_components"] == [
            {
                "type": "dipole",
                "B": 1.0,
                "lat": 0.0,
                "lon": 0.0
            }
        ]


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
            save_results_to_disk=True,
            filename="test"
        )
        
        assert factory.max_integration_radius == [2.0, 3.0]
        assert factory.save_results_to_disk is True
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
        """Test creating simple experiment series."""
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

        experiments = config.create_experiment_series()
        assert len(experiments) == 1
        assert experiments[0].name == "test_series_000"
        assert experiments[0].field == field_config
        assert experiments[0].grid == grid_config
        assert experiments[0].tracing == tracing_config
        assert experiments[0].template == template
        assert experiments[0].description == "Test description"
        assert experiments[0].metadata is not None
        assert experiments[0].metadata.get("test_key") == "test_value"
    
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
            save_results_to_disk=True,
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
        expected_names = [f"test_series_{i:03d}" for i in range(8)]
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
            config.create_directory_structure()

            # Save config
            config_file = Path(temp_dir) / "test_series" / "configs" / "test_series_config.yaml"
            config.save(str(config_file))

            # Check that file was created
            assert config_file.exists()

            # Load config
            loaded_config = ExperimentSeriesConfig.load(str(config_file))

            # Check that loaded config matches original
            assert loaded_config.name == config.name
            assert loaded_config.description == config.description
            assert loaded_config.metadata == config.metadata
            assert loaded_config.output_dir == config.output_dir
    
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
            save_results_to_disk=True,
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

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the directory structure that ExperimentSeriesRunner expects
            series_dir = Path(temp_dir) / "test_series"
            series_dir.mkdir(parents=True, exist_ok=True)

            series_config = ExperimentSeriesConfig(
                name="test_series",
                field=field_config,
                grid=grid_config,
                tracing=tracing_config,
                template=template,
                output_dir=temp_dir
            )

            runner = ExperimentSeriesRunner(
                series_config=series_config,
                save_individual_configs=True,
                continue_on_failure=True
            )

            assert runner.series_config == series_config
            assert runner.output_directory == Path(temp_dir)
    
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
        template = FieldLine.template_all()
        field_lines = [template] * 32  # Create 32 identical field lines
        mock_run_experiment.return_value = ExperimentResult(
            experiment_name="test_experiment_000",
            success=True,
            duration=1.0,
            results=FieldLineSet(field_lines)  # Use 32 field lines to match grid size
        )

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
                series_config=series_config
            )

            results = runner._run_experiments([experiment])

            assert len(results) == 1
            assert results[0].experiment_name == "test_experiment_000"
            assert results[0].success is True
            assert results[0].duration > 0
            # Results field is cleared after saving to disk
            assert results[0].results is None
    
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
        mock_run_experiment.return_value = ExperimentResult(
            experiment_name="test_experiment_000",
            success=False,
            duration=1.0,
            error_message="Test error"
        )

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
                continue_on_failure=True
            )

            results = runner._run_experiments([experiment])

            assert len(results) == 1
            assert results[0].experiment_name == "test_experiment_000"
            assert results[0].success is False
            assert results[0].duration > 0
            assert results[0].error_message == "Test error"
    
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
                save_individual_configs=True
            )

            runner._save_individual_configs(experiments)

            # Check that configs directory was created in series subdirectory
            configs_dir = Path(temp_dir) / "test_series" / "configs"
            assert configs_dir.exists()

            # Check that both series and experiment config files were saved
            config_files = list(configs_dir.glob("*.yaml"))
            assert len(config_files) == 2
            assert any(f.name == "test_series_config.yaml" for f in config_files)
            assert any(f.name == "test_series_000.yaml" for f in config_files)


class TestIntegration:
    """Integration tests for experiment series."""
    
    def test_full_series_workflow(self):
        """Test complete experiment series workflow."""
        # Create a simple series with factory configs
        field_factory = FieldFactoryConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0),  # Put B=1.0 first to match metadata
            FieldComponentConfig(moment_type="dipole", B_magnitude=2.0)
        ])

        grid_factory = GridFactoryConfig(grids=[
            SphericalGridConfig(nlon=5, nlat=3),
            SphericalGridConfig(nlon=10, nlat=5)
        ])

        tracing_factory = TracingFactoryConfig(
            max_integration_radius=[2.0, 3.0],
            save_results_to_disk=True,
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
        expected_names = [f"integration_test_{i:03d}" for i in range(8)]
        actual_names = [exp.name for exp in experiments]
        assert actual_names == expected_names

        # Check metadata
        for i, exp in enumerate(experiments):
            assert exp.metadata is not None
            assert exp.metadata["experiment_number"] == i
            assert exp.metadata["series_name"] == "integration_test"
            assert exp.metadata["test_type"] == "integration"
            # Components are sorted by B_magnitude in descending order
            assert exp.metadata["field_components"] == [
                {
                    "type": "dipole",
                    "B": 2.0 if i < 4 else 1.0,  # First 4 experiments have B=2.0, last 4 have B=1.0
                    "lat": 0.0,
                    "lon": 0.0
                }
            ]
            assert exp.metadata["grid_type"] == "spherical" 

    def test_create_directory_structure_overwrite_cleans_all_directories(self):
        """Test that overwrite=True cleans up configs, grids, and lines directories."""
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a config with output directory
            config = ExperimentSeriesConfig(
                name="test_series",
                field=FieldConfig(components=[]),
                grid=GridConfig(spherical=SphericalGridConfig(nlon=2, nlat=2)),
                tracing=TracingConfig(),
                template=FieldLine.template_coordinates_and_PA(),
                output_dir=temp_dir
            )
            
            # Create initial directory structure
            config.create_directory_structure(overwrite=False)
            
            # Create some dummy files in each directory
            configs_dir = config.get_configs_dir()
            grids_dir = config.get_grids_dir()
            lines_dir = config.get_lines_dir()
            
            (configs_dir / "dummy_config.yaml").write_text("dummy")
            (grids_dir / "dummy_grid.fits").write_text("dummy")
            (lines_dir / "dummy_line.fits").write_text("dummy")
            
            # Verify files exist
            assert (configs_dir / "dummy_config.yaml").exists()
            assert (grids_dir / "dummy_grid.fits").exists()
            assert (lines_dir / "dummy_line.fits").exists()
            
            # Now call with overwrite=True
            config.create_directory_structure(overwrite=True)
            
            # Verify all directories were cleaned and recreated
            assert configs_dir.exists()
            assert grids_dir.exists()
            assert lines_dir.exists()
            
            # Verify dummy files were removed
            assert not (configs_dir / "dummy_config.yaml").exists()
            assert not (grids_dir / "dummy_grid.fits").exists()
            assert not (lines_dir / "dummy_line.fits").exists() 