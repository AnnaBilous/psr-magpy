"""Extensive tests for experiment_config module."""

import pytest
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

from psr_magpy.experiment.experiment_config import (
    FieldComponentConfig, FieldConfig, SphericalGridConfig, RandomGridConfig,
    HealpixGridConfig, CustomGridConfig, GridConfig, TracingConfig,
    ExperimentConfig, FieldFactoryConfig, GridFactoryConfig, TemplateFactoryConfig,
    ExperimentSeriesConfig
)
from psr_magpy.line.field_line import FieldLine
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.grid.grid import SphericalGrid, RandomGrid, HealpixGrid, CustomGrid
from psr_magpy.line.tracing_parameters import TracingParameters

# Ensure I/O module is imported for registration
import psr_magpy.io


class TestFieldComponentConfig:
    """Test FieldComponentConfig class."""
    
    def test_basic_initialization(self):
        """Test basic initialization with required parameters."""
        config = FieldComponentConfig(
            moment_type="dipole",
            B_magnitude=1.5,
            moment_orientation_latitude_degree=30.0,
            moment_orientation_longitude_degree=45.0
        )
        
        assert config.moment_type == "dipole"
        assert config.B_magnitude == 1.5
        assert config.moment_orientation_latitude_degree == 30.0
        assert config.moment_orientation_longitude_degree == 45.0
        assert config.offset_r_ns == [0.0, 0.0, 0.0]
    
    def test_default_values(self):
        """Test initialization with default values."""
        config = FieldComponentConfig(moment_type="quadrupole")
        
        assert config.moment_type == "quadrupole"
        assert config.B_magnitude == 1.0
        assert config.moment_orientation_latitude_degree == 0.0
        assert config.moment_orientation_longitude_degree == 0.0
        assert config.offset_r_ns == [0.0, 0.0, 0.0]
    
    def test_custom_offset(self):
        """Test initialization with custom offset."""
        config = FieldComponentConfig(
            moment_type="dipole",
            offset_r_ns=[1.0, 2.0, 3.0]
        )
        
        assert config.offset_r_ns == [1.0, 2.0, 3.0]
    
    def test_invalid_moment_type(self):
        """Test that invalid moment types are handled appropriately."""
        # This should work as Pydantic allows any string
        config = FieldComponentConfig(moment_type="invalid_type")
        assert config.moment_type == "invalid_type"


class TestFieldConfig:
    """Test FieldConfig class."""
    
    def test_empty_components(self):
        """Test initialization with empty components list."""
        config = FieldConfig(components=[])
        assert len(config.components) == 0
    
    def test_single_component(self):
        """Test initialization with single component."""
        dipole = FieldComponentConfig(moment_type="dipole")
        config = FieldConfig(components=[dipole])
        
        assert len(config.components) == 1
        assert config.components[0].moment_type == "dipole"
    
    def test_multiple_components(self):
        """Test initialization with multiple components."""
        dipole = FieldComponentConfig(moment_type="dipole")
        quadrupole = FieldComponentConfig(moment_type="quadrupole")
        config = FieldConfig(components=[dipole, quadrupole])
        
        assert len(config.components) == 2
        assert config.components[0].moment_type == "dipole"
        assert config.components[1].moment_type == "quadrupole"
    
    def test_create_field_dipole(self):
        """Test creating MultipoleField with dipole component."""
        dipole = FieldComponentConfig(moment_type="dipole", B_magnitude=2.0)
        config = FieldConfig(components=[dipole])
        
        field = config.create_field()
        assert isinstance(field, MultipoleField)
        assert len(field.field_components) == 1
        # Check that the component was created correctly
        assert field.field_components[0].get_moment_type().value == "dipole"
    
    def test_create_field_quadrupole(self):
        """Test creating MultipoleField with quadrupole component."""
        quadrupole = FieldComponentConfig(moment_type="quadrupole", B_magnitude=1.5)
        config = FieldConfig(components=[quadrupole])
        
        field = config.create_field()
        assert isinstance(field, MultipoleField)
        assert len(field.field_components) == 1
        # Check that the component was created correctly
        assert field.field_components[0].get_moment_type().value == "quadrupole"
    
    def test_create_field_multiple_components(self):
        """Test creating MultipoleField with multiple components."""
        dipole = FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)
        quadrupole = FieldComponentConfig(moment_type="quadrupole", B_magnitude=0.5)
        config = FieldConfig(components=[dipole, quadrupole])
        
        field = config.create_field()
        assert isinstance(field, MultipoleField)
        assert len(field.field_components) == 2
    
    def test_create_field_invalid_type(self):
        """Test creating field with invalid component type."""
        invalid = FieldComponentConfig(moment_type="invalid_type")
        config = FieldConfig(components=[invalid])
        
        with pytest.raises(ValueError, match="Unknown field component type"):  # type: ignore
            config.create_field()
    
    def test_get_complete_components(self):
        """Test getting complete component configurations."""
        dipole = FieldComponentConfig(
            moment_type="dipole",
            B_magnitude=2.0,
            moment_orientation_latitude_degree=30.0,
            moment_orientation_longitude_degree=45.0,
            offset_r_ns=[1.0, 2.0, 3.0]
        )
        config = FieldConfig(components=[dipole])
        
        complete = config.get_complete_components()
        assert len(complete) == 1
        assert complete[0]["moment_type"] == "dipole"
        assert complete[0]["B_magnitude"] == 2.0
        assert complete[0]["moment_orientation_latitude_degree"] == 30.0
        assert complete[0]["moment_orientation_longitude_degree"] == 45.0
        # offset_r_ns might be a numpy array, convert to list for comparison
        offset = complete[0]["offset_r_ns"]
        if hasattr(offset, 'tolist'):
            offset = offset.tolist()
        assert offset == [1.0, 2.0, 3.0]


class TestSphericalGridConfig:
    """Test SphericalGridConfig class."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        config = SphericalGridConfig(nlon=10, nlat=20)
        
        assert config.nlon == 10
        assert config.nlat == 20
        assert config.pole_lat == 0.0  # default
    
    def test_with_pole_lat(self):
        """Test initialization with custom pole_lat."""
        config = SphericalGridConfig(nlon=5, nlat=15, pole_lat=30.0)
        
        assert config.nlon == 5
        assert config.nlat == 15
        assert config.pole_lat == 30.0


class TestRandomGridConfig:
    """Test RandomGridConfig class."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        config = RandomGridConfig(nline_ori=100)
        
        assert config.nline_ori == 100


class TestHealpixGridConfig:
    """Test HealpixGridConfig class."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        config = HealpixGridConfig(nside=8)
        
        assert config.nside == 8


class TestCustomGridConfig:
    """Test CustomGridConfig class."""
    
    def test_basic_initialization(self):
        """Test basic initialization without coordinates."""
        config = CustomGridConfig(nline_ori=50)
        
        assert config.nline_ori == 50
        assert config.coordinates is None
    
    def test_with_coordinates(self):
        """Test initialization with coordinates."""
        coords = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        config = CustomGridConfig(nline_ori=2, coordinates=coords)
        
        assert config.nline_ori == 2
        assert config.coordinates == coords


class TestGridConfig:
    """Test GridConfig class."""
    
    def test_spherical_grid(self):
        """Test spherical grid configuration."""
        spherical = SphericalGridConfig(nlon=5, nlat=10)
        config = GridConfig(spherical=spherical)
        
        assert config.spherical == spherical
        assert config.random is None
        assert config.healpix is None
        assert config.custom is None
        assert config.get_grid_type() == "spherical"
    
    def test_random_grid(self):
        """Test random grid configuration."""
        random = RandomGridConfig(nline_ori=100)
        config = GridConfig(random=random)
        
        assert config.random == random
        assert config.spherical is None
        assert config.healpix is None
        assert config.custom is None
        assert config.get_grid_type() == "random"
    
    def test_healpix_grid(self):
        """Test HEALPix grid configuration."""
        healpix = HealpixGridConfig(nside=8)
        config = GridConfig(healpix=healpix)
        
        assert config.healpix == healpix
        assert config.spherical is None
        assert config.random is None
        assert config.custom is None
        assert config.get_grid_type() == "healpix"
    
    def test_custom_grid(self):
        """Test custom grid configuration."""
        custom = CustomGridConfig(nline_ori=50)
        config = GridConfig(custom=custom)
        
        assert config.custom == custom
        assert config.spherical is None
        assert config.random is None
        assert config.healpix is None
        assert config.get_grid_type() == "custom"
    
    def test_no_grid_type(self):
        """Test configuration with no grid type specified."""
        # GridConfig requires exactly one grid type, so this should raise an error
        with pytest.raises(ValueError, match="Exactly one grid type must be specified"):  # type: ignore
            GridConfig()
    
    def test_multiple_grid_types(self):
        """Test that multiple grid types are handled appropriately."""
        spherical = SphericalGridConfig(nlon=5, nlat=10)
        random = RandomGridConfig(nline_ori=100)
        # GridConfig requires exactly one grid type, so this should raise an error
        with pytest.raises(ValueError, match="Exactly one grid type must be specified"):  # type: ignore
            GridConfig(spherical=spherical, random=random)
    
    def test_create_spherical_grid(self):
        """Test creating spherical grid."""
        spherical = SphericalGridConfig(nlon=5, nlat=10)
        config = GridConfig(spherical=spherical)
        
        grid = config.create_grid()
        assert isinstance(grid, SphericalGrid)
        assert grid.nlon == 5
        assert grid.nlat == 10
    
    def test_create_random_grid(self):
        """Test creating random grid."""
        random = RandomGridConfig(nline_ori=100)
        config = GridConfig(random=random)
        
        grid = config.create_grid()
        assert isinstance(grid, RandomGrid)
        assert grid.nline_ori == 100
    
    def test_create_healpix_grid(self):
        """Test creating HEALPix grid."""
        healpix = HealpixGridConfig(nside=8)
        config = GridConfig(healpix=healpix)
        
        grid = config.create_grid()
        assert isinstance(grid, HealpixGrid)
        assert grid.nside == 8
    
    def test_create_custom_grid(self):
        """Test creating custom grid."""
        # CustomGrid requires CartesianRepresentation, not list
        # This test should be skipped or modified based on actual implementation
        custom = CustomGridConfig(nline_ori=2, coordinates=None)
        config = GridConfig(custom=custom)
        
        grid = config.create_grid()
        assert isinstance(grid, CustomGrid)
        assert grid.nline_ori == 2


class TestTracingConfig:
    """Test TracingConfig class."""
    
    def test_default_values(self):
        """Test default values."""
        config = TracingConfig()
        
        # Integration parameters
        assert config.max_integration_radius == 10.0
        assert config.threshold_line_distance == 1.0
        assert config.t_span == (0.0, 1e8)
        assert config.rtol == 5.0e-9
        assert config.atol == 5.0e-9
        assert config.surface_radius == 1.0 - 1.e-8
        assert config.dt == 0.01
        assert config.max_step == 0.01
        assert config.trace_open_lines_backwards is False
        
        # Performance parameters
        assert config.max_memory_mb == 100.0
        assert config.safety_factor == 0.9
        assert config.max_batch_size == 100
        assert config.check_interval_sec == 1.0
        assert config.n_cores is None
        assert config.n_trial == 10
        assert config.progress_bar is True
        assert config.verbose is False
        assert config.skip_memory_estimation is False
        
        # Output parameters
        assert config.save_results is False
        assert config.filename is None
        assert config.output_format == "fits"
        assert config.overwrite is False
    
    def test_custom_values(self):
        """Test custom values."""
        config = TracingConfig(
            max_integration_radius=15.0,
            dt=0.1,
            n_cores=4,
            save_results=True,
            filename="test.fits"
        )
        
        assert config.max_integration_radius == 15.0
        assert config.dt == 0.1
        assert config.n_cores == 4
        assert config.save_results is True
        assert config.filename == "test.fits"
    
    def test_create_tracing_parameters(self):
        """Test creating TracingParameters."""
        config = TracingConfig(
            max_integration_radius=12.0,
            dt=0.05,
            n_cores=2,
            save_results=True,
            filename="test_output.fits"  # Required when save_results=True
        )
        
        params = config.create_tracing_parameters()
        assert isinstance(params, TracingParameters)
        assert params.integration.max_integration_radius == 12.0
        assert params.integration.dt == 0.05
        assert params.performance.n_cores == 2
        assert params.output.save_results is True
    
    def test_get_complete_config(self):
        """Test getting complete configuration."""
        config = TracingConfig(
            max_integration_radius=15.0,
            dt=0.1,
            n_cores=4
        )
        
        complete = config.get_complete_config()
        assert complete["max_integration_radius"] == 15.0
        assert complete["dt"] == 0.1
        assert complete["n_cores"] == 4
        assert complete["save_results"] is False  # default
        assert complete["output_format"] == "fits"  # default


class TestExperimentConfig:
    """Test ExperimentConfig class."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        field = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole")
        ])
        grid = GridConfig(spherical=SphericalGridConfig(nlon=5, nlat=10))
        tracing = TracingConfig()
        template = FieldLine.template_coordinates_and_PA()
        
        config = ExperimentConfig(
            name="test_experiment",
            field=field,
            grid=grid,
            tracing=tracing,
            template=template
        )
        
        assert config.name == "test_experiment"
        assert config.field == field
        assert config.grid == grid
        assert config.tracing == tracing
        assert config.template == template
        assert config.description is None
        assert config.metadata is None
    
    def test_with_description_and_metadata(self):
        """Test initialization with description and metadata."""
        field = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole")
        ])
        grid = GridConfig(spherical=SphericalGridConfig(nlon=5, nlat=10))
        tracing = TracingConfig()
        template = FieldLine.template_coordinates_and_PA()
        
        config = ExperimentConfig(
            name="test_experiment",
            description="Test experiment description",
            field=field,
            grid=grid,
            tracing=tracing,
            template=template,
            metadata={"version": "1.0", "author": "test"}
        )
        
        assert config.description == "Test experiment description"
        assert config.metadata == {"version": "1.0", "author": "test"}
    
    def test_save_and_load(self):
        """Test saving and loading configuration."""
        field = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=2.0)
        ])
        grid = GridConfig(spherical=SphericalGridConfig(nlon=5, nlat=10))
        tracing = TracingConfig(max_integration_radius=15.0, dt=0.1)
        template = FieldLine.template_all()
        
        config = ExperimentConfig(
            name="save_load_test",
            description="Test save/load functionality",
            field=field,
            grid=grid,
            tracing=tracing,
            template=template,
            metadata={"test": True}
        )
        
        # Save to temporary file
        temp_path = tempfile.mktemp(suffix='.yaml')
        
        try:
            config.save(temp_path)
            
            # Load from file
            loaded_config = ExperimentConfig.load(temp_path)
            
            # Verify all fields match
            assert loaded_config.name == config.name
            assert loaded_config.description == config.description
            assert loaded_config.metadata == config.metadata
            
            # Verify field configuration
            assert len(loaded_config.field.components) == len(config.field.components)
            assert loaded_config.field.components[0].moment_type == config.field.components[0].moment_type
            assert loaded_config.field.components[0].B_magnitude == config.field.components[0].B_magnitude
            
            # Verify grid configuration
            assert loaded_config.grid.get_grid_type() == config.grid.get_grid_type()
            assert loaded_config.grid.spherical is not None
            assert config.grid.spherical is not None
            assert loaded_config.grid.spherical.nlon == config.grid.spherical.nlon
            assert loaded_config.grid.spherical.nlat == config.grid.spherical.nlat
            
            # Verify tracing configuration
            assert loaded_config.tracing.max_integration_radius == config.tracing.max_integration_radius
            assert loaded_config.tracing.dt == config.tracing.dt
            
            # Verify template configuration
            loaded_flags = loaded_config.template.property_system.get_flags_dict()
            original_flags = config.template.property_system.get_flags_dict()
            assert loaded_flags == original_flags
            
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)
    
    def test_print_yaml(self, capsys):
        """Test printing YAML configuration."""
        field = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole")
        ])
        grid = GridConfig(spherical=SphericalGridConfig(nlon=5, nlat=10))
        tracing = TracingConfig()
        template = FieldLine.template_coordinates_and_PA()
        
        config = ExperimentConfig(
            name="yaml_print_test",
            field=field,
            grid=grid,
            tracing=tracing,
            template=template
        )
        
        config.print_yaml()
        captured = capsys.readouterr()
        
        # Verify output contains expected sections
        assert "=== Configuration ===" in captured.out
        assert "name: yaml_print_test" in captured.out
        assert "moment_type: dipole" in captured.out
        assert "nlon: 5" in captured.out
        assert "nlat: 10" in captured.out
    
    def test_convert_for_yaml(self):
        """Test YAML conversion utilities."""
        from psr_magpy.io import IOUtils
        
        # Test tuple conversion
        test_data = {"tuple": (1, 2, 3), "list": [1, 2, 3], "dict": {"a": 1}}
        converted: Any = IOUtils.convert_for_yaml(test_data)
        assert converted["tuple"] == [1, 2, 3]  # tuple converted to list
        assert converted["list"] == [1, 2, 3]   # list unchanged
        assert converted["dict"] == {"a": 1}    # dict unchanged
    
    def test_create_dipole_example(self):
        """Test creating dipole example."""
        config = ExperimentConfig.create_dipole_example()
        
        assert config.name == "dipole_example"
        assert config.description == "Simple dipole field experiment"
        assert len(config.field.components) == 1
        assert config.field.components[0].moment_type == "dipole"
        assert config.grid.get_grid_type() == "spherical"
        assert config.tracing.max_integration_radius == 10.0
        assert config.tracing.n_cores == 2


class TestFieldFactoryConfig:
    """Test FieldFactoryConfig class."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        dipole = FieldComponentConfig(moment_type="dipole")
        quadrupole = FieldComponentConfig(moment_type="quadrupole")
        
        config = FieldFactoryConfig(components=[
            [dipole],
            [dipole, quadrupole]
        ])
        
        assert len(config.components) == 2
        assert len(config.components[0]) == 1  # type: ignore
        assert len(config.components[1]) == 2  # type: ignore


class TestGridFactoryConfig:
    """Test GridFactoryConfig class."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        spherical = SphericalGridConfig(nlon=5, nlat=10)
        random = RandomGridConfig(nline_ori=100)
        
        config = GridFactoryConfig(grids=[spherical, random])
        
        assert len(config.grids) == 2
        assert isinstance(config.grids[0], SphericalGridConfig)
        assert isinstance(config.grids[1], RandomGridConfig)


class TestTemplateFactoryConfig:
    """Test TemplateFactoryConfig class."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        template1 = FieldLine.template_coordinates_and_PA()
        template2 = FieldLine.template_all()
        
        config = TemplateFactoryConfig(templates=[template1, template2])
        
        assert len(config.templates) == 2
        assert isinstance(config.templates[0], FieldLine)
        assert isinstance(config.templates[1], FieldLine)


class TestExperimentSeriesConfig:
    """Test ExperimentSeriesConfig class."""
    
    def test_basic_initialization(self):
        """Test basic initialization."""
        field = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole")
        ])
        grid = GridConfig(spherical=SphericalGridConfig(nlon=5, nlat=10))
        tracing = TracingConfig()
        template = FieldLine.template_coordinates_and_PA()
        
        config = ExperimentSeriesConfig(
            name="test_series",
            field=field,
            grid=grid,
            tracing=tracing,
            template=template
        )
        
        assert config.name == "test_series"
        assert config.field == field
        assert config.grid == grid
        assert config.tracing == tracing
        assert config.template == template
        assert config.description is None
        assert config.metadata is None
    
    def test_with_factory_configs(self):
        """Test initialization with factory configs."""
        field_factory = FieldFactoryConfig(components=[
            [FieldComponentConfig(moment_type="dipole")]
        ])
        grid_factory = GridFactoryConfig(grids=[
            SphericalGridConfig(nlon=5, nlat=10)
        ])
        template_factory = TemplateFactoryConfig(templates=[
            FieldLine.template_coordinates_and_PA()
        ])
        
        config = ExperimentSeriesConfig(
            name="factory_test",
            field=field_factory,
            grid=grid_factory,
            tracing=TracingConfig(),
            template=template_factory
        )
        
        assert config.name == "factory_test"
        assert isinstance(config.field, FieldFactoryConfig)
        assert isinstance(config.grid, GridFactoryConfig)
        assert isinstance(config.template, TemplateFactoryConfig)
    
    def test_create_experiment_series_simple(self):
        """Test creating experiment series with simple configs."""
        field = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole")
        ])
        grid = GridConfig(spherical=SphericalGridConfig(nlon=5, nlat=10))
        tracing = TracingConfig()
        template = FieldLine.template_coordinates_and_PA()
        
        config = ExperimentSeriesConfig(
            name="simple_series",
            field=field,
            grid=grid,
            tracing=tracing,
            template=template
        )
        
        experiments = config.create_experiment_series()
        assert len(experiments) == 1
        assert experiments[0].name == "simple_series_001"
        assert experiments[0].field == field
        assert experiments[0].grid == grid
        assert experiments[0].tracing == tracing
        assert experiments[0].template == template
    
    def test_create_experiment_series_with_factories(self):
        """Test creating experiment series with factory configs."""
        field_factory = FieldFactoryConfig(components=[
            [FieldComponentConfig(moment_type="dipole")],
            [FieldComponentConfig(moment_type="quadrupole")]
        ])
        grid_factory = GridFactoryConfig(grids=[
            SphericalGridConfig(nlon=5, nlat=10),
            RandomGridConfig(nline_ori=100)
        ])
        template_factory = TemplateFactoryConfig(templates=[
            FieldLine.template_coordinates_and_PA(),
            FieldLine.template_all()
        ])
        
        config = ExperimentSeriesConfig(
            name="factory_series",
            field=field_factory,
            grid=grid_factory,
            tracing=TracingConfig(),
            template=template_factory
        )
        
        experiments = config.create_experiment_series()
        # Should create 2 (field) * 2 (grid) * 2 (template) = 8 experiments
        assert len(experiments) == 8
        
        # Check that experiments have different configurations
        field_types = set()
        grid_types = set()
        template_props = set()
        
        for exp in experiments:
            field_types.add(exp.field.components[0].moment_type)
            grid_types.add(exp.grid.get_grid_type())
            template_props.add(len(exp.template.property_system.get_enabled_properties()))
        
        assert len(field_types) == 2  # dipole, quadrupole
        assert len(grid_types) == 2   # spherical, random
        assert len(template_props) == 2  # different template property counts


class TestIntegration:
    """Integration tests for experiment configuration."""
    
    def test_complete_experiment_workflow(self):
        """Test complete experiment workflow from config to execution."""
        # Create comprehensive configuration
        field = FieldConfig(components=[
            FieldComponentConfig(moment_type="dipole", B_magnitude=1.0),
            FieldComponentConfig(moment_type="quadrupole", B_magnitude=0.5)
        ])
        grid = GridConfig(spherical=SphericalGridConfig(nlon=3, nlat=4))
        tracing = TracingConfig(
            max_integration_radius=10.0,
            dt=0.1,
            n_cores=2,
            skip_memory_estimation=True,
            save_results=False
        )
        template = FieldLine.template_coordinates_and_PA()
        
        config = ExperimentConfig(
            name="integration_test",
            field=field,
            grid=grid,
            tracing=tracing,
            template=template
        )
        
        # Test field creation
        field_obj = config.field.create_field()
        assert isinstance(field_obj, MultipoleField)
        assert len(field_obj.field_components) == 2
        
        # Test grid creation
        grid_obj = config.grid.create_grid()
        assert isinstance(grid_obj, SphericalGrid)
        assert grid_obj.nlon == 3
        assert grid_obj.nlat == 4
        
        # Test tracing parameters creation
        tracing_params = config.tracing.create_tracing_parameters()
        assert isinstance(tracing_params, TracingParameters)
        assert tracing_params.integration.max_integration_radius == 10.0
        assert tracing_params.integration.dt == 0.1
        assert tracing_params.performance.n_cores == 2
        
        # Test template properties
        template_flags = config.template.property_system.get_flags_dict()
        assert template_flags["save_footpoint_id"] is True
        assert template_flags["save_r_line"] is True
        assert template_flags["save_pa_line"] is True
    
    def test_yaml_roundtrip(self):
        """Test complete YAML save/load roundtrip."""
        # Create complex configuration
        field = FieldConfig(components=[
            FieldComponentConfig(
                moment_type="dipole",
                B_magnitude=2.0,
                moment_orientation_latitude_degree=30.0,
                moment_orientation_longitude_degree=45.0,
                offset_r_ns=[1.0, 2.0, 3.0]
            ),
            FieldComponentConfig(
                moment_type="quadrupole",
                B_magnitude=1.0,
                moment_orientation_latitude_degree=60.0,
                moment_orientation_longitude_degree=90.0
            )
        ])
        grid = GridConfig(spherical=SphericalGridConfig(nlon=10, nlat=20, pole_lat=15.0))
        tracing = TracingConfig(
            max_integration_radius=15.0,
            dt=0.05,
            n_cores=4,
            save_results=True,
            filename="test_output.fits"
        )
        template = FieldLine.template_all()
        
        config = ExperimentConfig(
            name="yaml_roundtrip_test",
            description="Test YAML roundtrip functionality",
            field=field,
            grid=grid,
            tracing=tracing,
            template=template,
            metadata={"version": "1.0", "test": True}
        )
        
        # Save to temporary file
        temp_path = tempfile.mktemp(suffix='.yaml')
        
        try:
            config.save(temp_path)
            
            # Load from file
            loaded_config = ExperimentConfig.load(temp_path)
            
            # Verify complete roundtrip
            assert loaded_config.name == config.name
            assert loaded_config.description == config.description
            assert loaded_config.metadata == config.metadata
            
            # Verify field components
            assert len(loaded_config.field.components) == len(config.field.components)
            for i, (orig, loaded) in enumerate(zip(config.field.components, loaded_config.field.components)):
                assert orig.moment_type == loaded.moment_type
                assert orig.B_magnitude == loaded.B_magnitude
                assert orig.moment_orientation_latitude_degree == loaded.moment_orientation_latitude_degree
                assert orig.moment_orientation_longitude_degree == loaded.moment_orientation_longitude_degree
                assert orig.offset_r_ns == loaded.offset_r_ns
            
            # Verify grid configuration
            assert loaded_config.grid.get_grid_type() == config.grid.get_grid_type()
            assert loaded_config.grid.spherical is not None
            assert config.grid.spherical is not None
            assert loaded_config.grid.spherical.nlon == config.grid.spherical.nlon
            assert loaded_config.grid.spherical.nlat == config.grid.spherical.nlat
            assert loaded_config.grid.spherical.pole_lat == config.grid.spherical.pole_lat
            
            # Verify tracing configuration
            assert loaded_config.tracing.max_integration_radius == config.tracing.max_integration_radius
            assert loaded_config.tracing.dt == config.tracing.dt
            assert loaded_config.tracing.n_cores == config.tracing.n_cores
            assert loaded_config.tracing.save_results == config.tracing.save_results
            assert loaded_config.tracing.filename == config.tracing.filename
            
            # Verify template configuration
            orig_flags = config.template.property_system.get_flags_dict()
            loaded_flags = loaded_config.template.property_system.get_flags_dict()
            assert orig_flags == loaded_flags
            
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True) 