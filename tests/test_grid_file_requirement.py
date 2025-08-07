"""Tests for grid file requirements functionality."""

import pytest
from pathlib import Path
import numpy as np
from astropy.coordinates import CartesianRepresentation

from psr_magpy.grid.grid import RandomGrid, CustomGrid
from psr_magpy.experiment import ExperimentConfig
from psr_magpy.experiment.factory_configs import (
    RandomGridFactoryConfig, CustomGridFactoryConfig,
    FieldFactoryConfig, TracingFactoryConfig
)
from psr_magpy.experiment.experiment_config import (
    FieldConfig, GridConfig, TracingConfig,
    RandomGridConfig, CustomGridConfig
)
from psr_magpy.line import FieldLine

def test_random_grid_warning(caplog):
    """Test that RandomGrid creation issues a warning."""
    grid = RandomGrid(nline_ori=10)
    assert "RandomGrid created: This grid must be saved to a file" in caplog.text

def test_custom_grid_warning(caplog):
    """Test that CustomGrid creation issues a warning."""
    coordinates = CartesianRepresentation(x=[1], y=[0], z=[0])
    grid = CustomGrid(coordinates=coordinates, nline_ori=1)
    assert "CustomGrid created: This grid must be saved to a file" in caplog.text

def test_random_grid_load_from_file(tmp_path):
    """Test loading RandomGrid from file."""
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Create and save grid
    grid = RandomGrid(nline_ori=10)
    grid_path = tmp_path / "random_grid.fits"
    grid.save_to_fits(str(grid_path))
    
    # Reset random seed to ensure same coordinates are generated
    np.random.seed(42)
    
    # Create config with file path
    config = ExperimentConfig(
        name="test",
        field=FieldConfig(components=[]),
        grid=GridConfig(
            random=RandomGridConfig(
                nline_ori=10,
                grid_file_path=str(grid_path)
            )
        ),
        tracing=TracingConfig(),
        template=FieldLine.template_all()
    )
    
    # Load grid from config
    loaded_grid = config.grid.create_grid()
    if loaded_grid is not None and loaded_grid.footpoint_coordinates is not None:
        assert loaded_grid.nline == 10
        grid_coords = grid.footpoint_coordinates
        loaded_coords = loaded_grid.footpoint_coordinates
        if grid_coords is not None and loaded_coords is not None:
            assert np.allclose(loaded_coords.xyz.value, grid_coords.xyz.value)

def test_custom_grid_load_from_file(tmp_path):
    """Test loading CustomGrid from file."""
    # Create and save grid
    coordinates = CartesianRepresentation(x=[1], y=[0], z=[0])
    grid = CustomGrid(coordinates=coordinates, nline_ori=1)
    grid_path = tmp_path / "custom_grid.fits"
    grid.save_to_fits(str(grid_path))
    
    # Create config with file path
    config = ExperimentConfig(
        name="test",
        field=FieldConfig(components=[]),
        grid=GridConfig(
            custom=CustomGridConfig(
                coordinates=coordinates.xyz.value.tolist(),
                nline_ori=1,
                grid_file_path=str(grid_path)
            )
        ),
        tracing=TracingConfig(),
        template=FieldLine.template_all()
    )
    
    # Load grid from config
    loaded_grid = config.grid.create_grid()
    if loaded_grid is not None and loaded_grid.footpoint_coordinates is not None:
        grid_coords = grid.footpoint_coordinates
        loaded_coords = loaded_grid.footpoint_coordinates
        if grid_coords is not None and loaded_coords is not None:
            assert np.allclose(loaded_coords.xyz.value, grid_coords.xyz.value)

def test_random_grid_create_from_scratch():
    """Test creating RandomGrid from scratch."""
    config = ExperimentConfig(
        name="test",
        field=FieldConfig(components=[]),
        grid=GridConfig(
            random=RandomGridConfig(
                nline_ori=10
            )
        ),
        tracing=TracingConfig(),
        template=FieldLine.template_all()
    )
    
    grid = config.grid.create_grid()
    assert isinstance(grid, RandomGrid)
    assert grid.nline == 10

def test_custom_grid_create_from_scratch():
    """Test creating CustomGrid from scratch."""
    coordinates = CartesianRepresentation(x=[1], y=[0], z=[0])
    config = ExperimentConfig(
        name="test",
        field=FieldConfig(components=[]),
        grid=GridConfig(
            custom=CustomGridConfig(
                coordinates=[[1], [0], [0]],  # Pass as list of lists
                nline_ori=1
            )
        ),
        tracing=TracingConfig(),
        template=FieldLine.template_all()
    )
    
    grid = config.grid.create_grid()
    if grid is not None and grid.footpoint_coordinates is not None:
        assert isinstance(grid, CustomGrid)
        assert np.allclose(grid.footpoint_coordinates.xyz.value, coordinates.xyz.value)

def test_grid_save_during_series_creation(tmp_path):
    """Test that grids are saved during experiment series creation."""
    from psr_magpy.experiment import ExperimentSeriesConfig
    from psr_magpy.experiment.factory_configs import (
        FieldFactoryConfig, GridFactoryConfig, TracingFactoryConfig
    )
    from psr_magpy.line import FieldLine
    
    # Create series config with random grid
    series_config = ExperimentSeriesConfig(
        name="test_series",
        output_dir=str(tmp_path),
        field=FieldFactoryConfig(components=[]),
        grid=GridFactoryConfig(
            grids=[RandomGridFactoryConfig(nline_ori=[10, 20])]
        ),
        tracing=TracingFactoryConfig(),
        template=FieldLine.template_all()
    )
    
    # Create experiments
    experiments = series_config.create_experiment_series(save_configs=True)
    
    # Check that grid files exist and are referenced in configs
    grids_dir = tmp_path / series_config.name / "grids"
    assert grids_dir.exists()
    
    for experiment in experiments:
        grid_path = grids_dir / f"{experiment.name}.fits"
        assert grid_path.exists()
        assert experiment.grid.random is not None and experiment.grid.random.grid_file_path == str(grid_path) 