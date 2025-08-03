"""Tests for grid file requirement functionality."""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from psr_magpy.grid.grid import RandomGrid, CustomGrid
from psr_magpy.experiment import (
    RandomGridConfig, CustomGridConfig, GridConfig
)
from astropy.coordinates import CartesianRepresentation


class TestGridFileRequirement:
    """Test grid file requirement functionality."""
    
    def test_random_grid_creation_from_scratch(self):
        """Test that RandomGrid can be created from scratch without file path."""
        config = RandomGridConfig(nline_ori=100)
        grid_config = GridConfig(random=config)
        
        # Should create from scratch (no file path provided)
        grid = grid_config.create_grid()
        
        assert isinstance(grid, RandomGrid)
        assert grid.nline == 100
        assert grid.grid_type.value == "random"
    
    def test_custom_grid_creation_from_scratch(self):
        """Test that CustomGrid can be created from scratch without file path."""
        config = CustomGridConfig(nline_ori=50)
        grid_config = GridConfig(custom=config)
        
        # Should create from scratch (no file path provided)
        grid = grid_config.create_grid()
        
        assert isinstance(grid, CustomGrid)
        assert grid.nline == 50
        assert grid.grid_type.value == "custom"
    
    def test_random_grid_loading_from_file(self):
        """Test that RandomGrid can be loaded from file when path is provided."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create and save a grid
            original_grid = RandomGrid(nline_ori=100)
            grid_file_path = temp_path / "test_random_grid.fits"
            original_grid.save_to_fits(str(grid_file_path))
            
            # Create config with file path
            config = RandomGridConfig(
                nline_ori=100,
                grid_file_path=str(grid_file_path)
            )
            grid_config = GridConfig(random=config)
            
            # Should load from file
            loaded_grid = grid_config.create_grid()
            
            assert isinstance(loaded_grid, RandomGrid)
            assert loaded_grid.nline == 100
            assert grid_file_path.exists()
    
    def test_custom_grid_loading_from_file(self):
        """Test that CustomGrid can be loaded from file when path is provided."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create custom coordinates
            coords = CartesianRepresentation(
                x=np.linspace(0, 1, 25),
                y=np.linspace(0, 1, 25),
                z=np.zeros(25)
            )
            
            # Create and save a grid
            original_grid = CustomGrid(nline_ori=25, coordinates=coords)
            grid_file_path = temp_path / "test_custom_grid.fits"
            original_grid.save_to_fits(str(grid_file_path))
            
            # Create config with file path
            config = CustomGridConfig(
                nline_ori=25,
                grid_file_path=str(grid_file_path)
            )
            grid_config = GridConfig(custom=config)
            
            # Should load from file
            loaded_grid = grid_config.create_grid()
            
            assert isinstance(loaded_grid, CustomGrid)
            assert loaded_grid.nline == 25
            assert grid_file_path.exists()
    
    def test_random_grid_warning_on_creation(self, caplog):
        """Test that RandomGrid shows warning when created."""
        with caplog.at_level("WARNING"):
            grid = RandomGrid(nline_ori=50)
        
        assert "RandomGrid created: This grid must be saved to a file" in caplog.text
        assert grid.nline == 50
    
    def test_custom_grid_warning_on_creation(self, caplog):
        """Test that CustomGrid shows warning when created."""
        with caplog.at_level("WARNING"):
            grid = CustomGrid(nline_ori=25)
        
        assert "CustomGrid created: This grid must be saved to a file" in caplog.text
        assert grid.nline == 25
    
    def test_factory_config_without_file_path(self):
        """Test factory config works without file path."""
        from psr_magpy.experiment.factory_configs import (
            RandomGridFactoryConfig, CustomGridFactoryConfig, 
            GridFactoryConfig, expand_grid_configs
        )
        
        # Create factory configs without file paths
        random_factory = RandomGridFactoryConfig(nline_ori=[50, 100])
        custom_factory = CustomGridFactoryConfig(nline_ori=[25, 50])
        
        grid_factory_config = GridFactoryConfig(grids=[random_factory, custom_factory])
        expanded_configs = expand_grid_configs(grid_factory_config)
        
        assert len(expanded_configs) == 4  # 2 random + 2 custom
        
        # All should create from scratch
        for config in expanded_configs:
            grid = config.create_grid()
            assert grid.nline in [25, 50, 100]
    
    def test_factory_config_with_file_path(self):
        """Test factory config works with file paths."""
        from psr_magpy.experiment.factory_configs import (
            RandomGridFactoryConfig, CustomGridFactoryConfig, 
            GridFactoryConfig, expand_grid_configs
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create and save grids
            random_grid_50 = RandomGrid(50)
            random_grid_50.save_to_fits(str(temp_path / "random_50.fits"))
            
            random_grid_100 = RandomGrid(100)
            random_grid_100.save_to_fits(str(temp_path / "random_100.fits"))
            
            # Create factory configs with file paths
            random_factory = RandomGridFactoryConfig(
                nline_ori=[50, 100],
                grid_file_path=[str(temp_path / "random_50.fits"), str(temp_path / "random_100.fits")]
            )
            
            grid_factory_config = GridFactoryConfig(grids=[random_factory])
            expanded_configs = expand_grid_configs(grid_factory_config)
            

            
            # Should create 2 configs (paired nline_ori with grid_file_path)
            assert len(expanded_configs) == 2
            
            # All should load from files
            for config in expanded_configs:
                grid = config.create_grid()
                assert grid.nline in [50, 100] 