"""Tests for the unified I/O module."""

import pytest
import tempfile
import yaml
from pathlib import Path
import numpy as np

from psr_magpy.io import get_data_io, get_config_io, get_io_factory
from psr_magpy.io.base import IOUtils, DataIO, ConfigIO
from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line import FieldLine
from psr_magpy.grid.grid import SphericalGrid
from psr_magpy.field.multipole import MultipoleField


class TestIOUtils:
    """Test I/O utilities."""
    
    def test_ensure_directory(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "nested" / "dir" / "test.txt"
            result = IOUtils.ensure_directory(test_path)
            assert result == test_path
            assert test_path.parent.exists()
    
    def test_convert_for_yaml(self):
        """Test YAML conversion utilities."""
        # Test tuple conversion
        data = (1, 2, 3)
        result = IOUtils.convert_for_yaml(data)
        assert result == [1, 2, 3]
        
        # Test numpy array conversion
        data = np.array([1, 2, 3])
        result = IOUtils.convert_for_yaml(data)
        assert result == [1, 2, 3]
        
        # Test nested structure
        data = {
            "tuple": (1, 2),
            "array": np.array([3, 4]),
            "list": [(5, 6), np.array([7, 8])]
        }
        result = IOUtils.convert_for_yaml(data)
        assert result["tuple"] == [1, 2]
        assert result["array"] == [3, 4]
        assert result["list"][0] == [5, 6]
        assert result["list"][1] == [7, 8]
    
    def test_validate_path(self):
        """Test path validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = Path(tmpdir) / "test.txt"
            existing_file.write_text("test")
            
            # Test existing file
            result = IOUtils.validate_path(existing_file, must_exist=True)
            assert result == existing_file
            
            # Test non-existing file
            non_existing = Path(tmpdir) / "nonexistent.txt"
            with pytest.raises(FileNotFoundError):
                IOUtils.validate_path(non_existing, must_exist=True)
            
            # Test non-existing file without must_exist
            result = IOUtils.validate_path(non_existing, must_exist=False)
            assert result == non_existing
    
    def test_get_fits_format(self):
        """Test FITS format detection."""
        assert IOUtils.get_fits_format(1.0) == "D"
        assert IOUtils.get_fits_format(1) == "I"
        assert IOUtils.get_fits_format("test") == "10A"
        assert IOUtils.get_fits_format(None) == "D"
        
        # Test numpy arrays
        arr_1d = np.array([1, 2, 3])
        assert IOUtils.get_fits_format(arr_1d) == "D"
        
        arr_2d = np.array([[1, 2], [3, 4]])
        assert IOUtils.get_fits_format(arr_2d) == "2D"


class TestDataIO:
    """Test data I/O base class."""
    
    def test_data_io_interface(self):
        """Test data I/O interface methods."""
        # Test that abstract class cannot be instantiated
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DataIO()


class TestConfigIO:
    """Test config I/O base class."""
    
    def test_config_io_interface(self):
        """Test config I/O interface methods."""
        # Test that abstract class cannot be instantiated
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ConfigIO()


class TestIOFactory:
    """Test I/O factory."""
    
    def test_format_detection(self):
        """Test format detection."""
        factory = get_io_factory()
        assert factory.detect_format("test.fits") == "fits"
        assert factory.detect_format("test.fit") == "fits"
        assert factory.detect_format("test.yaml") == "yaml"
        assert factory.detect_format("test.yml") == "yaml"
        assert factory.detect_format("test.unknown") == "unknown"
    
    def test_factory_creation(self):
        """Test factory instance creation."""
        factory = get_io_factory()
        
        # Test data I/O creation
        data_io = factory.create_data_io(FieldLineSet)
        assert isinstance(data_io, DataIO)
        
        # Test config I/O creation
        config_io = factory.create_config_io(ExperimentConfig)
        assert isinstance(config_io, ConfigIO)
    
    def test_factory_errors(self):
        """Test factory error handling."""
        factory = get_io_factory()
        
        # Test unregistered data type
        with pytest.raises(ValueError, match="No I/O class registered"):
            factory.create_data_io(dict)
        
        # Test unregistered config type
        with pytest.raises(ValueError, match="No I/O class registered"):
            factory.create_config_io(dict)
    
    def test_factory_save_load(self):
        """Test factory save/load operations."""
        factory = get_io_factory()
        
        # Test config save/load
        config = ExperimentConfig.create_dipole_example()
        
        tmp_path = Path(tempfile.mktemp(suffix=".yaml"))
        
        try:
            # Save config
            factory.save_config(config, tmp_path)
            assert tmp_path.exists()
            
            # Load config
            loaded_config = factory.load_config(ExperimentConfig, tmp_path)
            assert isinstance(loaded_config, ExperimentConfig)
            assert loaded_config.name == config.name
            
        finally:
            tmp_path.unlink(missing_ok=True)


class TestIntegration:
    """Integration tests for the I/O system."""
    
    def test_experiment_config_integration(self):
        """Test experiment config with new I/O system."""
        config = ExperimentConfig.create_dipole_example()
        
        tmp_path = Path(tempfile.mktemp(suffix=".yaml"))
        
        try:
            # Save using new I/O system
            config.save(tmp_path)
            assert tmp_path.exists()
            
            # Load using new I/O system
            loaded_config = ExperimentConfig.load(tmp_path)
            assert isinstance(loaded_config, ExperimentConfig)
            assert loaded_config.name == config.name
            
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_grid_integration(self):
        """Test grid with new I/O system."""
        grid = SphericalGrid(nlon=10, nlat=5)
        
        tmp_path = Path(tempfile.mktemp(suffix=".fits"))
        
        try:
            # Save using new I/O system
            grid.save_to_fits(str(tmp_path))
            assert tmp_path.exists()
            
            # Load using new I/O system
            loaded_grid = SphericalGrid.load_from_fits(str(tmp_path))
            assert isinstance(loaded_grid, SphericalGrid)
            assert loaded_grid.nline == grid.nline
            
        finally:
            tmp_path.unlink(missing_ok=True) 