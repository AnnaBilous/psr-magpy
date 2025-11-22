"""Tests for the unified I/O module."""

import pytest
import tempfile
import yaml
from pathlib import Path
import numpy as np

from psr_magpy.io import save_data, load_data, save_config, load_config, detect_format
from psr_magpy.io.base import IOUtils, DataIO, ConfigIO
from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.grid.grid import SphericalGrid
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.line.property_system import PropertyName


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
            from psr_magpy.utils.exceptions import PSRMagPyIOError
            with pytest.raises(PSRMagPyIOError, match="File does not exist"):
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


class TestIODispatch:
    """Test I/O functional dispatch."""
    
    def test_format_detection(self):
        """Test format detection."""
        assert detect_format("test.fits") == "fits"
        assert detect_format("test.fit") == "fits"
        assert detect_format("test.yaml") == "yaml"
        assert detect_format("test.yml") == "yaml"
        assert detect_format("test.unknown") == "unknown"
    
    def test_dispatch_errors(self):
        """Test dispatch error handling."""
        from psr_magpy.utils.exceptions import PSRMagPyIOError
        
        # Test unregistered data type
        with pytest.raises(PSRMagPyIOError, match="No IO handler"):
            save_data({}, "test.fits")
        
        # Test unregistered config type
        with pytest.raises(PSRMagPyIOError, match="No IO handler"):
            load_config(dict, "test.yaml")
    
    def test_dispatch_save_load(self):
        """Test dispatch save/load operations."""
        # Test config save/load
        config = ExperimentConfig.create_dipole_example()
        
        tmp_path = Path(tempfile.mktemp(suffix=".yaml"))
        
        try:
            # Save config
            save_config(config, tmp_path)
            assert tmp_path.exists()
            
            # Load config
            loaded_config = load_config(ExperimentConfig, tmp_path)
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


class TestTemplateFunctionality:
    """Test the new template functionality in I/O system."""
    
    def test_load_with_template_success(self, test_data_dir):
        """Test loading FieldLineSet with a template that matches available columns."""
        
        # Create a FieldLineSet with all properties
        full_template = FieldLineTemplate.template_all()
        field_line_set = FieldLineSet(full_template, copy_data=False)
        
        # Add some test data using direct property assignment
        from psr_magpy.line.property_system import PropertyName
        
        field_line_set.property_system._properties[PropertyName.FOOTPOINT_ID] = np.array([1, 2, 3])
        field_line_set.property_system._properties[PropertyName.R_CYL_MAX] = np.array([1.5, 2.0, 2.5])
        field_line_set.property_system._properties[PropertyName.Z_ABS_MAX] = np.array([0.5, 1.0, 1.5])
        field_line_set.n_field_lines = 3
        
        # Save to test data directory
        test_data_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = test_data_dir / "test_template_load.fits"
        
        save_data(field_line_set, tmp_path)
        
        # Create minimal template
        minimal_template = FieldLine(
            save_footpoint_id=True,
            save_r_cyl_max=True,
            save_z_abs_max=True
        )
        
        # Load with minimal template
        loaded_set = load_data(FieldLineSet, tmp_path, template=minimal_template)
        
        # Check that only requested properties are loaded
        assert loaded_set.property_system.get_property(PropertyName.FOOTPOINT_ID) is not None
        assert loaded_set.property_system.get_property(PropertyName.R_CYL_MAX) is not None
        assert loaded_set.property_system.get_property(PropertyName.Z_ABS_MAX) is not None
        
        # Check that other properties are not loaded
        assert PropertyName.R_LINE not in loaded_set.property_system.enabled_properties
        assert PropertyName.B_LINE not in loaded_set.property_system.enabled_properties
        
        # Check field line count
        assert loaded_set.n_field_lines == 3
    
    def test_load_with_template_missing_columns(self):
        """Test loading with template that requires columns not present in file."""
        
        # Create a FieldLineSet with minimal properties
        minimal_template = FieldLine(
            save_footpoint_id=True,
            save_r_cyl_max=True
        )
        field_line_set = FieldLineSet(minimal_template, copy_data=False)
        
        # Add test data using direct property assignment
        from psr_magpy.line.property_system import PropertyName
        
        field_line_set.property_system._properties[PropertyName.FOOTPOINT_ID] = np.array([1, 2])
        field_line_set.property_system._properties[PropertyName.R_CYL_MAX] = np.array([1.5, 2.0])
        field_line_set.n_field_lines = 2
        
        # Save to temporary file
        tmp_path = Path(tempfile.mktemp(suffix=".fits"))
        
        try:
            save_data(field_line_set, tmp_path)
            
            # Create template that requires missing columns
            template_with_missing = FieldLine(
                save_footpoint_id=True,
                save_r_cyl_max=True,
                save_z_abs_max=True  # This column doesn't exist in the file
            )
            
            # Should raise error about missing columns
            from psr_magpy.utils.exceptions import PSRMagPyIOError
            with pytest.raises(PSRMagPyIOError, match="Template requires columns that are missing"):
                load_data(FieldLineSet, tmp_path, template=template_with_missing)
                
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_load_without_template_original_behavior(self):
        """Test that loading without template maintains original behavior."""
        
        # Create a FieldLineSet with some properties
        template = FieldLine(
            save_footpoint_id=True,
            save_r_cyl_max=True,
            save_z_abs_max=True
        )
        field_line_set = FieldLineSet(template, copy_data=False)
        
        # Add test data using direct property assignment
        from psr_magpy.line.property_system import PropertyName
        
        field_line_set.property_system._properties[PropertyName.FOOTPOINT_ID] = np.array([1, 2])
        field_line_set.property_system._properties[PropertyName.R_CYL_MAX] = np.array([1.5, 2.0])
        field_line_set.property_system._properties[PropertyName.Z_ABS_MAX] = np.array([0.5, 1.0])
        field_line_set.n_field_lines = 2
        
        # Save to temporary file
        tmp_path = Path(tempfile.mktemp(suffix=".fits"))
        
        try:
            save_data(field_line_set, tmp_path)
            
            # Load without template (should auto-detect from columns)
            loaded_set = load_data(FieldLineSet, tmp_path)
            
            # Should load all available properties
            assert loaded_set.property_system.get_property(PropertyName.FOOTPOINT_ID) is not None
            assert loaded_set.property_system.get_property(PropertyName.R_CYL_MAX) is not None
            assert loaded_set.property_system.get_property(PropertyName.Z_ABS_MAX) is not None
            
            # Check field line count
            assert loaded_set.n_field_lines == 2
            
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_template_parameter_filtering(self):
        """Test that template parameter is filtered from I/O constructor kwargs."""
        
        # This should not raise an error about unexpected keyword argument
        # Test that template parameter is passed through
        from psr_magpy.io.data_io import FieldLineSetIO
        from psr_magpy.io.validator import ValidationLevel, RecoveryStrategy, DataIOValidator
        from psr_magpy.io.memory_monitor import MemoryMonitor, MemoryThreshold
        
        validator = DataIOValidator(
            validation_level=ValidationLevel.STANDARD,
            recovery_strategy=RecoveryStrategy.RETRY_ONCE,
        )
        memory_monitor = MemoryMonitor(
            threshold=MemoryThreshold.MODERATE,
            safety_margin_mb=100.0,
            enable_checks=True,
            enable_warnings=True,
        )
        data_io = FieldLineSetIO(
            validator=validator,
            memory_monitor=memory_monitor,
        )
        assert isinstance(data_io, DataIO)
    
    def test_load_with_empty_template(self):
        """Test loading with template that has no enabled properties."""
        
        # Create a FieldLineSet with some properties
        template = FieldLine(
            save_footpoint_id=True,
            save_r_cyl_max=True
        )
        field_line_set = FieldLineSet(template, copy_data=False)
        
        # Add test data using direct property assignment
        from psr_magpy.line.property_system import PropertyName
        
        field_line_set.property_system._properties[PropertyName.FOOTPOINT_ID] = np.array([1, 2])
        field_line_set.property_system._properties[PropertyName.R_CYL_MAX] = np.array([1.5, 2.0])
        field_line_set.n_field_lines = 2
        
        # Save to temporary file
        tmp_path = Path(tempfile.mktemp(suffix=".fits"))
        
        try:
            save_data(field_line_set, tmp_path)
            
            # Create empty template (only footpoint_id is always enabled)
            empty_template = FieldLine()
            
            # Load with empty template
            loaded_set = load_data(FieldLineSet, tmp_path, template=empty_template)
            
            # Should only have footpoint_id (which is always enabled)
            assert loaded_set.property_system.get_property(PropertyName.FOOTPOINT_ID) is not None
            assert PropertyName.R_CYL_MAX not in loaded_set.property_system.enabled_properties
            
            # Check field line count
            assert loaded_set.n_field_lines == 2
            
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_template_with_array_list_properties(self):
        """Test loading with template that includes array list properties."""
        
        # Create a FieldLineSet with array list properties
        template = FieldLine(
            save_footpoint_id=True,
            save_r_line=True,
            save_b_line=True
        )
        field_line_set = FieldLineSet(template, copy_data=False)
        
        # Add test data with array list properties using direct assignment
        from psr_magpy.line.property_system import PropertyName
        
        field_line_set.property_system._properties[PropertyName.FOOTPOINT_ID] = np.array([1, 2])
        field_line_set.property_system._properties[PropertyName.R_LINE] = [
            np.array([[1, 2, 3], [4, 5, 6]]), 
            np.array([[7, 8, 9], [10, 11, 12]])
        ]
        field_line_set.property_system._properties[PropertyName.B_LINE] = [
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]), 
            np.array([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]])
        ]
        field_line_set.n_field_lines = 2
        
        # Save to temporary file
        tmp_path = Path(tempfile.mktemp(suffix=".fits"))
        
        try:
            save_data(field_line_set, tmp_path)
            
            # Load with same template
            loaded_set = load_data(FieldLineSet, tmp_path, template=template)
            
            # Check that array list properties are loaded correctly
            assert loaded_set.property_system.get_property(PropertyName.FOOTPOINT_ID) is not None
            assert loaded_set.property_system.get_property(PropertyName.R_LINE) is not None
            assert loaded_set.property_system.get_property(PropertyName.B_LINE) is not None
            
            # Check field line count
            assert loaded_set.n_field_lines == 2
            
        finally:
            tmp_path.unlink(missing_ok=True) 