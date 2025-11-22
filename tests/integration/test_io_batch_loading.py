"""Tests for batch loading functionality with templates."""

import pytest
import tempfile
import numpy as np
from pathlib import Path

from psr_magpy.io import save_data, load_data
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.line.property_system import PropertyName
from psr_magpy.experiment.experiment_config import ExperimentConfig


def create_test_batch_files(test_data_dir, experiment_name, n_batches=3, n_lines_per_batch=100):
    """Create test batch files for testing."""
    tmp_dir = test_data_dir / f"batch_loading_{experiment_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    lines_dir = tmp_dir / "lines"
    lines_dir.mkdir(parents=True, exist_ok=True)
    
    # Create template with all properties
    template = FieldLineTemplate.template_all()
    
    for batch_num in range(1, n_batches + 1):
        # Create field line set for this batch
        field_line_set = FieldLineSet(template, copy_data=False)
        
        # Generate test data
        start_idx = (batch_num - 1) * n_lines_per_batch
        end_idx = batch_num * n_lines_per_batch
        
        # Use direct property assignment for bulk data (like in I/O system)
        from psr_magpy.line.property_system import PropertyName
        
        field_line_set.property_system._properties[PropertyName.FOOTPOINT_ID] = np.arange(start_idx, end_idx)
        field_line_set.property_system._properties[PropertyName.R_CYL_MAX] = np.random.uniform(1.0, 5.0, n_lines_per_batch)
        field_line_set.property_system._properties[PropertyName.Z_ABS_MAX] = np.random.uniform(0.1, 2.0, n_lines_per_batch)
        field_line_set.property_system._properties[PropertyName.R_MAX] = np.random.uniform(2.0, 10.0, n_lines_per_batch)
        field_line_set.property_system._properties[PropertyName.B_INI] = np.random.uniform(0.5, 2.0, n_lines_per_batch)
        
        field_line_set.n_field_lines = n_lines_per_batch
        
        # Save batch file
        batch_file = lines_dir / f"{experiment_name}_batch_{batch_num:03d}_out_of_{n_batches:03d}.fits"
        save_data(field_line_set, batch_file)
    
    return tmp_dir, lines_dir


def test_load_combined_batch_data_function(test_data_dir):
    """Test the load_combined_batch_data function with template."""
    # Create test batch files
    experiment_name = "test_experiment"
    tmp_dir, lines_dir = create_test_batch_files(test_data_dir, experiment_name, n_batches=2, n_lines_per_batch=50)
    
    # Create minimal template
    minimal_template = FieldLine(
        save_footpoint_id=True,
        save_r_cyl_max=True,
        save_z_abs_max=True
    )
    
    # Create mock experiment config
    experiment = ExperimentConfig(
        name=experiment_name,
        field=None,  # Not needed for this test
        grid=None,   # Not needed for this test
        tracing=None, # Not needed for this test
        template=minimal_template
    )
    
    # Test the load_combined_batch_data function logic
    batch_pattern = f"{lines_dir}/{experiment_name}_batch_*.fits"
    batch_files = list(lines_dir.glob(f"{experiment_name}_batch_*.fits"))
    
    assert len(batch_files) == 2
    
    # Create combined field line set
    combined_field_line_set = FieldLineSet(minimal_template, copy_data=False)
    
    # Load and combine batch files
    for batch_file in sorted(batch_files):
        field_line_set = load_data(FieldLineSet, batch_file, template=minimal_template)
        combined_field_line_set.append(field_line_set)
    
    # Check results
    assert combined_field_line_set.n_field_lines == 100  # 2 batches * 50 lines each
    
    # Check that only requested properties are loaded
    assert combined_field_line_set.property_system.get_property(PropertyName.FOOTPOINT_ID) is not None
    assert combined_field_line_set.property_system.get_property(PropertyName.R_CYL_MAX) is not None
    assert combined_field_line_set.property_system.get_property(PropertyName.Z_ABS_MAX) is not None
    
    # Check that other properties are not loaded (check if they're in enabled properties)
    assert PropertyName.R_LINE not in combined_field_line_set.property_system.enabled_properties
    assert PropertyName.B_LINE not in combined_field_line_set.property_system.enabled_properties
    
    # Check data integrity
    footpoint_ids = combined_field_line_set.property_system.get_property(PropertyName.FOOTPOINT_ID)
    assert len(footpoint_ids) == 100
    assert np.all(footpoint_ids == np.arange(100))
    
    # No cleanup needed - test_data_dir is cleaned up automatically


def test_template_validation_missing_columns(test_data_dir):
    """Test that template validation catches missing columns."""
    # Create test batch files with minimal properties
    experiment_name = "test_minimal"
    tmp_dir, lines_dir = create_test_batch_files(test_data_dir, experiment_name, n_batches=1, n_lines_per_batch=10)
    
    # Create template that requires columns not in the file
    template_with_missing = FieldLine(
        save_footpoint_id=True,
        save_r_cyl_max=True,
        save_z_abs_max=True,
        save_r_line=True  # This won't be in the file
    )
    
    batch_file = list(lines_dir.glob(f"{experiment_name}_batch_*.fits"))[0]
    
    # Should raise error about missing columns
    from psr_magpy.utils.exceptions import PSRMagPyIOError
    with pytest.raises(PSRMagPyIOError, match="Template requires columns that are missing"):
        load_data(FieldLineSet, batch_file, template=template_with_missing)
    
    # No cleanup needed - test_data_dir is cleaned up automatically


def test_memory_efficiency_with_template(test_data_dir):
    """Test that loading with template is memory efficient."""
    # Create test batch files with many properties
    experiment_name = "test_memory"
    tmp_dir, lines_dir = create_test_batch_files(test_data_dir, experiment_name, n_batches=1, n_lines_per_batch=100)
    
    batch_file = list(lines_dir.glob(f"{experiment_name}_batch_*.fits"))[0]
    
    # Create template that matches what's actually in the test files
    full_template = FieldLine(
        save_footpoint_id=True,
        save_r_cyl_max=True,
        save_z_abs_max=True,
        save_r_max=True,
        save_b_ini=True
    )
    full_set = load_data(FieldLineSet, batch_file, template=full_template)
    
    # Load with minimal template (only 3 properties)
    minimal_template = FieldLine(
        save_footpoint_id=True,
        save_r_cyl_max=True,
        save_z_abs_max=True
    )
    minimal_set = load_data(FieldLineSet, batch_file, template=minimal_template)
    
    # Both should have the same number of field lines
    assert full_set.n_field_lines == minimal_set.n_field_lines == 100
    
    # Minimal set should have fewer properties loaded
    full_properties = list(full_set.property_system.enabled_properties)
    minimal_properties = list(minimal_set.property_system.enabled_properties)
    
    assert len(full_properties) > len(minimal_properties)
    assert len(minimal_properties) == 3  # footpoint_id, r_cyl_max, z_abs_max
    
    # No cleanup needed - test_data_dir is cleaned up automatically


if __name__ == "__main__":
    pytest.main([__file__])
