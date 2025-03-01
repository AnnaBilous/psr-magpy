#!/usr/bin/env python3
"""
Demo script showcasing the usage of FieldLineSet class for managing multiple field lines.
"""
import sys
from pathlib import Path

# Add project root to PYTHONPATH only if not already present
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
import numpy as np
from pathlib import Path
from src.line.field_line import FieldLine
from src.line.field_line_set import FieldLineSet
from src.line.io import FieldLineSetFitsIO
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_field_line(footpoint_id: int, points: int = 100) -> FieldLine:
    """Create a sample field line with synthetic data.
    
    Args:
        footpoint_id: Identifier for the field line
        points: Number of points in the trajectory
        
    Returns:
        FieldLine: A field line with synthetic data
    """
    # Create a helical trajectory
    t = np.linspace(0, 4*np.pi, points)
    r = np.array([
        np.cos(t),  # x
        np.sin(t),  # y
        t/4         # z
    ])
    
    # Initialize with all save flags enabled
    line = FieldLine(
        save_r_line=True,
        save_r_max=True,
        save_r_cyl_max=True,
        save_z_abs_max=True,
        save_b_ini=True,
        save_b_r_ini_angle=True
    )
    
    
    # Set properties using the update method
    kwargs = { 'footpoint_id': footpoint_id,
         'r_line': r,
         'r_max': np.max(np.sum(r**2, axis=0)**0.5),
         'r_cyl_max': np.max(np.sum(r[:2,]**2, axis=0)**0.5),
         'z_abs_max': np.max(np.abs(r[2, :])),
         'b_ini': 1.0,
         'b_r_ini_angle': np.pi/4
        }
    line.update(**kwargs)
    
    return line

def main():
    """Main demo function."""
    logger.info("Starting FieldLineSet demo")
    
    # Create several sample field lines
    field_lines = [create_sample_field_line(i) for i in range(5)]
    logger.info("Created %d sample field lines", len(field_lines))
    for i in range(5):
        logger.info(repr(field_lines[i]))
    
    # Create a FieldLineSet
    field_line_set = FieldLineSet(field_lines)
    logger.info("Created FieldLineSet")
    logger.info(repr(field_line_set))
    
    # Print some statistics
    size_mb = field_line_set.estimate_size() / (1024 * 1024)
    logger.info("FieldLineSet size: %.2f MB", size_mb)
    logger.info("Number of field lines: %d", len(field_line_set.footpoint_id))
    
    # Append more field lines
    new_lines = [create_sample_field_line(i+5) for i in range(3)]
    field_line_set.append(new_lines)
    logger.info("Appended %d more field lines", len(new_lines))
    
    # Save to FITS file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "field_lines.fits"
    field_line_set.write_to_fits(str(output_file))
    logger.info("Saved field lines to %s", output_file)
    
    # Demonstrate loading from FITS
    logger.info("Demonstrating loading from FITS file")
    loaded_field_lines = FieldLineSetFitsIO.load_from_fits(str(output_file))
    logger.info("Loaded FieldLineSet")
    logger.info(repr(loaded_field_lines))
    
    # Verify the loaded data matches the original
    logger.info("Verifying loaded data...")
    assert len(loaded_field_lines.footpoint_id) == len(field_line_set.footpoint_id), "Number of field lines mismatch"
    assert loaded_field_lines.save_flags.__dict__ == field_line_set.save_flags.__dict__, "Save flags mismatch"
    logger.info("Data verification successful")

if __name__ == "__main__":
    main() 