"""
Demo script showcasing the usage of FieldLineSet class.

This demo illustrates:
1. Creating individual field lines
2. Creating a field line set
3. Appending field lines to the set
4. Accessing properties from the set
5. Memory size estimation
"""

import logging
import numpy as np
import sys
from pathlib import Path

# Add src to path to import field_line
sys.path.append(str(Path(__file__).parent.parent))
from src.line.field_line import FieldLine, PropertyName
from src.line.field_line_set import FieldLineSet

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_field_line(footpoint_id: int, scale: float = 1.0) -> FieldLine:
    """Create a sample field line with some properties."""
    line = FieldLine(
        save_r_line=True,
        save_r_max=True,
        save_b_ini=True
    )
    
    coordinates = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2]
    ]) * scale
    
    line.set_property(
        footpoint_id=footpoint_id,
        r_line=coordinates,
        r_max=3.0 * scale,
        b_ini=np.array([0.1, 0.2, 0.3]) * scale
    )
    
    return line

def main():
    # Create several field lines
    logger.info("Creating sample field lines")
    field_lines = [
        create_sample_field_line(i, scale=1.0 + i*0.5) 
        for i in range(3)
    ]
    
    # Create field line set from a single line
    logger.info("Creating FieldLineSet from single line")
    field_set_single = FieldLineSet(field_lines[0], copy_data=True)
    print(f"\nField set from single line:")
    print(f"Number of lines: {field_set_single.n_field_lines}")
    print(f"First footpoint ID: {field_set_single.get_property('footpoint_id')}")
    
    # Create field line set from multiple lines
    logger.info("Creating FieldLineSet from multiple lines")
    field_set = FieldLineSet(field_lines, copy_data=True)
    print(f"\nField set from multiple lines:")
    print(f"Number of lines: {field_set.n_field_lines}")
    
    # Display properties
    print("\nField set properties:")
    print(f"Footpoint IDs: {field_set.get_property('footpoint_id')}")
    print(f"R Max values: {field_set.get_property('r_max')}")
    print(f"B Initial values:\n{field_set.get_property('b_ini')}")
    print("\nR Line coordinates (first two lines):")
    r_lines = field_set.get_property('r_line')
    for i, r_line in enumerate(r_lines[:2]):
        print(f"Line {i}:\n{r_line}")
    
    # Append more field lines
    logger.info("Appending additional field lines")
    new_lines = [
        create_sample_field_line(i, scale=3.0) 
        for i in range(3, 5)
    ]
    field_set.append(new_lines)
    
    print(f"\nAfter appending:")
    print(f"Number of lines: {field_set.n_field_lines}")
    print(f"Footpoint IDs: {field_set.get_property('footpoint_id')}")
    
    # Demonstrate size estimation
    logger.info("Estimating memory size")
    size_bytes = field_set.estimate_size()
    size_mb = size_bytes / (1024 * 1024)
    print(f"\nMemory usage:")
    print(f"Bytes: {size_bytes:,.0f}")
    print(f"Megabytes: {size_mb:.3f}")
    
    # Demonstrate error handling for incompatible field lines
    logger.info("Demonstrating error handling for incompatible field lines")
    incompatible_line = FieldLine(save_r_line=False)  # Different save flags
    incompatible_line.set_property(footpoint_id=99)
    
    try:
        field_set.append([incompatible_line])
    except ValueError as e:
        print(f"\nExpected error when appending incompatible line: {e}")

if __name__ == "__main__":
    main() 