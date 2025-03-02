"""
Demo script showcasing the usage of FieldLine class.

This demo illustrates:
1. Creating field lines with different save flags
2. Setting and getting properties
3. Copying field line structure
4. Error handling for disabled properties
"""

import logging
import numpy as np
import sys
from pathlib import Path

# Add src to path to import field_line
sys.path.append(str(Path(__file__).parent.parent))
from src.line.field_line import FieldLine, PropertyName

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Create a field line with default settings (only footpoint_id enabled)
    logger.info("Creating default field line")
    default_line = FieldLine()
    print(f"Default field line: {default_line}\n")

    # Create a field line with multiple properties enabled
    logger.info("Creating field line with multiple properties enabled")
    custom_line = FieldLine(
        save_r_line=True,
        save_r_max=True,
        save_b_ini=True
    )
    print(f"Custom field line: {custom_line}\n")

    # Set some properties
    logger.info("Setting properties")
    coordinates = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    r_max = 3.0
    b_ini = np.array([0.1, 0.2, 0.3])
    
    custom_line.set_property(
        footpoint_id=1,
        r_line=coordinates,
        r_max=r_max,
        b_ini=b_ini
    )

    # Get and display properties
    logger.info("Getting properties")
    print("Field line properties:")
    print(f"Footpoint ID: {custom_line.get_property(PropertyName.FOOTPOINT_ID)}")
    print(f"R Line:\n{custom_line.get_property(PropertyName.R_LINE)}")
    print(f"R Max: {custom_line.get_property(PropertyName.R_MAX)}")
    print(f"B Initial: {custom_line.get_property(PropertyName.B_INI)}\n")

    # Try to access a disabled property
    logger.info("Demonstrating error handling for disabled property")
    try:
        custom_line.get_property(PropertyName.Z_ABS_MAX)
    except ValueError as e:
        print(f"Expected error when accessing disabled property: {e}\n")

    # Copy structure and verify
    logger.info("Demonstrating structure copying")
    copied_line = custom_line.copy_structure()
    print(f"Copied field line (empty properties): {copied_line}")
    
    # Set properties on copied line
    copied_line.set_property(
        footpoint_id=2,
        r_line=coordinates * 2,
        r_max=r_max * 2,
        b_ini=b_ini * 2
    )
    
    # Compare original and copied properties
    print("\nComparing original and copied field lines:")
    for prop in [PropertyName.FOOTPOINT_ID, PropertyName.R_LINE, 
                 PropertyName.R_MAX, PropertyName.B_INI]:
        print(f"\n{prop.name}:")
        print(f"Original: {custom_line.get_property(prop)}")
        print(f"Copied  : {copied_line.get_property(prop)}")

    # Demonstrate string-based property access
    logger.info("Demonstrating string-based property access")
    print("\nAccessing properties using strings:")
    print(f"R Max (using string): {custom_line.get_property('r_max')}")
    print(f"B Initial (using string): {custom_line.get_property('b_ini')}")
    
    # Demonstrate error handling for invalid string property
    logger.info("Demonstrating error handling for invalid string property")
    try:
        custom_line.get_property('invalid_property')
    except KeyError as e:
        print(f"\nExpected error when using invalid property name: {e}")

if __name__ == "__main__":
    main() 