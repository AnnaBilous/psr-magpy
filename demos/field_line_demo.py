"""
Demo script showcasing the usage of FieldLine class.
"""
import sys
from pathlib import Path

# Add project root to PYTHONPATH only if not already present
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
import numpy as np
from src.line.field_line import FieldLine, PropertyName

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    # Create field line with properties enabled
    field_line = FieldLine(
        save_r_line=True,
        save_r_max=True,
        save_b_ini=True
    )
    
    # Create sample data
    r_line_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    
    # Set properties before trying to access them
    field_line.update(
        r_line=r_line_data,
        r_max=5.0,
        b_ini=0.1
    )
    
    # Now we can access the properties
    logger.info(f"footpoint_id: {field_line.footpoint_id} (type: {type(field_line.footpoint_id)})")
    logger.info(f"r_line: {field_line.r_line} (type: {type(field_line.r_line)})")
    logger.info(f"r_max: {field_line.r_max} (type: {type(field_line.r_max)})")
    
    # Try to access disabled property (should raise AttributeError)
    logger.info("Trying to access disabled property (z_abs_max)...")
    try:
        _ = field_line.z_abs_max
    except AttributeError as e:
        logger.error(f"Expected error: {e}")

    # Create a copy of the structure
    logger.info("Creating copy of field line structure")
    field_line_copy = field_line.copy_structure()

    # Verify save flags are copied
    logger.info("Verifying save flags in copied structure:")
    for prop in PropertyName:
        flag_name = prop.flag_name
        original_flag = getattr(field_line.save_flags, flag_name)
        copied_flag = getattr(field_line_copy.save_flags, flag_name)
        logger.info(f"{flag_name}: original={original_flag}, copied={copied_flag}")

    # Try invalid type assignments
    logger.info("Testing type validation...")
    
    # Try to assign array where scalar is expected
    try:
        field_line.update(r_max=np.array([1.0, 2.0]))  # Should fail: array for scalar
        logger.error("Failed to catch invalid array assignment to scalar property")
    except TypeError as e:
        logger.info(f"Caught expected error: {e}")
        
    # Try to assign wrong scalar type
    try:
        field_line.update(footpoint_id=1.5)  # Should fail: float for int
        logger.error("Failed to catch invalid float assignment to int property")
    except TypeError as e:
        logger.info(f"Caught expected error: {e}")
        
    # Try to assign string (invalid type)
    try:
        field_line.update(r_line="invalid_type")  # Should fail: string for array
        logger.error("Failed to catch invalid string assignment")
    except TypeError as e:
        logger.info(f"Caught expected error: {e}")

if __name__ == "__main__":
    main() 