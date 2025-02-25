from astropy.io import fits as pf

from src.fitsio.base import FitsIO
from src.line.field_line_base import PropertyName
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FieldLineSetFitsIO(FitsIO):
    """FITS I/O operations for field lines."""
    def __init__(self, data):
        self.data = data
        super().__init__(self.data)
        logger.info("Initialized FieldLineSetFitsIO")

    def generate_column_info(self):
        """Generate column information for field line data.
        
        Returns:
            dict: Mapping of property names to their FITS format strings
        """
        result = {}
        
        # Process each enabled property
        for prop in PropertyName:
            if not getattr(self.data.save_flags, prop.flag_name):
                logger.debug(f"Skipping disabled property: {prop.key_name}")
                continue
                
            if prop not in self.data._properties:
                logger.warning(f"Property {prop.key_name} is enabled but not set")
                continue
            
            value = self.data._properties[prop]
            logger.info(f"Processing property: {prop.key_name}, type: {type(value)}")
            
            # Special case for r_line which is a list of arrays
            result[prop.key_name] = FitsIO.get_fits_format(value)
            
            logger.info(f"Added format for {prop.key_name}: {result.get(prop.key_name)}")
        
        return result
    
    def create_column(self, key: str, format: str):
        """Create a FITS column for field line data.

        Args:
            key: Column name (property key name).
            format: FITS format string.

        Returns:
            FITS column object.
        """
        # Get property by key name
        prop = PropertyName[key.upper()]
        value = self.data._properties[prop]
        
        logger.info(f"Creating column for {key} with format {format}")
        if value is not None:
            logger.info(f"Sample values: {value[:10] if isinstance(value, (list, np.ndarray)) else value}")
            
        return pf.Column(name=key, format=format, array=value)
