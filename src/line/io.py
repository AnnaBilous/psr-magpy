from typing import List, Type, Union
from astropy.io import fits as pf
from numpy.typing import NDArray
from numpy import ndarray, integer, floating


from astropy.io import fits as pf

from src.fitsio.base import FitsIO
from src.line.field_line_base import PropertyName
from src.line.field_line_set import FieldLineSet
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FieldLineSetFitsIO(FitsIO):
    """FITS I/O operations for field lines."""
    def __init__(self, field_line_set: FieldLineSet):
        self.data = field_line_set
        logger.info("Initialized FieldLineSetFitsIO")

    def get_hdu_name(self) -> str:
        """Get the name for the FITS HDU.
        
        Returns
        -------
        str
            Name for the FITS HDU
        """
        return "FIELD_LINES"

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
    
        return pf.Column(name=key, format=format, array=value)


    @classmethod
    def load_from_fits(cls, fitsname: str) -> FieldLineSet:
        """Load field line data from a FITS file.

        Parameters
        ----------
        fitsname : str
            Path to the FITS file

        Returns
        -------
        FieldLineSet
            Loaded field line set object

        Raises
        ------
        ValueError
            If required HDU is not found in the FITS file
        """
        logger.info(f"Loading field lines from {fitsname}")
        
        with pf.open(fitsname) as hdul:
            # Look for the FIELD_LINES HDU
            hdu_name = "FIELD_LINES"
            if hdu_name not in [h.name for h in hdul]:
                raise ValueError(f"No {hdu_name} HDU found in {fitsname}")
            
            hdu = hdul[hdu_name]
            
            # Create template FieldLine based on column names
            from src.line.field_line import FieldLine  # Import here to avoid circular import
            template = FieldLine()
            column_names = [col.name.upper() for col in hdu.columns]
            for prop in PropertyName:
                flag_enabled = prop.name in column_names
                setattr(template.save_flags, prop.flag_name, flag_enabled)
                logger.debug(f"Template: setting {prop.flag_name} to {flag_enabled}")
            
            # Create a new FieldLineSet with the template
            field_line_set = FieldLineSet(template)
            
            # Load each column into the appropriate property
            for col in hdu.columns:
                try:
                    # Convert column name to PropertyName enum
                    prop = PropertyName[col.name.upper()]
                    
                    # Read the column data
                    data = hdu.data[col.name]
                    
                    # Convert to numpy array if needed
                    if isinstance(data, np.ndarray) and data.dtype.kind == 'O':
                        data = np.array([np.array(x) for x in data])
                    
                    # Set the property in the FieldLineSet using set_property method
                    field_line_set._set_property(prop, data)
                    
                    logger.info(f"Loaded property: {prop.key_name}")
                except KeyError:
                    logger.warning(f"Unknown property name: {col.name}")
                except Exception as e:
                    logger.error(f"Error loading property {col.name}: {str(e)}")
                    raise
        
        logger.info("Successfully loaded field line data")
        return field_line_set