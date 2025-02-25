from typing import List, Type, Union
from astropy.io import fits as pf
from FieldLineDefinition import FieldLine
from FieldDefinition import (
    DipoleFieldComponent,
    QuadrupoleFieldComponent,
    MultipoleType,
    FieldComponentProtocol
)
from numpy.typing import NDArray
from numpy import ndarray, integer, floating

# Define FITS format mapping at module level
FITS_TYPE_MAPPING = {
    float: "D",
    int: "I",
    str: "10A",
    ndarray: "PE",  # Variable-length array of doubles
    integer: "I",  # numpy integers
    floating: "D",  # numpy floats
}


def get_fits_format(value):
    """Helper function to determine FITS format for a value."""
    for type_class, format_str in FITS_TYPE_MAPPING.items():
        if isinstance(value, type_class):
            return format_str
    return FITS_TYPE_MAPPING[type(value)]


class FitsIO:
    """Base class for FITS I/O operations."""

    def __init__(self, data):
        self.data = data

    def save_to_fits(self, filename: str) -> None:
        """Save data to a FITS file.

        Args:
            filename: Path to the output FITS file.
        """
        hdu = self.create_hdu()
        hdu.writeto(filename, overwrite=True)

    def create_hdu(self):
        """Create a FITS HDU from the data."""
        column_info = self.generate_column_info()
        columns = [
            self.create_column(key, column_info[key]) for key in column_info.keys()
        ]
        hdu = pf.BinTableHDU.from_columns(columns, name="TEST")
        return hdu

    def generate_column_info(self):
        """Generate column information for the FITS table."""
        raise NotImplementedError("Subclasses must implement generate_column_info")

    def create_column(self, key: str, format: str):
        """Create a FITS column for the given key and format."""
        raise NotImplementedError("Subclasses must implement create_column")

    @classmethod
    def load_from_fits(cls, hdu, component_class: FieldComponentProtocol):
        """Load data from a FITS HDU.

        Args:
            hdu: FITS HDU containing the data.
            component_class: Class to use for creating components.

        Returns:
            Instance of the appropriate component class.
        """
        data = hdu.data
        n_components = data.shape[0]
        components = [
            component_class(**{key: hdu.data[key][i] for key in hdu.columns.names})
            for i in range(n_components)
        ]
        return cls(components)


class FieldComponentFitsIO(FitsIO):
    """FITS I/O operations for field components."""

    def generate_column_info(self):
        """Generate column information for field component data."""
        first_component = self.data[0]
        if not isinstance(first_component, FieldComponentProtocol):
            raise TypeError(
                f"Expected field component, got {type(first_component).__name__}"
            )
       
        params_dict = first_component.params._asdict()
        

        return {
            param_name: (
                FITS_TYPE_MAPPING[str]
                if param_name == "moment_type"
                else (
                    FITS_TYPE_MAPPING[ndarray]
                    if param_name == "field_lines"
                    else get_fits_format(param_value)
                )
            )
            for param_name, param_value in params_dict.items()
            if param_value is not None
        }

    def create_column(self, key: str, format: str):
        """Create a FITS column for field component data.

        Args:
            key: Column name.
            format: FITS format string.

        Returns:
            FITS column object.
        """
        if key == "moment_type":
            value = [v.params.moment_type.value for v in self.data]
        else:
            value = [getattr(v.params, key) for v in self.data]
        return pf.Column(name=key, format=format, array=value)

    @classmethod
    def load_from_fits(cls, hdu):
        """Load field components from a FITS HDU.

        Args:
            hdu: FITS HDU containing the field component data.

        Returns:
            List of field component instances.
        """
        data = hdu.data
        n_components = data.shape[0]
        components = []
        
        for i in range(n_components):
            # Determine the component type from the moment_type column
            moment_type = MultipoleType(data['moment_type'][i])
            component_class = (
                DipoleFieldComponent if moment_type == MultipoleType.DIPOLE
                else QuadrupoleFieldComponent
            )
            
            # Create component instance with parameters from FITS file
            component_params = {
                key: data[key][i] for key in hdu.columns.names
                if key != 'moment_type'  # Handle moment_type separately
            }
            component_params['moment_type'] = moment_type
            
            components.append(component_class(**component_params))
            
        return cls(components)


class FieldLineFitsIO(FitsIO):
    """FITS I/O operations for field lines."""

    def generate_column_info(self):
        """Generate column information for field line data."""
        first_line = self.data[0]
        if not isinstance(first_line, FieldLine):
            raise TypeError(f"Expected FieldLine, got {type(first_line).__name__}")

        params_dict = vars(first_line)

        return {
            param_name: get_fits_format(param_value)
            for param_name, param_value in params_dict.items()
            if param_value is not None
        }

    def create_column(self, key: str, format: str):
        """Create a FITS column for field line data.

        Args:
            key: Column name.
            format: FITS format string.

        Returns:
            FITS column object.
        """
        value = [getattr(v, key) for v in self.data]
        return pf.Column(name=key, format=format, array=value)

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
