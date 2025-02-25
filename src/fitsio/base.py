"""Base classes for FITS I/O operations."""

import astropy.io.fits as pf
from typing import Any, Dict, Type
import logging
from src.field.components import FieldComponentProtocol
from numpy import ndarray, integer, floating

# Setup logging
log_file = "test.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # This will print to console as well
    ]
)
logger = logging.getLogger(__name__)

class FitsIO:
    """Base class for FITS I/O operations."""

    # Move FITS_TYPE_MAPPING inside class as a class variable
    FITS_TYPE_MAPPING = {
        float: "D",
        int: "I",
        str: "10A",
        list: "PE",  # Variable-length array of doubles
        ndarray: "D",  
        integer: "I",  # numpy integers
        floating: "D",  # numpy floats
    }

    @staticmethod
    def get_fits_format(value):
        """Helper function to determine FITS format for a value.
        
        Parameters
        ----------
        value : Any
            The value to determine FITS format for
            
        Returns
        -------
        str
            FITS format string for the given value
        """
        for type_class, format_str in FitsIO.FITS_TYPE_MAPPING.items():
            if isinstance(value, type_class):
                return format_str
        return FitsIO.FITS_TYPE_MAPPING[type(value)]

    def __init__(self, data: Any):
        """Initialize FitsIO with data to be saved/loaded.
        
        Parameters
        ----------
        data : Any
            Data to be saved to or loaded from FITS files
        """
        self.data = data

    def save_to_fits(self, filename: str) -> None:
        """Save data to a FITS file.

        Parameters
        ----------
        filename : str
            Path to the output FITS file
            
        Raises
        ------
        Exception
            If saving to FITS file fails
        """
        logger.info(f"Saving data to {filename}")
        try:
            logger.debug(f"Creating HDU for {filename}")
            hdu = self.create_hdu()
            logger.debug(f"Writing HDU to {filename}")
            hdu.writeto(filename, overwrite=True)
            logger.info("Successfully saved data to FITS file")
        except Exception as e:
            logger.error(f"Failed to save data to FITS: {e}")
            raise

    def create_hdu(self) -> pf.BinTableHDU:
        """Create a FITS HDU from the data.
        
        Returns
        -------
        pf.BinTableHDU
            FITS binary table HDU containing the data
            
        Raises
        ------
        Exception
            If HDU creation fails
        """
        try:
            column_info = self.generate_column_info()
            columns = [
                self.create_column(key, column_info[key]) for key in column_info.keys()
            ]
            print(columns)
            for i in columns:
                print(i.array)
            #exit(1)
            hdu = pf.BinTableHDU.from_columns(columns, name="TEST")
            return hdu
        except Exception as e:
            logger.error(f"Failed to create HDU: {e}")
            raise

    def generate_column_info(self) -> Dict[str, str]:
        """Generate column information for the FITS table.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping column names to their FITS format strings
            
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement generate_column_info")

    def create_column(self, key: str, format: str) -> pf.Column:
        """Create a FITS column for the given key and format.
        
        Parameters
        ----------
        key : str
            Column name
        format : str
            FITS format string for the column
            
        Returns
        -------
        pf.Column
            FITS column object
            
        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement create_column")

    @classmethod
    def load_from_fits(cls, hdu: pf.BinTableHDU, component_class: Type[FieldComponentProtocol]) -> 'FitsIO':
        """Load data from a FITS HDU.

        Parameters
        ----------
        hdu : pf.BinTableHDU
            FITS HDU containing the data
        component_class : Type[FieldComponentProtocol]
            Class to use for creating components

        Returns
        -------
        FitsIO
            Instance of the appropriate component class
            
        Raises
        ------
        Exception
            If loading from FITS file fails
        """
        logger.info(f"Loading data from FITS HDU using {component_class.__name__}")
        try:
            data = hdu.data
            n_components = data.shape[0]
            components = [
                component_class(**{key: hdu.data[key][i] for key in hdu.columns.names})
                for i in range(n_components)
            ]
            logger.info(f"Successfully loaded {n_components} components")
            return cls(components)
        except Exception as e:
            logger.error(f"Failed to load data from FITS: {e}")
            raise 