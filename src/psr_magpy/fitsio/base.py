"""Base classes for FITS I/O operations."""

import logging
from typing import Dict

from numpy import ndarray, integer, floating

import astropy.io.fits as pf

# Setup logging
log_file = "test.log"

# Suppress matplotlib debug messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),  # This will print to console as well
    ],
)
logger = logging.getLogger(__name__)


class FitsIO:
    """Base class for FITS I/O operations."""

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

            hdu = pf.BinTableHDU.from_columns(columns, name=self.get_hdu_name())
            return hdu
        except Exception as e:
            logger.error(f"Failed to create HDU: {e}")
            raise

    def get_hdu_name(self) -> str:
        """Get the name for the FITS HDU.

        Returns
        -------
        str
            Name for the FITS HDU
        """
        return "DATA"

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
    def load_from_fits(cls, hdu: pf.BinTableHDU) -> "FitsIO":
        """Load data from a FITS HDU.

        Parameters
        ----------
        hdu : pf.BinTableHDU
            FITS HDU containing the data

        Returns
        -------
        FitsIO
            Instance of the class

        Raises
        ------
        Exception
            If loading from FITS file fails
        """
        logger.info(f"Loading data from FITS HDU")
        try:
            data = hdu.data
            n_components = data.shape[0]
            logger.info(f"Successfully loaded {n_components} components")
            return cls(data)
        except Exception as e:
            logger.error(f"Failed to load data from FITS: {e}")
            raise
