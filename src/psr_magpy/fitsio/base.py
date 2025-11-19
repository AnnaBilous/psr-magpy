"""Base classes for FITS I/O operations."""

import logging
from typing import Dict, Any, List, Optional, Union, Type, TypeVar, ClassVar
from pathlib import Path
import os

import numpy as np
from numpy.typing import NDArray
from numpy import ndarray, integer, floating

import astropy.io.fits as pf
from astropy.table import Table

from psr_magpy.utils.logging_config import get_logger

# Suppress matplotlib debug messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logger = get_logger(__name__, console_format_brief=True)

# Type variable for generic FitsIO subclasses
T = TypeVar("T", bound="FitsIO")


class FitsIO:
    """Base class for FITS I/O operations.

    This class provides common functionality for saving and loading data to/from
    FITS files. It handles type mapping, column creation, and basic I/O operations.

    Subclasses should implement the specific methods for their data types.
    """

    FITS_TYPE_MAPPING: ClassVar[Dict[Type, str]] = {
        float: "D",
        int: "I",
        str: "10A",
        list: "PE",  # Variable-length array of doubles
        ndarray: "D",
        integer: "I",  # numpy integers
        floating: "D",  # numpy floats
    }

    @staticmethod
    def get_fits_format(value: Any) -> str:
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
        logger.debug(
            f"Determining FITS format for value of type {type(value).__name__}"
        )

        # Handle None values
        if value is None:
            logger.debug("Value is None, using default format 'D'")
            return "D"

        # Handle arrays with shape
        if isinstance(value, ndarray) and value.size > 0:
            shape = value.shape
            if len(shape) > 1:
                format_str = f"{shape[1]}D"
                logger.debug(f"Array with shape {shape}, using format '{format_str}'")
                return format_str

        # Use type mapping
        for type_class, format_str in FitsIO.FITS_TYPE_MAPPING.items():
            if isinstance(value, type_class):
                logger.debug(
                    f"Found matching type {type_class.__name__}, using format '{format_str}'"
                )
                return format_str

        # Default case
        logger.warning(
            f"No format mapping found for type {type(value).__name__}, using default"
        )
        return FitsIO.FITS_TYPE_MAPPING[type(value)]

    def save_to_fits(self, filename: Union[str, Path]) -> None:
        """Save data to a FITS file.

        Parameters
        ----------
        filename : Union[str, Path]
            Path to the output FITS file

        Raises
        ------
        Exception
            If saving to FITS file fails
        """
        logger.debug(f"Saving data to {filename}")

        # Convert Path to string if needed
        filename_str = str(filename)

        # Create directory if it doesn't exist
        directory = os.path.dirname(filename_str)
        if directory and not os.path.exists(directory):
            logger.debug(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)

        try:
            logger.debug(f"Creating HDU for {filename_str}")
            hdu = self.create_hdu()

            # Check if file exists
            if os.path.exists(filename_str):
                logger.debug(f"File {filename_str} exists, will be overwritten")

            logger.debug(f"Writing HDU to {filename_str}")
            hdu.writeto(filename_str, overwrite=True)
            logger.debug(f"Successfully saved data to {filename_str}")

        except Exception as e:
            logger.error(f"Failed to save data to FITS: {e}")
            logger.debug("FITS save error details", exc_info=True)
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
            logger.debug("Generating column information")
            column_info = self.generate_column_info()
            logger.debug(f"Creating {len(column_info)} columns")

            columns: List[pf.Column] = []
            for key in column_info.keys():
                logger.debug(
                    f"Creating column '{key}' with format '{column_info[key]}'"
                )
                columns.append(self.create_column(key, column_info[key]))

            hdu_name = self.get_hdu_name()
            logger.debug(f"Creating binary table HDU with name '{hdu_name}'")
            hdu = pf.BinTableHDU.from_columns(columns, name=hdu_name)

            return hdu

        except Exception as e:
            logger.error(f"Failed to create HDU: {e}")
            logger.debug("HDU creation error details", exc_info=True)
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
    def load_from_fits(
        cls: Type[T], filename_or_hdu: Union[str, Path, pf.BinTableHDU]
    ) -> T:
        """Load data from a FITS file or HDU.

        Parameters
        ----------
        filename_or_hdu : Union[str, Path, pf.BinTableHDU]
            FITS filename or HDU containing the data

        Returns
        -------
        T
            Instance of the class

        Raises
        ------
        Exception
            If loading from FITS file fails
        """
        if isinstance(filename_or_hdu, (str, Path)):
            filename_str = str(filename_or_hdu)
            logger.info(f"Loading data from FITS file: {filename_str}")
            try:
                with pf.open(filename_str) as hdul:
                    # Find the first BinTableHDU
                    for hdu in hdul:
                        if isinstance(hdu, pf.BinTableHDU):
                            logger.debug(f"Found BinTableHDU: {hdu.name}")
                            return cls._load_from_hdu(hdu)

                    # If we get here, no BinTableHDU was found
                    logger.error(f"No BinTableHDU found in {filename_str}")
                    raise ValueError(f"No BinTableHDU found in {filename_str}")
            except Exception as e:
                logger.error(f"Failed to load data from FITS file {filename_str}: {e}")
                logger.debug("FITS load error details", exc_info=True)
                raise
        else:
            logger.info(f"Loading data from FITS HDU: {filename_or_hdu.name}")
            return cls._load_from_hdu(filename_or_hdu)

    @classmethod
    def _load_from_hdu(cls: Type[T], hdu: pf.BinTableHDU) -> T:
        """Load data from a FITS HDU.

        Parameters
        ----------
        hdu : pf.BinTableHDU
            FITS HDU containing the data

        Returns
        -------
        T
            Instance of the class

        Raises
        ------
        Exception
            If loading from FITS HDU fails
        """
        try:
            data: pf.FITS_rec = hdu.data
            n_components: int = data.shape[0]
            logger.info(f"Successfully loaded {n_components} components from HDU")

            return cls(data)  # type: ignore[call-arg]

        except Exception as e:
            logger.error(f"Failed to load data from FITS HDU: {e}")
            logger.debug("HDU load error details", exc_info=True)
            raise
