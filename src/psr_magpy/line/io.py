import logging
import psutil
from pathlib import Path

import numpy as np

from astropy.io import fits as pf

from psr_magpy.fitsio.base import FitsIO
from psr_magpy.line.field_line import PropertyName
from psr_magpy.line.field_line_set import FieldLineSet
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class FieldLineSetFitsIO(FitsIO):
    """FITS I/O operations for field lines."""

    # List of properties that are stored as lists of arrays
    ARRAY_LIST_PROPERTIES = [PropertyName.R_LINE, PropertyName.PA_LINE]

    def __init__(self, field_line_set: FieldLineSet):
        self.data = field_line_set
        logger.debug("Initialized FieldLineSetFitsIO")

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
            logger.debug(f"Processing property: {prop.key_name}, type: {type(value)}")

            # Get FITS format for the property
            result[prop.key_name] = FitsIO.get_fits_format(value)

            logger.debug(
                f"Added format for {prop.key_name}: {result.get(prop.key_name)}"
            )

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

        logger.debug(f"Creating column for {key} with format {format}")

        return pf.Column(name=key, format=format, array=value)

    @classmethod
    def load_from_fits(cls, fitsname: str) -> FieldLineSet:
        """Load field line data from one or more FITS files.

        Parameters
        ----------
        fitsname : str
            Base name for FITS files (without .fits extension)

        Returns
        -------
        FieldLineSet
            Combined field line set from all matching files

        Raises
        ------
        ValueError
            If no matching files found or if total size exceeds memory limit
        """
        logger.debug(f"Looking for FITS files matching {fitsname}*.fits")

        # Find all matching files
        base_path = Path(fitsname).parent
        pattern = f"{Path(fitsname).name}_?_of_?.fits"
        fits_files = list(base_path.glob(pattern))

        if not fits_files:
            raise ValueError(f"No FITS files found matching {fitsname}*.fits")

        # Calculate total size of all FITS files
        total_size = sum(f.stat().st_size for f in fits_files)
        available_memory = psutil.virtual_memory().available
        memory_threshold = 0.5  # 50% of available memory

        logger.info(
            f"Found {len(fits_files)} files, total size: {total_size/1e6:.1f} MB"
        )
        logger.info(f"Available memory: {available_memory/1e6:.1f} MB")

        if total_size > available_memory * memory_threshold:
            raise ValueError(
                f"Total FITS size ({total_size/1e6:.1f} MB) exceeds {memory_threshold*100}% "
                f"of available memory ({available_memory/1e6:.1f} MB)"
            )

        # Load first file to get template
        first_file = fits_files[0]
        logger.info(f"Loading first file {first_file} to create template")

        with pf.open(first_file) as hdul:
            if "FIELD_LINES" not in [h.name for h in hdul]:
                raise ValueError(f"No FIELD_LINES HDU found in {first_file}")

            hdu = hdul["FIELD_LINES"]

            # Create template based on column names
            from psr_magpy.line.field_line import FieldLine

            template = FieldLine()
            column_names = [col.name.upper() for col in hdu.columns]
            for prop in PropertyName:
                flag_enabled = prop.name in column_names
                setattr(template.save_flags, prop.flag_name, flag_enabled)
                logger.debug(f"Template: setting {prop.flag_name} to {flag_enabled}")

            # Create empty FieldLineSet with template
            field_line_set = FieldLineSet(template, copy_data=False)

        # Load data from all files
        for fits_file in fits_files:
            logger.info(f"Loading data from {fits_file}")
            with pf.open(fits_file) as hdul:
                hdu = hdul["FIELD_LINES"]
                
                # Load each column
                for col in hdu.columns:
                    try:
                        prop = PropertyName[col.name.upper()]
                        data = hdu.data[col.name]
                        
                        # Special handling for array list properties
                        if prop in cls.ARRAY_LIST_PROPERTIES:
                            # For R_LINE, reshape to 3D coordinates
                            if prop == PropertyName.R_LINE:
                                data = [d.reshape(-1, 3) for d in data]
                            # For other array properties, ensure they're properly shaped
                            else:
                                data = [np.array(d) for d in data]

                        # For first file, set the property
                        
                        if fits_file == first_file:
                            if prop in cls.ARRAY_LIST_PROPERTIES:
                                field_line_set._properties[prop] = data
                            else:
                                field_line_set._properties[prop] = data

                        # For subsequent files, append the data
                        else:
                            if prop in cls.ARRAY_LIST_PROPERTIES:
                                field_line_set._properties[prop].extend(data)
                            else:
                                field_line_set._properties[prop] = np.append(
                                    field_line_set._properties[prop], data, axis=0
                                )

                        logger.debug(
                            f"Loaded property {prop.key_name} from {fits_file}"
                        )
                    except KeyError:
                        logger.warning(f"Unknown property name: {col.name}")
                    except Exception as e:
                        logger.error(f"Error loading property {col.name}: {str(e)}")
                        raise

                # Update field line count
                field_line_set.n_field_lines += len(hdu.data)

        logger.info(
            f"Successfully loaded {field_line_set.n_field_lines} field lines from {len(fits_files)} files"
        )
        return field_line_set
