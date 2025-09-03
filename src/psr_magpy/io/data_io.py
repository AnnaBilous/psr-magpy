"""FITS I/O implementations for scientific data."""

from abc import abstractmethod
from pathlib import Path
from typing import Any, Union, Optional
import numpy as np
from astropy.io import fits

from .base import DataIO, IOUtils
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class FitsIO(DataIO):
    """Base class for FITS I/O operations."""

    def __init__(self, data=None, validator=None, memory_monitor=None):
        """Initialize FITS I/O with data, optional validator, and memory monitor."""
        super().__init__(validator, memory_monitor)
        self.data = data

    def save(self, data: Any, path: Union[str, Path], **kwargs) -> None:
        """Save data to FITS file with validation and recovery."""
        # Set overwrite=True by default
        kwargs.setdefault("overwrite", True)

        # Validate with recovery options
        if not self.validate_with_recovery(data, path, "save", **kwargs):
            error_msg = f"Validation failed for save operation: {path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        path = IOUtils.ensure_directory(path)

        with self._track_io_progress("save", path, data):
            with IOUtils.safe_file_operation("save", path):
                hdu = self.create_hdu()
                hdu.writeto(str(path), overwrite=True)

    def load(self, path: Union[str, Path], **kwargs) -> Any:
        """Load data from FITS file with validation and recovery."""
        # Validate with recovery options
        if not self.validate_with_recovery(None, path, "load", **kwargs):
            error_msg = f"Validation failed for load operation: {path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        path = IOUtils.validate_path(path, must_exist=True)

        with self._track_io_progress("load", path):
            with IOUtils.safe_file_operation("load", path):
                with fits.open(str(path)) as hdul:
                    return self._load_from_hdul(hdul, **kwargs)

    def validate(self, data: Any) -> bool:
        """Validate data before saving."""
        return True  # Override in subclasses

    @abstractmethod
    def create_hdu(self) -> fits.HDUList:
        """Create FITS HDU for the data."""
        pass

    @abstractmethod
    def _load_from_hdul(self, hdul: fits.HDUList, **kwargs) -> Any:
        """Load data from FITS HDU list."""
        pass


class FieldLineSetIO(FitsIO):
    """FITS I/O for field line sets."""

    # List of properties that are stored as lists of arrays
    ARRAY_LIST_PROPERTIES = []  # Will be set when we import PropertyName

    def __init__(self, field_line_set=None, validator=None, memory_monitor=None):
        super().__init__(validator=validator, memory_monitor=memory_monitor)
        self.data = field_line_set
        self.header_keywords = {}

    def validate(self, data: Any) -> bool:
        """Validate field line set data."""
        try:
            if data is None:
                logger.error("Field line set data is None")
                return False

            # Check required attributes for new FieldLineSet structure
            required_attrs = ["property_system", "save_flags", "n_field_lines"]
            for attr in required_attrs:
                if not hasattr(data, attr):
                    logger.error(f"Missing required attribute: {attr}")
                    return False

            # Additional validation for non-None data
            if data is not None:
                if not hasattr(data, "save_flags") or not hasattr(
                    data.save_flags, "save_footpoint_id"
                ):
                    logger.error("Missing save_flags attribute")
                    return False

                if not hasattr(data, "property_system") or data.property_system is None:
                    logger.error("Missing or None property_system attribute")
                    return False

            logger.debug("Field line set validation passed")
            return True
        except Exception as e:
            logger.error(f"Field line set validation failed: {e}")
            return False

    def create_hdu(self) -> fits.HDUList:
        """Create FITS HDU for field line set."""
        from psr_magpy.line.property_system import PropertyName

        if self.data is None:
            raise ValueError("Field line set data is None")

        # Set ARRAY_LIST_PROPERTIES if not already set
        if not self.ARRAY_LIST_PROPERTIES:
            self.ARRAY_LIST_PROPERTIES = [
                PropertyName.R_LINE,
                PropertyName.PA_LINE,
                PropertyName.B_LINE,
                PropertyName.CURVATURE_LINE,
                PropertyName.TANGENT_THETA,
                PropertyName.TANGENT_PHI,
            ]

        # Generate column information
        column_info = self._generate_column_info()

        # Create columns
        columns = []
        for key, format_str in column_info.items():
            prop = PropertyName[key.upper()]
            value = self.data.property_system.get_property(prop)
            columns.append(fits.Column(name=key, format=format_str, array=value))

        # Create HDU
        hdu = fits.BinTableHDU.from_columns(columns, name="FIELD_LINES")

        # Add user-specified keywords to header if provided
        if hasattr(self, "header_keywords") and self.header_keywords:
            for keyword, value in self.header_keywords.items():
                if isinstance(value, tuple) and len(value) == 2:
                    # Value is (value, comment) tuple
                    hdu.header[keyword] = value
                else:
                    # Value is just the value
                    hdu.header[keyword] = value
                logger.debug(f"Added keyword to header: {keyword} = {value}")

        return fits.HDUList([fits.PrimaryHDU(), hdu])

    def _generate_column_info(self) -> dict:
        """Generate column information for field line data."""
        from psr_magpy.line.property_system import PropertyName

        result = {}

        # Process each enabled property
        for prop in PropertyName:
            if not getattr(self.data.save_flags, prop.flag_name):
                logger.debug(f"Skipping disabled property: {prop.key_name}")
                continue

            # Check if property exists in PropertySystem
            value = self.data.property_system.get_property(prop)
            if value is None:
                logger.warning(f"Property {prop.key_name} is enabled but not set")
                continue

            logger.debug(f"Processing property: {prop.key_name}, type: {type(value)}")

            # Get FITS format for the property
            result[prop.key_name] = IOUtils.get_fits_format(value)

            logger.debug(
                f"Added format for {prop.key_name}: {result.get(prop.key_name)}"
            )

        return result

    def add_header_keyword(self, keyword: str, value, comment: Optional[str] = None):
        """
        Add a keyword to the FITS header.

        Args:
            keyword: Header keyword name
            value: Header keyword value
            comment: Optional comment for the keyword
        """
        if comment:
            self.header_keywords[keyword] = (value, comment)
        else:
            self.header_keywords[keyword] = value
        logger.debug(f"Added header keyword: {keyword} = {value}")

    def _load_from_hdul(self, hdul: fits.HDUList, **kwargs) -> Any:
        """Load field line set from FITS HDU list."""
        from psr_magpy.line.field_line_set import FieldLineSet
        from psr_magpy.line.field_line import FieldLine
        from psr_magpy.line.property_system import PropertyName
        import psutil

        # Find FIELD_LINES HDU
        if "FIELD_LINES" not in [h.name for h in hdul]:
            error_msg = "No FIELD_LINES HDU found in FITS file"
            logger.error(error_msg)
            raise ValueError(error_msg)

        hdu = hdul["FIELD_LINES"]
        column_names = [col.name.upper() for col in hdu.columns]

        # Check if template is provided in kwargs
        template = kwargs.get("template")
        if template is not None:
            # Validate that all required columns from template exist in FITS file
            missing_columns = []
            for prop in PropertyName:
                if getattr(template.save_flags, prop.flag_name):
                    if prop.name not in column_names:
                        missing_columns.append(prop.name)

            if missing_columns:
                error_msg = f"Template requires columns that are missing from FITS file: {missing_columns}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.debug(
                f"Using provided template with enabled properties: {template.property_system.get_enabled_property_names()}"
            )
        else:
            # Create template based on column names (original behavior)
            template = FieldLine()
            for prop in PropertyName:
                flag_enabled = prop.name in column_names
                setattr(template.save_flags, prop.flag_name, flag_enabled)
                logger.debug(f"Template: setting {prop.flag_name} to {flag_enabled}")

        # Create empty FieldLineSet with template
        field_line_set = FieldLineSet(template, copy_data=False)

        # Load data from HDU (only for enabled properties in template)
        for col in hdu.columns:
            try:
                prop = PropertyName[col.name.upper()]

                # Only load properties that are enabled in the template
                if not getattr(template.save_flags, prop.flag_name):
                    logger.debug(
                        f"Skipping property {prop.key_name} (not enabled in template)"
                    )
                    continue

                data = hdu.data[col.name]

                # Special handling for array list properties
                if prop in self.ARRAY_LIST_PROPERTIES:
                    if prop.name == "R_LINE":
                        data = [d.reshape(-1, 3) for d in data]
                    else:
                        data = [np.array(d, dtype=object) for d in data]

                field_line_set.property_system._properties[prop] = data
                logger.debug(f"Loaded property {prop.key_name}")

            except KeyError:
                logger.warning(f"Unknown property name: {col.name}")
            except Exception as e:
                error_msg = f"Error loading property {col.name}: {str(e)}"
                logger.error(error_msg)
                raise

        # Update field line count
        field_line_set.n_field_lines = len(hdu.data)

        logger.info(f"Successfully loaded {field_line_set.n_field_lines} field lines")
        return field_line_set


class FieldComponentIO(FitsIO):
    """FITS I/O for field components."""

    def __init__(
        self,
        data=None,
        field=None,
        validator=None,
        memory_monitor=None,
        progress_tracker=None,
    ):
        super().__init__(
            data=data or field, validator=validator, memory_monitor=memory_monitor
        )
        self.progress_tracker = progress_tracker

    def validate(self, data: Any) -> bool:
        """Validate field component data."""
        # Add validation logic here
        return True

    def create_hdu(self) -> fits.HDUList:
        """Create FITS HDU for field components."""
        if self.data is None:
            raise ValueError("Field data is None")

        # Create primary HDU
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["NCOMPONENTS"] = len(self.data.field_components)

        # Generate column information
        column_info = self._generate_column_info()

        # Create columns
        columns = []
        for param_name, format_str in column_info.items():
            if param_name == "moment_type":
                value = [v.params.moment_type.value for v in self.data.field_components]
            else:
                value = [
                    getattr(v.params, param_name) for v in self.data.field_components
                ]
            columns.append(fits.Column(name=param_name, format=format_str, array=value))

        # Create data HDU
        data_hdu = fits.BinTableHDU.from_columns(columns, name="FIELD_COMPONENTS")

        return fits.HDUList([primary_hdu, data_hdu])

    def _generate_column_info(self) -> dict:
        """Generate column information for field component data."""
        if self.data is None or not self.data.field_components:
            return {}

        first_component = self.data.field_components[0]
        params_dict = (
            first_component.params.model_dump()
        )  # Use model_dump() for Pydantic

        column_info = {}
        for param_name, param_value in params_dict.items():
            if param_name == "moment_type":
                column_info[param_name] = "10A"  # String format
            elif param_name == "offset_r_ns":
                column_info[param_name] = "PE"  # Variable-length array
            else:
                column_info[param_name] = IOUtils.get_fits_format(param_value)

        return column_info

    def _load_from_hdul(self, hdul: fits.HDUList, **kwargs) -> Any:
        """Load field components from FITS HDU list."""
        from psr_magpy.field.components import (
            MultipoleType,
            DipoleFieldComponent,
            QuadrupoleFieldComponent,
        )
        from psr_magpy.field.multipole import MultipoleField

        if "FIELD_COMPONENTS" not in hdul:
            raise ValueError("No FIELD_COMPONENTS HDU found in FITS file")

        field_data = hdul["FIELD_COMPONENTS"].data
        n_components = len(field_data)
        logger.debug(f"Found {n_components} field components in FITS file")

        components = []
        for i in range(n_components):
            moment_type = MultipoleType(field_data["moment_type"][i])
            logger.debug(f"Loading component {i+1}/{n_components}: {moment_type.value}")

            if moment_type == MultipoleType.DIPOLE:
                component_class = DipoleFieldComponent
            elif moment_type == MultipoleType.QUADRUPOLE:
                component_class = QuadrupoleFieldComponent
            else:
                error_msg = f"Unsupported multipole type: {moment_type}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Extract component parameters
            component_params = {}
            for col_name in field_data.columns.names:
                if col_name != "moment_type":
                    component_params[col_name] = field_data[col_name][i]

            logger.debug(f"Component parameters: {component_params}")
            component = component_class(**component_params)
            components.append(component)

        logger.info(f"Successfully loaded {n_components} field components")
        return MultipoleField(components)


class GridIO(FitsIO):
    """FITS I/O for grids."""

    def __init__(self, data=None, validator=None, memory_monitor=None):
        super().__init__(validator=validator, memory_monitor=memory_monitor)
        self.data = data

    def validate(self, data: Any) -> bool:
        """Validate grid data."""
        try:
            if data is None:
                logger.error("Grid data is None")
                return False

            # Check required attributes
            required_attrs = ["grid_type", "footpoint_coordinates", "indices", "nline"]
            for attr in required_attrs:
                if not hasattr(data, attr):
                    logger.error(f"Missing required attribute: {attr}")
                    return False

            logger.debug("Grid validation passed")
            return True
        except Exception as e:
            logger.error(f"Grid validation failed: {e}")
            return False

    def create_hdu(self) -> fits.HDUList:
        """Create FITS HDU for grid."""
        if self.data is None:
            raise ValueError("Grid data is None")

        # Create primary HDU with header
        hdu_main = fits.PrimaryHDU()
        header_main = hdu_main.header
        header_main.set("GRIDTYPE", self.data.grid_type.value, "Grid type")
        header_main.set(
            "NLINE", self.data.nline_ori, "Number of field lines in a full grid"
        )
        header_main.set("PARTIAL", self.data.partial, "Partial grid")

        # Add grid-specific parameters to header
        self.data._add_params_to_header(header_main)

        # Create coordinates HDU
        from astropy.nddata import CCDData

        data = CCDData(
            self.data.footpoint_coordinates.xyz,
            unit=self.data.footpoint_coordinates.xyz.unit,
        )
        hdu_footpoints = data.to_hdu(as_image_hdu=True)[0]
        hdu_footpoints.name = "FTP_CRD"

        # Create indices HDU
        col = fits.Column(name="footpoint_id", format="K", array=self.data.indices)
        hdu_indices = fits.BinTableHDU.from_columns([col], name="FTP_ID")

        # Create HDU list
        return fits.HDUList([hdu_main, hdu_indices, hdu_footpoints])

    def _load_from_hdul(self, hdul: fits.HDUList, **kwargs) -> Any:
        """Load grid from FITS HDU list."""
        from psr_magpy.grid.grid import FootpointGrid, GridType
        from astropy.coordinates import CartesianRepresentation

        # Extract header information
        header_main = hdul[0].header
        grid_type = GridType(header_main["GRIDTYPE"])

        # Create grid instance based on type
        grid = self._create_grid_from_header(header_main, grid_type)

        # Load coordinates
        footpoint_coordinates = hdul["FTP_CRD"].data
        grid.footpoint_coordinates = CartesianRepresentation(footpoint_coordinates)

        # Load indices
        grid.indices = np.array([row[0] for row in hdul["FTP_ID"].data])
        grid.partial = header_main["PARTIAL"]
        grid.nline = len(grid.indices)

        return grid

    def _create_grid_from_header(self, header: fits.Header, grid_type) -> Any:
        """Create grid instance from header based on type."""
        from psr_magpy.grid.grid import (
            SphericalGrid,
            HealpixGrid,
            RandomGrid,
            CustomGrid,
            GridType,
        )

        grid_classes = {
            GridType.SPHERICAL: SphericalGrid,
            GridType.HEALPIX: HealpixGrid,
            GridType.RANDOM: RandomGrid,
            GridType.CUSTOM: CustomGrid,
        }

        grid_class = grid_classes.get(grid_type)
        if grid_class is None:
            error_msg = f"Unknown grid type: {grid_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return grid_class.from_header(header)
