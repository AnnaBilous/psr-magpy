"""Grid configuration classes for experiment setup."""

from typing import Optional, Any, TYPE_CHECKING
from pydantic import BaseModel, Field, model_validator, field_validator

if TYPE_CHECKING:
    from astropy.coordinates import CartesianRepresentation

from psr_magpy.grid.grid import (
    SphericalGrid,
    RandomGrid,
    HealpixGrid,
    CustomGrid,
    FootpointGrid,
    GridType,
)
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyGridError

logger = get_logger(__name__, console_format_brief=True)


class SphericalGridConfig(BaseModel):
    """Configuration for spherical grid."""

    nlon: int
    nlat: int
    pole_lat: float = 0.0
    subset: bool = Field(default=False, description="Whether this is a subset grid")
    grid_file_path: Optional[str] = Field(
        default=None, description="Path to saved grid file (required for subset grids)"
    )


class RandomGridConfig(BaseModel):
    """Configuration for random grid."""

    nline_ori: int
    grid_file_path: Optional[str] = Field(
        default=None,
        description="Path to saved grid file (required when saving results to disk)",
    )


class HealpixGridConfig(BaseModel):
    """Configuration for HEALPix grid."""

    nside: int
    subset: bool = Field(default=False, description="Whether this is a subset grid")
    grid_file_path: Optional[str] = Field(
        default=None, description="Path to saved grid file (required for subset grids)"
    )

    @model_validator(mode="after")
    def validate_nside_power_of_two(self):
        """Validate that nside is a power of 2."""
        if not (self.nside & (self.nside - 1) == 0):
            raise ValueError(f"nside must be a power of 2, got {self.nside}")
        return self


class CustomGridConfig(BaseModel):
    """Configuration for custom grid."""

    nline_ori: int
    coordinates: Optional[Any] = None
    grid_file_path: Optional[str] = Field(
        default=None,
        description="Path to saved grid file (required when saving results to disk)",
    )

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("coordinates", mode="before")
    @classmethod
    def validate_coordinates(cls, v):
        """Validate and normalize coordinates format."""
        if v is None:
            return v
        
        from astropy.coordinates import CartesianRepresentation
        import numpy as np
        
        if isinstance(v, list):
            # Check if it's a list of points (each point is [x, y, z]) or [x_array, y_array, z_array]
            if len(v) > 0:
                first_elem = v[0]
                # If first element is a list/array, assume it's a list of points
                if isinstance(first_elem, (list, np.ndarray)) and len(first_elem) == 3:
                    # Format: [[x1, y1, z1], [x2, y2, z2], ...]
                    # Convert to [x_array, y_array, z_array] format
                    x_coords = [point[0] for point in v]
                    y_coords = [point[1] for point in v]
                    z_coords = [point[2] for point in v]
                    return CartesianRepresentation(
                        x=np.array(x_coords),
                        y=np.array(y_coords),
                        z=np.array(z_coords),
                    )
                elif len(v) == 3:
                    # Format: [x_array, y_array, z_array]
                    return CartesianRepresentation(
                        x=np.array(v[0]),
                        y=np.array(v[1]),
                        z=np.array(v[2]),
                    )
            raise ValueError(
                f"Coordinates list must be either list of [x,y,z] points or [x_array, y_array, z_array], got {type(v)}"
            )
        elif isinstance(v, CartesianRepresentation):
            return v
        else:
            raise ValueError(f"Coordinates must be list or CartesianRepresentation, got {type(v)}")

    @model_validator(mode="after")
    def validate_coordinates_length(self):
        """Validate that coordinates length matches nline_ori."""
        if self.coordinates is not None:
            if len(self.coordinates.x) != self.nline_ori:
                raise ValueError(
                    f"Coordinates length {len(self.coordinates.x)} != nline_ori {self.nline_ori}"
                )
        return self


class GridConfig(BaseModel):
    """Configuration for field line grid."""

    spherical: Optional[SphericalGridConfig] = Field(
        default=None, description="Spherical grid parameters"
    )
    random: Optional[RandomGridConfig] = Field(
        default=None, description="Random grid parameters"
    )
    healpix: Optional[HealpixGridConfig] = Field(
        default=None, description="HEALPix grid parameters"
    )
    custom: Optional[CustomGridConfig] = Field(
        default=None, description="Custom grid parameters"
    )

    @model_validator(mode="after")
    def validate_grid_type(self):
        """Validate that exactly one grid type is specified."""
        configs = [self.spherical, self.random, self.healpix, self.custom]
        non_none_configs = [c for c in configs if c is not None]

        # Skip validation for temporary objects (all fields are None or default)
        if len(non_none_configs) == 0:
            # This might be a temporary object during I/O operations
            # Only log a warning instead of raising an error
            logger.warning(
                "GridConfig created with no grid type specified (likely temporary object)"
            )
            return self

        if len(non_none_configs) != 1:
            raise ValueError(
                f"Exactly one grid type must be specified, got {len(non_none_configs)}"
            )
        
        return self

    def get_grid_type(self) -> str:
        """Get the grid type from the configuration."""
        if self.spherical is not None:
            return GridType.SPHERICAL.value
        elif self.random is not None:
            return GridType.RANDOM.value
        elif self.healpix is not None:
            return GridType.HEALPIX.value
        elif self.custom is not None:
            return GridType.CUSTOM.value

    def create_grid(self) -> Optional[FootpointGrid]:
        """Create grid from config."""
        if self.random is not None:
            config = self.random
            if config.grid_file_path:
                logger.info(f"Loading RandomGrid from {config.grid_file_path}")
                return RandomGrid.load_from_fits(config.grid_file_path)
            else:
                logger.info("Creating RandomGrid from scratch")
                params = config.model_dump()
                params.pop(
                    "grid_file_path", None
                )  # Remove file path when creating from scratch
                return RandomGrid(**params)
        elif self.custom is not None:
            config = self.custom
            if config.grid_file_path:
                logger.info(f"Loading CustomGrid from {config.grid_file_path}")
                return CustomGrid.load_from_fits(config.grid_file_path)
            else:
                logger.info("Creating CustomGrid from scratch")
                params = config.model_dump()
                params.pop(
                    "grid_file_path", None
                )  # Remove file path when creating from scratch
                # Coordinates are already validated and normalized by CustomGridConfig validator
                return CustomGrid(**params)
        elif self.spherical is not None:
            config = self.spherical
            if config.subset and config.grid_file_path:
                logger.info(
                    f"Loading subset SphericalGrid from {config.grid_file_path}"
                )
                return SphericalGrid.load_from_fits(config.grid_file_path)
            else:
                logger.info("Creating SphericalGrid from scratch")
                params = config.model_dump()
                params.pop(
                    "grid_file_path", None
                )  # Remove file path when creating from scratch
                params.pop(
                    "subset", None
                )  # Remove subset flag when creating from scratch
                return SphericalGrid(**params)
        elif self.healpix is not None:
            config = self.healpix
            if config.subset and config.grid_file_path:
                logger.info(f"Loading subset HealpixGrid from {config.grid_file_path}")
                return HealpixGrid.load_from_fits(config.grid_file_path)
            else:
                logger.info("Creating HealpixGrid from scratch")
                params = config.model_dump()
                params.pop(
                    "grid_file_path", None
                )  # Remove file path when creating from scratch
                params.pop(
                    "subset", None
                )  # Remove subset flag when creating from scratch
                return HealpixGrid(**params)
        else:
            raise PSRMagPyGridError("No grid type specified")

