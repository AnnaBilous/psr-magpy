"""Field configuration classes for experiment setup."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field

from psr_magpy.field.components import (
    DipoleFieldComponent,
    QuadrupoleFieldComponent,
    MultipoleType,
)
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyFieldError

logger = get_logger(__name__, console_format_brief=True)


class FieldComponentConfig(BaseModel):
    """Configuration for a single field component."""

    moment_type: str
    B_magnitude: float = 1.0
    moment_orientation_latitude_degree: float = 0.0
    moment_orientation_longitude_degree: float = 0.0
    offset_r_ns: List[float] = [0.0, 0.0, 0.0]


class FieldConfig(BaseModel):
    """Configuration for magnetic field components."""

    components: List[FieldComponentConfig] = Field(
        description="List of field components"
    )

    def create_field(self) -> MultipoleField:
        """Create MultipoleField from configuration."""
        logger.debug(f"Creating field with {len(self.components)} components")

        field_components = []
        for i, comp in enumerate(self.components):
            comp_type = comp.moment_type
            logger.debug(f"Creating {comp_type} component {i+1}")

            if comp_type == MultipoleType.DIPOLE.value:
                component = DipoleFieldComponent(
                    B_magnitude=comp.B_magnitude,
                    moment_orientation_latitude_degree=comp.moment_orientation_latitude_degree,
                    moment_orientation_longitude_degree=comp.moment_orientation_longitude_degree,
                    offset_r_ns=comp.offset_r_ns,
                )
            elif comp_type == MultipoleType.QUADRUPOLE.value:
                component = QuadrupoleFieldComponent(
                    B_magnitude=comp.B_magnitude,
                    moment_orientation_latitude_degree=comp.moment_orientation_latitude_degree,
                    moment_orientation_longitude_degree=comp.moment_orientation_longitude_degree,
                    offset_r_ns=comp.offset_r_ns,
                )
            else:
                raise PSRMagPyFieldError(f"Unknown field component type: {comp_type}")

            field_components.append(component)

        return MultipoleField(field_components)

    def get_complete_components(self) -> List[Dict[str, Any]]:
        """Get complete component configurations including all default parameters."""
        logger.debug("Getting complete component configurations")

        complete_components = []
        for comp in self.components:
            comp_type = comp.moment_type

            # Create component to get all parameters (including defaults)
            if comp_type == MultipoleType.DIPOLE.value:
                component = DipoleFieldComponent(
                    B_magnitude=comp.B_magnitude,
                    moment_orientation_latitude_degree=comp.moment_orientation_latitude_degree,
                    moment_orientation_longitude_degree=comp.moment_orientation_longitude_degree,
                    offset_r_ns=comp.offset_r_ns,
                )
            elif comp_type == MultipoleType.QUADRUPOLE.value:
                component = QuadrupoleFieldComponent(
                    B_magnitude=comp.B_magnitude,
                    moment_orientation_latitude_degree=comp.moment_orientation_latitude_degree,
                    moment_orientation_longitude_degree=comp.moment_orientation_longitude_degree,
                    offset_r_ns=comp.offset_r_ns,
                )
            else:
                raise PSRMagPyFieldError(f"Unknown field component type: {comp_type}")

            # Get complete parameters including defaults
            complete_params = component.params.model_dump()
            # Convert enum to string for JSON serialization
            complete_params["moment_type"] = complete_params["moment_type"].value

            complete_components.append(complete_params)
            logger.debug(f"Complete params for {comp_type}: {complete_params}")

        return complete_components

