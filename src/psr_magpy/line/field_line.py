import logging
from typing import Optional, Any, Union

from psr_magpy.utils.logging_config import get_logger
from psr_magpy.line.property_system import PropertySystem, PropertyName

logger = get_logger(__name__, console_format_brief=True)


class FieldLine:
    """Base class for field lines with configurable properties.

    This class provides the basic structure for storing field line properties
    and managing which properties should be saved.
    """

    def __init__(self, **kwargs):
        """Initialize field line with optional save flags.

        Args:
            **kwargs: Boolean flags for which properties to save.
                     save_footpoint_id will always be True regardless of input.
                     
                     Property groups can be specified:
                     - save_all: Enable all properties
                     - save_line_data: Enable r_line, pa_line
                     - save_line_max: Enable r_max, r_cyl_max, z_abs_max
                     - save_b_ini: Enable b_ini, b_r_ini_angle
        """
        # Initialize property system, ensuring footpoint_id is True
        kwargs["save_footpoint_id"] = True
        self.property_system = PropertySystem(**kwargs)
        self.save_flags = self.property_system.save_flags
        logger.debug(f"Initialized FieldLine with save flags: {self.save_flags}")

        # Log enabled properties at debug level
        if logger.isEnabledFor(logging.DEBUG):
            enabled_props = self.property_system.get_enabled_property_names()
            logger.debug(f"Enabled properties: {enabled_props}")

    def __repr__(self) -> str:
        """Return string representation of the field line."""
        enabled_flags = [
            flag for flag in vars(self.save_flags) if getattr(self.save_flags, flag)
        ]
        
        # Add property values to the representation
        property_values = self.property_system.get_property_values()
        
        return (f"FieldLine(enabled_properties={enabled_flags}, "
                f"values={property_values})")

    def get_property(self, prop: Union[PropertyName, str]) -> Optional[Any]:
        """Get a property value if it's enabled.

        Args:
            prop: Property to get, either as PropertyName enum or string
                 (e.g., PropertyName.R_MAX or 'r_max')

        Returns:
            Property value or None if not enabled/set

        Raises:
            ValueError: If property is not enabled but accessed
            KeyError: If string property name is not valid
        """
        return self.property_system.get_property(prop)

    def set_property(self, **kwargs: Any) -> None:
        """Set multiple property values if they're enabled.

        Args:
            **kwargs: Property values to set, where keys are PropertyName.key_name
                     Example: set_property(r_line=coordinates, r_max=max_value)

        Raises:
            ValueError: If any property is not enabled but attempted to be set
            KeyError: If property name is not valid
        """
        self.property_system.set_property(**kwargs)

    def copy_structure(self) -> "FieldLine":
        """Create a new instance with the same save flags configuration.

        Returns:
            A new FieldLineBase instance with the same save flags but empty properties
        """
        # Get the current save flags as a dictionary
        save_flags_dict = self.property_system.get_flags_dict()

        # Create new instance with same flags
        logger.debug(f"Copying structure with flags: {save_flags_dict}")
        new_instance = self.__class__(**save_flags_dict)
        logger.debug(f"Created new {self.__class__.__name__} instance with same structure")
        return new_instance

    @classmethod
    def template_all(cls) -> "FieldLine":
        """Create template with all properties enabled.
        
        Returns:
            FieldLine with all properties enabled
        """
        return cls(save_all=True)
    
    @classmethod
    def template_all_except_line_and_PA(cls) -> "FieldLine":
        """Create template with all properties enabled except line coordinates and PA.
        
        Returns:
            FieldLine with all properties enabled except line coordinates and PA
        """
        return cls(save_r_max=True, save_r_cyl_max=True, save_z_abs_max=True, save_b_ini=True, save_b_r_ini_angle=True)
    
    @classmethod
    def template_OFL_calc(cls) -> "FieldLine":
        """Create template with properties for open field line calculation (r_max, r_cyl_max, z_abs_max).
        
        Returns:
            FieldLine with basic properties enabled
        """
        return cls(save_r_max=True, save_r_cyl_max=True, save_z_abs_max=True)
    
    @classmethod
    def template_coordinates_only(cls) -> "FieldLine":
        """Create template with line coordinates only (r_line).
        
        Returns:
            FieldLine with line coordinates enabled
        """
        return cls(save_r_line=True)
    
    @classmethod
    def template_coordinates_and_PA(cls) -> "FieldLine":
        """Create template with line coordinates and PA (r_line, pa_line).
        
        Returns:
            FieldLine with line coordinates and PA enabled
        """
        return cls(save_r_line=True, save_pa_line=True)
    
    @classmethod
    def template_surface_properties(cls) -> "FieldLine":
        """Create template with surface properties (b_ini, b_r_ini_angle).
        
        Returns:
            FieldLine with surface properties enabled
        """
        return cls(save_b_ini=True, save_b_r_ini_angle=True)
