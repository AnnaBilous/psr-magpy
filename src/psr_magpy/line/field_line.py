import logging
import warnings
from typing import Optional, Any, Union, TYPE_CHECKING

from psr_magpy.utils.logging_config import get_logger
from psr_magpy.line.property_system import PropertySystem, PropertyName

if TYPE_CHECKING:
    from psr_magpy.line.field_line_template import FieldLineTemplate

logger = get_logger(__name__, console_format_brief=True)


class FieldLine:
    """Data container for field line tracing results.

    FieldLine holds actual field line data (coordinates, properties, etc.)
    from individual field line tracing operations. Field lines are combined
    into FieldLineSet for bulk data storage.

    **Role in I/O System:**
    - **Data Container**: Holds actual field line data (coordinates, properties, etc.)
    - **FITS Storage**: Data is stored in FITS format via FieldLineSet
    - **Individual Results**: Used to store single field line results

    **Configuration:**
    Use FieldLineTemplate to configure which properties to save during tracing.
    Create FieldLine instances from templates using FieldLine.from_template().
    """

    def __init__(self, **kwargs):
        """Initialize field line with optional save flags.

        Parameters
        ----------
        **kwargs : bool
            Boolean flags for which properties to save.
            save_footpoint_id will always be True regardless of input.

        Examples
        --------
        >>> field_line = FieldLine(save_r_line=True, save_r_max=True)
        >>> # save_footpoint_id is always enabled
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

        return (
            f"FieldLine(enabled_properties={enabled_flags}, "
            f"values={property_values})"
        )

    def get_property(self, prop: Union[PropertyName, str]) -> Optional[Any]:
        """Get a property value if it's enabled.

        Parameters
        ----------
        prop : Union[PropertyName, str]
            Property to get, either as PropertyName enum or string
            (e.g., PropertyName.R_MAX or 'r_max').

        Returns
        -------
        Optional[Any]
            Property value or None if not enabled/set.

        Raises
        ------
        ValueError
            If property is not enabled but accessed.
        KeyError
            If string property name is not valid.
        """
        return self.property_system.get_property(prop)

    def set_property(self, **kwargs: Any) -> None:
        """Set multiple property values if they're enabled.

        Parameters
        ----------
        **kwargs : Any
            Property values to set, where keys are PropertyName.key_name.

        Examples
        --------
        >>> field_line.set_property(r_line=coordinates, r_max=max_value)

        Raises
        ------
        ValueError
            If any property is not enabled but attempted to be set.
        KeyError
            If property name is not valid.
        """
        self.property_system.set_property(**kwargs)

    def copy_structure(self) -> "FieldLine":
        """Create a new instance with the same save flags configuration."""
        # Get the current save flags as a dictionary
        save_flags_dict = self.property_system.get_flags_dict()

        # Create new instance with same flags
        logger.debug(f"Copying structure with flags: {save_flags_dict}")
        new_instance = self.__class__(**save_flags_dict)
        logger.debug(
            f"Created new {self.__class__.__name__} instance with same structure"
        )
        return new_instance

    @classmethod
    def from_template(cls, template: "FieldLineTemplate") -> "FieldLine":
        """Create a FieldLine instance from a FieldLineTemplate.

        Parameters
        ----------
        template : FieldLineTemplate
            Template defining which properties to save.

        Returns
        -------
        FieldLine
            New FieldLine instance with same configuration as template.

        Examples
        --------
        >>> template = FieldLineTemplate.template_all()
        >>> field_line = FieldLine.from_template(template)
        """
        flags_dict = template.property_system.get_flags_dict()
        return cls(**flags_dict)

    @classmethod
    def template_all(cls) -> "FieldLineTemplate":
        """Create configuration template with all properties enabled.

        .. deprecated:: 
            Use FieldLineTemplate.template_all() instead.
            This method will be removed in a future version.

        Returns
        -------
        FieldLineTemplate
            Template with all properties enabled.
        """
        from psr_magpy.line.field_line_template import FieldLineTemplate

        warnings.warn(
            "FieldLine.template_all() is deprecated. Use FieldLineTemplate.template_all() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FieldLineTemplate.template_all()

    @classmethod
    def template_all_except_along_line(cls) -> "FieldLineTemplate":
        """Create configuration template with all properties enabled except line coordinates and PA.

        .. deprecated:: 
            Use FieldLineTemplate.template_all_except_along_line() instead.
            This method will be removed in a future version.

        Returns
        -------
        FieldLineTemplate
            Template with all properties except line coordinates and PA.
        """
        from psr_magpy.line.field_line_template import FieldLineTemplate

        warnings.warn(
            "FieldLine.template_all_except_along_line() is deprecated. Use FieldLineTemplate.template_all_except_along_line() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FieldLineTemplate.template_all_except_along_line()

    @classmethod
    def template_OFL_calc(cls) -> "FieldLineTemplate":
        """Create configuration template with properties for open field line calculation.

        .. deprecated:: 
            Use FieldLineTemplate.template_OFL_calc() instead.
            This method will be removed in a future version.

        Returns
        -------
        FieldLineTemplate
            Template with r_max, r_cyl_max, z_abs_max enabled.
        """
        from psr_magpy.line.field_line_template import FieldLineTemplate

        warnings.warn(
            "FieldLine.template_OFL_calc() is deprecated. Use FieldLineTemplate.template_OFL_calc() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FieldLineTemplate.template_OFL_calc()

    @classmethod
    def template_OFL_subset(cls) -> "FieldLineTemplate":
        """Create minimal configuration template for OFL subset grid creation.

        .. deprecated:: 
            Use FieldLineTemplate.template_OFL_subset() instead.
            This method will be removed in a future version.

        Returns
        -------
        FieldLineTemplate
            Template with r_cyl_max and z_abs_max enabled.
        """
        from psr_magpy.line.field_line_template import FieldLineTemplate

        warnings.warn(
            "FieldLine.template_OFL_subset() is deprecated. Use FieldLineTemplate.template_OFL_subset() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FieldLineTemplate.template_OFL_subset()

    @classmethod
    def template_line_coordinates_only(cls) -> "FieldLineTemplate":
        """Create configuration template with line coordinates only.

        .. deprecated:: 
            Use FieldLineTemplate.template_line_coordinates_only() instead.
            This method will be removed in a future version.

        Returns
        -------
        FieldLineTemplate
            Template with r_line enabled.
        """
        from psr_magpy.line.field_line_template import FieldLineTemplate

        warnings.warn(
            "FieldLine.template_line_coordinates_only() is deprecated. Use FieldLineTemplate.template_line_coordinates_only() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FieldLineTemplate.template_line_coordinates_only()

    @classmethod
    def template_properties_along_line(cls) -> "FieldLineTemplate":
        """Create configuration template with line coordinates and PA.

        .. deprecated:: 
            Use FieldLineTemplate.template_properties_along_line() instead.
            This method will be removed in a future version.

        Returns
        -------
        FieldLineTemplate
            Template with r_line, pa_line, b_line, curvature_line, tangent properties enabled.
        """
        from psr_magpy.line.field_line_template import FieldLineTemplate

        warnings.warn(
            "FieldLine.template_properties_along_line() is deprecated. Use FieldLineTemplate.template_properties_along_line() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FieldLineTemplate.template_properties_along_line()

    @classmethod
    def template_surface_field_properties(cls) -> "FieldLineTemplate":
        """Create configuration template with surface properties.

        .. deprecated:: 
            Use FieldLineTemplate.template_surface_field_properties() instead.
            This method will be removed in a future version.

        Returns
        -------
        FieldLineTemplate
            Template with b_ini and b_r_ini_angle enabled.
        """
        from psr_magpy.line.field_line_template import FieldLineTemplate

        warnings.warn(
            "FieldLine.template_surface_field_properties() is deprecated. Use FieldLineTemplate.template_surface_field_properties() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return FieldLineTemplate.template_surface_field_properties()
