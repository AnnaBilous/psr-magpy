"""Field line template for configuration.

FieldLineTemplate defines which properties to save during field line tracing.
It is an immutable configuration object used to configure the tracing process.
Templates are stored as YAML configuration files.
"""

import logging
from typing import Dict, List

from psr_magpy.utils.logging_config import get_logger
from psr_magpy.line.property_system import PropertySystem, PropertyName

logger = get_logger(__name__, console_format_brief=True)


class FieldLineTemplate:
    """Immutable configuration template for field line tracing.

    FieldLineTemplate defines which properties to save during field line tracing.
    It is used as a configuration object, not for storing actual field line data.

    **Role in I/O System:**
    - **Configuration Template**: Defines which properties to save
    - **YAML Storage**: Templates are stored as YAML configuration files
    - **Human-readable**: Configuration format for experiments

    **Usage:**
    - Pass to FieldLineTracer.trace_single_line() to configure tracing
    - Pass to FieldLineCalculator.get_field_lines() to configure calculation
    - Pass to FieldLineSet.__init__() to initialize data structure
    - Store in ExperimentConfig for reproducible experiments
    """

    def __init__(self, **kwargs):
        """Initialize template with optional save flags.

        Parameters
        ----------
        **kwargs : bool
            Boolean flags for which properties to save.
            save_footpoint_id will always be True regardless of input.

        Examples
        --------
        >>> template = FieldLineTemplate(save_r_line=True, save_r_max=True)
        >>> # save_footpoint_id is always enabled
        """
        # Initialize property system, ensuring footpoint_id is True
        kwargs["save_footpoint_id"] = True
        self._property_system = PropertySystem(**kwargs)
        self._save_flags = self._property_system.save_flags
        logger.debug(f"Initialized FieldLineTemplate with save flags: {self._save_flags}")

        # Log enabled properties at debug level
        if logger.isEnabledFor(logging.DEBUG):
            enabled_props = self._property_system.get_enabled_property_names()
            logger.debug(f"Enabled properties: {enabled_props}")

    @property
    def property_system(self) -> PropertySystem:
        """Get the property system (read-only)."""
        return self._property_system

    @property
    def save_flags(self):
        """Get the save flags (read-only)."""
        return self._save_flags

    def get_flags_dict(self) -> Dict[str, bool]:
        """Get save flags as a dictionary.

        Returns
        -------
        Dict[str, bool]
            Dictionary mapping flag names to boolean values.
        """
        return self._property_system.get_flags_dict()

    def get_enabled_property_names(self) -> List[str]:
        """Get list of enabled property names.

        Returns
        -------
        List[str]
            List of enabled property names.
        """
        return self._property_system.get_enabled_property_names()

    def __repr__(self) -> str:
        """Return string representation of the template."""
        enabled_flags = [
            flag for flag in vars(self._save_flags) if getattr(self._save_flags, flag)
        ]
        return f"FieldLineTemplate(enabled_properties={enabled_flags})"

    @classmethod
    def template_all(cls) -> "FieldLineTemplate":
        """Create configuration template with all properties enabled.

        This template defines which properties to save during field line tracing.
        Used as configuration, not for storing actual field line data.
        """
        return cls(
            save_r_line=True,
            save_pa_line=True,
            save_r_max=True,
            save_r_cyl_max=True,
            save_z_abs_max=True,
            save_b_ini=True,
            save_b_r_ini_angle=True,
            save_b_line=True,
            save_curvature_line=True,
            save_tangent_theta=True,
            save_tangent_phi=True,
            save_reversed_error=True,
        )

    @classmethod
    def template_all_except_along_line(cls) -> "FieldLineTemplate":
        """Create configuration template with all properties enabled except line coordinates and PA.

        This template defines which properties to save during field line tracing.
        Used as configuration, not for storing actual field line data.
        """
        return cls(
            save_r_max=True,
            save_r_cyl_max=True,
            save_z_abs_max=True,
            save_b_ini=True,
            save_b_r_ini_angle=True,
            save_reversed_error=True,
        )

    @classmethod
    def template_OFL_calc(cls) -> "FieldLineTemplate":
        """Create configuration template with properties for open field line calculation (r_max, r_cyl_max, z_abs_max).

        This template defines which properties to save during field line tracing.
        Used as configuration, not for storing actual field line data.
        """
        return cls(save_r_max=True, save_r_cyl_max=True, save_z_abs_max=True, save_reversed_error=True)

    @classmethod
    def template_OFL_subset(cls) -> "FieldLineTemplate":
        """Create minimal configuration template for OFL subset grid creation.

        Includes only footpoint_id (always enabled), r_cyl_max, and z_abs_max.
        This is the minimal template needed for extracting OFL footpoints from
        existing field line data to create subset grids.

        This template defines which properties to save during field line tracing.
        Used as configuration, not for storing actual field line data.
        """
        return cls(save_r_cyl_max=True, save_z_abs_max=True, save_reversed_error=True)

    @classmethod
    def template_line_coordinates_only(cls) -> "FieldLineTemplate":
        """Create configuration template with line coordinates only (r_line).

        This template defines which properties to save during field line tracing.
        Used as configuration, not for storing actual field line data.
        """
        return cls(save_r_line=True, save_reversed_error=True)

    @classmethod
    def template_properties_along_line(cls) -> "FieldLineTemplate":
        """Create configuration template with line coordinates and PA (r_line, pa_line).

        This template defines which properties to save during field line tracing.
        Used as configuration, not for storing actual field line data.
        """
        return cls(
            save_r_line=True,
            save_pa_line=True,
            save_b_line=True,
            save_curvature_line=True,
            save_tangent_theta=True,
            save_tangent_phi=True,
            save_reversed_error=True,
        )

    @classmethod
    def template_surface_field_properties(cls) -> "FieldLineTemplate":
        """Create configuration template with surface properties (b_ini, b_r_ini_angle).

        This template defines which properties to save during field line tracing.
        Used as configuration, not for storing actual field line data.
        """
        return cls(save_b_ini=True, save_b_r_ini_angle=True, save_reversed_error=True)

