import logging
from typing import List, Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

from psr_magpy.field.components import FieldComponent
from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyFieldError

if TYPE_CHECKING:
    from psr_magpy.viz import MultipoleFieldVisualizer

logger = get_logger(__name__, console_format_brief=True)


class MultipoleFieldParams(BaseModel):
    """Parameters for multipole field configuration."""

    field_components: List[FieldComponent] = Field(
        description="List of field components that make up this multipole field"
    )

    @field_validator("field_components")
    @classmethod
    def validate_field_components(cls, v):
        """Validate that all components are FieldComponent instances and list is not empty."""
        if not v:
            raise ValueError("Field components list cannot be empty")

        for i, component in enumerate(v):
            if not isinstance(component, FieldComponent):
                raise ValueError(
                    f"Component {i+1} is not a FieldComponent instance, got {type(component).__name__}"
                )

        return v

    model_config = {"arbitrary_types_allowed": True}


class MultipoleField:
    """
    Magnetic field composed of multiple field components.

    This class manages a collection of field components (dipole, quadrupole, etc.) and
    provides methods to calculate their combined field effect.

    Parameters
    ----------
    field_components : List[FieldComponent]
        List of field components that make up this multipole field.
        Each component must be a FieldComponent instance.

    Raises
    ------
    ValueError
        If field_components list is empty or contains invalid components.

    Examples
    --------
    >>> from psr_magpy.field.components import DipoleFieldComponent
    >>> dipole = DipoleFieldComponent(B_magnitude=1.0)
    >>> field = MultipoleField([dipole])
    >>> field.get_total_field(np.array([1.0, 0.0, 0.0]))
    array([...])
    """

    def __init__(self, field_components: List[FieldComponent]):
        """Initialize MultipoleField with validated field components."""
        logger.info(
            f"Initializing MultipoleField with {len(field_components)} components"
        )

        # Validate components using Pydantic
        try:
            self.params = MultipoleFieldParams(field_components=field_components)
            self.field_components = self.params.field_components
        except Exception as e:
            raise PSRMagPyFieldError(f"Failed to initialize MultipoleField parameters: {e}") from e

        # Lazy visualizer instance
        self._visualizer: Optional["MultipoleFieldVisualizer"] = None

        # Log component details at debug level
        if logger.isEnabledFor(logging.DEBUG):
            for i, comp in enumerate(self.field_components):
                logger.debug(f"Component {i+1}: {comp}")

        logger.debug("MultipoleField initialized successfully")

    @property
    def n_components(self) -> int:
        """Get number of field components."""
        return len(self.field_components)

    def get_numba_config(self) -> List[dict]:
        """Get Numba-compatible configuration for all field components.
        
        Collects Numba-compatible parameters from all components and validates
        that all components support Numba compilation.
        
        Returns
        -------
        List[dict]
            List of parameter dictionaries, one per component. Each dict contains:
            - 'B_magnitude': float
            - 'rotation_matrix': NDArray[float64] (3x3, magn to spin frame)
            - 'rotation_matrix_inv': NDArray[float64] (3x3, spin to magn frame)
            - 'offset_vector': NDArray[float64] (3-element)
            - 'moment_type': str ("dipole" or "quadrupole")
        
        Raises
        ------
        ValueError
            If any component does not support Numba compilation (missing get_numba_params method).
        """
        config = []
        for i, component in enumerate(self.field_components):
            if not hasattr(component, "get_numba_params"):
                raise PSRMagPyFieldError(
                    f"Component {i+1} ({type(component).__name__}) does not support "
                    "Numba compilation. Missing get_numba_params() method."
                )
            
            try:
                params = component.get_numba_params()
                config.append(params)
            except Exception as e:
                raise PSRMagPyFieldError(
                    f"Failed to get Numba parameters from component {i+1} "
                    f"({type(component).__name__}): {e}"
                ) from e
        
        logger.debug(f"Generated Numba config for {len(config)} components")
        return config

    @property
    def visualizer(self) -> "MultipoleFieldVisualizer":
        """Get the visualizer instance for this field.
        
        Returns
        -------
        MultipoleFieldVisualizer
            Visualizer instance for plotting this field
        """
        if self._visualizer is None:
            from psr_magpy.viz import MultipoleFieldVisualizer
            self._visualizer = MultipoleFieldVisualizer(self)
        return self._visualizer

    def get_total_field(
        self, r: NDArray[np.float32], K: int = 1, normalize: bool = True
    ) -> NDArray[np.float32]:
        """Fast calculation of total magnetic field at position r.
        
        Assumes r is a valid 3-element float32 array. No validation or error handling.

        Parameters
        ----------
        r : NDArray[np.float32]
            Position vector (3-element array)
        K : int, optional
            Field direction multiplier (1 or -1). Default is 1.
        normalize : bool, optional
            Whether to normalize the field vector. Default is True.

        Returns
        -------
        NDArray[np.float32]
            Total magnetic field vector
        """
        F = np.zeros(3, dtype=np.float32)
        # Convert r to float64 for get_field call (component expects float64)
        r_float64 = np.asarray(r, dtype=np.float64)
        for component in self.field_components:
            F += component.get_field(r_float64).astype(np.float32)
        F *= K
        if normalize:
            mag = np.linalg.norm(F)
            if mag > 0:
                F /= mag

        return F

    def __repr__(self) -> str:
        """Return string representation of the multipole field."""
        component_types = [type(c).__name__ for c in self.field_components]
        return f"MultipoleField(components=[{', '.join(component_types)}], n_components={self.n_components})"
