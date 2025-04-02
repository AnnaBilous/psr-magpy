import logging
from typing import List

import numpy as np
from numpy.typing import NDArray

from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class MultipoleField:
    """
    Magnetic field composed of multiple field components.
    
    This class manages a collection of field components (dipole, quadrupole, etc.) and
    provides methods to calculate their combined field effect.
    
    Parameters
    ----------
    field_components : List
        List of field components that make up this multipole field.
        Each component must be a subclass of FieldComponent.
    
    Raises
    ------
    TypeError
        If any component isn't a subclass of FieldComponent.
    
    Examples
    --------
    >>> from psr_magpy.field.components import DipoleFieldComponent
    >>> dipole = DipoleFieldComponent(strength=1.0)
    >>> field = MultipoleField([dipole])
    >>> field.get_total_field(np.array([1.0, 0.0, 0.0]))
    array([...])
    """

    def __init__(self, field_components: List):
        logger.info(
            f"Initializing MultipoleField with {len(field_components)} components"
        )
        
        # Check if components are field components by checking class name
        for i, comp in enumerate(field_components):
            comp_class = type(comp)
            comp_module = comp_class.__module__
            comp_name = comp_class.__name__
            
            logger.debug(f"Validating component {i+1}: {comp_name} from {comp_module}")
            
            # Check if it's a FieldComponent by name
            if not (comp_name.endswith('FieldComponent') or 
                   'FieldComponent' in [cls.__name__ for cls in comp_class.__mro__]):
                error_msg = f"Component {comp_name} from {comp_module} is not a FieldComponent"
                logger.error(error_msg)
                raise TypeError(error_msg)
        
        self.field_components = field_components
        
        # Log component details at debug level
        if logger.isEnabledFor(logging.DEBUG):
            for i, comp in enumerate(field_components):
                logger.debug(f"Component {i+1}: {comp}")
                
        logger.debug("MultipoleField initialized successfully")

    def get_total_field(
        self, r: NDArray[np.float32], K: float = 1, normalize: bool = True
    ) -> NDArray[np.float32]:
        """
        Calculate total magnetic field at position r.
        
        This method combines the field contributions from all components
        to calculate the total field at the specified position.

        Parameters
        ----------
        r : NDArray[np.float32]
            Position vector where to calculate the field.
        K : float, optional
            Field direction. Can be 1 or -1. Default is 1.
        normalize : bool, optional
            Whether to normalize the resulting field vector. Default is True.

        Returns
        -------
        NDArray[np.float32]
            Total magnetic field vector at position r.
            If normalize=True, the vector will have unit length.
            
        Notes
        -----
        The total field is calculated as the sum of individual component
        fields, multiplied by the scaling factor K, and optionally normalized.
        
        Examples
        --------
        >>> field.get_total_field(np.array([1.0, 0.0, 0.0]), K=1.0)
        array([...])
        """
        if K not in [-1, 1]:
            error_msg = f"K must be either -1 or 1, got {K}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        #logger.debug(f"Calculating total field at r={r}, K={K}, normalize={normalize}")
        try:
            F = np.array([0.0, 0.0, 0.0])
            
            # Calculate field from each component
            for i, component in enumerate(self.field_components):
                component_field = component.get_field(r)
                #logger.debug(f"Component {i+1} field: {component_field}")
                F += component_field
                
            # Apply field direction
            F = F * K
            
            # Normalize if requested
            if normalize:
                magnitude = np.sum(F**2) ** 0.5
                if magnitude > 0:
                    F /= magnitude
                    #logger.debug(f"Normalized field vector, magnitude was {magnitude:.6g}")
                else:
                    logger.warning("Field magnitude is zero, cannot normalize")
            
            #logger.debug(f"Total field calculated: F={F}")
            return F
            
        except Exception as e:
            logger.error(f"Error calculating total field: {e}")
            logger.debug("Field calculation error details", exc_info=True)
            raise

    def __repr__(self) -> str:
        """Return string representation of the multipole field."""
        component_types = [type(c).__name__ for c in self.field_components]
        return f"MultipoleField(components=[{', '.join(component_types)}])"

    
