import logging
from typing import List

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


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
        logger.debug(
            f"Initializing MultipoleField with {len(field_components)} components"
        )
        
        # Check if components are field components by checking class name
        for comp in field_components:
            comp_class = type(comp)
            comp_module = comp_class.__module__
            comp_name = comp_class.__name__
            
            # Check if it's a FieldComponent by name
            if not (comp_name.endswith('FieldComponent') or 
                   'FieldComponent' in [cls.__name__ for cls in comp_class.__mro__]):
                error_msg = f"Component {comp_name} from {comp_module} is not a FieldComponent"
                logger.error(error_msg)
                raise TypeError(error_msg)
        
        self.field_components = field_components
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
            raise ValueError("K must be either -1 or 1")
        
        logger.debug(f"Calculating total field at r={r}, K={K}, normalize={normalize}")
        try:
            F = np.array([0.0, 0.0, 0.0])
            for component in self.field_components:
                F += component.get_field(r)
            F = F * K
            if normalize:
                F /= np.sum(F**2) ** 0.5
            logger.debug(f"Total field calculated: F={F}")
            return F
        except Exception as e:
            logger.error(f"Error calculating total field: {e}")
            raise

    
