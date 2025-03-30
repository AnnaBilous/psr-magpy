from typing import List
import numpy as np
from numpy.typing import NDArray
import logging

from .components import FieldComponent

logger = logging.getLogger(__name__)

class MultipoleField:
    """Class representing a multipole magnetic field composed of multiple field components.
    
    This class manages a collection of field components (dipole, quadrupole, etc.) and 
    provides methods to calculate their combined field effect.
    
    Parameters
    ----------
    field_components : List[FieldComponent]
        List of field components that make up this multipole field
        
    Raises
    ------
    TypeError
        If any component isn't a subclass of FieldComponent
    """

    def __init__(self, field_components: List[FieldComponent]):
        logger.debug(f"Initializing MultipoleField with {len(field_components)} components")
        if not all(isinstance(comp, FieldComponent) for comp in field_components):
            error_msg = "All components must be subclasses of FieldComponent"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        self.field_components = field_components
        logger.debug("MultipoleField initialized successfully")

    def get_total_field(self, r: NDArray[np.float32], K: float = 1, normalize: bool = True) -> NDArray[np.float32]:
        """Calculate total magnetic field at position r.
        
        Parameters
        ----------
        r : NDArray[np.float32]
            Position vector where to calculate the field
        K : float, optional
            Field strength scaling factor, by default 1
        normalize : bool, optional
            Whether to normalize the resulting field vector, by default True
            
        Returns
        -------
        NDArray[np.float32]
            Total magnetic field vector at position r
        """
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