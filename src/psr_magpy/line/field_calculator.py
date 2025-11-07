"""
Field calculation abstraction layer.

Provides unified interface for field calculation with support for both
Python and Numba-compiled backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import numpy as np
from numpy.typing import NDArray

from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class FieldCalculator(ABC):
    """Abstract base class for magnetic field calculation.
    
    Provides unified interface for field calculation that supports both
    Python and Numba-compiled implementations. All calculations use
    float64 for precision, with conversion to float32 handled at storage layer.
    
    Attributes
    ----------
    field : MultipoleField
        The magnetic field object to calculate from.
    """
    
    def __init__(self, field):
        """Initialize field calculator.
        
        Parameters
        ----------
        field : MultipoleField
            The magnetic field object to calculate from.
        """
        self.field = field
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def get_field(
        self, r: np.ndarray, K: int, normalize: bool = True
    ) -> NDArray[np.float64]:
        """Calculate magnetic field at position r.
        
        All calculations are performed in float64 for precision.
        Input arrays are converted to float64 internally if needed.
        
        Parameters
        ----------
        r : np.ndarray
            Position vector (3-element array). Can be any dtype, will be
            converted to float64 internally.
        K : int
            Field direction multiplier (typically 1 or -1).
        normalize : bool, optional
            Whether to normalize the field vector. Default is True.
            
        Returns
        -------
        NDArray[np.float64]
            Magnetic field vector (3-element array, float64).
        """
        pass
    
    def get_numba_callable(self) -> Optional[Callable[[np.ndarray, int], NDArray[np.float64]]]:
        """Get Numba-compiled field calculation function if available.
        
        Returns a compiled function with signature (r, K) -> normalized field.
        This is optimized for integration use cases where normalize=True is always used.
        
        Returns
        -------
        Optional[Callable]
            Compiled function (r: np.ndarray, K: int) -> NDArray[np.float64],
            or None if not available (e.g., Python backend).
            
        Notes
        -----
        The returned function always returns normalized field vectors.
        For non-normalized fields, use get_field() instead.
        """
        return None


class PythonFieldCalculator(FieldCalculator):
    """Python implementation of field calculator.
    
    Wraps MultipoleField components and performs calculations in float64.
    Maintains same logic as MultipoleField.get_total_field() but uses float64
    throughout for precision.
    """
    
    def __init__(self, field):
        """Initialize Python field calculator.
        
        Parameters
        ----------
        field : MultipoleField
            The magnetic field object to calculate from.
        """
        super().__init__(field)
        # Pre-allocate field vector for reuse (zeroed on each call)
        self._F = np.zeros(3, dtype=np.float64)
    
    def get_field(
        self, r: np.ndarray, K: int, normalize: bool = True
    ) -> NDArray[np.float64]:
        """Calculate magnetic field at position r using Python implementation.
        
        Parameters
        ----------
        r : np.ndarray
            Position vector (3-element array). Assumes float64.
        K : int
            Field direction multiplier (typically 1 or -1).
        normalize : bool, optional
            Whether to normalize the field vector. Default is True.
            
        Returns
        -------
        NDArray[np.float64]
            Magnetic field vector (3-element array, float64).
        """
        self._F.fill(0.0)
        for component in self.field.field_components:
            self._F += component.get_field(r)  
        self._F *= K  
        if normalize:
            mag = np.linalg.norm(self._F)
            if mag > 0:
                self._F /= mag
        
        return self._F.copy()  # Return copy to avoid modification by subsequent calls  
