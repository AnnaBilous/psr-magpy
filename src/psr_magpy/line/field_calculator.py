"""
Field calculation abstraction layer.

Provides unified interface for field calculation with support for both
Python and Numba-compiled backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, cast
import numpy as np
from numpy.typing import NDArray

from psr_magpy.utils.logging_config import get_logger
from psr_magpy.utils.exceptions import PSRMagPyFieldError, PSRMagPyTracingError
from psr_magpy.line.field_numba import get_total_field_numba

logger = get_logger(__name__, console_format_brief=True)


class FieldCalculator(ABC):
    """Abstract base class for magnetic field calculation.
    
    Provides unified interface for field calculation that supports both
    Python and Numba-compiled implementations. All calculations use
    float64 for precision; ``r_line`` is stored as float64 at save time.
    
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


class NumbaFieldCalculator(FieldCalculator):
    """Numba-compiled implementation of field calculator.
    
    Pre-compiles field calculation on initialization and uses Numba-compiled
    functions for maximum performance. All calculations are performed in float64.
    """
    
    def __init__(self, field):
        """Initialize Numba field calculator.
        
        Parameters
        ----------
        field : MultipoleField
            The magnetic field object to calculate from.
            
        Raises
        ------
        ValueError
            If field components are not Numba-compatible.
        """
        super().__init__(field)
        
        # Get Numba-compatible configuration
        try:
            config = field.get_numba_config()
        except ValueError as e:
            raise PSRMagPyFieldError(f"Field is not Numba-compatible: {e}") from e
        
        # Convert config list to arrays for Numba efficiency
        self._n_components = len(config)
        self._B_magnitudes = np.array(
            [p["B_magnitude"] for p in config], dtype=np.float64
        )
        self._rotation_matrices = np.array(
            [p["rotation_matrix"] for p in config], dtype=np.float64
        )
        self._rotation_matrices_inv = np.array(
            [p["rotation_matrix_inv"] for p in config], dtype=np.float64
        )
        self._offset_vectors = np.array(
            [p["offset_vector"] for p in config], dtype=np.float64
        )
        self._moment_types = np.array(
            [0 if p["moment_type"] == "dipole" else 1 for p in config],
            dtype=np.int32
        )
        
        # Warmup: call Numba function once to trigger compilation
        # Use a dummy position to avoid first-call overhead during integration
        dummy_r = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        try:
            _ = get_total_field_numba(
                dummy_r, 1, True,
                self._n_components,
                self._B_magnitudes,
                self._rotation_matrices,
                self._rotation_matrices_inv,
                self._offset_vectors,
                self._moment_types,
            )
            logger.debug(f"Numba field calculator warmed up for {self._n_components} components")
        except Exception as e:
            raise PSRMagPyTracingError(f"Failed to warmup Numba field calculator: {e}") from e
    
    def get_field(
        self, r: np.ndarray, K: int, normalize: bool = True
    ) -> NDArray[np.float64]:
        """Calculate magnetic field at position r using Numba-compiled implementation.
        
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
            
        Notes
        -----
        No input validation is performed for maximum performance. Inputs are
        assumed to be valid (3-element float64 array, K in {1, -1}).
        """
        # Ensure r is float64
        r_f64 = np.asarray(r, dtype=np.float64)
        
        # Call Numba-compiled function
        return cast(NDArray[np.float64], get_total_field_numba(
            r_f64,
            K,
            normalize,
            self._n_components,
            self._B_magnitudes,
            self._rotation_matrices,
            self._rotation_matrices_inv,
            self._offset_vectors,
            self._moment_types,
        ))
