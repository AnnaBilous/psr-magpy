"""Numba-compiled field calculation functions.

This module provides Numba-compiled implementations of magnetic field calculations
for performance-critical integration tasks.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit(fastmath=True, cache=True)
def _transform_to_magnetic_frame_numba(
    r_spin: NDArray[np.float64],
    rotation_matrix_inv: NDArray[np.float64],
    offset_vector: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transform coordinates from spin frame to magnetic frame.
    
    This function applies offset subtraction and rotation to transform
    a position vector from the spin frame to the magnetic frame.
    
    Parameters
    ----------
    r_spin : NDArray[np.float64]
        3-element position vector in spin frame [x, y, z]
    rotation_matrix_inv : NDArray[np.float64]
        3x3 rotation matrix from spin to magnetic frame (inverse of magn_to_spin)
    offset_vector : NDArray[np.float64]
        3-element offset vector in spin frame
        
    Returns
    -------
    NDArray[np.float64]
        3-element position vector in magnetic frame [x, y, z]
        
    Notes
    -----
    Transformation steps:
    1. Subtract offset: r_offset = r_spin - offset_vector
    2. Rotate: r_magn = rotation_matrix_inv @ r_offset
    """
    # Subtract offset
    r_offset = np.array([
        r_spin[0] - offset_vector[0],
        r_spin[1] - offset_vector[1],
        r_spin[2] - offset_vector[2],
    ], dtype=np.float64)
    
    # Apply rotation matrix: r_magn = R @ r_offset
    r_magn = np.array([
        rotation_matrix_inv[0, 0] * r_offset[0] + rotation_matrix_inv[0, 1] * r_offset[1] + rotation_matrix_inv[0, 2] * r_offset[2],
        rotation_matrix_inv[1, 0] * r_offset[0] + rotation_matrix_inv[1, 1] * r_offset[1] + rotation_matrix_inv[1, 2] * r_offset[2],
        rotation_matrix_inv[2, 0] * r_offset[0] + rotation_matrix_inv[2, 1] * r_offset[1] + rotation_matrix_inv[2, 2] * r_offset[2],
    ], dtype=np.float64)
    
    return r_magn


@njit(fastmath=True, cache=True)
def _rotate_field_to_spin_frame_numba(
    B_magn: NDArray[np.float64],
    rotation_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rotate magnetic field vector from magnetic frame to spin frame.
    
    This function applies rotation to transform a field vector from
    the magnetic frame to the spin frame.
    
    Parameters
    ----------
    B_magn : NDArray[np.float64]
        3-element magnetic field vector in magnetic frame [Bx, By, Bz]
    rotation_matrix : NDArray[np.float64]
        3x3 rotation matrix from magnetic to spin frame
        
    Returns
    -------
    NDArray[np.float64]
        3-element magnetic field vector in spin frame [Bx, By, Bz]
        
    Notes
    -----
    Rotation: B_spin = rotation_matrix @ B_magn
    """
    # Apply rotation matrix: B_spin = R @ B_magn
    B_spin = np.array([
        rotation_matrix[0, 0] * B_magn[0] + rotation_matrix[0, 1] * B_magn[1] + rotation_matrix[0, 2] * B_magn[2],
        rotation_matrix[1, 0] * B_magn[0] + rotation_matrix[1, 1] * B_magn[1] + rotation_matrix[1, 2] * B_magn[2],
        rotation_matrix[2, 0] * B_magn[0] + rotation_matrix[2, 1] * B_magn[1] + rotation_matrix[2, 2] * B_magn[2],
    ], dtype=np.float64)
    
    return B_spin


@njit(fastmath=True, cache=True)
def _calculate_dipole_field_numba(
    r_magn_frame: NDArray[np.float64], B_magnitude: float
) -> NDArray[np.float64]:
    """Calculate dipole magnetic field in the magnetic frame.
    
    This function computes the dipole field at position r_magn_frame in the
    magnetic frame coordinate system. The field is NOT rotated to the spin frame.
    
    Parameters
    ----------
    r_magn_frame : NDArray[np.float64]
        3-element position vector in magnetic frame [x, y, z]
    B_magnitude : float
        Field strength scaling factor
        
    Returns
    -------
    NDArray[np.float64]
        3-element magnetic field vector in magnetic frame [Bx, By, Bz]
        
    Notes
    -----
    The dipole field formula is:
        B = B_magnitude * r^-5 * [1.5*z*x, 1.5*z*y, z^2 - 0.5*r_cyl^2]
    
    where:
        r = sqrt(x^2 + y^2 + z^2)
        r_cyl = sqrt(x^2 + y^2)
    
    For positions very close to the origin (r < 1e-10), returns zero field
    to avoid singularity.
    """
    x = r_magn_frame[0]
    y = r_magn_frame[1]
    z = r_magn_frame[2]
    
    # Calculate radial distance
    r_sq = x * x + y * y + z * z
    r = np.sqrt(r_sq)
    
    # Avoid singularity at origin
    if r < 1e-10:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    # Calculate cylindrical radius squared
    r_cyl_sq = x * x + y * y
    
    # Pre-compute r^-5 for efficiency
    r_inv5 = 1.0 / (r_sq * r_sq * r)
    
    # Calculate field components in magnetic frame
    Bx = B_magnitude * r_inv5 * 1.5 * z * x
    By = B_magnitude * r_inv5 * 1.5 * z * y
    Bz = B_magnitude * r_inv5 * (z * z - 0.5 * r_cyl_sq)
    
    return np.array([Bx, By, Bz], dtype=np.float64)

