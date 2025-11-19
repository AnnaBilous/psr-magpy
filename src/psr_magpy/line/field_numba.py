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


@njit(fastmath=True, cache=True)
def _calculate_quadrupole_field_numba(
    r_magn_frame: NDArray[np.float64], B_magnitude: float
) -> NDArray[np.float64]:
    """Calculate quadrupole magnetic field in the magnetic frame.
    
    This function computes the quadrupole field at position r_magn_frame in the
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
    The quadrupole field formula is:
        denom = (0.5 * (5 * z^2 / r^2 - 1) / r)^-1
        B = B_magnitude * r^-4 * [x/denom, y/denom, z/denom - z/r]
    
    where:
        r = sqrt(x^2 + y^2 + z^2)
    
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
    
    # Calculate denominator: denom = (0.5 * (5 * z^2 / r^2 - 1) / r)^-1
    # Simplified: denom = 2 * r^3 / (5 * z^2 - r^2)
    z_sq = z * z
    denom_inv = 0.5 * (5.0 * z_sq / r_sq - 1.0) / r
    denom = 1.0 / denom_inv
    
    # Pre-compute r^-4 for efficiency
    r_inv4 = 1.0 / (r_sq * r_sq)
    
    # Calculate field components in magnetic frame
    Bx = B_magnitude * r_inv4 * x / denom
    By = B_magnitude * r_inv4 * y / denom
    Bz = B_magnitude * r_inv4 * (z / denom - z / r)
    
    return np.array([Bx, By, Bz], dtype=np.float64)


@njit(fastmath=True, cache=True)
def _get_component_field_numba(
    r_spin: NDArray[np.float64],
    B_magnitude: float,
    rotation_matrix: NDArray[np.float64],
    rotation_matrix_inv: NDArray[np.float64],
    offset_vector: NDArray[np.float64],
    moment_type: int,  # 0 = dipole, 1 = quadrupole
) -> NDArray[np.float64]:
    """Calculate magnetic field for a single component in spin frame.
    
    This function performs the complete field calculation for a single component:
    1. Transform coordinates from spin frame to magnetic frame
    2. Calculate field in magnetic frame
    3. Rotate field back to spin frame
    
    Parameters
    ----------
    r_spin : NDArray[np.float64]
        3-element position vector in spin frame [x, y, z]
    B_magnitude : float
        Field strength scaling factor
    rotation_matrix : NDArray[np.float64]
        3x3 rotation matrix from magnetic to spin frame
    rotation_matrix_inv : NDArray[np.float64]
        3x3 rotation matrix from spin to magnetic frame (inverse)
    offset_vector : NDArray[np.float64]
        3-element offset vector in spin frame
    moment_type : int
        Component type: 0 for dipole, 1 for quadrupole
        
    Returns
    -------
    NDArray[np.float64]
        3-element magnetic field vector in spin frame [Bx, By, Bz]
    """
    # Step 1: Transform coordinates from spin to magnetic frame
    r_magn = _transform_to_magnetic_frame_numba(
        r_spin, rotation_matrix_inv, offset_vector
    )
    
    # Step 2: Calculate field in magnetic frame
    if moment_type == 0:  # dipole
        B_magn = _calculate_dipole_field_numba(r_magn, B_magnitude)
    elif moment_type == 1:  # quadrupole
        B_magn = _calculate_quadrupole_field_numba(r_magn, B_magnitude)
    else:
        # Unknown moment type - return zero field
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    # Step 3: Rotate field back to spin frame
    B_spin = _rotate_field_to_spin_frame_numba(B_magn, rotation_matrix)
    
    return B_spin


@njit(fastmath=True, cache=True)
def get_total_field_numba(
    r: NDArray[np.float64],
    K: int,
    normalize: bool,
    n_components: int,
    B_magnitudes: NDArray[np.float64],
    rotation_matrices: NDArray[np.float64],  # (n_components, 3, 3)
    rotation_matrices_inv: NDArray[np.float64],  # (n_components, 3, 3)
    offset_vectors: NDArray[np.float64],  # (n_components, 3)
    moment_types: NDArray[np.int32],  # (n_components,)
) -> NDArray[np.float64]:
    """Calculate total magnetic field from multiple components in spin frame.
    
    This function sums the fields from all components, applies K multiplier,
    and optionally normalizes the result.
    
    Parameters
    ----------
    r : NDArray[np.float64]
        3-element position vector in spin frame [x, y, z]
    K : int
        Field direction multiplier (1 or -1)
    normalize : bool
        Whether to normalize the field vector
    n_components : int
        Number of field components
    B_magnitudes : NDArray[np.float64]
        Array of field strength scaling factors (n_components,)
    rotation_matrices : NDArray[np.float64]
        3D array of rotation matrices from magnetic to spin frame (n_components, 3, 3)
    rotation_matrices_inv : NDArray[np.float64]
        3D array of inverse rotation matrices from spin to magnetic frame (n_components, 3, 3)
    offset_vectors : NDArray[np.float64]
        2D array of offset vectors in spin frame (n_components, 3)
    moment_types : NDArray[np.int32]
        Array of component types: 0 for dipole, 1 for quadrupole (n_components,)
        
    Returns
    -------
    NDArray[np.float64]
        3-element total magnetic field vector in spin frame [Bx, By, Bz]
    """
    # Initialize total field
    F = np.zeros(3, dtype=np.float64)
    
    # Sum fields from all components
    for i in range(n_components):
        B_component = _get_component_field_numba(
            r,
            B_magnitudes[i],
            rotation_matrices[i],
            rotation_matrices_inv[i],
            offset_vectors[i],
            moment_types[i],
        )
        F[0] += B_component[0]
        F[1] += B_component[1]
        F[2] += B_component[2]
    
    # Apply K multiplier
    F[0] *= K
    F[1] *= K
    F[2] *= K
    
    # Normalize if requested
    if normalize:
        mag = np.sqrt(F[0] * F[0] + F[1] * F[1] + F[2] * F[2])
        if mag > 0.0:
            F[0] /= mag
            F[1] /= mag
            F[2] /= mag
    
    return F

