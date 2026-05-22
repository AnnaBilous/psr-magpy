"""
Property calculator for field line analysis.

Calculates various properties of field lines including maximum distances,
magnetic field properties, and geometric characteristics.
"""

from typing import Optional, Any, Dict, cast
import numpy as np

from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.property_system import PropertyName
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)

_MIN_ARC_SEGMENT = 1.0e-15


def _arc_length_samples(r: np.ndarray) -> np.ndarray:
    """Cumulative arc length ``s`` along ``r`` with shape ``(3, N)``; returns length ``N``."""
    n_pts = int(r.shape[1])
    if n_pts == 0:
        return np.array([], dtype=np.float64)
    if n_pts == 1:
        return np.zeros(1, dtype=np.float64)
    dr = np.diff(r, axis=1)
    ds_seg = np.linalg.norm(dr, axis=0)
    ds_seg = np.maximum(ds_seg, _MIN_ARC_SEGMENT)
    return np.concatenate(([0.0], np.cumsum(ds_seg)))


def calculate_pa_line(
    r: np.ndarray,
    line_R_max: float,
    threshold_line_distance: float,
    template: FieldLine,
) -> Optional[np.ndarray]:
    """Calculate position angle along the field line (standalone function).

    Computes the position angle (PA) along the field line using
    the tangent vector and its gradient. Handles numerical instabilities
    near coordinate singularities.

    Parameters
    ----------
    r : np.ndarray
        Field line coordinates, shape (3, N) where N is number of points.
    line_R_max : float
        Maximum distance reached by the field line.
    threshold_line_distance : float
        Distance threshold for line tracing.
    template : FieldLine
        Field line template with save flags.

    Returns
    -------
    Optional[np.ndarray]
        Position angle array in degrees, or None if PA saving is disabled.
    """
    if getattr(template.save_flags, PropertyName.PA_LINE.flag_name):
        if line_R_max >= threshold_line_distance:
            
            r64 = r.astype(np.float64)  # Shape: (3, n_points)
            s = _arc_length_samples(r64)
            r1 = np.gradient(r64, s, axis=1)
            r2 = np.gradient(r1, s, axis=1)
            
            unstable_threshold = 1.1e-10
            
            # Tangent vector along whole line
            t_abs = np.linalg.norm(r1, axis=0, keepdims=True)
            t_unstable = t_abs <= unstable_threshold
            t_norm = r1 / t_abs
            
            # Curvature normal along whole line
            proj = np.sum(r2 * t_norm, axis=0, keepdims=True) * t_norm
            n = r2 - proj
            n_abs = np.linalg.norm(n, axis=0, keepdims=True)
            n_unstable = n_abs <= unstable_threshold
            n_norm = n / n_abs
            
            
            Omega = np.array([[0.0], [0.0], [1.0]]) 
            # Project Ω onto plane perpendicular to t
            Omega_dot_t = np.sum(Omega * t_norm, axis=0, keepdims=True)
            Omega_p = Omega - Omega_dot_t * t_norm

    
            # Binormal vector
            b = np.cross(t_norm.T, n_norm.T).T
            b_abs = np.linalg.norm(b, axis=0, keepdims=True)
            b_unstable = b_abs <= unstable_threshold
            b_norm = b / b_abs

            # Project curvature normal onto this basis
            nom = np.sum(Omega_p * b_norm, axis=0)
            denom = np.sum(Omega_p * n_norm, axis=0)
            
            PA = np.arctan2(nom, denom)
            PA = np.rad2deg(PA) % 180
            PA[PA >= 90] -= 180
            PA[PA < -90] += 180
                                  
            unstable_points = n_unstable[0, :] | t_unstable[0, :] | b_unstable[0, :]
            PA[unstable_points] = np.nan
            
            
            logger.debug(f"Calculated PA line with {len(PA)} points")

           
        else:
            PA = np.array([], dtype=np.float32)
            logger.debug(
                "Line did not reach threshold_line_distance, saving empty PA array"
            )
    else:
        PA = None
        logger.debug("PA line saving disabled")
    return cast(Optional[np.ndarray], PA)


class PropertyCalculator:
    """Calculates properties for field lines."""

    def __init__(self, threshold_line_distance: float, field=None):
        """Initialize property calculator.

        Parameters
        ----------
        threshold_line_distance : float
            Distance threshold for line tracing.
        field : optional
            Magnetic field object for calculating B_line and curvature_line. Default is None.
        """
        self.threshold_line_distance = threshold_line_distance
        self.field = field
        logger.debug(
            f"Initialized PropertyCalculator with threshold_line_distance={threshold_line_distance}"
        )

    def calculate_line_properties(
        self,
        footpoint_id: int,
        r: np.ndarray,
        B_initial: np.ndarray,
        dotprod: float,
        dt: float,
        template: FieldLine,
    ) -> Dict[str, Any]:
        """Calculate properties for a field line."""
        line_R_max = np.max(np.sum(r**2, axis=0) ** 0.5)
        props: Dict[str, Any] = {}

        logger.debug(
            f"Calculating properties for field line {footpoint_id} with R_max={line_R_max}"
        )

        # Pre-calculate tangent vector if needed for any property
        needs_tangent = any(
            [
                getattr(template.save_flags, PropertyName.TANGENT_THETA.flag_name),
                getattr(template.save_flags, PropertyName.TANGENT_PHI.flag_name),
            ]
        )

        tangent_norm = None
        if needs_tangent and line_R_max >= self.threshold_line_distance:
            t = np.gradient(r, axis=1)
            t_abs = np.linalg.norm(t, axis=0, keepdims=True)
            safe_t_abs = np.where(t_abs == 0.0, 1.0, t_abs)
            tangent_norm = t / safe_t_abs

        for prop in PropertyName:
            if prop.key_name == "footpoint_id":
                props[prop.key_name] = footpoint_id
            elif prop.key_name == "r_line":
                props[prop.key_name] = self.prep_r_line(r, line_R_max, template)
            elif prop.key_name == "r_max":
                props[prop.key_name] = (
                    line_R_max if getattr(template.save_flags, prop.flag_name) else None
                )
            elif prop.key_name == "r_cyl_max":
                props[prop.key_name] = (
                    np.max(np.sum(r[:2,] ** 2, axis=0) ** 0.5)
                    if getattr(template.save_flags, prop.flag_name)
                    else None
                )
            elif prop.key_name == "z_abs_max":
                props[prop.key_name] = (
                    np.max(np.abs(r[2, :]))
                    if getattr(template.save_flags, prop.flag_name)
                    else None
                )
            elif prop.key_name == "b_ini":
                props[prop.key_name] = (
                    np.sum(B_initial**2) ** 0.5
                    if getattr(template.save_flags, prop.flag_name)
                    else None
                )
            elif prop.key_name == "b_line":
                props[prop.key_name] = self.prep_b_line(r, line_R_max, template)
            elif prop.key_name == "b_r_ini_angle":
                if getattr(template.save_flags, prop.flag_name):
                    # Check for suspiciously large dot product values
                    if abs(dotprod) > 1.0 + 1e-3:
                        error_msg = f"Dot product {dotprod:.6f} is significantly outside [-1, 1] range"
                        logger.error(f"{error_msg} for field line {footpoint_id}")
                        # Add error to properties
                        props["dotprod_error"] = error_msg

                    # Clip dotprod to [-1, 1] to avoid numerical precision issues
                    clipped_dotprod = np.clip(dotprod, -1.0, 1.0)
                    props[prop.key_name] = np.arccos(clipped_dotprod)
                else:
                    props[prop.key_name] = None
            elif prop.key_name == "pa_line":
                props[prop.key_name] = self.prep_pa_line(r, line_R_max, template)
            elif prop.key_name == "curvature_line":
                if getattr(template.save_flags, prop.flag_name):
                    curvature = self.prep_curvature_line(r, dt, line_R_max, template)
                    props[prop.key_name] = curvature
                else:
                    props[prop.key_name] = None
            elif prop.key_name == "tangent_theta":
                props[prop.key_name] = self.prep_tangent_theta(
                    r, line_R_max, template, tangent_norm
                )
            elif prop.key_name == "tangent_phi":
                props[prop.key_name] = self.prep_tangent_phi(
                    r, line_R_max, template, tangent_norm
                )

        # Remove None values from kwargs
        props = {k: v for k, v in props.items() if v is not None}
        logger.debug(
            f"Calculated {len(props)} properties for field line {footpoint_id}"
        )
        return props

    def prep_r_line(
        self, r: np.ndarray, line_R_max: float, template: FieldLine
    ) -> Optional[np.ndarray]:
        """Prepare position vectors for saving."""
        if getattr(template.save_flags, PropertyName.R_LINE.flag_name):
            if line_R_max >= self.threshold_line_distance:
                r_line_save = r.T.astype(np.float32)
                logger.debug(f"Prepared r_line with {r_line_save.shape[0]} points")
            else:
                r_line_save = np.array(
                    [], dtype=np.float32
                )  # Line traced but didn't reach threshold_line_distance
                logger.debug(
                    "Line did not reach threshold_line_distance, saving empty array"
                )
        else:
            r_line_save = None  # Don't save line at all
            logger.debug("r_line saving disabled")
        return r_line_save

    def prep_curvature_line(
        self, r: np.ndarray, dt: float, line_R_max: float, template: FieldLine
    ) -> Optional[np.ndarray]:
        """Calculate curvature for a 3D parametric curve.
        
        Parameters
        ----------
        r : np.ndarray
            3D parametric curve r[k] = (x,y,z) with shape ``(3, N)``.
        
        Returns
        -------
        Optional[np.ndarray]
            Radius of curvature.
        """
        if getattr(template.save_flags, PropertyName.CURVATURE_LINE.flag_name):
            if line_R_max >= self.threshold_line_distance:
                r64 = r.astype(np.float64)
                s = _arc_length_samples(r64)
                r1 = np.gradient(r64, s, axis=1)
                r2 = np.gradient(r1, s, axis=1)     
                t_abs = np.linalg.norm(r1, axis=0)    
                t_norm = r1 / t_abs[None, :]           
                proj = (np.sum(r2 * t_norm, axis=0, keepdims=True)) * t_norm
                k_vec = r2 - proj                     
                k_abs = np.linalg.norm(k_vec, axis=0) 
                # Curvature κ and radius ρ
                eps = 1e-15                            # to avoid 0-division for straight segments
                kappa = k_abs / np.maximum(t_abs**2, eps)
                curvature_radius   = np.where(kappa > 0.0, 1.0 / kappa, np.inf)


                # Return as a list containing a single 1D array
                logger.debug(
                    f"Calculated curvature line with {len(curvature_radius)} points"
                )
                return curvature_radius.astype(np.float32)

            else:
                curvature_radius = np.array([], dtype=np.float32)
                logger.debug(
                    "Line did not reach threshold_line_distance, saving empty curvature array"
                )
                return curvature_radius
        else:
            curvature_radius = None
            logger.debug("curvature line saving disabled")
            return curvature_radius

    def prep_pa_line(
        self,
        r: np.ndarray,
        line_R_max: float,
        template: FieldLine,
    ) -> Optional[np.ndarray]:
        """Calculate position angle along the field line.

        Computes the position angle (PA) along the field line using
        the tangent vector and its gradient. Handles numerical instabilities
        near coordinate singularities.
        """
        return calculate_pa_line(
            r=r,
            line_R_max=line_R_max,
            threshold_line_distance=self.threshold_line_distance,
            template=template,
        )

    def prep_b_line(
        self, r: np.ndarray, line_R_max: float, template: FieldLine
    ) -> Optional[np.ndarray]:
        """Calculate magnetic field strength along the field line.

        Computes the absolute value of the magnetic field at each point
        along the field line.
        """
        if getattr(template.save_flags, PropertyName.B_LINE.flag_name):
            if self.field is not None:
                if line_R_max >= self.threshold_line_distance:
                    # Calculate magnetic field at each point along the field line
                    num_points = r.shape[1]
                    b_values = np.zeros(num_points, dtype=np.float32)
                    for i in range(num_points):
                        point = r[:, i]  # Extract single 3D point
                        b = self.field.get_total_field(point, K=1, normalize=False)
                        b_values[i] = np.linalg.norm(b)
                    logger.debug(f"Calculated B_line with {len(b_values)} points")

                else:
                    b_values = np.array([], dtype=np.float32)
                    logger.debug(
                        "Line did not reach threshold_line_distance, saving empty B array"
                    )
                return b_values
            else:
                logger.warning("Field object not provided, cannot calculate b_line")
                return None
        else:
            logger.debug("B_line saving disabled")
            return None

    def prep_tangent_theta(
        self,
        r: np.ndarray,
        line_R_max: float,
        template: FieldLine,
        tangent_norm: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Calculate theta (colatitude) of tangent vector along the field line.

        Returns theta angle in radians for each point along the field line.
        """
        if getattr(template.save_flags, PropertyName.TANGENT_THETA.flag_name):
            if line_R_max >= self.threshold_line_distance:
                # Use pre-calculated tangent vector or calculate if not provided
                if tangent_norm is None:
                    t = np.gradient(r, axis=1)
                    t_abs = np.linalg.norm(t, axis=0, keepdims=True)
                    safe_t_abs = np.where(t_abs == 0.0, 1.0, t_abs)
                    t_norm = t / safe_t_abs
                else:
                    t_norm = tangent_norm

                # Convert to spherical coordinates
                x, y, z = t_norm
                r_sph = np.sqrt(x**2 + y**2 + z**2)
                theta = np.rad2deg(np.arccos(z / np.where(r_sph == 0, 1, r_sph)))
                logger.debug(f"Calculated tangent theta with {len(theta)} points")
                return cast(Optional[np.ndarray], theta.astype(np.float32))
            else:
                theta = np.array([], dtype=np.float32)
                logger.debug(
                    "Line did not reach threshold_line_distance, saving empty tangent theta array"
                )
                return cast(Optional[np.ndarray], theta)
        else:
            logger.debug("tangent theta saving disabled")
            return None

    def prep_tangent_phi(
        self,
        r: np.ndarray,
        line_R_max: float,
        template: FieldLine,
        tangent_norm: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Calculate phi (azimuth) of tangent vector along the field line.

        Returns phi angle in radians for each point along the field line.
        """
        if getattr(template.save_flags, PropertyName.TANGENT_PHI.flag_name):
            if line_R_max >= self.threshold_line_distance:
                # Use pre-calculated tangent vector or calculate if not provided
                if tangent_norm is None:
                    t = np.gradient(r, axis=1)
                    t_abs = np.linalg.norm(t, axis=0, keepdims=True)
                    safe_t_abs = np.where(t_abs == 0.0, 1.0, t_abs)
                    t_norm = t / safe_t_abs
                else:
                    t_norm = tangent_norm

                # Convert to spherical coordinates
                x, y, z = t_norm
                phi = np.rad2deg(np.arctan2(y, x))
                logger.debug(f"Calculated tangent phi with {len(phi)} points")
                return cast(Optional[np.ndarray], phi.astype(np.float32))
            else:
                phi = np.array([], dtype=np.float32)
                logger.debug(
                    "Line did not reach threshold_line_distance, saving empty tangent phi array"
                )
                return cast(Optional[np.ndarray], phi)
        else:
            logger.debug("tangent phi saving disabled")
            return None
