"""
Property calculator for field line analysis.

Calculates various properties of field lines including maximum distances,
magnetic field properties, and geometric characteristics.
"""

from typing import Optional, Any, Dict
import numpy as np

from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.property_system import PropertyName
from psr_magpy.utils.logging_config import get_logger

logger = get_logger(__name__, console_format_brief=True)


class PropertyCalculator:
    """Calculates properties for field lines."""

    def __init__(self, threshold_line_distance: float):
        """Initialize property calculator."""
        self.threshold_line_distance = threshold_line_distance
        logger.debug(
            f"Initialized PropertyCalculator with threshold_line_distance={threshold_line_distance}"
        )

    def calculate_line_properties(
        self,
        footpoint_id: int,
        r: np.ndarray,
        B_initial: np.ndarray,
        dotprod: float,
        K: float,
        template: FieldLine,
    ) -> Dict[str, Any]:
        """Calculate properties for a field line."""
        line_R_max = np.max(np.sum(r**2, axis=0) ** 0.5)
        props: Dict[str, Any] = {}

        logger.debug(
            f"Calculating properties for field line {footpoint_id} with R_max={line_R_max}"
        )

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
                props[prop.key_name] = self.prep_pa_line(r, K, line_R_max, template)

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
        self, r: np.ndarray, line_R_max: float, template: FieldLine
    ) -> Optional[np.ndarray]:
        """Calculate curvature along field line.

        To be added later. This is not used in the current implementation.
        """
        # Note: CURVATURE_LINE is not in PropertyName enum, so this method is not used
        # Keeping for future implementation
        return None

    def prep_pa_line(
        self, r: np.ndarray, B_dir: float, line_R_max: float, template: FieldLine
    ) -> Optional[np.ndarray]:
        """Calculate position angle along the field line.

        Computes the position angle (PA) along the field line using
        the tangent vector and its gradient. Handles numerical instabilities
        near coordinate singularities.
        """
        if getattr(template.save_flags, PropertyName.PA_LINE.flag_name):
            if line_R_max >= self.threshold_line_distance:
                Npoints = r.shape[1]
                t = np.gradient(r, axis=1)
                t_abs = np.linalg.norm(t, axis=0, keepdims=True)
                safe_t_abs = np.where(t_abs == 0.0, 1.0, t_abs)
                t_norm = t / safe_t_abs

                n = np.gradient(t_norm, axis=1)
                n_abs = np.linalg.norm(n, axis=0, keepdims=True)
                safe_n_abs = np.where(n_abs == 0.0, 1.0, n_abs)
                n_norm = n / safe_n_abs

                # Convert to spherical coordinates manually to avoid astropy issues
                x, y, z = t_norm
                r_sph = np.sqrt(x**2 + y**2 + z**2)
                t_phi = np.arctan2(y, x)
                t_theta = np.arccos(z / np.where(r_sph == 0, 1, r_sph))

                l0 = np.zeros((3, Npoints), dtype=np.float32)
                l0[0, :] = -np.sin(t_phi)
                l0[1, :] = np.cos(t_phi)
                l0[2, :] = 0

                m0 = np.zeros((3, Npoints), dtype=np.float32)
                m0[0, :] = np.cos(t_theta) * np.cos(t_phi)
                m0[1, :] = np.cos(t_theta) * np.sin(t_phi)
                m0[2, :] = -np.sin(t_theta)
                nom = np.sum(n_norm * l0, axis=0)
                denom = np.sum(n_norm * m0, axis=0)
                PA = np.arctan2(nom, denom).astype(np.float32)

                # PA is prone to numerical instability when FL tangent's azimuth is close to multiple of pi
                # and LOS colatitude is close to 90 degrees.
                # This is a hack to fix it.
                eps = 0.01
                cr_t_phi = (
                    (t_phi < eps)
                    | (t_phi > 2 * np.pi - eps)
                    | (np.abs(t_phi - np.pi) < eps)
                )
                cr = cr_t_phi & (np.abs(t_theta - 1.57) < 0.01)
                PA[cr] = np.arctan2(-n[1, cr], n[0, cr])

                if B_dir == -1:
                    PA = PA + np.pi
                PA[PA <= -np.pi] += 2 * np.pi
                PA[PA >= np.pi] -= 2 * np.pi

                logger.debug(f"Calculated PA line with {len(PA)} points")

            else:
                PA = np.array([], dtype=np.float32)
                logger.debug(
                    "Line did not reach threshold_line_distance, saving empty PA array"
                )
        else:
            PA = None
            logger.debug("PA line saving disabled")
        return PA
