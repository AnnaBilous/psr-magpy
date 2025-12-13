import numpy as np
from typing import Union


def get_R_LC(P_ms: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the light cylinder radius (in R_NS) for a given pulsar period.
    
    Parameters
    ----------
    P_ms : Union[float, np.ndarray]
        Pulsar period in milliseconds.
    
    Returns
    -------
    Union[float, np.ndarray]
        Light cylinder radius in R_NS.
    """
    return 4.77 * P_ms


def get_P_ms(R_LC: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the pulsar period in milliseconds for a given light cylinder radius (in R_NS)."""
    return R_LC / 4.77


def PA_RVM(alpha: Union[float, np.ndarray], observer_angle: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate the Rotating Vector Model (RVM) polarization angle.
    
    Parameters
    ----------
    alpha : Union[float, np.ndarray]
        Magnetic inclination angle in degrees.
    observer_angle : Union[float, np.ndarray]
        Observer angle (zeta) in degrees. Impact parameter beta = observer_angle - alpha.
    phi : Union[float, np.ndarray]
        Spin phase (azimuthal angle) in degrees.
    
    Returns
    -------
    Union[float, np.ndarray]
        Polarization angle in degrees, wrapped to [-90, 90].
    """
    alpha_rad = np.deg2rad(alpha)
    beta = observer_angle - alpha  # Impact parameter
    beta_rad = np.deg2rad(beta)
    phi_rad = np.deg2rad(phi)
    PA_rad = np.arctan2(
        np.sin(alpha_rad) * np.sin(phi_rad),
        np.sin(alpha_rad + beta_rad) * np.cos(alpha_rad) - np.cos(alpha_rad + beta_rad) * np.sin(alpha_rad) * np.cos(phi_rad)
    )
    PA_deg = np.rad2deg(PA_rad)
    PA_deg = (PA_deg + 90) % 180 - 90
    return PA_deg
