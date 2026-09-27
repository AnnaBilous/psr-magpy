"""Observer line-of-sight overlay on equirectangular panels."""

from __future__ import annotations

from typing import Optional

from matplotlib.axes import Axes


def overlay_observer_los(
    ax: Axes,
    theta_obs: Optional[float],
    *,
    color: str = "b",
    linewidth: float = 0.7,
    zorder: int = 3,
) -> None:
    """Draw horizontal LOS at observer colatitude ``theta_obs`` (degrees).

    Equirectangular panels use ``extent=(0, 360, 180, 0)`` so colatitude is the
    vertical axis and a constant-θ slice is an ``axhline``.
    """
    if theta_obs is None:
        return
    ax.axhline(y=theta_obs, color=color, linewidth=linewidth, zorder=zorder)
