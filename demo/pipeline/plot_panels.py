"""Plot field lines, an open-field-line map, a skymap, and a PA curve from arrays."""

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from psr_magpy.field.components import DipoleFieldComponent
from psr_magpy.field.multipole import MultipoleField
from psr_magpy.viz.field_2d import plot_field_lines_2d
from psr_magpy.viz.oflr import plot_OFLR_map
from psr_magpy.viz.pa import plot_PA_vs_phi
from psr_magpy.viz.skymap import plot_skymap


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="psr_magpy_plots_"))
    field = MultipoleField(
        [DipoleFieldComponent(moment_orientation_latitude_degree=30.0)]
    )
    fig, ax = plt.subplots()
    plot_field_lines_2d(
        field,
        plane="xy",
        fixed_coord=0.0,
        x_range=(1.0, 2.0),
        y_range=(1.0, 2.0),
        n_points=3,
        ax=ax,
    )
    fig.savefig(out / "field_lines.png")
    plt.close(fig)

    image = np.linspace(0.0, 1.0, 8).reshape(2, 4)
    longitude = np.linspace(0.0, 360.0, 4, endpoint=False)
    latitude = np.array([60.0, 120.0])
    fig, _, _ = plot_OFLR_map(image, longitude, latitude)
    fig.savefig(out / "oflr.png")
    plt.close(fig)

    fig, _, _ = plot_skymap(image, longitude, latitude)
    fig.savefig(out / "skymap.png")
    plt.close(fig)

    phi = np.linspace(0.0, 360.0, 12, endpoint=False)
    fig, _ = plot_PA_vs_phi(
        np.linspace(-40.0, 40.0, phi.size),
        phi,
        target_theta=90.0,
        alpha_angle=30.0,
    )
    fig.savefig(out / "pa.png")
    plt.close(fig)
    print(f"output={out}")


if __name__ == "__main__":
    main()
