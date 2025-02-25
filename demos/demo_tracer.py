import sys
from pathlib import Path

# Add project root to PYTHONPATH only if not already present
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from src.field.multipole import MultipoleField
from src.footpoint_grid.grid import SphericalGrid
from src.field.components import DipoleFieldComponent, QuadrupoleFieldComponent
from src.line.tracer import FieldLineTracer
from src.line.field_line import FieldLine

def main():
    # Create magnetic field (example with dipole)
    dipole = DipoleFieldComponent(
        B_magnitude=1.0,
        moment_orientation_latitude_degree=60.0,
    )
    quadrupole = QuadrupoleFieldComponent(
        B_magnitude=3.0,
        moment_orientation_latitude_degree=45.0,
    )
    field = MultipoleField([dipole, quadrupole])

    # Initialize field line tracer
    tracer = FieldLineTracer(field, R_max=3)

    # Create spherical grid of footpoints
    grid = SphericalGrid(nlon=8, nlat=20, pole_lat=60.0)
    footpoints = grid.footpoint_coordinates

    # Create template for field line properties to save
    template = FieldLine(save_r_line=True)

    # Calculate field lines
    field_lines = []
    for i, footpoint in enumerate(footpoints):
        line = tracer.trace_single_line(footpoint, i, template)
        field_lines.append(line.r_line)

    # Plot results
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot field lines
    for line in field_lines:
        ax.plot(line[:, 0], line[:, 1], line[:, 2], "b-", alpha=0.5)

    # Plot NS surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="r", alpha=0.1)

    # Plot footpoints
    xyz = footpoints.xyz.value
    ax.scatter(xyz[0], xyz[1], xyz[2], c='black', s=20, alpha=1.0)

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()

if __name__ == "__main__":
    main() 