import numpy as np
from astropy import units as u
from astropy.coordinates import PhysicsSphericalRepresentation
from astropy.coordinates.representation import CartesianRepresentation
from scipy.spatial.transform import Rotation as rot

r_ns = u.def_unit("Neutron star radius, 10 km", 10 * u.km)
u.add_enabled_units([r_ns])


class FieldComponent:
    def __init__(self, **kwargs):

        default_values = {
            "component_number": 0,
            "moment_type": "dipole",
            "B_magnitude": 1,
            "moment_orientation_latitude_degree": 45.0,
            "moment_orientation_longitude_degree": 0,
            "offset_x_r_ns": 0,
            "offset_y_r_ns": 0,
            "offset_z_r_ns": 0,
        }

        expected_types = {
            "component_number": int,
            "moment_type": str,
            "B_magnitude": float,
            "moment_orientation_latitude_degree": float,
            "moment_orientation_longitude_degree": float,
            "offset_x_r_ns": float,
            "offset_y_r_ns": float,
            "offset_z_r_ns": float,
        }

        for key in kwargs:
            if key not in default_values:
                raise ValueError(f"Invalid parameter name: {key}")

        for key, value in kwargs.items():
            expected_type = expected_types[key]
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Incorrect type for parameter {key}: expected {expected_type}"
                )

        assert kwargs["moment_type"] in [
            "dipole",
            "quadrupole",
        ], "Multipole must be dipole or quadrupole"

        print("Setting up field components with the following vars:")
        for key in default_values:
            print(f"\t\t{key}: {default_values[key]}")
        print("--------------------------------------")
        default_values.update(kwargs)

        self.component_number = default_values["component_number"]
        self.moment_type = default_values["moment_type"]
        self.B_magnitude = default_values["B_magnitude"]
        self.moment_orientation = PhysicsSphericalRepresentation(
            phi=default_values["moment_orientation_longitude_degree"] * u.deg,
            theta=default_values["moment_orientation_latitude_degree"] * u.deg,
            r=0,
        )
        self.moment_offset_vector = CartesianRepresentation(
            x=default_values["offset_x_r_ns"],
            y=default_values["offset_y_r_ns"],
            z=default_values["offset_z_r_ns"],
        )

        self.rotate_magn_to_spin_frame = rot.from_euler(
            "yzx",
            [self.moment_orientation.theta.rad, self.moment_orientation.phi.rad, 0],
        )
        self.rotate_spin_to_magn_frame = self.rotate_magn_to_spin_frame.inv()

    def get_field(self, r_spin_frame: np.array):
        r_spin_frame = r_spin_frame * 1
        r_spin_frame = CartesianRepresentation(x=r_spin_frame)
        r_offset_spin_frame = r_spin_frame - self.moment_offset_vector
        r_magn_frame_value = self.rotate_spin_to_magn_frame.apply(
            r_offset_spin_frame.xyz.T
        )
        r_magn_frame = CartesianRepresentation(x=r_magn_frame_value)
        r_magn_frame_cyl = CartesianRepresentation(
            x=r_magn_frame.x, y=r_magn_frame.y, z=0 * r_magn_frame.z
        )
        r = r_magn_frame.norm()
        r_cyl = r_magn_frame_cyl.norm()

        x = r_magn_frame.x
        y = r_magn_frame.y
        z = r_magn_frame.z

        # Magneic field defined in cartesian coordinates
        if self.moment_type == "dipole":
            B_magnetic_frame = (
                self.B_magnitude
                * r**-5
                * np.array(
                    [
                        1.5 * z * x,
                        1.5 * z * y,
                        z**2 - 0.5 * r_cyl**2,
                    ]
                )
            )
        if self.moment_type == "quadrupole":
            denom = (0.5 * (5 * z**2 / r**2 - 1) / r) ** -1
            B_magnetic_frame = (
                self.B_magnitude
                * r**-4
                * np.array(
                    [
                        x / denom,
                        y / denom,
                        z / denom - z / r,
                    ]
                )
            )

        B_spin_frame = self.rotate_magn_to_spin_frame.apply(B_magnetic_frame)

        return B_spin_frame
