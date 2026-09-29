"""Dipole, quadrupole, and both in one field."""

from psr_magpy.experiment.field_config import FieldComponentConfig, FieldConfig
from psr_magpy.field.components import DipoleFieldComponent, QuadrupoleFieldComponent
from psr_magpy.field.multipole import MultipoleField


def main() -> None:
    dipole = DipoleFieldComponent(
        B_magnitude=1.0,
        moment_orientation_latitude_degree=30.0,
    )
    quadrupole = QuadrupoleFieldComponent(B_magnitude=0.5)
    field = MultipoleField([dipole, quadrupole])
    config = FieldConfig(
        components=[
            FieldComponentConfig(
                moment_type="dipole",
                B_magnitude=1.0,
                moment_orientation_latitude_degree=30.0,
            ),
            FieldComponentConfig(moment_type="quadrupole", B_magnitude=0.5),
        ]
    )
    kinds = [component.get_moment_type().value for component in field.field_components]
    print(f"components={kinds}")
    print(f"config_types={[c.moment_type for c in config.components]}")


if __name__ == "__main__":
    main()
