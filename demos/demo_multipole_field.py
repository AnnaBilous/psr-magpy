import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
import numpy as np
from astropy.io import fits as pf
from FieldComponent import (
    FieldComponent,
)  # Adjust the import based on your project structure
from MultipoleField import (
    MultipoleField,
    MultipoleType,
)  # Adjust the import based on your project structure


if __name__ == "__main__":

    dipole = FieldComponent(
        moment_type=MultipoleType.DIPOLE,
        B_magnitude=1.0,
        moment_orientation_latitude_degree=0.0,
    )
    quadrupole = FieldComponent(moment_type=MultipoleType.QUADRUPOLE, B_magnitude=3.0)
    field = MultipoleField([dipole, quadrupole])

    # Specify the output FITS file name
    output_filename = "multipole_field.fits"
    field.save_to_fits(output_filename)

    # hdu = pf.open(output_filename)
    # field_new = MultipoleField.load_from_fits(hdu[1], FieldComponent)
