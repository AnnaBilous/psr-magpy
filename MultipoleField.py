import numpy as np
from astropy.io import fits as pf
from typing import List

from FieldComponent import *


class MultipoleField:
    def __init__(self, field_components: List[FieldComponent]):
        self.field_components = field_components
        for index, component in enumerate(self.field_components):
            component.params.component_number = index

    def get_total_field(self, r, K=1, normalize=True):
        """Calculate total magnetic field at position r."""
        F = np.array([0.0, 0.0, 0.0])
        for multipole in self.field_components:
            F += multipole.get_field(r)
        F = F * K
        if normalize:
            F /= np.sum(F**2) ** 0.5
        return F

    # @classmethod
    # def load_from_fits(cls, hdu):
    #    """Load data from a FITS file using FitsIO."""
    #    from FitsIO import FitsIO  # Import only when needed
    #    fits_io = FitsIO()  # Initialize FitsIO
    #    return fits_io.load_from_fits(hdu)

    def save_to_fits(self, filename: str):
        """Save field configuration to FITS file.

        Parameters
        ----------
        filename : str
            Path to output FITS file
        """
        from FitsIO import FieldComponentFitsIO  # Import only when needed

        fits_io = FieldComponentFitsIO(self.field_components)
        fits_io.save_to_fits(filename)
