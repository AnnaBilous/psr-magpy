
import numpy as np

from FieldComponent import *


class MultipoleField:
    def __init__(self, field_components: list):
        self.field_components = field_components

    def get_total_field(self, r, K=1, normalize=True):
        """Calculate total magnetic field at position r."""
        F = np.array([0.0, 0.0, 0.0])
        for multipole in self.field_components:
            F += multipole.get_field(r)
        F = F * K
        if normalize:
            F /= np.sum(F**2) ** 0.5
        return F
