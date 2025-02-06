from typing import List

import numpy as np

from FieldComponent import ConcreteFieldComponent, DipoleFieldComponent, QuadrupoleFieldComponent



class MultipoleField:
    """Class representing a multipole magnetic field composed of multiple field components.

    Attributes
    ----------
    field_components : List[FieldComponent]
        List of field components (DipoleFieldComponent or QuadrupoleFieldComponent instances)
        that make up the multipole field.
    """

    def __init__(self, field_components: List[ConcreteFieldComponent]):
        """Initialize MultipoleField with a list of field components.

        Args:
            field_components: List of field components (must be instances of DipoleFieldComponent
                            or QuadrupoleFieldComponent).

        Raises:
            TypeError: If any component is not an instance of DipoleFieldComponent
                      or QuadrupoleFieldComponent.
        """
        if not all(isinstance(comp, (DipoleFieldComponent, QuadrupoleFieldComponent)) for comp in field_components):
            raise TypeError(
                "All components must be instances of DipoleFieldComponent or QuadrupoleFieldComponent"
            )
        
        self.field_components = field_components
        for index, component in enumerate(self.field_components):
            component.params.component_number = index

    def get_total_field(self, r: np.ndarray, K: float = 1, normalize: bool = True) -> np.ndarray:
        """Calculate total magnetic field at position r.

        Args:
            r: Position vector in Cartesian coordinates.
            K: Field direction multiplier (default: 1).
            normalize: Whether to normalize the field vector (default: True).

        Returns:
            np.ndarray: The total magnetic field vector.
        """
        F = np.array([0.0, 0.0, 0.0])
        for component in self.field_components:
            F += component.get_field(r)
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

    def save_to_fits(self, filename: str) -> None:
        """Save field configuration to FITS file.

        Args:
            filename: Path to output FITS file.
        """
        from FitsIO import FieldComponentFitsIO  # Import only when needed
        fits_io = FieldComponentFitsIO(self.field_components)
        fits_io.save_to_fits(filename)
