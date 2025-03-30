from astropy.io import fits as pf

from src.fitsio.base import FitsIO
from src.field.components import MultipoleType, DipoleFieldComponent, QuadrupoleFieldComponent
from src.field.multipole import MultipoleField



class FieldComponentFitsIO(FitsIO):
    """FITS I/O operations for field components."""

    def __init__(self, field: MultipoleField):
        """Initialize with a MultipoleField instance.
        
        Parameters
        ----------
        field : MultipoleField
            The multipole field to save/load
        
        Raises
        ------
        TypeError
            If field is not a MultipoleField instance
        """
        if not isinstance(field, MultipoleField):
            raise TypeError(f"Expected MultipoleField, got {type(field).__name__}")
        self.field = field

    def get_hdu_name(self) -> str:
        """Get the name for the FITS HDU.
        
        Returns
        -------
        str
            Name for the FITS HDU
        """
        return "FIELD_COMPONENTS"

    def generate_column_info(self):
        """Generate column information for field component data."""
        first_component = self.field.field_components[0]
       
        params_dict = first_component.params._asdict()
        
        column_info = {
            param_name: (
                FitsIO.FITS_TYPE_MAPPING[str]
                if param_name == "moment_type"
                else (
                    FitsIO.FITS_TYPE_MAPPING[list]
                    if param_name == "offset_r_ns"
                    else FitsIO.get_fits_format(param_value)
                )
            )
            for param_name, param_value in params_dict.items()         
        }
        
        return column_info

    def create_column(self, key: str, format: str):
        """Create a FITS column for field component data."""
        if key == "moment_type":
            value = [v.params.moment_type.value for v in self.field.field_components]
        else:
            value = [getattr(v.params, key) for v in self.field.field_components]
        column = pf.Column(name=key, format=format, array=value)
        return column

    @classmethod
    def load_from_fits(cls, fitsname: str) -> MultipoleField:
        """Load MultipoleField from a FITS file.
        
        Parameters
        ----------
        fitsname : str
            Path to the FITS file
            
        Returns
        -------
        MultipoleField
            Loaded multipole field instance
        """
        with pf.open(fitsname) as hdulist:
            field_data = hdulist['FIELD_COMPONENTS'].data
            n_components = field_data.shape[0]
            components = []
            
            for i in range(n_components):
                moment_type = MultipoleType(field_data['moment_type'][i])
                if moment_type == MultipoleType.DIPOLE:
                    component_class = DipoleFieldComponent
                elif moment_type == MultipoleType.QUADRUPOLE:
                    component_class = QuadrupoleFieldComponent
                else:
                    raise ValueError(f"Unsupported multipole type: {moment_type}")
                
                component_params = {
                    key: field_data[key][i] for key in field_data.columns.names
                    if key != 'moment_type'  # Handle moment_type separately
                }

                component = component_class(**component_params)
                components.append(component)
            
            return MultipoleField(components)


