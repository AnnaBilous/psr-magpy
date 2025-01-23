import numpy as np
from astropy.io import fits as pf
from typing import List

from FieldComponent import *


class FitsIO:
    def __init__(self, field_components: List[FieldComponent]):
        self.field_components = field_components

    def write_to_fits(self, filename: str):
        """Write the field components to a FITS file."""
        hdu_field_components = self.create_hdu_field_components()
        hdu_field_components.writeto(filename, overwrite=True)

    def create_hdu_field_components(self):
        field_component_param_names = [field.name for field in FieldComponentParams.__dataclass_fields__.values()]
        column_info = self.generate_column_info()

        magnetic_field_columns = [
            self.create_column(key, field_component_param_names, column_info[key]) 
            for key in column_info.keys()
        ]

        hdu_field_components = pf.BinTableHDU.from_columns(magnetic_field_columns, name="FLD_PRM")
        return hdu_field_components

    def generate_column_info(self):
        """Generate column information for FITS HDU."""
        type_mapping = {
            float: "D",
            int: "I",
            str: "10A",
        }
        print( [field.name for field in FieldComponentParams.__dataclass_fields__.values()])
        return {
            field.name: type_mapping[str] if field.name == "moment_type" else type_mapping[type(field.default)]
            for field in FieldComponentParams.__dataclass_fields__.values()
        }

    def create_column(self, key: str, field_component_param_names: list, format: str):
        # Custom extraction
        value_extraction = {
            "moment_type": lambda: [v.params.moment_type.value for v in self.field_components],
        }

        # Default extraction for standard fields
        value = value_extraction.get(key, lambda: [getattr(v.params, key) for v in self.field_components])()

        return pf.Column(name=key, format=format, array=value)

    def retrieve_field_components(hdu):

        data = hdu.data
        n_components = np.shape(data)[0]
        names_dict = {
            "component_number": "component_number",
            "moment_type": "moment_type",
            "B_magnitude": "B_magnitude",
            "moment_orientation_latitude_degree": "orientation_theta_deg",
            "moment_orientation_longitude_degree": "orientation_phi_deg",
            "offset_x_r_ns": "offset_x_R_NS",
            "offset_y_r_ns": "offset_y_R_NS",
            "offset_z_r_ns": "offset_z_R_NS",
        }
        components_data = [
            {key: None for key in names_dict.keys()} for _ in range(n_components)
        ]

        for key, val in names_dict.items():
            curr_val = data[val]
            for i in range(n_components):
                components_data[i][key] = curr_val[i]
                # Some weird mismatch in integer types
                if key == "component_number":
                    components_data[i][key] = int(curr_val[i])

        magnetic_field_components = MultipoleField()
        magnetic_field_components.initialize(components_data)
        return magnetic_field_components


class MultipoleField:
    def __init__(self, field_components: List[FieldComponent]):
        self.field_components = field_components
        for index, component in enumerate(self.field_components):
            component.params.component_number = index  
        self.fits_io = FitsIO(self.field_components)  

    def get_total_field(self, r, K=1, normalize=True):
        """Calculate total magnetic field at position r."""
        F = np.array([0.0, 0.0, 0.0])
        for multipole in self.field_components:
            F += multipole.get_field(r)
        F = F * K
        if normalize:
            F /= np.sum(F**2) ** 0.5
        return F

