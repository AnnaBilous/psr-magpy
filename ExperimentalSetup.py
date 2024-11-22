import itertools as it
import numpy as np


class ExperimentSetup:
    """ """

    def __init__(
        self,
        grid_type: str,
        grid_params: dict,
        params: dict,
        include_in_filename: list,
        filename_prefix="",
        path="",
    ):
        self.grid_type = grid_type
        for key, value in grid_params.items():
            setattr(self, key, value)
        for key, value in params.items():
            setattr(self, key, value)
        self.include_in_filename = include_in_filename
        self.filename_prefix = filename_prefix
        self.path = path

        not_expandable = set(["include_in_filename", "filename_prefix", "path"])
        expandable = set(vars(self).keys()) - not_expandable
        attribute_values = [getattr(self, attr_name) for attr_name in expandable]

        combinations = it.product(*attribute_values)
        print(combinations)

        run_parameters = [{**dict(zip(expandable, values))} for values in combinations]

        for parameter_dict in run_parameters:
            filename = self.filename_prefix
            if not (self.filename_prefix == ""):
                filename += "_"
            for fname_key in self.include_in_filename:
                if fname_key == "BQ":
                    filename += f"{fname_key}_{parameter_dict[fname_key]:.1f}_"
                elif fname_key == "inclination":
                    filename += f"{fname_key}_{parameter_dict[fname_key]:02g}_deg_"
                else:
                    filename += f"{fname_key}_{parameter_dict[fname_key]:g}_"
            parameter_dict["filename"] = filename[:-1]
        self.run_parameters = run_parameters
