import itertools as it
import numpy as np


class ExperimentSetup:
    """ """

    def __init__(
        self,
        grid_type: str,
        grid_params: dict,
        params: dict,
        saved_vars: str,
        filename_formatter: str,
        path="",
    ):
        self.grid_type = grid_type
        for key, value in grid_params.items():
            setattr(self, key, value)
        for key, value in params.items():
            setattr(self, key, value)
        self.path = path
        self.saved_vars = saved_vars
        self.filename_formatter = filename_formatter

    def get_parameters_dict(self):
        not_expandable = set(["path", "saved_vars"])
        expandable = set(vars(self).keys()) - not_expandable
        attribute_values = [
            (
                value
                if (isinstance(value, list) | isinstance(value, np.ndarray))
                else [value]
            )
            for value in (getattr(self, attr_name) for attr_name in expandable)
        ]

        combinations = it.product(*attribute_values)
        run_parameters = [
            {
                **dict(zip(expandable, values)),
                "filename": self.filename_formatter.format(
                    **dict(zip(expandable, values))
                ),
            }
            for values in combinations
        ]
        return run_parameters
