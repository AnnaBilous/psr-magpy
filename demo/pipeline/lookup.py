"""Match a dipole-plus-quadrupole experiment, and convert period and phase."""

import tempfile
from pathlib import Path

from psr_magpy.experiment.experiment_lookup import (
    experiment_matches_thD_fQ,
    find_experiment_numbers_by_thD_fQ,
    infer_experiment_count,
)
from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.field_config import FieldComponentConfig, FieldConfig
from psr_magpy.experiment.grid_config import GridConfig, SphericalGridConfig
from psr_magpy.experiment.tracing_config import TracingConfig
from psr_magpy.line.field_line_template import FieldLineTemplate
from psr_magpy.utils.physical_conversions import (
    PA_RVM,
    get_P_ms,
    get_R_LC,
    tangent_phi_to_observer_phi,
)


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="psr_magpy_lookup_"))
    series = ExperimentSeriesConfig(
        name="lookup_series",
        output_dir=str(out),
        field=FieldConfig(
            components=[
                FieldComponentConfig(
                    moment_type="dipole",
                    B_magnitude=1.0,
                    moment_orientation_latitude_degree=30.0,
                ),
                FieldComponentConfig(moment_type="quadrupole", B_magnitude=0.5),
            ]
        ),
        grid=GridConfig(spherical=SphericalGridConfig(nlon=4, nlat=3)),
        tracing=TracingConfig(save_results_to_disk=False),
        template=FieldLineTemplate.template_OFL_calc(),
    )
    experiments = series.create_experiment_series(save_configs=True)
    experiment = experiments[0]
    print(f"match={experiment_matches_thD_fQ(experiment, thD=30.0, fQ=0.5)}")
    print(f"n_experiments={infer_experiment_count(series)}")
    print(f"numbers={find_experiment_numbers_by_thD_fQ(series, thD=30.0, fQ=0.5)}")
    print(f"R_LC={get_R_LC(1.0)}")
    print(f"P_ms={get_P_ms(4.77)}")
    print(f"observer_phi={tangent_phi_to_observer_phi(90.0)}")
    print(f"PA={float(PA_RVM(30.0, 90.0, 0.0)):.3f}")


if __name__ == "__main__":
    main()
