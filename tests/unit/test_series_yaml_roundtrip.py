"""YAML round-trip for a series grid factory and a dict field component."""

from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.factory_configs import (
    FieldFactoryConfig,
    GridFactoryConfig,
    SphericalGridFactoryConfig,
    expand_field_configs,
)
from psr_magpy.experiment.field_config import FieldComponentConfig, FieldConfig
from psr_magpy.experiment.tracing_config import TracingConfig
from psr_magpy.line.field_line_template import FieldLineTemplate


def test_spherical_grid_factory_survives_yaml(tmp_path):
    series = ExperimentSeriesConfig(
        name="dipole_series",
        output_dir=str(tmp_path),
        field=FieldConfig(
            components=[FieldComponentConfig(moment_type="dipole", B_magnitude=1.0)]
        ),
        grid=GridFactoryConfig(
            grids=[SphericalGridFactoryConfig(nlon=4, nlat=[3, 4])]
        ),
        tracing=TracingConfig(),
        template=FieldLineTemplate.template_OFL_calc(),
    )
    path = tmp_path / "dipole_series_config.yaml"
    series.save(str(path))
    loaded = ExperimentSeriesConfig.load(path)
    experiments = loaded.create_experiment_series(
        save_configs=False,
        create_directories=False,
    )
    assert len(experiments) == 2
    assert {exp.grid.spherical.nlat for exp in experiments} == {3, 4}


def test_dict_field_component_keeps_offset_vector():
    configs = expand_field_configs(
        FieldFactoryConfig(
            components=[
                {
                    "moment_type": "dipole",
                    "B_magnitude": 1.0,
                    "moment_orientation_latitude_degree": [0.0, 30.0],
                    "offset_r_ns": [0.0, 0.0, 0.0],
                }
            ]
        )
    )
    assert len(configs) == 1
    assert len(configs[0].components) == 2
    assert list(configs[0].components[0].offset_r_ns) == [0.0, 0.0, 0.0]
