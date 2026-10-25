"""Tests for thD/fQ experiment lookup."""

import tempfile
from pathlib import Path
import pytest

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_lookup import (
    experiment_matches_thD_fQ,
    find_experiment_numbers_by_thD_fQ,
    infer_experiment_count,
)
from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.field_config import FieldComponentConfig, FieldConfig
from psr_magpy.experiment.grid_config import GridConfig, HealpixGridConfig
from psr_magpy.experiment.tracing_config import TracingConfig
from psr_magpy.line.field_line_template import FieldLineTemplate


def _minimal_experiment(
    name: str, dip_lat: float, quad_b: float, nside: int
) -> ExperimentConfig:
    field = FieldConfig(
        components=[
            FieldComponentConfig(
                moment_type="dipole",
                B_magnitude=1.0,
                moment_orientation_latitude_degree=dip_lat,
                moment_orientation_longitude_degree=0.0,
            ),
            FieldComponentConfig(
                moment_type="quadrupole",
                B_magnitude=quad_b,
                moment_orientation_latitude_degree=0.0,
                moment_orientation_longitude_degree=0.0,
            ),
        ]
    )
    grid = GridConfig(healpix=HealpixGridConfig(nside=nside))
    return ExperimentConfig(
        name=name,
        field=field,
        grid=grid,
        tracing=TracingConfig(),
        template=FieldLineTemplate.template_line_coordinates_only(),
    )


class TestExperimentMatchesThDFQ:
    def test_match(self):
        exp = _minimal_experiment("s_000", 30.0, 1.5, 16)
        assert experiment_matches_thD_fQ(exp, 30.0, 1.5)
        assert not experiment_matches_thD_fQ(exp, 31.0, 1.5)


class TestExperimentLookupOnDisk:
    @pytest.mark.filterwarnings(
        "ignore:ExperimentConfig model I/O/workflow helpers.*:DeprecationWarning"
    )
    def test_find_by_thD_fQ(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            series_name = "lookup_test"
            configs = root / series_name / "configs"
            configs.mkdir(parents=True)

            for i, (th, fq) in enumerate([(10.0, 0.0), (20.0, 1.0)]):
                exp = _minimal_experiment(f"{series_name}_{i:06d}", th, fq, 32)
                exp.save(configs / f"{series_name}_{i:06d}.yaml", overwrite=True)

            summary_path = root / series_name / f"{series_name}_results_summary.yaml"
            summary_path.write_text("total_experiments: 2\n", encoding="utf-8")

            series = ExperimentSeriesConfig(
                name=series_name,
                field=FieldConfig(components=[]),
                grid=GridConfig(healpix=HealpixGridConfig(nside=32)),
                tracing=TracingConfig(),
                template=FieldLineTemplate.template_line_coordinates_only(),
                output_dir=str(root),
            )

            assert infer_experiment_count(series) == 2
            assert find_experiment_numbers_by_thD_fQ(series, 20.0, 1.0) == [1]
            assert find_experiment_numbers_by_thD_fQ(series, 10.0, 0.0) == [0]

    @pytest.mark.filterwarnings(
        "ignore:ExperimentConfig model I/O/workflow helpers.*:DeprecationWarning"
    )
    def test_infer_count_uses_configs_when_summary_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            series_name = "lookup_from_configs"
            configs = root / series_name / "configs"
            configs.mkdir(parents=True)

            for i, (th, fq) in enumerate([(10.0, 0.0), (20.0, 1.0), (30.0, 2.0)]):
                exp = _minimal_experiment(f"{series_name}_{i:06d}", th, fq, 16)
                exp.save(configs / f"{series_name}_{i:06d}.yaml", overwrite=True)

            series = ExperimentSeriesConfig(
                name=series_name,
                field=FieldConfig(components=[]),
                grid=GridConfig(healpix=HealpixGridConfig(nside=16)),
                tracing=TracingConfig(),
                template=FieldLineTemplate.template_line_coordinates_only(),
                output_dir=str(root),
            )

            assert infer_experiment_count(series) == 3
            assert find_experiment_numbers_by_thD_fQ(series, 30.0, 2.0) == [2]

    def test_infer_count_uses_summary_when_configs_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            series_name = "lookup_from_summary"
            series_root = root / series_name
            series_root.mkdir(parents=True)
            (series_root / f"{series_name}_results_summary.yaml").write_text(
                "total_experiments: 4\n", encoding="utf-8"
            )

            series = ExperimentSeriesConfig(
                name=series_name,
                field=FieldConfig(components=[]),
                grid=GridConfig(healpix=HealpixGridConfig(nside=16)),
                tracing=TracingConfig(),
                template=FieldLineTemplate.template_line_coordinates_only(),
                output_dir=str(root),
            )

            assert infer_experiment_count(series) == 4
            # Without config files, lookup has no per-experiment payload to match.
            assert find_experiment_numbers_by_thD_fQ(series, 10.0, 0.0) == []
