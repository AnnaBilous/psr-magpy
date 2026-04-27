"""Contract tests for YAML round-trip backward compatibility."""

from pathlib import Path
import yaml
import pytest

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig


@pytest.mark.filterwarnings(
    "ignore:ExperimentConfig model I/O/workflow helpers.*:DeprecationWarning"
)
def test_experiment_config_old_yaml_roundtrip(tmp_path):
    old_config_data = {
        "name": "roundtrip_experiment",
        "description": "legacy config",
        "field": {
            "components": [
                {
                    "moment_type": "dipole",
                    "B_magnitude": 1.0,
                    "moment_orientation_latitude_degree": 0.0,
                    "moment_orientation_longitude_degree": 0.0,
                }
            ]
        },
        "grid": {"spherical": {"nlon": 2, "nlat": 5, "pole_lat": 0.0}},
        "tracing": {
            "max_integration_radius": 10.0,
            "threshold_line_distance": 1.0,
            "t_span": [0.0, 1e6],
            "rtol": 1e-6,
            "atol": 1e-8,
            "surface_radius": 1.0,
            "dt": 0.1,
            "max_step": 1.0,
            "trace_open_lines_backwards": True,
            "verify_by_reversing": False,
            "max_memory_mb": 100.0,
            "safety_factor": 0.1,
            "max_batch_size": 1000,
            "check_interval_sec": 60,
            "n_cores": 1,
            "save_results_to_disk": True,
            "keep_results_in_memory": False,
            "skip_memory_estimation": False,
            "method": "RK45",
            "batch_size_safety_factor": 5,
        },
        "template": {
            "save_r_line": True,
            "save_pa_line": True,
            "save_r_max": True,
            "save_r_cyl_max": True,
            "save_z_abs_max": True,
            "save_b_ini": True,
            "save_b_r_ini_angle": True,
        },
        "metadata": {"source": "legacy"},
    }

    input_path = tmp_path / "old_experiment.yaml"
    output_path = tmp_path / "roundtrip_experiment.yaml"
    input_path.write_text(yaml.dump(old_config_data), encoding="utf-8")

    loaded = ExperimentConfig.load(input_path)
    loaded.save(output_path, overwrite=True)
    roundtripped = ExperimentConfig.load(output_path)

    assert roundtripped.name == "roundtrip_experiment"
    assert roundtripped.metadata["source"] == "legacy"
    assert roundtripped.template.property_system.save_flags.save_r_line is True
    # Missing legacy fields remain backward-compatible with defaults.
    assert roundtripped.template.property_system.save_flags.save_b_line is False


@pytest.mark.filterwarnings(
    "ignore:ExperimentSeriesConfig model I/O/workflow helpers.*:DeprecationWarning"
)
def test_experiment_series_old_yaml_roundtrip(tmp_path):
    old_series_data = {
        "name": "roundtrip_series",
        "description": "legacy series",
        "field": {
            "components": [
                {
                    "moment_type": "dipole",
                    "B_magnitude": 1.0,
                    "moment_orientation_latitude_degree": 0.0,
                    "moment_orientation_longitude_degree": 0.0,
                }
            ]
        },
        "grid": {"spherical": {"nlon": 2, "nlat": 5, "pole_lat": 0.0}},
        "tracing": {
            "max_integration_radius": 10.0,
            "threshold_line_distance": 1.0,
            "t_span": [0.0, 1e6],
            "rtol": 1e-6,
            "atol": 1e-8,
            "surface_radius": 1.0,
            "dt": 0.1,
            "max_step": 1.0,
            "trace_open_lines_backwards": True,
            "verify_by_reversing": False,
            "max_memory_mb": 100.0,
            "safety_factor": 0.1,
            "max_batch_size": 1000,
            "check_interval_sec": 60,
            "n_cores": 1,
            "save_results_to_disk": True,
            "keep_results_in_memory": False,
            "skip_memory_estimation": False,
            "method": "RK45",
            "batch_size_safety_factor": 5,
        },
        "template": {
            "save_r_line": True,
            "save_pa_line": True,
            "save_r_max": True,
            "save_r_cyl_max": True,
            "save_z_abs_max": True,
            "save_b_ini": True,
            "save_b_r_ini_angle": True,
        },
        "metadata": {"source": "legacy_series"},
        "output_dir": str(tmp_path),
    }

    input_path = tmp_path / "old_series.yaml"
    output_path = tmp_path / "roundtrip_series.yaml"
    input_path.write_text(yaml.dump(old_series_data), encoding="utf-8")

    loaded = ExperimentSeriesConfig.load(input_path)
    loaded.save(str(output_path))
    roundtripped = ExperimentSeriesConfig.load(output_path)

    assert roundtripped.name == "roundtrip_series"
    assert roundtripped.metadata["source"] == "legacy_series"
    assert roundtripped.template.property_system.save_flags.save_pa_line is True
    assert roundtripped.template.property_system.save_flags.save_curvature_line is False
