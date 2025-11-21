"""Tests for backwards compatibility of YAML configuration loading."""

import pytest
import yaml
from pathlib import Path
import tempfile

from psr_magpy.experiment.experiment_config import ExperimentConfig
from psr_magpy.experiment.experiment_series import ExperimentSeriesConfig
from psr_magpy.experiment.factory_configs import TemplateFactoryConfig
from psr_magpy.line.field_line import FieldLine
from psr_magpy.line.property_system import PropertyName


class TestBackwardsCompatibility:
    """Test backwards compatibility for YAML configuration loading."""

    def test_old_experiment_config_missing_new_properties(self):
        """Test that old experiment configs without new properties load correctly."""
        # Create an old-style config without new properties
        old_config_data = {
            "name": "test_experiment",
            "description": "Test experiment",
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
            "grid": {
                "spherical": {
                    "nlon": 2,
                    "nlat": 5,
                    "pole_lat": 0.0
                }
            },
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
                "batch_size_safety_factor": 5
            },
            "template": {
                "save_r_line": True,
                "save_pa_line": True,
                "save_r_max": True,
                "save_r_cyl_max": True,
                "save_z_abs_max": True,
                "save_b_ini": True,
                "save_b_r_ini_angle": True,
                # Note: save_b_line and save_curvature_line are missing
            },
            "metadata": {}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(old_config_data, f)
            config_path = f.name

        try:
            # Load the config
            config = ExperimentConfig.load(config_path)
            
            # Check that new properties have default values
            assert config.template.property_system.save_flags.save_b_line is False
            assert config.template.property_system.save_flags.save_curvature_line is False
            assert config.template.property_system.save_flags.save_tangent_theta is False
            assert config.template.property_system.save_flags.save_tangent_phi is False
            
            # Check that old properties are preserved
            assert config.template.property_system.save_flags.save_r_line is True
            assert config.template.property_system.save_flags.save_pa_line is True
            
        finally:
            Path(config_path).unlink()

    def test_old_series_config_missing_new_properties(self):
        """Test that old series configs without new properties load correctly."""
        # Create an old-style series config without new properties
        old_series_data = {
            "name": "test_series",
            "description": "Test series",
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
            "grid": {
                "spherical": {
                    "nlon": 2,
                    "nlat": 5,
                    "pole_lat": 0.0
                }
            },
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
                "batch_size_safety_factor": 5
            },
            "template": {
                "save_r_line": True,
                "save_pa_line": True,
                "save_r_max": True,
                "save_r_cyl_max": True,
                "save_z_abs_max": True,
                "save_b_ini": True,
                "save_b_r_ini_angle": True,
                # Note: save_b_line and save_curvature_line are missing
            },
            "metadata": {}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(old_series_data, f)
            config_path = f.name

        try:
            # Load the config
            config = ExperimentSeriesConfig.load(config_path)
            
            # Check that new properties have default values
            assert config.template.property_system.save_flags.save_b_line is False
            assert config.template.property_system.save_flags.save_curvature_line is False
            assert config.template.property_system.save_flags.save_tangent_theta is False
            assert config.template.property_system.save_flags.save_tangent_phi is False
            
            # Check that old properties are preserved
            assert config.template.property_system.save_flags.save_r_line is True
            assert config.template.property_system.save_flags.save_pa_line is True
            
        finally:
            Path(config_path).unlink()

    def test_template_factory_config_missing_new_properties(self):
        """Test that old template factory configs without new properties load correctly."""
        # Create an old-style template factory config without new properties
        old_factory_data = {
            "templates": [
                {
                    "property_system": {
                        "save_r_line": True,
                        "save_pa_line": True,
                        "save_r_max": True,
                        "save_r_cyl_max": True,
                        "save_z_abs_max": True,
                        "save_b_ini": True,
                        "save_b_r_ini_angle": True,
                        # Note: save_b_line and save_curvature_line are missing
                    }
                }
            ]
        }

        # Load the config using the factory method
        factory_config = TemplateFactoryConfig.from_yaml_dict(old_factory_data)
        
        # Check that new properties have default values
        template = factory_config.templates[0]
        assert template.property_system.save_flags.save_b_line is False
        assert template.property_system.save_flags.save_curvature_line is False
        assert template.property_system.save_flags.save_tangent_theta is False
        assert template.property_system.save_flags.save_tangent_phi is False
        
        # Check that old properties are preserved
        assert template.property_system.save_flags.save_r_line is True
        assert template.property_system.save_flags.save_pa_line is True

    def test_new_config_with_new_properties(self):
        """Test that new configs with new properties load correctly."""
        # Create a new-style config with new properties
        new_config_data = {
            "name": "test_experiment",
            "description": "Test experiment",
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
            "grid": {
                "spherical": {
                    "nlon": 2,
                    "nlat": 5,
                    "pole_lat": 0.0
                }
            },
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
                "batch_size_safety_factor": 5
            },
            "template": {
                "save_r_line": True,
                "save_pa_line": True,
                "save_r_max": True,
                "save_r_cyl_max": True,
                "save_z_abs_max": True,
                "save_b_ini": True,
                "save_b_r_ini_angle": True,
                "save_b_line": True,  # New property
                "save_curvature_line": False,  # New property
            },
            "metadata": {}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(new_config_data, f)
            config_path = f.name

        try:
            # Load the config
            config = ExperimentConfig.load(config_path)
            
            # Check that new properties have correct values
            assert config.template.property_system.save_flags.save_b_line is True
            assert config.template.property_system.save_flags.save_curvature_line is False
            assert config.template.property_system.save_flags.save_tangent_theta is False
            assert config.template.property_system.save_flags.save_tangent_phi is False
            
            # Check that old properties are preserved
            assert config.template.property_system.save_flags.save_r_line is True
            assert config.template.property_system.save_flags.save_pa_line is True
            
        finally:
            Path(config_path).unlink()
