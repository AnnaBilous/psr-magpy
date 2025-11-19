#!/usr/bin/env python3
"""
Test script to verify both integration systems work correctly.
"""

import numpy as np
import time
from psr_magpy.line.field_line import FieldLine
from psr_magpy import *
import numpy as np
import matplotlib.pyplot as plt

def create_field_config():


    aligned_dipole_config = FieldComponentConfig(
        moment_type="dipole",
        B_magnitude=1, 
        moment_orientation_latitude_degree=0.0, 
    )
    
    orthogonal_dipole_config = FieldComponentConfig(
        moment_type="dipole",
        B_magnitude=1, 
        moment_orientation_latitude_degree=0.0, 
    )
    
    inclined_dipole_config = FieldComponentConfig(
        moment_type="dipole",
        B_magnitude=1, 
        moment_orientation_latitude_degree=30.0, 
    )
    
    quadrupole_config = FieldComponentConfig(
        moment_type="quadrupole",
        B_magnitude=3, 
        moment_orientation_latitude_degree=0.0, 
    )
    field_config = FieldFactoryConfig(components=[
        [orthogonal_dipole_config],
        [aligned_dipole_config],
        [inclined_dipole_config, quadrupole_config]
        ]
    )
  
    return field_config


if __name__ == "__main__":

    field_factory = create_field_config()

    def make_grid_factory(nlon: int) -> GridFactoryConfig:
        grid = SphericalGridFactoryConfig(nlon=nlon, nlat=50, pole_lat=0.0)
        return GridFactoryConfig(grids=[grid])

    def run_and_report(system_label: str, *, max_integration_radius: float, nlon: int):
        tracing_config = TracingFactoryConfig(
            max_integration_radius=max_integration_radius,
            save_results_to_disk=False,
            max_memory_mb=100,
            skip_memory_estimation=True,
            max_step=1.0,
            integration_system=system_label,
            verify_by_reversing=True,
        )

        series_config = ExperimentSeriesConfig(
            name=f"test_integration_systems_{system_label}_R{int(max_integration_radius)}_nlon{nlon}",
            field=field_factory,
            grid=make_grid_factory(nlon),
            tracing=tracing_config,
            template=FieldLine.template_line_coordinates_only(),
            description=f"Test integration system {system_label}",
        )

        t0 = time.perf_counter()
        series_result = run_experiment_series_in_memory(
            series_config=series_config,
            save_individual_configs=False,
            save_grids=False,
            continue_on_failure=True,
        )
        elapsed = time.perf_counter() - t0

        rev_errors = []
        for exp in series_result.experiment_results:
            if not getattr(exp, 'success', False) or exp.results is None:
                continue
            results_obj = exp.results
            try:
                err = results_obj.get_property('reversed_error')
            except Exception:
                err = getattr(results_obj, 'reversed_error', None)
            if err is not None and hasattr(err, '__len__') and len(err) > 0:
                rev_errors.append(np.max(err))
        if rev_errors:
            max_rev_err = np.max(rev_errors)
        else:
            max_rev_err = None

        print(f"{system_label}: {elapsed:.3f}s, max_reversed_error={max_rev_err}")
        return system_label, elapsed, max_rev_err

    results_table = []
    # Benchmark requested cases
    cases = [
        {"max_integration_radius": 500, "nlon": 2},
        {"max_integration_radius": 500, "nlon": 50},
    ]

    for case in cases:
        R = case["max_integration_radius"]
        nlon = case["nlon"]
        for sys_lbl in ["A", "B"]:
            label, elapsed, max_err = run_and_report(sys_lbl, max_integration_radius=R, nlon=nlon)
            results_table.append({
                "case": f"Rmax={R} nlon={nlon}",
                "system": label,
                "time": elapsed,
                "max_reversed_error": max_err
            })

    # Pretty print one table at the end
    print("\nSummary:")
    print(f"{'Case':<18} {'System':<6} {'Time[s]':>8} {'MaxReversedError':>18}")
    for row in results_table:
        print(f"{row['case']:<18} {row['system']:>6} {row['time']:8.3f} {str(row['max_reversed_error']):>18}")
