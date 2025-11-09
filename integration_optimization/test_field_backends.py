#!/usr/bin/env python3
"""
Benchmark script comparing field_backend="python" vs "numba" performance.
"""

import numpy as np
import time
from psr_magpy.line.field_line import FieldLine
from psr_magpy import *


def create_field_config():
    """Create field configuration for benchmarking."""
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
        grid = SphericalGridFactoryConfig(nlon=nlon, nlat=10, pole_lat=0.0)
        return GridFactoryConfig(grids=[grid])

    def run_and_report(backend_label: str, *, max_integration_radius: float, nlon: int):
        """Run experiment with specified field backend and report results."""
        tracing_config = TracingFactoryConfig(
            max_integration_radius=max_integration_radius,
            save_results_to_disk=False,
            max_memory_mb=100,
            skip_memory_estimation=True,
            max_step=1.0,
            field_backend=backend_label,
            verify_by_reversing=True,
        )

        series_config = ExperimentSeriesConfig(
            name=f"test_field_backend_{backend_label}_R{int(max_integration_radius)}_nlon{nlon}",
            field=field_factory,
            grid=make_grid_factory(nlon),
            tracing=tracing_config,
            template=FieldLine.template_line_coordinates_only(),
            description=f"Test field backend {backend_label}",
        )

        # Verify field configurations
        experiments = series_config.create_experiment_series()
        field_configs = set()
        for exp in experiments:
            field_desc = f"{len(exp.field.components)}_comp"
            for comp in exp.field.components:
                field_desc += f"_{comp.moment_type}"
            field_configs.add(field_desc)
        
        if backend_label == "python":  # Only print once
            print(f"  Field configurations: {sorted(field_configs)}")
            print(f"  Total experiments: {len(experiments)}")

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

        print(f"{backend_label}: {elapsed:.3f}s, max_reversed_error={max_rev_err}")
        return backend_label, elapsed, max_rev_err, series_result

    results_table = []
    series_results = {}  # Store series results for comparison
    # Benchmark requested cases
    cases = [
        {"max_integration_radius": 500, "nlon": 2},
        #{"max_integration_radius": 500, "nlon": 10},
    ]

    for case in cases:
        R = case["max_integration_radius"]
        nlon = case["nlon"]
        case_key = f"Rmax={R} nlon={nlon}"
        series_results[case_key] = {}
        
        for backend_lbl in ["python", "numba"]:
            label, elapsed, max_err, series_result = run_and_report(backend_lbl, max_integration_radius=R, nlon=nlon)
            results_table.append({
                "case": case_key,
                "backend": label,
                "time": elapsed,
                "max_reversed_error": max_err
            })
            series_results[case_key][backend_lbl] = series_result

    def compare_r_line_trajectories(python_result, numba_result):
        """Compare r_line trajectories between python and numba backends."""
        max_distance = 0.0
        total_comparisons = 0
        
        # Get all experiment results (filter successful ones)
        python_exps = [exp for exp in python_result.experiment_results 
                      if getattr(exp, 'success', False) and exp.results is not None]
        numba_exps = [exp for exp in numba_result.experiment_results 
                     if getattr(exp, 'success', False) and exp.results is not None]
        
        # Check same number of successful experiments
        if len(python_exps) != len(numba_exps):
            error_msg = (
                f"Mismatched number of successful experiments: "
                f"python={len(python_exps)}, numba={len(numba_exps)}"
            )
            print(f"ERROR: {error_msg}")
            exit(1)
        
        # Match experiments by index (they should be in the same order since they use
        # the same field/grid/tracing configs, only field_backend differs)
        for i in range(len(python_exps)):
            python_exp = python_exps[i]
            numba_exp = numba_exps[i]
            
            # Get r_line from both - it's always a list of arrays (one per field line)
            python_r_line = python_exp.results.get_property('r_line')
            numba_r_line = numba_exp.results.get_property('r_line')
            
            if python_r_line is None:
                error_msg = (
                    f"Experiment {i} ({python_exp.experiment_name}): "
                    f"python r_line is None"
                )
                print(f"ERROR: {error_msg}")
                exit(1)
            
            if numba_r_line is None:
                error_msg = (
                    f"Experiment {i} ({numba_exp.experiment_name}): "
                    f"numba r_line is None"
                )
                print(f"ERROR: {error_msg}")
                exit(1)
            
            # r_line is always a list (one array per field line)
            if not isinstance(python_r_line, list):
                error_msg = (
                    f"Experiment {i} ({python_exp.experiment_name}): "
                    f"python r_line is not a list, got {type(python_r_line)}"
                )
                print(f"ERROR: {error_msg}")
                exit(1)
            
            if not isinstance(numba_r_line, list):
                error_msg = (
                    f"Experiment {i} ({numba_exp.experiment_name}): "
                    f"numba r_line is not a list, got {type(numba_r_line)}"
                )
                print(f"ERROR: {error_msg}")
                exit(1)
            
            # Check same number of field lines
            if len(python_r_line) != len(numba_r_line):
                error_msg = (
                    f"Experiment {i} ({python_exp.experiment_name}): "
                    f"Mismatched number of field lines: "
                    f"python={len(python_r_line)}, numba={len(numba_r_line)}"
                )
                print(f"ERROR: {error_msg}")
                exit(1)
            
            # Compare each field line
            for line_idx, (py_line, nb_line) in enumerate(zip(python_r_line, numba_r_line)):
                # Handle empty arrays (field lines that didn't reach threshold)
                if py_line is None:
                    error_msg = (
                        f"Experiment {i}, field line {line_idx}: "
                        f"python r_line[{line_idx}] is None"
                    )
                    print(f"ERROR: {error_msg}")
                    exit(1)
                
                if nb_line is None:
                    error_msg = (
                        f"Experiment {i}, field line {line_idx}: "
                        f"numba r_line[{line_idx}] is None"
                    )
                    print(f"ERROR: {error_msg}")
                    exit(1)
                
                if hasattr(py_line, '__len__') and len(py_line) == 0:
                    # Empty array is acceptable (field line didn't reach threshold)
                    continue
                
                if hasattr(nb_line, '__len__') and len(nb_line) == 0:
                    # Empty array is acceptable (field line didn't reach threshold)
                    continue
                
                # Convert to numpy arrays
                py_arr = np.asarray(py_line, dtype=np.float64)
                nb_arr = np.asarray(nb_line, dtype=np.float64)
                
      
                
                # Calculate distances at each point
                distances = np.linalg.norm(py_arr - nb_arr, axis=1)
          
                if len(distances) > 0:
                    max_distance = max(max_distance, np.max(distances))
                    total_comparisons += len(distances)
        
        return max_distance, total_comparisons, 0

    # Calculate speedups and r_line comparisons
    speedup_table = []
    for case in cases:
        R = case["max_integration_radius"]
        nlon = case["nlon"]
        case_key = f"Rmax={R} nlon={nlon}"
        
        python_row = next(r for r in results_table if r["case"] == case_key and r["backend"] == "python")
        numba_row = next(r for r in results_table if r["case"] == case_key and r["backend"] == "numba")
        
        speedup = python_row["time"] / numba_row["time"] if numba_row["time"] > 0 else None
        
        # Compare r_line trajectories
        max_r_line_distance, total_points, mismatched = compare_r_line_trajectories(
            series_results[case_key]["python"],
            series_results[case_key]["numba"]
        )
        
        speedup_table.append({
            "case": case_key,
            "python_time": python_row["time"],
            "numba_time": numba_row["time"],
            "speedup": speedup,
            "max_reversed_error": python_row["max_reversed_error"],  # Should be same for both
            "max_r_line_distance": max_r_line_distance,
            "total_points_compared": total_points,
            "mismatched_lengths": mismatched
        })

    # Pretty print summary table
    print("\n" + "="*70)
    print("Summary:")
    print("="*70)
    print(f"{'Case':<18} {'Backend':<8} {'Time[s]':>10} {'MaxReversedError':>18}")
    print("-"*70)
    for row in results_table:
        print(f"{row['case']:<18} {row['backend']:<8} {row['time']:10.3f} {str(row['max_reversed_error']):>18}")
    
    print("\n" + "="*90)
    print("Speedup Analysis:")
    print("="*90)
    print(f"{'Case':<18} {'Python[s]':>10} {'Numba[s]':>10} {'Speedup':>10} {'MaxReversedError':>18} {'MaxRLineDist':>12}")
    print("-"*90)
    for row in speedup_table:
        speedup_str = f"{row['speedup']:.2f}x" if row['speedup'] is not None else "N/A"
        max_dist_str = f"{row['max_r_line_distance']:.2e}" if row['max_r_line_distance'] >= 0 else "N/A"
        print(f"{row['case']:<18} {row['python_time']:10.3f} {row['numba_time']:10.3f} {speedup_str:>10} {str(row['max_reversed_error']):>18} {max_dist_str:>12}")
    
    print("\n" + "="*90)
    print("r_line Trajectory Comparison Details:")
    print("="*90)
    print(f"{'Case':<18} {'MaxDistance':>12} {'TotalPoints':>12} {'Mismatched':>12}")
    print("-"*90)
    for row in speedup_table:
        max_dist_str = f"{row['max_r_line_distance']:.2e}" if row['max_r_line_distance'] >= 0 else "N/A"
        print(f"{row['case']:<18} {max_dist_str:>12} {row['total_points_compared']:>12} {row['mismatched_lengths']:>12}")

