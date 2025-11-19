#!/usr/bin/env python3
"""
Benchmark script comparing field_backend and solver_backend combinations:
- scipy + python field
- scipy + numba field
- cyrk + python field
- cyrk + numba field
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from psr_magpy.line.field_line import FieldLine
from psr_magpy import *


def create_field_config():
    """Create field configuration for benchmarking."""
    aligned_dipole_config = FieldComponentConfig(
        moment_type="dipole",
        B_magnitude=1, 
        moment_orientation_latitude_degree=60.0, 
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
        [orthogonal_dipole_config, aligned_dipole_config],
        [aligned_dipole_config],
        [inclined_dipole_config, quadrupole_config]
        ]
    )
  
    return field_config


if __name__ == "__main__":
    # Configuration
    PLOT_RESULTS = False  # Set to True to generate x-z plots

    field_factory = create_field_config()

    def make_grid_factory(nlon: int) -> GridFactoryConfig:
        grid = SphericalGridFactoryConfig(nlon=nlon, nlat=50, pole_lat=60.0)
        return GridFactoryConfig(grids=[grid])

    def run_and_report(field_backend: str, solver_backend: str, *, max_integration_radius: float, nlon: int):
        """Run experiment with specified field and solver backends and report results."""
        tracing_config = TracingFactoryConfig(
            max_integration_radius=max_integration_radius,
            save_results_to_disk=False,
            max_memory_mb=100,
            skip_memory_estimation=True,
            max_step=1.0,
            field_backend=field_backend,
            solver_backend=solver_backend,
            verify_by_reversing=True,
        )

        backend_label = f"{solver_backend}+{field_backend}"
        series_config = ExperimentSeriesConfig(
            name=f"test_{backend_label}_R{int(max_integration_radius)}_nlon{nlon}",
            field=field_factory,
            grid=make_grid_factory(nlon),
            tracing=tracing_config,
            template=FieldLine.template_line_coordinates_only(),
            description=f"Test {backend_label}",
        )

        # Verify field configurations
        experiments = series_config.create_experiment_series()
        field_configs = set()
        for exp in experiments:
            field_desc = f"{len(exp.field.components)}_comp"
            for comp in exp.field.components:
                field_desc += f"_{comp.moment_type}"
            field_configs.add(field_desc)
        
        if field_backend == "python" and solver_backend == "scipy":  # Only print once
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
        return field_backend, solver_backend, elapsed, max_rev_err, series_result

    results_table = []
    series_results = {}  # Store series results for comparison
    # Benchmark requested cases
    cases = [
        {"max_integration_radius": 500, "nlon": 10},
        #{"max_integration_radius": 500, "nlon": 10},
    ]

    # All combinations to test
    field_backends = ["python", "numba"]
    solver_backends = ["scipy", "cyrk"]

    for case in cases:
        R = case["max_integration_radius"]
        nlon = case["nlon"]
        case_key = f"Rmax={R} nlon={nlon}"
        series_results[case_key] = {}
        
        for solver_backend in solver_backends:
            for field_backend in field_backends:
                f_backend, s_backend, elapsed, max_err, series_result = run_and_report(
                    field_backend, solver_backend, max_integration_radius=R, nlon=nlon
                )
                backend_label = f"{solver_backend}+{field_backend}"
                results_table.append({
                    "case": case_key,
                    "solver_backend": solver_backend,
                    "field_backend": field_backend,
                    "backend": backend_label,
                    "time": elapsed,
                    "max_reversed_error": max_err
                })
                series_results[case_key][backend_label] = series_result

    def compare_r_line_trajectories(result1, result2, label1="result1", label2="result2"):
        """Compare r_line trajectories between two results."""
        max_distance = 0.0
        total_comparisons = 0
        
        # Get all experiment results (filter successful ones)
        exps1 = [exp for exp in result1.experiment_results 
                 if getattr(exp, 'success', False) and exp.results is not None]
        exps2 = [exp for exp in result2.experiment_results 
                 if getattr(exp, 'success', False) and exp.results is not None]
        
        # Check same number of successful experiments
        if len(exps1) != len(exps2):
            error_msg = (
                f"Mismatched number of successful experiments: "
                f"{label1}={len(exps1)}, {label2}={len(exps2)}"
            )
            print(f"ERROR: {error_msg}")
            exit(1)
        
        # Match experiments by index (they should be in the same order since they use
        # the same field/grid/tracing configs, only backends differ)
        for i in range(len(exps1)):
            exp1 = exps1[i]
            exp2 = exps2[i]
            
            # Get r_line from both - it's always a list of arrays (one per field line)
            r_line1 = exp1.results.get_property('r_line')
            r_line2 = exp2.results.get_property('r_line')
            
            if r_line1 is None:
                error_msg = (
                    f"Experiment {i} ({exp1.experiment_name}): "
                    f"{label1} r_line is None"
                )
                print(f"ERROR: {error_msg}")
                exit(1)
            
            if r_line2 is None:
                error_msg = (
                    f"Experiment {i} ({exp2.experiment_name}): "
                    f"{label2} r_line is None"
                )
                print(f"ERROR: {error_msg}")
                exit(1)
            
            # r_line is always a list (one array per field line)
            if not isinstance(r_line1, list):
                error_msg = (
                    f"Experiment {i} ({exp1.experiment_name}): "
                    f"{label1} r_line is not a list, got {type(r_line1)}"
                )
                print(f"ERROR: {error_msg}")
                exit(1)
            
            if not isinstance(r_line2, list):
                error_msg = (
                    f"Experiment {i} ({exp2.experiment_name}): "
                    f"{label2} r_line is not a list, got {type(r_line2)}"
                )
                print(f"ERROR: {error_msg}")
                exit(1)
            
            # Check same number of field lines
            if len(r_line1) != len(r_line2):
                error_msg = (
                    f"Experiment {i} ({exp1.experiment_name}): "
                    f"Mismatched number of field lines: "
                    f"{label1}={len(r_line1)}, {label2}={len(r_line2)}"
                )
                print(f"ERROR: {error_msg}")
                exit(1)
            
            # Compare each field line
            for line_idx, (line1, line2) in enumerate(zip(r_line1, r_line2)):
                # Handle empty arrays (field lines that didn't reach threshold)
                if line1 is None:
                    error_msg = (
                        f"Experiment {i}, field line {line_idx}: "
                        f"{label1} r_line[{line_idx}] is None"
                    )
                    print(f"ERROR: {error_msg}")
                    exit(1)
                
                if line2 is None:
                    error_msg = (
                        f"Experiment {i}, field line {line_idx}: "
                        f"{label2} r_line[{line_idx}] is None"
                    )
                    print(f"ERROR: {error_msg}")
                    exit(1)
                
                if hasattr(line1, '__len__') and len(line1) == 0:
                    # Empty array is acceptable (field line didn't reach threshold)
                    continue
                
                if hasattr(line2, '__len__') and len(line2) == 0:
                    # Empty array is acceptable (field line didn't reach threshold)
                    continue
                
                # Convert to numpy arrays
                arr1 = np.asarray(line1, dtype=np.float64)
                arr2 = np.asarray(line2, dtype=np.float64)
                
                # Ensure correct shape (N, 3)
                if arr1.ndim == 1 and len(arr1) == 3:
                    arr1 = arr1.reshape(1, 3)
                elif arr1.ndim == 2 and arr1.shape[1] != 3:
                    arr1 = arr1.T
                if arr2.ndim == 1 and len(arr2) == 3:
                    arr2 = arr2.reshape(1, 3)
                elif arr2.ndim == 2 and arr2.shape[1] != 3:
                    arr2 = arr2.T
                
                # Check same number of points
                if arr1.shape[0] != arr2.shape[0]:
                    # Mismatched lengths - skip this line but continue
                    continue
                
                # Calculate distances at each point
                distances = np.linalg.norm(arr1 - arr2, axis=1)
          
                if len(distances) > 0:
                    max_distance = max(max_distance, np.max(distances))
                    total_comparisons += len(distances)
        
        return max_distance, total_comparisons, 0

    # Calculate speedups and r_line comparisons
    # Compare all combinations against scipy+python (baseline)
    baseline_label = "scipy+python"
    comparison_table = []
    
    for case in cases:
        R = case["max_integration_radius"]
        nlon = case["nlon"]
        case_key = f"Rmax={R} nlon={nlon}"
        
        baseline_row = next(r for r in results_table 
                            if r["case"] == case_key and r["backend"] == baseline_label)
        baseline_result = series_results[case_key][baseline_label]
        
        for backend_label in series_results[case_key].keys():
            if backend_label == baseline_label:
                continue  # Skip baseline itself
            
            current_row = next(r for r in results_table 
                              if r["case"] == case_key and r["backend"] == backend_label)
            current_result = series_results[case_key][backend_label]
            
            speedup = baseline_row["time"] / current_row["time"] if current_row["time"] > 0 else None
            
            # Compare r_line trajectories against baseline
            max_r_line_distance, total_points, mismatched = compare_r_line_trajectories(
                baseline_result, current_result, label1=baseline_label, label2=backend_label
            )
            
            comparison_table.append({
                "case": case_key,
                "baseline": baseline_label,
                "backend": backend_label,
                "baseline_time": baseline_row["time"],
                "backend_time": current_row["time"],
                "speedup": speedup,
                "max_reversed_error": current_row["max_reversed_error"],
                "max_r_line_distance": max_r_line_distance,
                "total_points_compared": total_points,
                "mismatched_lengths": mismatched
            })

    # Pretty print summary table
    print("\n" + "="*80)
    print("Summary (all backend combinations):")
    print("="*80)
    print(f"{'Case':<18} {'Backend':<20} {'Time[s]':>10} {'MaxReversedError':>18}")
    print("-"*80)
    for row in results_table:
        print(f"{row['case']:<18} {row['backend']:<20} {row['time']:10.3f} {str(row['max_reversed_error']):>18}")
    
    print("\n" + "="*100)
    print(f"Speedup Analysis (relative to {baseline_label} baseline):")
    print("="*100)
    print(f"{'Case':<18} {'Backend':<20} {'Baseline[s]':>12} {'Backend[s]':>12} {'Speedup':>10} {'MaxReversedError':>18} {'MaxRLineDist':>12}")
    print("-"*100)
    for row in comparison_table:
        speedup_str = f"{row['speedup']:.2f}x" if row['speedup'] is not None else "N/A"
        max_dist_str = f"{row['max_r_line_distance']:.2e}" if row['max_r_line_distance'] >= 0 else "N/A"
        print(f"{row['case']:<18} {row['backend']:<20} {row['baseline_time']:12.3f} {row['backend_time']:12.3f} {speedup_str:>10} {str(row['max_reversed_error']):>18} {max_dist_str:>12}")
    
    print("\n" + "="*90)
    print("r_line Trajectory Comparison Details (vs baseline):")
    print("="*90)
    print(f"{'Case':<18} {'Backend':<20} {'MaxDistance':>12} {'TotalPoints':>12} {'Mismatched':>12}")
    print("-"*90)
    for row in comparison_table:
        max_dist_str = f"{row['max_r_line_distance']:.2e}" if row['max_r_line_distance'] >= 0 else "N/A"
        print(f"{row['case']:<18} {row['backend']:<20} {max_dist_str:>12} {row['total_points_compared']:>12} {row['mismatched_lengths']:>12}")
    
    # Check y-component is zero and plot x vs z for each field configuration
    # Process each case separately
    for case in cases:
        R = case["max_integration_radius"]
        nlon = case["nlon"]
        case_key = f"Rmax={R} nlon={nlon}"
        
        print("\n" + "="*90)
        print(f"Checking y-component (r_line[:, 1]) is zero and plotting x vs z for {case_key}:")
        print("="*90)
        
        # Get field configurations by recreating the series config to access experiment configs
        baseline_series_config = ExperimentSeriesConfig(
            name=f"test_{baseline_label}_R{int(R)}_nlon{nlon}",
            field=field_factory,
            grid=make_grid_factory(nlon),
            tracing=TracingFactoryConfig(
                max_integration_radius=R,
                save_results_to_disk=False,
                max_memory_mb=100,
                skip_memory_estimation=True,
                max_step=1.0,
                field_backend="python",
                solver_backend="scipy",
                verify_by_reversing=True,
            ),
            template=FieldLine.template_line_coordinates_only(),
            description=f"Test {baseline_label}",
        )
        baseline_experiments = baseline_series_config.create_experiment_series()
        
        # Get field configuration identifiers
        field_configs = {}
        for exp_idx, exp_config in enumerate(baseline_experiments):
            field_desc = f"{len(exp_config.field.components)}_comp"
            for comp in exp_config.field.components:
                field_desc += f"_{comp.moment_type}"
            if field_desc not in field_configs:
                field_configs[field_desc] = []
            field_configs[field_desc].append(exp_idx)
        
        # Backend combinations with colors
        backend_colors = {
            "scipy+python": "blue",
            "scipy+numba": "green",
            "cyrk+python": "orange",
            "cyrk+numba": "red"
        }
        
        # Field configuration line thicknesses (smallest thickness = largest zorder)
        field_thicknesses = {}
        sorted_field_descs = sorted(field_configs.keys())
        for idx, field_desc in enumerate(sorted_field_descs):
            # Thickness increases with index, zorder decreases (so smallest thickness on top)
            thickness = 0.5 + idx * 2  # 0.5, 1.0, 1.5, ...
            zorder = len(sorted_field_descs) - idx  # Largest zorder for smallest thickness
            field_thicknesses[field_desc] = (thickness, zorder)
        
        # Check y-component and collect data for plotting from all backends
        y_component_errors = []
        plot_data = {}  # field_desc -> backend_label -> list of (x, z) arrays
        
        for backend_label in series_results[case_key].keys():
            result = series_results[case_key][backend_label]
            
            for exp_idx, exp_result in enumerate(result.experiment_results):
                if not getattr(exp_result, 'success', False) or exp_result.results is None:
                    continue
                
                if exp_idx >= len(baseline_experiments):
                    continue
                
                exp_config = baseline_experiments[exp_idx]
                field_desc = f"{len(exp_config.field.components)}_comp"
                for comp in exp_config.field.components:
                    field_desc += f"_{comp.moment_type}"
                
                if field_desc not in plot_data:
                    plot_data[field_desc] = {}
                if backend_label not in plot_data[field_desc]:
                    plot_data[field_desc][backend_label] = []
                
                r_line = exp_result.results.get_property('r_line')
                if r_line is None:
                    continue
                
                for line_idx, line in enumerate(r_line):
                    if line is None or len(line) == 0:
                        continue
                    
                    line_arr = np.asarray(line, dtype=np.float64)
                    # Ensure shape is (N, 3)
                    if line_arr.ndim == 1 and len(line_arr) == 3:
                        line_arr = line_arr.reshape(1, 3)
                    elif line_arr.ndim == 2 and line_arr.shape[1] != 3:
                        line_arr = line_arr.T
                    
                    # Check y-component (index 1) is zero (only check once per line)
                    if backend_label == baseline_label:  # Only check once
                        y_vals = line_arr[:, 1]
                        max_y_abs = np.max(np.abs(y_vals))
                        if max_y_abs > 1e-10:  # Allow small numerical errors
                            y_component_errors.append({
                                'field_desc': field_desc,
                                'exp_name': exp_result.experiment_name,
                                'line_idx': line_idx,
                                'max_y_abs': max_y_abs
                            })
                    
                    # Collect x and z for plotting
                    x_vals = line_arr[:, 0]
                    z_vals = line_arr[:, 2]
                    plot_data[field_desc][backend_label].append((x_vals, z_vals))
        
        # Report y-component check results
        if y_component_errors:
            print(f"WARNING: Found {len(y_component_errors)} field lines with non-zero y-component:")
            for err in y_component_errors[:10]:  # Show first 10
                print(f"  {err['field_desc']}, exp={err['exp_name']}, line={err['line_idx']}, max|y|={err['max_y_abs']:.2e}")
            if len(y_component_errors) > 10:
                print(f"  ... and {len(y_component_errors) - 10} more")
        else:
            print("✓ All field lines have y-component ≈ 0 (within 1e-10 tolerance)")
        
        # Create plots for each field configuration (if enabled)
        if PLOT_RESULTS:
            n_configs = len(field_configs)
            if n_configs > 0:
                fig, axes = plt.subplots(1, n_configs, figsize=(6*n_configs, 6))
                if n_configs == 1:
                    axes = [axes]
                
                for ax_idx, (field_desc, _) in enumerate(sorted(plot_data.items())):
                    ax = axes[ax_idx]
                    thickness, zorder_base = field_thicknesses[field_desc]
                    
                    # Plot all backend combinations for this field configuration
                    for backend_label in sorted(backend_colors.keys()):
                        if backend_label not in plot_data[field_desc]:
                            continue
                        
                        color = backend_colors[backend_label]
                        # Use zorder_base - small offset for backend distinction
                        backend_zorder = zorder_base - 0.1 * list(backend_colors.keys()).index(backend_label)
                        
                        # Plot all field lines for this backend
                        for line_idx, (x_vals, z_vals) in enumerate(plot_data[field_desc][backend_label]):
                            # Only add label for the first line of each backend
                            label = backend_label if line_idx == 0 else ""
                            ax.plot(x_vals, z_vals, color=color, alpha=0.6, marker='.', markersize=1, linestyle='-',
                                   linewidth=thickness, zorder=backend_zorder, 
                                   label=label)
                    
                    ax.set_xlabel('x (r_line[:, 0])')
                    ax.set_ylabel('z (r_line[:, 2])')
                    ax.set_title(f'Field Config: {field_desc}')
                    ax.grid(True, alpha=0.3)
                    ax.set_aspect('auto')
                    ax.legend(loc='best')
                
                plt.tight_layout()
                plot_filename = f'field_lines_xz_plot_{case_key.replace(" ", "_").replace("=", "")}.png'
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                print(f"\n✓ Saved plot to {plot_filename}")
                plt.show()
                plt.close()

