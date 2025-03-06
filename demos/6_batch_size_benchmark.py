from utils import setup_demo
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup demo environment
fits_dir, plot_dir, log_dir, logger = setup_demo(__file__)

from src.field.multipole import MultipoleField
from src.field.components import DipoleFieldComponent, QuadrupoleFieldComponent
from src.footpoint_grid.grid import HealpixGrid
from src.line.field_line import FieldLine
from src.line.calculator import FieldLineCalculator, TracingConfig

def run_benchmark(batch_size: int, calculator: FieldLineCalculator, template: FieldLine) -> float:
    """Run field line calculation with given batch size and measure time.
    
    Parameters
    ----------
    batch_size : int
        Batch size to test
    calculator : FieldLineCalculator
        Calculator instance to use
    template : FieldLine
        Template specifying which properties to calculate
        
    Returns
    -------
    float
        Total calculation time in seconds
    """
    logger.info(f"Testing batch size: {batch_size}")
    
    config = TracingConfig(
        R_max=10.0,
        R_min=1.0,
        dt=0.01,
        max_memory_mb=100.0,  # Large enough to not be a constraint
        n_cores=4,
        max_batch_size=batch_size
    )
    
    start_time = time.perf_counter()
    calculator.get_field_lines(
        template=template,
        config=config
    )
    elapsed = time.perf_counter() - start_time
    
    logger.info(f"Completed in {elapsed:.2f} seconds")
    return elapsed

def main():
    # Create field configuration
    dipole = DipoleFieldComponent(
        B_magnitude=1.0,
        moment_orientation_latitude_degree=45.0
    )
    field = MultipoleField([dipole])
    
    # Create smaller grid for benchmarking
    grid = HealpixGrid(nside=32)  
    logger.info(f"Testing with {grid.nline} field lines")
    
    # Configure template
    template = FieldLine(
        save_r_max=True,
        save_b_r_ini_angle=True,
        save_r_line=True
    )
    
    # Initialize calculator
    calculator = FieldLineCalculator(field, grid)
    
    # Test different batch sizes
    batch_sizes = [100, 500, 1000, 5000]
    timings = []
    
    for batch_size in batch_sizes:
        try:
            elapsed = run_benchmark(batch_size, calculator, template)
            timings.append(elapsed)
        except Exception as e:
            logger.error(f"Error with batch size {batch_size}: {e}")
            timings.append(None)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'batch_size': batch_sizes,
        'time_seconds': timings
    })
    print(results)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(batch_sizes, timings, 'o-')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Calculation Time (seconds)')
    ax.set_title('Field Line Calculation Performance vs Batch Size')
    ax.grid(True)
    ax.set_xscale('log')
    
    # Add timing labels
    for x, y in zip(results['batch_size'], results['time_seconds']):
        ax.annotate(f'{y:.1f}s', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.savefig(plot_dir / 'batch_size_benchmark.png')
    plt.show()
    logger.info("Saved benchmark plot")
    
    # Print results table
    logger.info("\nBenchmark Results:")
    logger.info(results.to_string(index=False))
    
    # Find optimal batch size
    best_idx = results['time_seconds'].argmin()
    logger.info(f"\nOptimal batch size: {results.iloc[best_idx]['batch_size']}")
    logger.info(f"Best time: {results.iloc[best_idx]['time_seconds']:.2f} seconds")

if __name__ == "__main__":
    try:
        main()
        logger.info("Benchmark completed successfully")
    except Exception as e:
        logger.exception("Benchmark failed")
        raise 