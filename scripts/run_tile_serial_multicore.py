#!/usr/bin/env python3
"""
Run tile processing serially with multicore mode for memory isolation.

This script:
1. Divides tile into chunks (sources_per_task)
2. Runs each chunk as a separate subprocess in MULTICORE mode (uses all 4 GPUs)
3. Waits for each process to complete before starting next
4. Memory is released between tasks (process isolation)

Usage:
    python scripts/run_tile_serial_multicore.py \
        --parquet-file /path/to/tile.parquet \
        --sources-per-task 1000 \
        --batch-size 200 \
        --datestr tile_01_test
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime


def count_sources(parquet_file: str) -> int:
    """Count number of sources in parquet file using metadata (no data loading)."""
    import pyarrow.parquet as pq
    
    print(f"Reading parquet metadata: {parquet_file}", flush=True)
    sys.stdout.flush()
    
    try:
        # Use PyArrow to read only metadata (very fast, no data loaded)
        # This is critical for large files (e.g., 40GB+) - avoids OOM
        pf = pq.ParquetFile(parquet_file)
        n_sources = pf.metadata.num_rows
        print(f"Found {n_sources:,} sources (from metadata, no data loaded)", flush=True)
    except Exception as e:
        # Fallback: count unique SPHERExRefID (slower but works if metadata fails)
        print(f"PyArrow metadata read failed ({e}), using column read fallback", flush=True)
        df = pd.read_parquet(parquet_file, columns=['SPHERExRefID'])
        n_sources = len(df)
        print(f"Found {n_sources:,} sources (from column read)", flush=True)
        del df
        import gc
        gc.collect()
    
    sys.stdout.flush()
    return n_sources


def run_task(
    task_id: int,
    parquet_file: str,
    sources_per_task: int,
    batch_size: int,
    sampling_batch_size: int,
    datestr: str,
    use_robust_reinit: bool = True,
    init_reinit: bool = True,
    filter_set: str = "SPHEREx_filter_306",
    nf_alpha: float = 1.0,
    use_weighted_mean: bool = False,
    no_abs_norm: bool = False,
    random_seed: int = None,
    mask_wavelength_ranges: list = None,
) -> bool:
    """Run a single task in multicore mode."""
    
    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        "scripts/redshift_job_batched.py",
        "--parquet-file", parquet_file,
        "--datestr", datestr,
        "--job-array-task-id", str(task_id),
        "--sources-per-task", str(sources_per_task),
        "--batch-size", str(batch_size),
        "--sampling-batch-size", str(sampling_batch_size),
        "--use-multicore",  # Use all 4 GPUs
        "--n-devices", "4",
        "--filter-set", filter_set,
        "--nf-alpha", str(nf_alpha),
    ]
    
    if use_robust_reinit:
        cmd.append("--use_robust_reinit")
    
    if init_reinit:
        cmd.append("--init-reinit")
    
    if use_weighted_mean:
        cmd.append("--use-weighted-mean")
    
    if no_abs_norm:
        cmd.append("--no-abs-norm")
    
    if random_seed is not None:
        cmd.extend(["--random-seed", str(random_seed)])

    if mask_wavelength_ranges:
        # Pass each "lmin,lmax" pair as a separate token after the flag
        cmd.append("--mask-wavelength-ranges")
        cmd.extend(mask_wavelength_ranges)

    print(f"\n{'='*60}", flush=True)
    print(f"TASK {task_id}: Starting", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)
    print(flush=True)
    sys.stdout.flush()  # Force flush before subprocess
    
    # Run subprocess and stream output
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=False  # Stream to console
        )
        sys.stdout.flush()  # Force flush after subprocess
        print(f"\n{'='*60}", flush=True)
        print(f"TASK {task_id}: ✓ Completed successfully", flush=True)
        print(f"{'='*60}\n", flush=True)
        sys.stdout.flush()
        return True
        
    except subprocess.CalledProcessError as e:
        sys.stdout.flush()
        print(f"\n{'='*60}", flush=True)
        print(f"TASK {task_id}: ✗ Failed with exit code {e.returncode}", flush=True)
        print(f"{'='*60}\n", flush=True)
        sys.stdout.flush()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run tile processing serially with multicore mode",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--parquet-file",
        type=str,
        required=True,
        help="Path to parquet file with source data"
    )
    
    parser.add_argument(
        "--sources-per-task",
        type=int,
        default=1000,
        help="Number of sources to process per task (default: 1000)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Batch size for processing (default: 200)"
    )
    
    parser.add_argument(
        "--sampling-batch-size",
        type=int,
        default=200,
        help="Sampling batch size (default: 200)"
    )
    
    parser.add_argument(
        "--datestr",
        type=str,
        default=None,
        help="Date string for output directory (default: auto-generated)"
    )
    
    parser.add_argument(
        "--filter-set",
        type=str,
        default="SPHEREx_filter_306",
        help="Filter set to use (default: SPHEREx_filter_306)"
    )
    
    parser.add_argument(
        "--nf-alpha",
        type=float,
        default=1.0,
        help="Normalizing flow prior multiplier (0=no flow prior, 1=full flow prior, default: 1.0)"
    )
    
    parser.add_argument(
        "--use-weighted-mean",
        action="store_true",
        help="Use inverse-variance weighted mean for normalization (more robust for noisy data)"
    )
    
    parser.add_argument(
        "--no-abs-norm",
        action="store_true",
        help="Disable absolute normalization (raw fluxes/errors, amplitude marginalization only)"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If None, uses system time for different initializations."
    )
    
    parser.add_argument(
        "--no-robust-reinit",
        action="store_true",
        help="Disable robust reinitialization"
    )
    
    parser.add_argument(
        "--no-init-reinit",
        action="store_true",
        help="Disable initial reinitialization"
    )
    
    parser.add_argument(
        "--start-task",
        type=int,
        default=0,
        help="Start from this task ID (default: 0)"
    )
    
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to run (default: all)"
    )

    parser.add_argument(
        "--mask-wavelength-ranges",
        type=str,
        nargs="*",
        default=None,
        metavar="LMIN,LMAX",
        help="Wavelength ranges to exclude from the likelihood (micron). "
             "Provide one or more 'lmin,lmax' pairs, e.g. 2.42,3.82. "
             "Channels whose central wavelength falls inside any range are "
             "zero-weighted. Band 4 shorthand: 2.42,3.82."
    )

    args = parser.parse_args()
    
    # Validate parquet file exists
    parquet_path = Path(args.parquet_file)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {args.parquet_file}")
    
    # Auto-generate datestr if not provided
    if args.datestr is None:
        tile_name = parquet_path.stem
        datestr = f"{tile_name}_{datetime.now().strftime('%m%d%y_%H%M%S')}"
    else:
        datestr = args.datestr
    
    # Count sources and compute tasks
    n_sources = count_sources(args.parquet_file)
    n_tasks = (n_sources + args.sources_per_task - 1) // args.sources_per_task
    
    # Apply start_task and max_tasks
    start_task = args.start_task
    end_task = min(n_tasks, start_task + args.max_tasks) if args.max_tasks else n_tasks
    
    print(f"\n{'='*60}")
    print("SERIAL MULTICORE PROCESSING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Parquet file: {args.parquet_file}")
    print(f"Total sources: {n_sources}")
    print(f"Sources per task: {args.sources_per_task}")
    print(f"Total tasks: {n_tasks}")
    print(f"Running tasks: {start_task} to {end_task-1}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sampling batch size: {args.sampling_batch_size}")
    print(f"Datestr: {datestr}")
    print(f"Robust reinit: {not args.no_robust_reinit}")
    print(f"Init reinit: {not args.no_init_reinit}")
    print(f"Filter set: {args.filter_set}")
    print(f"NF alpha: {args.nf_alpha}")
    print(f"Use weighted mean: {args.use_weighted_mean}")
    if args.mask_wavelength_ranges:
        print(f"Masking wavelength ranges: {args.mask_wavelength_ranges}")

    # Initialize random seed
    if args.random_seed is None:
        import time
        seed = int(time.time() * 1000000) % (2**31)
        print(f"Random seed: {seed} (auto-generated from system time)")
    else:
        seed = args.random_seed
        print(f"Random seed: {seed} (user-specified)")
    
    print(f"Mode: MULTICORE (4 GPUs per task)")
    print(f"{'='*60}\n")
    sys.stdout.flush()
    
    # Run tasks serially
    successful_tasks = 0
    failed_tasks = []
    
    for task_id in range(start_task, end_task):
        print(f"\n[SERIAL WRAPPER] About to launch task {task_id} of {end_task-1}...", flush=True)
        sys.stdout.flush()
        
        success = run_task(
            task_id=task_id,
            parquet_file=args.parquet_file,
            sources_per_task=args.sources_per_task,
            batch_size=args.batch_size,
            sampling_batch_size=args.sampling_batch_size,
            datestr=datestr,
            use_robust_reinit=not args.no_robust_reinit,
            init_reinit=not args.no_init_reinit,
            filter_set=args.filter_set,
            nf_alpha=args.nf_alpha,
            use_weighted_mean=args.use_weighted_mean,
            no_abs_norm=args.no_abs_norm,
            random_seed=seed,
            mask_wavelength_ranges=args.mask_wavelength_ranges,
        )
        
        print(f"[SERIAL WRAPPER] Task {task_id} returned with success={success}", flush=True)
        sys.stdout.flush()
        if success:
            successful_tasks += 1
        else:
            failed_tasks.append(task_id)
    
    # Final summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total tasks: {end_task - start_task}")
    print(f"Successful: {successful_tasks}")
    print(f"Failed: {len(failed_tasks)}")
    
    if failed_tasks:
        print(f"\nFailed task IDs: {failed_tasks}")
        print("\nTo retry failed tasks:")
        for task_id in failed_tasks:
            print(f"  --start-task {task_id} --max-tasks 1")
    
    print(f"\nTo collate results:")
    print(f"  python scripts/collate_tile_results.py --datestr {datestr}")
    print(f"{'='*60}\n")
    
    # Exit with error if any tasks failed
    if failed_tasks:
        sys.exit(1)


if __name__ == "__main__":
    main()
