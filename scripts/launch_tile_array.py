#!/usr/bin/env python3
"""
Helper script to launch SLURM job array for tile processing.

This script:
1. Reads the parquet file to determine number of sources
2. Computes required number of array tasks
3. Submits the job array with correct parameters

Usage:
    python scripts/launch_tile_array.py \
        --parquet-file /path/to/tile.parquet \
        --sources-per-task 1000 \
        --datestr tile_01_010125 \
        --dry-run
"""

import argparse
import subprocess
from pathlib import Path
import pandas as pd


def count_sources(parquet_file: str) -> int:
    """Count number of sources in parquet file."""
    print(f"Reading parquet file: {parquet_file}")
    df = pd.read_parquet(parquet_file)
    n_sources = len(df)
    print(f"Found {n_sources} sources")
    return n_sources


def compute_array_tasks(n_sources: int, sources_per_task: int) -> int:
    """Compute number of array tasks needed."""
    n_tasks = (n_sources + sources_per_task - 1) // sources_per_task  # Ceiling division
    return n_tasks


def submit_job_array(
    parquet_file: str,
    n_sources: int,
    sources_per_task: int,
    datestr: str,
    script_path: str,
    time_limit: str = "02:00:00",
    dry_run: bool = False
) -> None:
    """Submit SLURM job array."""
    
    n_tasks = compute_array_tasks(n_sources, sources_per_task)
    array_spec = f"0-{n_tasks - 1}"
    
    print("\n" + "=" * 60)
    print("JOB ARRAY CONFIGURATION")
    print("=" * 60)
    print(f"Parquet file: {parquet_file}")
    print(f"Total sources: {n_sources}")
    print(f"Sources per task: {sources_per_task}")
    print(f"Number of tasks: {n_tasks}")
    print(f"Array spec: {array_spec}")
    print(f"Time limit: {time_limit}")
    print(f"Datestr: {datestr}")
    print("=" * 60)
    
    # Build sbatch command
    cmd = [
        "sbatch",
        f"--array={array_spec}",
        f"--time={time_limit}",
        script_path,
        parquet_file,
        str(n_sources),
        str(sources_per_task),
        datestr
    ]
    
    print("\nCommand:")
    print(" ".join(cmd))
    
    if dry_run:
        print("\n[DRY RUN] Would submit job but --dry-run flag set")
        return
    
    # Submit job
    print("\nSubmitting job array...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Job submitted successfully")
        print(result.stdout)
    else:
        print("✗ Job submission failed")
        print(result.stderr)
        raise RuntimeError("Job submission failed")


def main():
    parser = argparse.ArgumentParser(
        description="Launch SLURM job array for tile processing",
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
        help="Number of sources to process per array task (default: 1000)"
    )
    
    parser.add_argument(
        "--datestr",
        type=str,
        default=None,
        help="Date string for output directory (default: auto-generated)"
    )
    
    parser.add_argument(
        "--script-path",
        type=str,
        default=None,
        help="Path to SLURM script (default: scripts/run_tile_job_array.sh)"
    )
    
    parser.add_argument(
        "--time-limit",
        type=str,
        default="02:00:00",
        help="Time limit per task (default: 02:00:00)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration but don't submit job"
    )
    
    args = parser.parse_args()
    
    # Validate parquet file exists
    parquet_path = Path(args.parquet_file)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {args.parquet_file}")
    
    # Auto-generate datestr if not provided
    if args.datestr is None:
        from datetime import datetime
        tile_name = parquet_path.stem
        datestr = f"{tile_name}_{datetime.now().strftime('%m%d%y')}"
    else:
        datestr = args.datestr
    
    # Default script path
    if args.script_path is None:
        script_dir = Path(__file__).parent
        script_path = str(script_dir / "run_tile_job_array.sh")
    else:
        script_path = args.script_path
    
    # Validate script exists
    if not Path(script_path).exists():
        raise FileNotFoundError(f"SLURM script not found: {script_path}")
    
    # Count sources
    n_sources = count_sources(args.parquet_file)
    
    # Submit job
    submit_job_array(
        parquet_file=args.parquet_file,
        n_sources=n_sources,
        sources_per_task=args.sources_per_task,
        datestr=datestr,
        script_path=script_path,
        time_limit=args.time_limit,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
