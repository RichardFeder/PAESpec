#!/usr/bin/env python
"""
Submit multiple PAE redshift jobs for processing multiple parquet files.

This script generates and submits SLURM jobs (or job arrays) for each input file.
Use cases:
1. Process multiple separate parquet files (one job per file)
2. Process a single large file split across multiple nodes (job array)

Usage:
    # Multiple files, one job each:
    python submit_multifile_jobs.py --files file1.parquet file2.parquet file3.parquet
    
    # Single file, split across N nodes:
    python submit_multifile_jobs.py --files large.parquet --split-into 10
    
    # Multiple files, each split across nodes:
    python submit_multifile_jobs.py --files file1.parquet file2.parquet --split-into 5
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd


def count_parquet_sources(parquet_file, filter_specz=False):
    """Count sources in a parquet file (optionally after filtering)."""
    try:
        # Read just the necessary columns
        cols = ['SPHERExRefID']
        if filter_specz:
            cols.append('z_specz')
        
        df = pd.read_parquet(parquet_file, columns=cols)
        
        if filter_specz:
            # Filter out sources without spec-z
            df = df[df['z_specz'].notna() & (df['z_specz'] > 0)]
        
        return len(df)
    except Exception as e:
        print(f"Warning: Could not count sources in {parquet_file}: {e}")
        return None


def generate_job_script(
    parquet_file,
    sources_per_task,
    total_sources,
    output_dir,
    job_name,
    array_spec=None,
    **kwargs
):
    """Generate a SLURM job script for one file (optionally as job array)."""
    
    # Create job script content
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={kwargs.get('account', 'm4031_g')}",
        f"#SBATCH --qos={kwargs.get('qos', 'regular')}",
        "#SBATCH --constraint=gpu",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=128",
        "#SBATCH --gpus-per-node=4",
        f"#SBATCH --time={kwargs.get('time', '06:00:00')}",
    ]
    
    # Add array spec if splitting across nodes
    if array_spec:
        script_lines.append(f"#SBATCH --array={array_spec}")
    
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if array_spec:
        script_lines.extend([
            f"#SBATCH --output={log_dir}/{job_name}_%A_%a.out",
            f"#SBATCH --error={log_dir}/{job_name}_%A_%a.err",
        ])
    else:
        script_lines.extend([
            f"#SBATCH --output={log_dir}/{job_name}_%j.out",
            f"#SBATCH --error={log_dir}/{job_name}_%j.err",
        ])
    
    # Setup and execution
    script_lines.extend([
        "",
        "# Setup",
        "module load conda",
        f"conda activate {kwargs.get('conda_env', 'jax-env')}",
        "umask 077",
        "",
        f"cd {Path(__file__).parent.parent.absolute()}",
        "",
        "# Run",
    ])
    
    # Build Python command
    cmd_parts = [
        "python scripts/redshift_job_batched.py",
        f"--parquet-file {parquet_file}",
        f"--filter-set {kwargs.get('filter_set', 'SPHEREx_filter_306')}",
        f"--datestr {kwargs.get('datestr', '$(date +%m%d%y)')}",
        f"--batch-size {kwargs.get('batch_size', 800)}",
        f"--sampling-batch-size {kwargs.get('sampling_batch_size', 200)}",
    ]
    
    # Add job array parameters if splitting
    if array_spec:
        cmd_parts.extend([
            "--job-array-task-id ${SLURM_ARRAY_TASK_ID}",
            f"--sources-per-task {sources_per_task}",
        ])
    
    # Add filtering options
    if kwargs.get('filter_specz'):
        cmd_parts.append("--filter-specz")
    if kwargs.get('z_min') is not None:
        cmd_parts.append(f"--z-min {kwargs['z_min']}")
    if kwargs.get('z_max') is not None:
        cmd_parts.append(f"--z-max {kwargs['z_max']}")
    
    # Add prior options
    cmd_parts.extend([
        f"--prior-type {kwargs.get('prior_type', 1)}",
        f"--z0-prior {kwargs.get('z0_prior', 0.65)}",
        f"--sigma-prior {kwargs.get('sigma_prior', 0.6)}",
    ])
    
    # Add multicore options
    if kwargs.get('use_multicore', True):
        cmd_parts.append("--use-multicore")
    cmd_parts.append(f"--n-devices {kwargs.get('n_devices', 4)}")
    
    script_lines.append(" \\\n    ".join(cmd_parts))
    
    return "\n".join(script_lines)


def main():
    parser = argparse.ArgumentParser(
        description='Submit multiple PAE redshift jobs for multiple files or multi-node processing'
    )
    
    # Input files
    parser.add_argument('--files', nargs='+', required=True,
                       help='Parquet files to process')
    parser.add_argument('--split-into', type=int, default=None,
                       help='Split each file across N nodes (creates job array). '
                            'If not specified, processes each file on one node.')
    parser.add_argument('--sources-per-task', type=int, default=10000,
                       help='Sources per job array task (when using --split-into)')
    
    # Output
    parser.add_argument('--output-dir', type=Path, 
                       default=Path('/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched'),
                       help='Base output directory')
    parser.add_argument('--script-dir', type=Path, default=Path('job_scripts'),
                       help='Directory to save generated job scripts')
    
    # Job options
    parser.add_argument('--dry-run', action='store_true',
                       help='Generate scripts but do not submit')
    parser.add_argument('--account', default='m4031_g')
    parser.add_argument('--qos', default='regular')
    parser.add_argument('--time', default='06:00:00')
    parser.add_argument('--max-concurrent', type=int, default=5,
                       help='Max concurrent array tasks (for --split-into)')
    
    # Processing options
    parser.add_argument('--filter-specz', action='store_true')
    parser.add_argument('--z-min', type=float, default=None)
    parser.add_argument('--z-max', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=800)
    parser.add_argument('--sampling-batch-size', type=int, default=200)
    
    args = parser.parse_args()
    
    # Create script directory
    args.script_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PAE MULTI-FILE/MULTI-NODE JOB SUBMISSION")
    print("=" * 70)
    print(f"Files to process: {len(args.files)}")
    if args.split_into:
        print(f"Each file split across: {args.split_into} nodes")
        print(f"Sources per task: {args.sources_per_task}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)
    
    submitted_jobs = []
    
    for file_path in args.files:
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"\n❌ File not found: {file_path}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Processing: {file_path.name}")
        print(f"{'='*70}")
        
        # Count sources
        print("Counting sources...")
        total_sources = count_parquet_sources(str(file_path), args.filter_specz)
        if total_sources is None:
            print("⚠️  Could not count sources, skipping")
            continue
        
        print(f"Total sources: {total_sources:,}")
        
        # Determine job array specification
        if args.split_into:
            # Manual specification: split into N tasks
            n_tasks = args.split_into
            sources_per_task = (total_sources + n_tasks - 1) // n_tasks
            array_spec = f"0-{n_tasks-1}%{args.max_concurrent}"
            print(f"Splitting into {n_tasks} tasks ({sources_per_task} sources/task)")
        else:
            # Single job, no array
            array_spec = None
            sources_per_task = total_sources
            n_tasks = 1
            print("Processing on single node (no splitting)")
        
        # Generate job name
        job_name = f"pae_{file_path.stem}"
        if len(job_name) > 20:
            job_name = job_name[:20]
        
        # Generate script
        script_content = generate_job_script(
            parquet_file=str(file_path),
            sources_per_task=sources_per_task,
            total_sources=total_sources,
            output_dir=args.output_dir,
            job_name=job_name,
            array_spec=array_spec,
            account=args.account,
            qos=args.qos,
            time=args.time,
            filter_specz=args.filter_specz,
            z_min=args.z_min,
            z_max=args.z_max,
            batch_size=args.batch_size,
            sampling_batch_size=args.sampling_batch_size,
        )
        
        # Save script
        script_path = args.script_dir / f"{job_name}.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o700)
        print(f"Generated: {script_path}")
        
        # Submit
        if not args.dry_run:
            print("Submitting job...")
            result = subprocess.run(
                ['sbatch', str(script_path)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"✓ Submitted: Job ID {job_id}")
                submitted_jobs.append((file_path.name, job_id, n_tasks))
            else:
                print(f"❌ Submission failed: {result.stderr}")
        else:
            print("(Dry run - not submitted)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUBMISSION SUMMARY")
    print("=" * 70)
    if args.dry_run:
        print("Dry run mode - scripts generated but not submitted")
        print(f"Scripts saved to: {args.script_dir.absolute()}")
    else:
        print(f"Submitted {len(submitted_jobs)} job(s):")
        for filename, job_id, n_tasks in submitted_jobs:
            tasks_str = f" ({n_tasks} tasks)" if n_tasks > 1 else ""
            print(f"  {filename:40s} → Job {job_id}{tasks_str}")
    print("=" * 70)


if __name__ == '__main__':
    main()
