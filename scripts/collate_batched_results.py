#!/usr/bin/env python
"""
Collate results from a batched redshift run into a single combined file.

This script takes per-batch result files (PAE_results_batch*.npz) and combines them
into a single file (PAE_results_combined_*.npz).

Usage:
    python scripts/collate_batched_results.py test_df_122925
    python scripts/collate_batched_results.py test_df_122925 --results-dir /custom/path
    python scripts/collate_batched_results.py test_df_122925 --include-samples
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from pathlib import Path
from config import scratch_basepath


def collate_results(results_dir, run_name, include_samples=False, verbose=True):
    """
    Collate per-batch result files into a single combined file.
    
    Supports two directory structures:
    1. Single directory: batched/run_name/ with PAE_results_batch*.npz files
    2. Nested with tasks: batched/run_name/task0/, task1/, ... each with PAE_results_batch*.npz files
    
    Parameters
    ----------
    results_dir : Path
        Directory containing the batch result files (or parent of task subdirs)
    run_name : str
        Name of the run (e.g., 'test_df_122925')
    include_samples : bool
        If True, also collate sample files (PAE_samples_*.npz)
    verbose : bool
        Print detailed progress information
        
    Returns
    -------
    dict
        Dictionary with keys 'results_file' and optionally 'samples_file'
        containing paths to the created combined files
    """
    
    output_files = {}
    
    # ==================== Find Result Files ====================
    if verbose:
        print("\n" + "="*70)
        print("COLLATING PAE RESULTS")
        print("="*70)
        print('Directory:', results_dir)
        print('Run name:', run_name)
    
    pattern = 'PAE_results_batch*.npz'
    
    # Strategy 1: Look in current directory (single task case)
    files = sorted(list(results_dir.glob(pattern)))
    
    if verbose and len(files) > 0:
        print(f"\nFound {len(files)} batch files in current directory")
    
    # Strategy 2: Look in task*/ subdirectories (multi-task case)
    if len(files) == 0:
        task_pattern = 'task*/PAE_results_batch*.npz'
        files = sorted(list(results_dir.glob(task_pattern)))
        if verbose and len(files) > 0:
            print(f"\nFound {len(files)} batch files in nested task directories")
    
    if len(files) == 0:
        print(f"❌ No batch result files found!")
        print(f"   Searched:")
        print(f"     - {results_dir / pattern}")
        print(f"     - {results_dir / 'task*' / pattern}")
        return output_files
    
    if verbose:
        print(f"\nFound {len(files)} batch result files:")
        for f in files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.2f} MB)")
    
    # Load all batch files
    if verbose:
        print("\nLoading batch files...")
    
    all_dicts = []
    for i, f in enumerate(files):
        try:
            data = dict(np.load(str(f), allow_pickle=True))
            all_dicts.append(data)
            if verbose and (i+1) % 10 == 0:
                print(f"  Loaded {i+1}/{len(files)} files...")
        except Exception as e:
            print(f"⚠️  Warning: Failed to load {f.name}: {e}")
            continue
    
    if len(all_dicts) == 0:
        print("❌ No valid batch files could be loaded")
        return output_files
    
    if verbose:
        print(f"✓ Successfully loaded {len(all_dicts)} batch files")
    
    # Get all keys from all dictionaries
    keys = set().union(*[d.keys() for d in all_dicts])
    
    if verbose:
        print(f"\nMerging {len(keys)} arrays:")
        # Show first dict's keys and shapes
        sample_dict = all_dicts[0]
        for k in sorted(keys):
            if k in sample_dict:
                v = sample_dict[k]
                if hasattr(v, 'shape'):
                    print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"  - {k}: type={type(v)}")
    
    # Merge arrays
    merged = {}
    n_failed = 0
    
    for k in keys:
        parts = [d.get(k, None) for d in all_dicts]
        parts_nonnull = [p for p in parts if p is not None]
        
        if len(parts_nonnull) == 0:
            merged[k] = None
            n_failed += 1
            continue
        
        # Try different concatenation strategies
        try:
            # First try: simple concatenation along axis 0
            merged[k] = np.concatenate(parts_nonnull, axis=0)
        except Exception:
            try:
                # Second try: vstack
                merged[k] = np.vstack(parts_nonnull)
            except Exception:
                try:
                    # Third try: object array
                    merged[k] = np.array(parts_nonnull, dtype=object)
                    if verbose:
                        print(f"  ⚠️  {k}: stored as object array")
                except Exception as e:
                    if verbose:
                        print(f"  ❌ {k}: failed to merge ({e})")
                    merged[k] = None
                    n_failed += 1
    
    # Remove None values
    save_dict = {k: v for k, v in merged.items() if v is not None}
    
    if verbose:
        print(f"\n✓ Successfully merged {len(save_dict)}/{len(keys)} arrays")
        if n_failed > 0:
            print(f"  ⚠️  {n_failed} arrays could not be merged")
        
        print("\nMerged array shapes:")
        for k in sorted(save_dict.keys()):
            v = save_dict[k]
            if hasattr(v, 'shape'):
                print(f"  - {k}: {v.shape}")
    
    # Save combined results
    outpath = results_dir / f'PAE_results_combined_{run_name}.npz'
    
    if verbose:
        print(f"\nSaving combined results to:")
        print(f"  {outpath}")
    
    np.savez_compressed(str(outpath), **save_dict)
    
    size_mb = outpath.stat().st_size / (1024 * 1024)
    if verbose:
        print(f"✓ Saved ({size_mb:.2f} MB)")
    
    output_files['results_file'] = outpath
    
    # ==================== Optionally Collate Samples ====================
    if include_samples:
        if verbose:
            print("\n" + "="*70)
            print("COLLATING PAE SAMPLES")
            print("="*70)
        
        pattern = 'PAE_samples_batch*.npz'
        sample_files = sorted(list(results_dir.glob(pattern)))
        
        if len(sample_files) == 0:
            print(f"⚠️  No sample files found matching pattern: {pattern}")
        else:
            if verbose:
                print(f"Found {len(sample_files)} batch sample files")
            
            # Load all sample files
            all_sample_dicts = []
            for f in sample_files:
                try:
                    data = dict(np.load(str(f), allow_pickle=True))
                    all_sample_dicts.append(data)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to load {f.name}: {e}")
                    continue
            
            if len(all_sample_dicts) > 0:
                # Merge sample arrays
                sample_keys = set().union(*[d.keys() for d in all_sample_dicts])
                merged_samples = {}
                
                for k in sample_keys:
                    parts = [d.get(k, None) for d in all_sample_dicts]
                    parts_nonnull = [p for p in parts if p is not None]
                    
                    if len(parts_nonnull) == 0:
                        continue
                    
                    try:
                        merged_samples[k] = np.concatenate(parts_nonnull, axis=0)
                    except Exception:
                        try:
                            merged_samples[k] = np.vstack(parts_nonnull)
                        except Exception:
                            try:
                                merged_samples[k] = np.array(parts_nonnull, dtype=object)
                            except Exception:
                                continue
                
                # Save combined samples
                sample_outpath = results_dir / f'PAE_samples_combined_{run_name}.npz'
                np.savez_compressed(str(sample_outpath), **merged_samples)
                
                size_mb = sample_outpath.stat().st_size / (1024 * 1024)
                if verbose:
                    print(f"✓ Saved combined samples ({size_mb:.2f} MB)")
                    print(f"  {sample_outpath}")
                
                output_files['samples_file'] = sample_outpath
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description='Collate batched redshift estimation results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('run_name', type=str,
                       help='Name of the batched run (e.g., test_df_122925)')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Custom results directory (default: scratch_basepath/data/pae_sample_results/MCLMC/batched/RUN_NAME)')
    parser.add_argument('--include-samples', action='store_true',
                       help='Also collate sample files (PAE_samples_*.npz)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    # Determine results directory
    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(scratch_basepath) / 'data' / 'pae_sample_results' / 'MCLMC' / 'batched' / args.run_name
    
    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Run collation
    output_files = collate_results(
        results_dir,
        args.run_name,
        include_samples=args.include_samples,
        verbose=not args.quiet
    )
    
    # Summary
    if output_files and not args.quiet:
        print("\n" + "="*70)
        print("COLLATION COMPLETE")
        print("="*70)
        print("Created files:")
        for key, filepath in output_files.items():
            print(f"  - {filepath.name}")
        print("="*70)
    
    if not output_files:
        print("\n❌ Collation failed - no output files created")
        sys.exit(1)


if __name__ == '__main__':
    main()
