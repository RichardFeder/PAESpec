#!/usr/bin/env python
"""
Batched redshift estimation for mock/simulated SPHEREx data.

This script processes mock data in the original format (loaded via load_spherex_data())
rather than parquet format. It's designed for validation, testing, and development
on simulated data with known redshifts.

Key differences from redshift_job_batched.py:
- Uses load_spherex_data() instead of parquet loading
- All data loaded at once (not streamed from disk)
- Supports filtering by redshift range, SNR, etc.
- Single-GPU optimized (use redshift_job_batched.py for multi-GPU)

Usage Examples:
    # Quick test with 100 sources in single task
    python scripts/redshift_job_mock_batched.py --sources-per-task 100 --max-tasks 1 --batch-size 50
    
    # Specific redshift range
    python scripts/redshift_job_mock_batched.py --sources-per-task 1000 --z-min 0.5 --z-max 2.0
    
    # High SNR sources only, multiple tasks
    python scripts/redshift_job_mock_batched.py --sources-per-task 500 --max-tasks 3 --snr-min 100 --snr-max 300
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import jax.numpy as jnp
import time
import json
from pathlib import Path

from models.pae_jax import initialize_PAE, load_spherex_data
from training.train_ae_jax import param_dict_gen
from sampling.sample_pae_batch_refactor import MCLMCSamplingConfig, sample_mclmc_wrapper
from data_proc.dataloader_jax import SPHERExData
from config import scratch_basepath
from utils.memory_utils import (
    get_detailed_memory_info,
    print_memory_status
)
from utils.config_loader import apply_yaml_defaults


def get_memory_info():
    """Get current memory usage (wrapper for compatibility)."""
    return get_detailed_memory_info()


def load_bpz_prior_from_json(json_path):
    """Load BPZ prior parameters from fit results JSON.
    
    Parameters
    ----------
    json_path : str or Path
        Path to redshift_prior_fit_results.json
        
    Returns
    -------
    dict
        Dictionary with 'z0', 'alpha', 'beta' keys
    """
    with open(json_path, 'r') as f:
        fit_results = json.load(f)
    
    if 'bpz' not in fit_results:
        raise ValueError(f"No BPZ results found in {json_path}")
    
    bpz = fit_results['bpz']
    return {
        'z0': bpz['z0'],
        'alpha': bpz['alpha'],
        'beta': bpz['beta']
    }


def apply_filters(spherex_dat, property_cat_df, args):
    """Apply filtering criteria to select subset of sources.
    
    Parameters
    ----------
    spherex_dat : SPHERExData
        Full dataset
    property_cat_df : DataFrame
        Property catalog
    args : Namespace
        Command line arguments with filter specifications
        
    Returns
    -------
    valid_mask : ndarray
        Boolean mask for valid sources
    """
    nsrc_total = len(spherex_dat.redshift)
    valid_mask = np.ones(nsrc_total, dtype=bool)
    
    print(f"\nApplying filters:")
    print(f"  Starting with {nsrc_total} sources")
    
    # Redshift filters
    if args.z_min is not None:
        z_mask = spherex_dat.redshift >= args.z_min
        valid_mask &= z_mask
        print(f"  z >= {args.z_min}: {np.sum(z_mask)} sources")
        
    if args.z_max is not None:
        z_mask = spherex_dat.redshift <= args.z_max
        valid_mask &= z_mask
        print(f"  z <= {args.z_max}: {np.sum(z_mask)} sources")
    
    # SNR filters (if phot_snr column exists)
    if hasattr(spherex_dat, 'phot_snr') and spherex_dat.phot_snr is not None:
        if args.snr_min is not None:
            snr_mask = spherex_dat.phot_snr >= args.snr_min
            valid_mask &= snr_mask
            print(f"  SNR >= {args.snr_min}: {np.sum(snr_mask)} sources")
            
        if args.snr_max is not None:
            snr_mask = spherex_dat.phot_snr <= args.snr_max
            valid_mask &= snr_mask
            print(f"  SNR <= {args.snr_max}: {np.sum(snr_mask)} sources")
    
    n_valid = np.sum(valid_mask)
    print(f"  Final filtered count: {n_valid} sources ({100*n_valid/nsrc_total:.1f}%)")
    
    return valid_mask


def process_batch_mock(
    batch_indices,
    batch_idx,
    spherex_dat,
    property_cat_df,
    PAE_model,
    cfg,
    save_dir,
    args
):
    """Process a single batch of mock data sources."""
    
    nsrc_batch = len(batch_indices)
    start_idx = batch_indices[0] if len(batch_indices) > 0 else 0
    end_idx = batch_indices[-1] + 1 if len(batch_indices) > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"PROCESSING BATCH {batch_idx}: sources {start_idx} to {end_idx}")
    print(f"{'='*70}")
    
    # Memory check at batch start
    print_memory_status("START", batch_idx)
    
    batch_start_time = time.time()
    
    # Create source index array for this batch
    src_idxs_sub = batch_indices
    
    # Generate file paths for this batch (with start index like production)
    batch_suffix = f'batch{batch_idx}_start{start_idx}'
    save_fpath = save_dir / f'PAE_results_{batch_suffix}_{args.datestr}.npz'
    sample_fpath = save_dir / f'PAE_samples_{batch_suffix}_{args.datestr}.npz'
    
    print(f"\nSaving results to:")
    print(f"  {save_fpath.name}")
    
    # Run MCLMC sampling on this batch
    t_sampling_start = time.time()
    print_memory_status("Before sampling", batch_idx)
    
    try:
        results = sample_mclmc_wrapper(
            PAE_model,
            spherex_dat,
            cfg,
            ngal=nsrc_batch,
            batch_size=args.sampling_batch_size,
            save_results=True,
            save_fpath=str(save_fpath),
            sample_fpath=str(sample_fpath),
            return_results=True,
            do_cleanup=False,  # Match production setting
            src_idxs_sub=src_idxs_sub,
            property_cat_df=property_cat_df
        )
        
        print_memory_status("After sampling", batch_idx)
        
        t_sampling = time.time() - t_sampling_start
        batch_time = time.time() - batch_start_time
        
        # Memory check at batch end
        print_memory_status("END (before cleanup)", batch_idx)
        
        print(f"\n{'='*70}")
        print(f"BATCH {batch_idx} TIMING")
        print(f"{'='*70}")
        print(f"Sampling time:  {t_sampling:8.2f}s ({t_sampling/nsrc_batch:6.2f}s per galaxy)")
        print(f"Total time:     {batch_time:8.2f}s ({batch_time/60:6.2f} min)")
        print(f"{'='*70}\n")
        
        return True, batch_time
        
    except Exception as e:
        print(f"\n✗ Batch {batch_idx} FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        
        if args.stop_on_error:
            raise
        return False, 0.0


def process_single_task(
    task_id,
    args,
    spherex_dat,
    property_cat_df_obs,
    valid_indices,
    PAE_COSMOS,
    cfg,
    base_save_dir
):
    """Process a single task with its own subdirectory."""
    
    # Create task-specific directory
    task_save_dir = base_save_dir / f'task{task_id}'
    task_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate source range for this task
    start_idx = task_id * args.sources_per_task
    end_idx = min(start_idx + args.sources_per_task, len(valid_indices))
    selected_indices = valid_indices[start_idx:end_idx]
    n_selected = len(selected_indices)
    
    if n_selected == 0:
        print(f"\n⚠ Task {task_id}: No sources in range")
        return True, []  # Not a failure, just no sources
    
    print(f"\n{'='*70}")
    print(f"TASK {task_id}: Processing {n_selected} sources")
    print(f"{'='*70}")
    print(f"  Source range: {start_idx} to {end_idx-1}")
    print(f"  Index range: {selected_indices[0]} to {selected_indices[-1]}")
    print(f"  Redshift range: z={spherex_dat.redshift[selected_indices].min():.3f} to {spherex_dat.redshift[selected_indices].max():.3f}")
    print(f"  Output directory: {task_save_dir.name}")
    
    # Process sources in batches
    n_batches = int(np.ceil(n_selected / args.batch_size))
    batch_times = []
    failed_batches = []
    
    for batch_idx in range(n_batches):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, n_selected)
        batch_indices = selected_indices[start:end]
        
        success, batch_time = process_batch_mock(
            batch_indices,
            batch_idx,
            spherex_dat,
            property_cat_df_obs,
            PAE_COSMOS,
            cfg,
            task_save_dir,  # Use task-specific directory
            args
        )
        
        if success:
            batch_times.append(batch_time)
        else:
            failed_batches.append(batch_idx)
    
    # Report task completion
    if failed_batches:
        print(f"\n⚠ Task {task_id}: {len(failed_batches)} batches failed: {failed_batches}")
        return False, batch_times
    else:
        print(f"\n✓ Task {task_id}: All {n_batches} batches completed successfully!")
        return True, batch_times


def main(args):
    """Main processing function."""
    
    t_start_total = time.time()
    print(f"\n{'='*70}")
    print(f"MOCK DATA BATCHED REDSHIFT ESTIMATION")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Filter set: {args.filter_set}")
    print(f"  Latent dims: {args.nlatent}")
    print(f"  Sig level: {args.sig_level_norm}")
    print(f"  Selection: {args.sel_str}")
    print(f"  Processing tasks: {args.start_task} to {args.start_task + args.max_tasks - 1}")
    print(f"  Sources per task: {args.sources_per_task}")
    print(f"  Total sources to process: {args.sources_per_task * args.max_tasks}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sampling batch size: {args.sampling_batch_size}")
    print(f"  Multicore mode: {args.use_multicore}")
    if args.use_multicore:
        print(f"  Number of devices: {args.n_devices}")
    print(f"  Date string: {args.datestr}")
    
    # Create base output directory
    base_save_dir = Path(scratch_basepath) / 'data' / 'pae_sample_results' / 'MCLMC' / 'batched' / args.datestr
    base_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nBase output directory: {base_save_dir}")
    
    # Save copy of calling shell script for full traceability
    import subprocess
    try:
        # Get the parent process command line (the shell script that called this Python script)
        parent_pid = os.getppid()
        result = subprocess.run(['ps', '-p', str(parent_pid), '-o', 'cmd='], 
                              capture_output=True, text=True, check=True)
        parent_cmd = result.stdout.strip()
        
        # Check if parent is a bash script
        if parent_cmd and ('.sh' in parent_cmd or 'bash' in parent_cmd):
            # Extract script path from command
            parts = parent_cmd.split()
            script_path = None
            for part in parts:
                if part.endswith('.sh') and os.path.isfile(part):
                    script_path = part
                    break
            
            if script_path:
                import shutil
                dest_path = base_save_dir / 'run_script_copy.sh'
                shutil.copy2(script_path, dest_path)
                print(f"Saved calling script to: {dest_path.name}")
    except Exception as e:
        # Silently skip if we can't determine/copy the parent script
        pass
    
    # Save run configuration for reproducibility
    run_params = {
        'filter_set': args.filter_set,
        'nlatent': args.nlatent,
        'sig_level_norm': args.sig_level_norm,
        'sel_str': args.sel_str,
        'sources_per_task': args.sources_per_task,
        'start_task': args.start_task,
        'max_tasks': args.max_tasks,
        'batch_size': args.batch_size,
        'sampling_batch_size': args.sampling_batch_size,
        'datestr': args.datestr,
        'with_ext_phot': args.with_ext_phot,
        'filename_flow': args.filename_flow,
        # Filtering
        'z_min': args.z_min,
        'z_max': args.z_max,
        'snr_min': args.snr_min,
        'snr_max': args.snr_max,
        # Sampling config
        'num_steps': args.num_steps,
        'burn_in': args.burn_in,
        'nsamp_init': args.nsamp_init,
        'chi2_red_threshold': args.chi2_red_threshold,
        'gr_threshold': args.gr_threshold,
        'fix_z': args.fix_z,
        'nf_alpha': args.nf_alpha,
        'nf_alpha_burnin': args.nf_alpha_burnin,
        'prior_type': args.prior_type,
        'z0_prior': args.z0_prior,
        'sigma_prior': args.sigma_prior,
        'use_multicore': args.use_multicore,
        'n_devices': args.n_devices if args.use_multicore else 1,
        'sample_log_redshift': args.sample_log_redshift,
        'use_snr_prefit_init': args.use_snr_prefit_init,
        'snr_prefit_json': args.snr_prefit_json,
        'snr_prefit_column': args.snr_prefit_column,
    }
    config_file = base_save_dir / 'run_params.npz'
    np.savez(config_file, **run_params)
    print(f"Saved run configuration to: {config_file.name}")
    
    # Load BPZ parameters from JSON if provided
    z0_prior = args.z0_prior
    sigma_prior = args.sigma_prior
    # Set dummy values for alpha_prior and beta_prior to allow JAX JIT compilation
    # (even when not using BPZ prior, all branches need to be compilable)
    alpha_prior = 1.0
    beta_prior = 1.0
    
    if args.prior_type == 2 and args.bpz_prior_json is not None:
        print(f"\nLoading BPZ prior from: {args.bpz_prior_json}")
        bpz_params = load_bpz_prior_from_json(args.bpz_prior_json)
        sigma_prior = bpz_params['z0']  # Note: z0 is stored in sigma_prior for BPZ
        alpha_prior = bpz_params['alpha']
        beta_prior = bpz_params['beta']
        print(f"  z0 = {sigma_prior:.4f}")
        print(f"  alpha = {alpha_prior:.4f}")
        print(f"  beta = {beta_prior:.4f}")
    
    # Initialize sampling configuration
    cfg = MCLMCSamplingConfig(
        num_steps=args.num_steps,
        fix_z=args.fix_z,
        nf_alpha=args.nf_alpha,
        nf_alpha_burnin=args.nf_alpha_burnin,
        nsamp_init=args.nsamp_init,
        burn_in=args.burn_in,
        chi2_red_threshold=args.chi2_red_threshold,
        gr_threshold=args.gr_threshold,
        init_reinit=True,
        # Redshift prior
        redshift_prior_type=args.prior_type,
        z0_prior=z0_prior,
        sigma_prior=sigma_prior,
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
        # Amplitude handling
        sample_log_amplitude=args.sample_log_amplitude,
        log_amplitude_prior_std=args.log_amplitude_prior_std,
        # Log-redshift sampling
        sample_log_redshift=args.sample_log_redshift,
        # Performance
        use_batched_logdensity=True,
        use_multicore=args.use_multicore,
        n_devices_per_node=args.n_devices if args.use_multicore else 1,
        # Optional SNR-prefit initialization path
        use_snr_prefit_init=args.use_snr_prefit_init,
        snr_prefit_json=args.snr_prefit_json,
        snr_prefit_column=args.snr_prefit_column,
        skip_autotune_with_prefit=args.use_snr_prefit_init,
    )
    
    prior_names = {
        0: "None", 
        1: f"Gaussian(z0={z0_prior}, σ={sigma_prior})",
        2: f"BPZ(z0={sigma_prior:.4f}, α={cfg.alpha_prior:.2f}, β={cfg.beta_prior:.2f})",
        3: f"Amplitude-dependent BPZ(α={cfg.alpha_amp}, slope={cfg.z0_amp_slope:.4f}, intercept={cfg.z0_amp_intercept:.4f})"
    }
    print(f"\nSampling configuration:")
    print(f"  Redshift prior: {prior_names.get(args.prior_type, 'Unknown')}")
    print(f"  Sample log(amplitude): {args.sample_log_amplitude}")
    if args.sample_log_amplitude:
        print(f"  log(amplitude) prior std: {args.log_amplitude_prior_std}")
    print(f"  Sample log(redshift): {args.sample_log_redshift}")
    print(f"  MCMC steps: {args.num_steps}")
    print(f"  Burn-in: {args.burn_in}")
    print(f"  χ²_red threshold: {args.chi2_red_threshold}")
    print(f"  G-R threshold: {args.gr_threshold}")
    
    # Load mock SPHEREx data
    print(f"\n{'='*70}")
    print("LOADING MOCK DATA")
    print(f"{'='*70}")
    t_load_start = time.time()
    print_memory_status("Before data load")
    
    dat_obs, property_cat_df_obs, property_cat_df_restframe, \
        central_wavelengths, wave_obs = load_spherex_data(
            sig_level_norm=args.sig_level_norm,
            sel_str=args.sel_str,
            abs_norm=True,
            with_ext_phot=args.with_ext_phot,
            load_rf_dat=False,
            load_obs_dat=True,
            weight_soft=5e-4,
            data_fpath=args.data_fpath
        )
    
    # Prepare SPHEREx data wrapper
    spherex_dat = SPHERExData.from_prep(
        dat_obs,
        property_cat_df_obs,
        property_cat_df_restframe,
        phot_snr_min=None,  # Apply filters later
        phot_snr_max=None,
        zmin=None,
        zmax=None
    )
    
    t_load = time.time() - t_load_start
    print(f"✓ Data loaded in {t_load:.2f}s ({t_load/60:.2f} min)")
    print(f"  Total sources available: {len(spherex_dat.redshift)}")
    print_memory_status("After data load")
    
    # Apply filters to select sources
    valid_mask = apply_filters(spherex_dat, property_cat_df_obs, args)
    valid_indices = np.where(valid_mask)[0]
    
    print(f"\nTotal sources after filtering: {len(valid_indices)}")
    
    # Initialize PAE model
    print(f"\n{'='*70}")
    print("INITIALIZING PAE MODEL")
    print(f"{'='*70}")
    t_model_start = time.time()
    
    # Use model run configured by CLI arg (defaults set in argparse)
    run_name = args.run_name
    # Old models (trained with rescale.transform convention):
    # run_name = f'jax_conv1_nlatent={args.nlatent}_siglevelnorm={args.sig_level_norm}_newAllen_all_091325'
    # run_name = f'jax_conv1_nlatent={args.nlatent}_siglevelnorm={args.sig_level_norm}_newAllen_all'
    # run_name = f'jax_conv1_nlatent={args.nlatent}_siglevelnorm={args.sig_level_norm}_newAllen'
    
    PAE_COSMOS = initialize_PAE(
        run_name,
        filter_set_name=args.filter_set,
        with_ext_phot=args.with_ext_phot,
        inference_dtype=jnp.float32,
        lam_min_rest=0.15,
        lam_max_rest=5.0,
        nlam_rest=500,
        filename_flow=args.filename_flow
    )
    
    t_model = time.time() - t_model_start
    print(f"✓ PAE model initialized in {t_model:.2f}s")
    print_memory_status("After model init")
    
    # Process multiple tasks serially
    n_total_sources = len(valid_indices)
    n_total_tasks = (n_total_sources + args.sources_per_task - 1) // args.sources_per_task
    end_task = min(args.start_task + args.max_tasks, n_total_tasks)
    
    print(f"\n{'='*70}")
    print("TASK-BASED PROCESSING")
    print(f"{'='*70}")
    print(f"Total available sources: {n_total_sources}")
    print(f"Sources per task: {args.sources_per_task}")
    print(f"Total possible tasks: {n_total_tasks}")
    print(f"Running tasks: {args.start_task} to {end_task-1}")
    print(f"{'='*70}")
    
    successful_tasks = 0
    failed_tasks = []
    all_batch_times = []
    
    for task_id in range(args.start_task, end_task):
        success, batch_times = process_single_task(
            task_id,
            args,
            spherex_dat,
            property_cat_df_obs,
            valid_indices,
            PAE_COSMOS,
            cfg,
            base_save_dir
        )
        
        if success:
            successful_tasks += 1
            all_batch_times.extend(batch_times)
        else:
            failed_tasks.append(task_id)
    
    # Collate results across tasks if requested
    if args.collate_results and successful_tasks > 0:
        print(f"\n{'='*70}")
        print("COLLATING RESULTS ACROSS TASKS")
        print(f"{'='*70}")
        
        # Collect all batch files from all task directories
        all_files = []
        for task_id in range(args.start_task, end_task):
            if task_id not in failed_tasks:
                task_dir = base_save_dir / f'task{task_id}'
                pattern = f'PAE_results_batch*_start*_{args.datestr}.npz'
                task_files = sorted(list(task_dir.glob(pattern)))
                all_files.extend(task_files)
        
        if len(all_files) > 0:
            print(f"Found {len(all_files)} batch files across {successful_tasks} tasks")
            all_dicts = [dict(np.load(str(f), allow_pickle=True)) for f in all_files]
            keys = set().union(*[d.keys() for d in all_dicts])
            merged = {}
            
            for k in keys:
                parts = [d.get(k, None) for d in all_dicts]
                parts_nonnull = [p for p in parts if p is not None]
                if len(parts_nonnull) == 0:
                    continue
                try:
                    merged[k] = np.concatenate(parts_nonnull, axis=0)
                except:
                    try:
                        merged[k] = np.vstack(parts_nonnull)
                    except:
                        merged[k] = np.array(parts_nonnull, dtype=object)
            
            outpath = base_save_dir / f'PAE_results_combined_{args.datestr}.npz'
            save_dict = {kk: vv for kk, vv in merged.items() if vv is not None}
            print(f"Saving combined results to {outpath.name}")
            np.savez_compressed(str(outpath), **save_dict)
            print(f"✓ Collation complete: {len(save_dict)} arrays merged")
        else:
            print("⚠ No result files found to collate")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total tasks: {end_task - args.start_task}")
    print(f"Successful: {successful_tasks}")
    print(f"Failed: {len(failed_tasks)}")
    
    if failed_tasks:
        print(f"\nFailed task IDs: {failed_tasks}")
    
    if all_batch_times:
        print(f"\nTotal batches processed: {len(all_batch_times)}")
        print(f"Avg time per batch: {np.mean(all_batch_times):.2f}s")
    
    print(f"\nResults saved to: {base_save_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        '--config-yaml',
        type=str,
        default=None,
        help='Path to YAML config file. Values are used as defaults and can be overridden by CLI flags.'
    )
    pre_args, remaining_argv = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description='Run batched redshift estimation on mock SPHEREx data',
        parents=[pre_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument('--filter-set', type=str, default='spherex_filters102/',
                       help='Filter set name')
    parser.add_argument('--nlatent', type=int, default=5,
                       help='Number of latent dimensions')
    parser.add_argument('--sig-level-norm', type=float, default=0.01,
                       help='Noise level for mock data')
    parser.add_argument('--sel-str', type=str, default='zlt22.5',
                       help='Selection string for data loading')
    parser.add_argument('--with-ext-phot', action='store_true',
                       help='Include external photometry')
    parser.add_argument('--data-fpath', type=str, default=None,
                       help='Custom path to catgrid data file (default: auto-detect from sel-str)')
    parser.add_argument('--filename-flow', type=str, default='flow_model_iaf',
                       help='Flow model filename')
    parser.add_argument('--run-name', type=str, default='fp_nlatent=10_031326',
                       help='Model run directory name under modl_runs/')
    
    # Source selection (task-based structure matching production)
    parser.add_argument('--sources-per-task', type=int, default=None,
                       help='Number of sources per task')
    parser.add_argument('--start-task', type=int, default=0,
                       help='Starting task ID (default: 0)')
    parser.add_argument('--max-tasks', type=int, default=1,
                       help='Maximum number of tasks to run (default: 1)')
    parser.add_argument('--batch-size', type=int, default=200,
                       help='Sources per batch (outer loop)')
    parser.add_argument('--sampling-batch-size', type=int, default=100,
                       help='Sources per sampling batch (inner loop)')
    
    # Filtering
    parser.add_argument('--z-min', type=float, default=None,
                       help='Minimum redshift')
    parser.add_argument('--z-max', type=float, default=None,
                       help='Maximum redshift')
    parser.add_argument('--snr-min', type=float, default=None,
                       help='Minimum photometric SNR')
    parser.add_argument('--snr-max', type=float, default=None,
                       help='Maximum photometric SNR')
    
    # Sampling configuration
    parser.add_argument('--num-steps', type=int, default=2000,
                       help='MCMC steps')
    parser.add_argument('--burn-in', type=int, default=1000,
                       help='Burn-in steps')
    parser.add_argument('--nsamp-init', type=int, default=500,
                       help='Initial samples for reinitialization')
    parser.add_argument('--chi2-red-threshold', type=float, default=1.5,
                       help='Chi-squared threshold for convergence')
    parser.add_argument('--gr-threshold', type=float, default=1.5,
                       help='Gelman-Rubin threshold for convergence')
    parser.add_argument('--fix-z', action='store_true',
                       help='Fix redshift (for testing)')
    parser.add_argument('--nf-alpha', type=float, default=0.0,
                       help='Normalizing flow alpha parameter (final sampling)')
    parser.add_argument('--nf-alpha-burnin', type=float, default=1.0,
                       help='NF alpha for burn-in phase only (default 1.0 for stable initialization)')
    
    # Redshift prior
    parser.add_argument('--prior-type', type=int, default=1,
                       help='Prior type: 0=none, 1=Gaussian, 2=BPZ, 3=amplitude-dependent BPZ')
    parser.add_argument('--z0-prior', type=float, default=0.65,
                       help='Gaussian prior center (type=1) or BPZ z0 (type=2)')
    parser.add_argument('--sigma-prior', type=float, default=0.6,
                       help='Gaussian prior width (type=1) or BPZ z0 (type=2)')
    parser.add_argument('--bpz-prior-json', type=str, default=None,
                       help='Path to JSON file with fitted BPZ parameters (overrides --z0-prior and --sigma-prior when --prior-type=2)')
    
    # Amplitude handling
    parser.add_argument('--sample-log-amplitude', action='store_true',
                       help='Sample log(amplitude) instead of marginalizing (enforces positivity)')
    parser.add_argument('--log-amplitude-prior-std', type=float, default=2.0,
                       help='Std of Gaussian prior on log(amplitude)')
    
    # Log-redshift sampling
    parser.add_argument('--sample-log-redshift', action='store_true',
                       help='Sample ln(z) instead of z (eliminates BPZ prior singularity near z=0)')
    
    # Performance
    parser.add_argument('--use-multicore', action='store_true',
                       help='Use multicore mode (multiple GPUs)')
    parser.add_argument('--n-devices', type=int, default=4,
                       help='Number of GPUs to use in multicore mode')
    parser.add_argument('--use-snr-prefit-init', action='store_true',
                       help='Use SNR->(L, step_size) prefit model for per-source burn-in init and skip final autotuning')
    parser.add_argument('--snr-prefit-json', type=str, default=None,
                       help='Path to fit JSON generated by fit_mclmc_tuning_from_snr.py')
    parser.add_argument('--snr-prefit-column', type=str, default='phot_snr',
                       help='Property catalog column used as SNR predictor for prefit init')
    
    # Output
    parser.add_argument('--datestr', type=str, default='mock_test',
                       help='Date string for output files')
    parser.add_argument('--collate-results', action='store_true',
                       help='Collate per-batch results into single file')
    parser.add_argument('--stop-on-error', action='store_true',
                       help='Stop if any batch fails')
    
    if pre_args.config_yaml:
        apply_yaml_defaults(parser, pre_args.config_yaml, section='redshift_mock')

    args = parser.parse_args(remaining_argv)

    if args.sources_per_task is None:
        parser.error('Missing required setting: --sources-per-task (or provide sources_per_task in --config-yaml)')

    if args.use_snr_prefit_init and args.snr_prefit_json is None:
        parser.error('--use-snr-prefit-init requires --snr-prefit-json')
    
    # Print task configuration
    print(f"Task mode: will run {args.max_tasks} task(s) starting from task {args.start_task}")
    print(f"Each task processes {args.sources_per_task} sources")
    
    # Set restrictive permissions
    os.umask(0o077)
    
    main(args)
