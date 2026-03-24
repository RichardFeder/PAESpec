#!/usr/bin/env python
"""
Example: Running redshift estimation on large parquet files using batched loading.

This script demonstrates how to process 100k+ sources efficiently by:
1. Loading data in batches (to manage memory)
2. Running redshift estimation on each batch
3. Saving results incrementally

OPTIMIZING PARALLEL CHAIN COUNT (sampling_batch_size):
======================================================

The sampling_batch_size parameter controls how many MCMC chains run in parallel per GPU.
This is critical for performance optimization:

MEMORY CONSIDERATIONS:
- Shared GPU nodes (< 40GB): Typically limited to ~800 chains/GPU
- Exclusive A100-40GB: Can handle ~1200-1600 chains/GPU
- Exclusive A100-80GB: Can handle ~2000-3200 chains/GPU

HOW TO FIND YOUR OPTIMAL BATCH SIZE:
1. Run this script - it will print a batch size guide at startup
2. Monitor GPU memory with the detailed printouts during execution
3. Look for "Peak" memory usage, not just "Current"
4. Target 70-85% peak utilization for best performance

SIGNS YOU'RE USING TOO MUCH MEMORY:
- Peak GPU memory > 90%: Risk of OOM (Out of Memory) errors
- Peak GPU memory > 85%: Performance degradation (GPU swapping to host RAM)
- "CUDA out of memory" errors
- Significantly slower sampling times per galaxy

SIGNS YOU CAN INCREASE BATCH SIZE:
- Peak GPU memory < 70%: Substantial headroom available
- Consistent performance across batches
- No memory warnings in the output

PERFORMANCE VS MEMORY TRADEOFF:
- Larger batches = Better GPU utilization = Faster per-galaxy time
- Too large = Memory pressure = Slower or crashes
- Sweet spot: Peak memory around 75-80% of GPU limit

MULTICORE MODE:
With --use-multicore and 4 GPUs:
  Total parallel sources = sampling_batch_size (divided across GPUs)
  Each GPU gets sampling_batch_size ÷ 4 sources
  Example: --sampling-batch-size 1000 → 250 sources per GPU, 1000 total parallel

Usage:
    # Conservative (shared nodes or testing)
    python scripts/redshift_job_batched.py --parquet-file /path/to/file.parquet \\
        --batch-size 10000 --sampling-batch-size 800
    
    # Optimized for exclusive A100-80GB nodes
    python scripts/redshift_job_batched.py --parquet-file /path/to/file.parquet \\
        --batch-size 10000 --sampling-batch-size 2000 --use-multicore --n-devices 4
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import jax.numpy as jnp
import time
from pathlib import Path

from models.pae_jax import initialize_PAE, load_filter_central_wavelengths
from training.train_ae_jax import param_dict_gen
from sampling.sample_pae_batch_refactor import MCLMCSamplingConfig, sample_mclmc_wrapper
from data_proc.dataloader_jax import SPHERExData, load_real_spherex_parquet
from data_proc.batch_parquet_loader import (
    ParquetBatchLoader, 
    filter_has_specz, 
    filter_redshift_range,
    filter_good_fits,
    combine_filters
)
from config import scratch_basepath
from utils.memory_utils import (
    get_gpu_memory_limit,
    get_detailed_memory_info,
    print_batch_size_guide,
    print_memory_status
)


# Memory utility functions moved to utils/memory_utils.py
# Import them at the top of this file:
#   from utils.memory_utils import (
#       get_gpu_memory_limit,
#       print_batch_size_guide, 
#       print_memory_status
#   )


def process_batch(
    batch_df,
    batch_idx,
    start_idx,
    PAE_COSMOS,
    cfg,
    filter_set_name,
    wave_obs,
    save_dir,
    datestr_test,
    sampling_batch_size=200,
    gpu_limit_gb=80.0,
    use_weighted_mean=False,
    abs_norm=True,
    channel_mask=None,
):
    """Process a single batch of sources.
    
    Parameters
    ----------
    sampling_batch_size : int
        Number of sources to process per GPU in parallel.
        With multicore=True and 4 GPUs: total parallel = sampling_batch_size * 4.
        Default 200 sources/GPU = 800 parallel sources with 4 GPUs.
    channel_mask : np.ndarray or None
        Boolean array of length nbands. True = exclude that channel from the
        likelihood (its weight is forced to zero).
    """
    
    print(f"[DEBUG] ENTERED process_batch for batch {batch_idx}", flush=True)
    sys.stdout.flush()
    
    print(f"\n{'='*70}")
    print(f"PROCESSING BATCH {batch_idx}: sources {start_idx} to {start_idx + len(batch_df)}")
    print(f"{'='*70}")
    
    # Memory check at batch start
    print_memory_status("START", batch_idx)
    
    t_batch_start = time.time()
    
    # Load batch data
    print_memory_status("Before data load", batch_idx)
    # Set max_normflux appropriately: high value if no normalization, 100 if normalized
    max_normflux_val = 1e4 if not abs_norm else 100
    dat_obs, property_cat_df = load_real_spherex_parquet(
        parquet_file=None,
        filter_set_name=filter_set_name,
        wave_obs=wave_obs,
        weight_soft=5e-4,
        abs_norm=abs_norm,
        max_normflux=max_normflux_val,
        df=batch_df,
        use_weighted_mean=use_weighted_mean,
        channel_mask=channel_mask,
    )
    print_memory_status("After data load", batch_idx)
    
    # Extract normalization values for later saving
    phot_norms = dat_obs.phot_dict['phot_norms'].squeeze()  # Shape: (n_sources,)
    
    # Prepare SPHERExData
    spherex_dat = SPHERExData.from_prep(
        dat_obs,
        property_cat_df,
        None,  # No rest frame data for real obs
        phot_snr_min=None,
        phot_snr_max=None,
        zmin=None,
        zmax=None
    )
    
    ngal = len(property_cat_df)
    src_idxs_sub = np.arange(ngal)

    print('ngal in property_cat_df is = ', ngal)
    
    # Generate file paths for this batch
    batch_suffix = f'batch{batch_idx}_start{start_idx}'
    save_fpath = save_dir / f'PAE_results_{batch_suffix}_{datestr_test}.npz'
    sample_fpath = save_dir / f'PAE_samples_{batch_suffix}_{datestr_test}.npz'
    
    print(f"\nSaving results to:")
    print(f"  {save_fpath.name}")
    
    # Run MCLMC sampling
    t_sampling_start = time.time()
    print_memory_status("Before sampling", batch_idx, gpu_limit_gb)
    
    results = sample_mclmc_wrapper(
        PAE_COSMOS,
        spherex_dat,
        cfg,
        ngal=ngal,
        batch_size=sampling_batch_size,  # Total sources across all GPUs (divided in multicore mode)
        keyidx=seed,  # Use random seed for MCMC initialization
        save_results=True,
        save_fpath=str(save_fpath),
        sample_fpath=str(sample_fpath),
        return_results=True,
        do_cleanup=False,
        src_idxs_sub=src_idxs_sub,
        property_cat_df=property_cat_df,
        phot_norms=phot_norms
    )
    
    print_memory_status("After sampling", batch_idx, gpu_limit_gb)
    
    t_sampling = time.time() - t_sampling_start
    t_batch_total = time.time() - t_batch_start
    
    # Memory check at batch end
    print_memory_status("END (before cleanup)", batch_idx, gpu_limit_gb)
    
    print(f"\n{'='*70}")
    print(f"BATCH {batch_idx} TIMING")
    print(f"{'='*70}")
    print(f"Sampling time:  {t_sampling:8.2f}s ({t_sampling/ngal:6.2f}s per galaxy)")
    print(f"Total time:     {t_batch_total:8.2f}s ({t_batch_total/60:6.2f} min)")
    print(f"{'='*70}\n")
    
    sys.stdout.flush()  # Force flush before returning
    
    return results


def main(args):
    """Main execution."""
    
    print("=" * 70)
    print("BATCHED REDSHIFT ESTIMATION")
    print("=" * 70)
    print(f"Parquet file: {args.parquet_file}")
    print(f"Batch size: {args.batch_size:,}")
    print(f"Sampling batch size (total): {args.sampling_batch_size}")
    print(f"Max batches: {args.max_batches or 'all'}")
    
    # Detect GPU memory limit and print optimization guide
    gpu_limit = get_gpu_memory_limit()
    if args.use_multicore:
        print(f"\nMulti-core mode: {args.n_devices} GPUs")
        print(f"Sources per GPU: {args.sampling_batch_size} ÷ {args.n_devices} = {args.sampling_batch_size // args.n_devices}")
        print(f"Total parallel sources: {args.sampling_batch_size}")
    print_batch_size_guide(args.sampling_batch_size, gpu_limit)
    
    # ==================== Configuration ====================
    filter_set_name = args.filter_set
    nlatent = 5
    sig_level_norm = 0.01
    
    # Sampling configuration
    cfg = MCLMCSamplingConfig(
        num_steps=700,          # Total inference steps (200 burn-in + 500 kept)
        nsamp_init=500,         # Initialization/tuning steps (separate from inference)
        nchain_per_gal=4,
        burn_in=200,            # Burn-in steps to discard (leaves 500 for inference)
        chi2_red_threshold=1.5,
        gr_threshold=1.5,
        init_reinit=args.init_reinit,
        fix_z=args.fix_z,
        nf_alpha=args.nf_alpha,
        redshift_prior_type=args.prior_type,
        z0_prior=args.z0_prior,
        sigma_prior=args.sigma_prior,
        use_batched_logdensity=True,
        use_multicore=args.use_multicore,
        n_devices_per_node=args.n_devices,
        # Log-redshift sampling
        sample_log_redshift=args.sample_log_redshift,
        # Robust reinitialization parameters
        use_robust_reinit=args.use_robust_reinit,
        reinit_min_chains_agree=2,
        reinit_logL_tolerance=5.0,
        reinit_scatter=0.05
    )

    print('USE ROBUST REINIT:', args.use_robust_reinit)
    
    print(f"\nRedshift prior: type={args.prior_type}")
    print(f"Fix redshift: {args.fix_z}")
    print(f"Multi-core: {'Enabled' if args.use_multicore else 'Disabled'}")
    
    # Load filters
    wave_obs, _ = load_filter_central_wavelengths(filter_set_name, filtfiles=None)
    nbands = len(wave_obs)
    print(f"Loaded {nbands} filters from {filter_set_name}")

    # Build channel mask from --mask-wavelength-ranges (True = exclude channel)
    channel_mask = np.zeros(nbands, dtype=bool)
    if args.mask_wavelength_ranges:
        for pair in args.mask_wavelength_ranges:
            try:
                lmin_str, lmax_str = pair.split(',')
                lmin, lmax = float(lmin_str), float(lmax_str)
            except ValueError:
                raise ValueError(
                    f"Invalid --mask-wavelength-ranges entry '{pair}'. "
                    "Expected 'lmin,lmax' (e.g. '2.42,3.82')."
                )
            in_range = (wave_obs >= lmin) & (wave_obs <= lmax)
            channel_mask |= in_range
            print(f"  Masking range [{lmin}, {lmax}] um: "
                  f"{in_range.sum()} channels excluded "
                  f"(λ = {wave_obs[in_range].min():.4f}–{wave_obs[in_range].max():.4f} um)")
        n_masked = channel_mask.sum()
        print(f"Channel mask: {n_masked}/{nbands} channels excluded "
              f"({100*n_masked/nbands:.1f}%)")
    else:
        channel_mask = None  # No masking


    print("\nInitializing PAE model...")
    
    # Use new model trained with rescale.inverse convention (Jan 2026)
    run_name = 'fp_nlatent=5_013126'
    # Old models (trained with rescale.transform convention):
    # run_name = 'jax_conv1_nlatent=5_siglevelnorm=0.01_newAllen_all_091325'
    
    # filename_flow = 'flow_model_iaf_092225'
    filename_flow = 'flow_model_iaf'
    
    PAE_COSMOS = initialize_PAE(
        run_name,
        filter_set_name=filter_set_name,
        with_ext_phot=False,
        inference_dtype=jnp.float32,
        lam_min_rest=0.15,
        lam_max_rest=5.0,
        nlam_rest=500,
        filename_flow=filename_flow
    )
    
    # Create output directory
    # If job_array_task_id is set, create nested structure: datestr/task{N}/
    if args.job_array_task_id is not None:
        save_dir = Path(scratch_basepath) / 'data' / 'pae_sample_results' / 'MCLMC' / 'batched' / args.datestr / f'task{args.job_array_task_id}'
    else:
        save_dir = Path(scratch_basepath) / 'data' / 'pae_sample_results' / 'MCLMC' / 'batched' / args.datestr
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults directory: {save_dir}")
    
    # Save run configuration for reproducibility
    run_params = {
        'run_name': run_name,
        'filter_set_name': filter_set_name,
        'filename_flow': filename_flow,
        'nlatent': nlatent,
        'sig_level_norm': sig_level_norm,
        'batch_size': args.batch_size,
        'sampling_batch_size': args.sampling_batch_size,
        'max_batches': args.max_batches,
        'fix_z': args.fix_z,
        'parquet_file': str(args.parquet_file),
        'datestr': args.datestr,
        # Sampling config
        'num_steps': cfg.num_steps,
        'nsamp_init': cfg.nsamp_init,
        'nchain_per_gal': cfg.nchain_per_gal,
        'burn_in': cfg.burn_in,
        'chi2_red_threshold': cfg.chi2_red_threshold,
        'gr_threshold': cfg.gr_threshold,
        'init_reinit': cfg.init_reinit,
        'use_robust_reinit': cfg.use_robust_reinit,
        'reinit_min_chains_agree': cfg.reinit_min_chains_agree,
        'reinit_logL_tolerance': cfg.reinit_logL_tolerance,
        'reinit_scatter': cfg.reinit_scatter,
        'redshift_prior_type': cfg.redshift_prior_type,
        'z0_prior': cfg.z0_prior,
        'sigma_prior': cfg.sigma_prior,
        'use_batched_logdensity': cfg.use_batched_logdensity,
        'use_multicore': cfg.use_multicore,
        'n_devices_per_node': cfg.n_devices_per_node,
        # Filter criteria
        'filter_specz': args.filter_specz,
        'z_min': args.z_min,
        'z_max': args.z_max,
        'chi2_red_max': args.chi2_red_max,
        'prior_type': args.prior_type,
        'z0_prior': args.z0_prior,
        'sigma_prior': args.sigma_prior,
        'random_seed': seed,
        # Channel masking
        'mask_wavelength_ranges': str(args.mask_wavelength_ranges),
        'n_masked_channels': int(channel_mask.sum()) if channel_mask is not None else 0,
    }
    
    config_file = save_dir / 'run_params.npz'
    np.savez(config_file, **run_params)
    print(f"Saved run configuration to: {config_file.name}")
    
    # Set up batch loader with optional filtering
    filter_fn = None
    if args.filter_specz:
        print("\nApplying filters:")
        filters = []
        if args.filter_specz:
            filters.append(filter_has_specz)
            print("  - Has spectroscopic redshift")
        if args.z_min is not None or args.z_max is not None:
            z_min = args.z_min if args.z_min is not None else 0.0
            z_max = args.z_max if args.z_max is not None else 10.0
            filters.append(filter_redshift_range(z_min, z_max))
            print(f"  - Redshift range: {z_min} < z < {z_max}")
        if args.chi2_red_max is not None:
            filters.append(filter_good_fits(args.chi2_red_max))
            print(f"  - χ²_red < {args.chi2_red_max}")
        
        if filters:
            filter_fn = combine_filters(*filters)
    

    print('batch size is ', args.batch_size)

    print('INIT REINIT is', args.init_reinit)
    
    # Specify only the columns we actually need to reduce memory usage
    # This is critical for large files with many columns!
    # First check which columns actually exist in the file
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(args.parquet_file)
    available_columns = set(pf.schema_arrow.names)
    
    required_columns = [
        'SPHERExRefID',           # Source ID (required)
        'flux_dered_fiducial',    # Flux arrays (required)
        'flux_err_dered_fiducial', # Flux error arrays (required)
        'flags',                  # Quality flags per measurement (required for filtering)
        'ra', 'dec',              # Coordinates (required)
        'z_specz',                # Spectroscopic redshift (optional but useful)
        'z_best_gals',            # Photometric redshift (optional)
        'z_err_std_gals',         # Redshift error (optional)
        'minchi2_minchi2_gals',   # Chi-squared (optional)
        'Nsamples',               # Number of samples (optional)
        'frac_sampled_102',       # Fraction of 102 bands sampled (optional)
    ]
    
    # Filter to only columns that exist in the file
    existing_columns = [col for col in required_columns if col in available_columns]
    missing_columns = [col for col in required_columns if col not in available_columns]
    
    print(f"\nColumn selection for memory optimization:")
    print(f"  Total columns in file: {len(available_columns)}")
    print(f"  Requested columns: {len(required_columns)}")
    print(f"  Found in file: {len(existing_columns)}")
    if missing_columns:
        print(f"  Missing (will skip): {', '.join(missing_columns)}")
    print(f"  Columns being skipped: {len(available_columns) - len(existing_columns)}")
    
    # Compute task row slice for fast single-read init (avoids per-batch O(N) re-scan).
    # args.start_source / end_source are already set by the job-array auto-compute block.
    _row_offset = args.start_source if args.start_source is not None else None
    _num_rows   = (
        (args.end_source - args.start_source)
        if (args.start_source is not None and args.end_source is not None)
        else None
    )
    loader = ParquetBatchLoader(
        args.parquet_file,
        batch_size=args.batch_size,
        columns=existing_columns,  # Only load columns that exist!
        filter_fn=filter_fn,
        row_offset=_row_offset,
        num_rows=_num_rows,
    )
    
    # Determine source range to process (for multi-node parallelization)
    if args.start_source is not None or args.end_source is not None:
        start_source = args.start_source if args.start_source is not None else 0
        end_source = args.end_source if args.end_source is not None else loader.total_sources
        print(f"\n{'='*70}")
        print(f"SOURCE RANGE RESTRICTION (for multi-node parallelization)")
        print(f"{'='*70}")
        print(f"Processing sources {start_source} to {end_source} (of {loader.total_sources} total)")
        print(f"This is chunk for job array task or multi-node distribution")
        print(f"{'='*70}\n")
    else:
        start_source = 0
        end_source = loader.total_sources
    
    # Process batches
    t_total_start = time.time()
    
    # Calculate number of batches to process based on actual source range for this task
    if args.max_batches:
        n_batches_to_process = args.max_batches
    else:
        # For job array mode: only process batches within this task's source range
        n_sources_this_task = end_source - start_source
        n_batches_to_process = (n_sources_this_task + args.batch_size - 1) // args.batch_size
    
    print(f"Will process {n_batches_to_process} batches for source range {start_source}-{end_source}")
    
    # Track memory across batches for diagnostics
    memory_history = []
    
    # Detect GPU limit once for all memory monitoring
    # gpu_limit = get_gpu_memory_limit()
    
    # print_memory_status("BEFORE BATCH LOOP", None, gpu_limit)
    
    batches_processed = 0  # Track number of batches we've actually processed
    
    for batch_idx, (batch_df, start_idx, end_idx) in enumerate(loader):
        # Skip batches outside our assigned source range (for multi-node)
        # Check this FIRST to avoid printing debug messages for skipped batches
        if end_idx <= start_source or start_idx >= end_source:
            continue  # This batch is outside our chunk

        # Global batch index derived from start_idx so filenames are identical
        # whether we use the fast-path loader (local batch_idx 0..N) or the
        # legacy full-file loader (batch_idx already equals start_idx // batch_size).
        global_batch_idx = start_idx // args.batch_size

        # Debug: mark start of batch iteration (only for batches we're actually processing)
        print(f"\n[DEBUG] Processing batch {batches_processed} (global batch_idx={global_batch_idx}, indices {start_idx}-{end_idx})", flush=True)
        # sys.stdout.flush()
        
        # Force GC immediately after loader yields batch (frees previous batch's parquet data)
        # import gc
        # gc.collect()
        
        # Clip batch to our assigned range if it overlaps
        if start_idx < start_source or end_idx > end_source:
            # Partial overlap - need to filter batch_df
            local_start = max(0, start_source - start_idx)
            local_end = min(len(batch_df), end_source - start_idx)
            batch_df = batch_df.iloc[local_start:local_end].copy()
            if len(batch_df) == 0:
                continue
            # Update indices for logging
            start_idx = max(start_idx, start_source)
            end_idx = min(end_idx, end_source)
        
        # print_memory_status(f"After loader yield (batch {batch_idx})", None, gpu_limit)
        
        if len(batch_df) == 0:
            print(f"\nBatch {global_batch_idx} is empty after filtering, skipping.")
            continue

        # --resume: skip this batch if its result file already exists on disk
        if args.resume:
            expected_suffix = f'batch{global_batch_idx}_start{start_idx}'
            expected_results_file = save_dir / f'PAE_results_{expected_suffix}_{args.datestr}.npz'
            if expected_results_file.exists():
                print(f"[RESUME] Batch {global_batch_idx} (start={start_idx}) already saved — skipping.",
                      flush=True)
                batches_processed += 1
                if batches_processed >= n_batches_to_process:
                    print(f"\nAll {batches_processed} batches already complete.")
                    break
                continue

        print(f"[DEBUG] About to call process_batch with {len(batch_df)} sources", flush=True)
        # sys.stdout.flush()
        
        try:
            results = process_batch(
                batch_df,
                global_batch_idx,
                start_idx,
                PAE_COSMOS,
                cfg,
                filter_set_name,
                wave_obs,
                save_dir,
                args.datestr,
                sampling_batch_size=args.sampling_batch_size,
                gpu_limit_gb=gpu_limit,
                use_weighted_mean=args.use_weighted_mean,
                abs_norm=not args.no_abs_norm,
                channel_mask=channel_mask,
            )
            
            # Debug: confirm we returned from process_batch
            print(f"\n[DEBUG] Batch {batches_processed} completed, continuing to next batch...", flush=True)
            sys.stdout.flush()
            
            # Increment counter for batches we've actually processed
            batches_processed += 1
            
            # Check if we've processed enough batches BEFORE fetching next batch from loader
            # (avoids hanging on loader.__next__() when we're done)
            if batches_processed >= n_batches_to_process:
                print(f"\nProcessed {batches_processed} batches, stopping.")
                break
            
            # Explicit memory cleanup after each batch
            # NOTE: If cleanup hangs, you can disable this section
            # print("\n[CLEANUP] Releasing batch memory...")
            # del results
            # del batch_df
            
            # Force Python garbage collection
            # import gc
            # gc.collect()
            
            # Try to return memory to OS (Python 3.13+ or with malloc_trim on Linux)
            # try:
            #     import ctypes
            #     libc = ctypes.CDLL("libc.so.6")
            #     libc.malloc_trim(0)  # Force glibc to return freed memory to OS
            #     print("[CLEANUP] Called malloc_trim to return memory to OS")
            # except Exception as e:
            #     pass  # malloc_trim not available on this system
            
            # Also collect garbage after every batch
            # gc.collect()
            
            # print_memory_status(f"After cleanup (batch {batch_idx})", None, gpu_limit)
            
            # Track memory for diagnostics
            # mem = get_detailed_memory_info()
            # if mem:
            #     memory_history.append({
            #         'batch': batch_idx,
            #         'rss_gb': mem['rss_gb'],
            #         'vms_gb': mem['vms_gb']
            #     })
                
        except Exception as e:
            import traceback
            print(f"\n\u274c ERROR processing batch {global_batch_idx}:")
            print(f"   {str(e)}")
            print("   Full traceback:")
            traceback.print_exc()
            if args.stop_on_error:
                raise
            else:
                print("   Continuing to next batch...")
                continue
    
    t_total = time.time() - t_total_start
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Total runtime: {t_total/60:.2f} minutes")
    print(f"Processed {batches_processed} batches")
    print(f"Results saved to: {save_dir}")
    print("=" * 70)
    
    # Print memory growth summary
    if memory_history:
        print("\n" + "=" * 70)
        print("MEMORY USAGE SUMMARY")
        print("=" * 70)
        print(f"{'Batch':<8} {'RSS (GB)':<12} {'VMS (GB)':<12} {'RSS Δ (GB)':<12}")
        print("-" * 70)
        
        for i, mem in enumerate(memory_history):
            delta = mem['rss_gb'] - memory_history[0]['rss_gb'] if i > 0 else 0.0
            print(f"{mem['batch']:<8} {mem['rss_gb']:<12.2f} {mem['vms_gb']:<12.2f} {delta:+12.2f}")
        
        # Compute average memory growth per batch
        if len(memory_history) > 1:
            total_growth = memory_history[-1]['rss_gb'] - memory_history[0]['rss_gb']
            avg_growth_per_batch = total_growth / (len(memory_history) - 1)
            print("-" * 70)
            print(f"Total memory growth: {total_growth:+.2f} GB over {len(memory_history)} batches")
            print(f"Average per batch: {avg_growth_per_batch:+.3f} GB")
            
            if avg_growth_per_batch > 0.5:
                print("\n  WARNING: Significant memory leak detected!")
                print("   Memory is growing by >0.5 GB per batch.")
                print("   Check for accumulating arrays, unclosed figures, or cached compilations.")
        
        print("=" * 70)

    # Optionally collate per-batch PAE result files into one combined file
    if args.collate_results:
        print('\nCollating per-batch PAE result files...')
        pattern = 'PAE_results_*.npz'
        files = sorted(list(save_dir.glob(pattern)))
        if len(files) == 0:
            print('No per-batch result files found to collate.')
        else:
            all_dicts = [dict(np.load(str(f), allow_pickle=True)) for f in files]
            keys = set().union(*[d.keys() for d in all_dicts])
            merged = {}
            for k in keys:
                parts = [d.get(k, None) for d in all_dicts]
                parts_nonnull = [p for p in parts if p is not None]
                if len(parts_nonnull) == 0:
                    merged[k] = None
                    continue
                try:
                    merged[k] = np.concatenate(parts_nonnull, axis=0)
                except Exception:
                    try:
                        merged[k] = np.vstack(parts_nonnull)
                    except Exception:
                        merged[k] = np.array(parts_nonnull, dtype=object)

            outpath = save_dir / f'PAE_results_combined_{args.datestr}.npz'
            save_dict = {kk:vv for kk,vv in merged.items() if vv is not None}
            print(f'Saving combined results to {outpath}')
            np.savez_compressed(str(outpath), **save_dict)
            print('Collation complete.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run redshift estimation on large parquet files in batches',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output
    parser.add_argument('--parquet-file', type=str, 
                       default='/pscratch/sd/r/rmfeder/data/l3_data/selection_df.parquet',
                       help='Path to parquet file')
    parser.add_argument('--filter-set', type=str, default='SPHEREx_filter_306',
                       help='Filter set name')
    parser.add_argument('--datestr', type=str, default='122725_batched',
                       help='Date string for output files')
    parser.add_argument('--use-weighted-mean', action='store_true',
                       help='Use weighted mean fluxes instead of median fluxes when loading data')
    parser.add_argument('--no-abs-norm', action='store_true',
                       help='Disable absolute normalization (raw fluxes/errors, amplitude marginalization only)')
    parser.add_argument('--random-seed', type=int, default=None,
                       help='Random seed for reproducibility. If None, uses system time.')
    
    # Batching
    parser.add_argument('--batch-size', type=int, default=10000,
                       help='Sources per batch for parquet loading (outer batch, 10k ≈ 10-15 MB memory)')
    parser.add_argument('--sampling-batch-size', type=int, default=800,
                       help='Total sources processed in parallel across all GPUs (inner batch). '
                            'With 4 GPUs: each GPU gets sampling_batch_size ÷ 4. '
                            'Default 800 total = 200 per GPU with 4 GPUs.')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum number of batches to process (for testing)')
    parser.add_argument('--resume', action='store_true', default=False,
                       help='Skip batches whose output files already exist in save_dir. '
                            'Lets you restart a partial run without reprocessing completed batches.')
    parser.add_argument('--collate-results', action='store_true',
                       help='After run, collate per-batch PAE result files into one combined file')
    
    # Multi-node parallelization
    parser.add_argument('--start-source', type=int, default=None,
                       help='Start source index (for multi-node: process sources [start_source, end_source))')
    parser.add_argument('--end-source', type=int, default=None,
                       help='End source index (exclusive, for multi-node parallelization)')
    parser.add_argument('--job-array-task-id', type=int, default=None,
                       help='SLURM_ARRAY_TASK_ID (auto-compute start/end from this and --sources-per-task)')
    parser.add_argument('--sources-per-task', type=int, default=10000,
                       help='Sources per job array task (used with --job-array-task-id)')
    
    # Filtering
    parser.add_argument('--filter-specz', action='store_true',
                       help='Only process sources with spectroscopic redshift')
    parser.add_argument('--z-min', type=float, default=None,
                       help='Minimum redshift')
    parser.add_argument('--z-max', type=float, default=None,
                       help='Maximum redshift')
    parser.add_argument('--chi2-red-max', type=float, default=None,
                       help='Maximum reduced chi-squared for SED fit quality')
    
    # Init-reinit defaults to True for production use
    parser.add_argument('--no-init-reinit', dest='init_reinit', action='store_false',
                       help='Disable chain reinitialization after burn-in (default: enabled)')
    parser.add_argument('--init-reinit', dest='init_reinit', action='store_true',
                       help='Enable chain reinitialization after burn-in (default: enabled)')
    parser.set_defaults(init_reinit=True)
    
    parser.add_argument('--use_robust_reinit', action='store_true',
                       help='Use robust reinitialization logic (default: False)')
    # Redshift prior
    parser.add_argument('--prior-type', type=int, default=1,
                       help='Redshift prior type: 0=none, 1=Gaussian')
    parser.add_argument('--z0-prior', type=float, default=0.65,
                       help='Gaussian prior center')
    parser.add_argument('--sigma-prior', type=float, default=0.6,
                       help='Gaussian prior width')
    parser.add_argument('--nf-alpha', type=float, default=1.0,
                       help='Normalizing flow prior multiplier (0=no flow prior, 1=full flow prior)')
    
    # Log-redshift sampling
    parser.add_argument('--sample-log-redshift', action='store_true',
                       help='Sample ln(z) instead of z (eliminates BPZ prior singularity near z=0)')
    
    # Redshift fitting
    parser.add_argument('--fix-z', action='store_true',
                       help='Fix redshift to spec-z value (default: False, redshift is fitted)')
    
    # Performance
    parser.add_argument('--use-multicore', action='store_true',
                       help='Use multi-core parallelization')
    parser.add_argument('--n-devices', type=int, default=4,
                       help='Number of GPU devices to use')
    
    # Error handling
    parser.add_argument('--stop-on-error', action='store_true',
                       help='Stop if any batch fails (default: continue)')

    # Channel masking
    parser.add_argument('--mask-wavelength-ranges', type=str, nargs='*', default=None,
                       metavar='LMIN,LMAX',
                       help='Wavelength ranges to exclude from the likelihood (micron). '
                            'Provide one or more "lmin,lmax" pairs, e.g. 2.42,3.82. '
                            'Channels whose central wavelength falls within any range are '
                            'zero-weighted before inference. Band 4 shorthand: 2.42,3.82.')

    args = parser.parse_args()
    
    # Initialize random seed
    if args.random_seed is None:
        import time
        seed = int(time.time() * 1000000) % (2**31)
        print(f"Using time-based random seed: {seed}")
    else:
        seed = args.random_seed
        print(f"Using user-specified random seed: {seed}")
    
    # Auto-compute start/end from job array task ID if provided
    if args.job_array_task_id is not None:
        args.start_source = args.job_array_task_id * args.sources_per_task
        args.end_source = (args.job_array_task_id + 1) * args.sources_per_task
        print(f"Job array task {args.job_array_task_id}: processing sources {args.start_source}-{args.end_source}")
    
    # Set restrictive permissions for all output files
    import os
    os.umask(0o077)  # New files will be owner-only (600), directories owner-only (700)
    
    main(args)
