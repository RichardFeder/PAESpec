#!/usr/bin/env python
"""
Generate per-source reconstruction plots for PAE redshift estimation results.

Creates detailed diagnostic plots for individual sources, automatically categorizing
them as "good_fits" or "bad_fits" based on chi-squared and z-score metrics.

Usage:
    # Using datestr (auto-detects files)
    python scripts/generate_source_reconstructions.py --datestr multicore_test_16k_123025
    
    # Using explicit file paths
    python scripts/generate_source_reconstructions.py \\
        --result-file path/to/PAE_results_combined.npz \\
        --sample-file path/to/PAE_samples_combined.npz \\
        --data-file path/to/observed_data.npz
    
    # Customize selection criteria
    python scripts/generate_source_reconstructions.py --datestr XXX \\
        --n-good 50 --n-bad 50 \\
        --chi2-percentile 90 --zscore-threshold 3.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import jax.numpy as jnp

from config import scratch_basepath
from models.pae_jax import initialize_PAE, set_up_pae_wrapper
from data_proc.dataloader_jax import SPHERExData
from diagnostics.diagnostics_jax import proc_spec_post
from visualization.result_plotting_fns import (
    plot_post_spec_recon, 
    make_corner_plot, 
    plot_redshift_posterior
)


def load_batch_samples(result_dir, source_idx, batch_size=800):
    """
    Load samples for a specific source from batched sample files.
    
    Parameters:
    -----------
    result_dir : Path
        Directory containing batch result files
    source_idx : int
        Global index of source
    batch_size : int
        Number of sources per batch
    
    Returns:
    --------
    samples : array
        MCMC samples for the source
    batch_info : dict
        Information about which batch was loaded
    """
    # Determine which batch this source belongs to
    batch_idx = source_idx // batch_size
    local_idx = source_idx % batch_size
    
    # Find the batch sample file
    batch_pattern = f'PAE_samples_batch{batch_idx}_*.npz'
    batch_files = list(result_dir.glob(batch_pattern))
    
    if not batch_files:
        raise FileNotFoundError(f"No sample file found for batch {batch_idx} (source {source_idx})")
    
    # Load the batch file
    batch_file = batch_files[0]
    print(f"  Loading samples for source {source_idx} from batch {batch_idx} file: {batch_file}")
    with np.load(batch_file) as data:
        all_samples = data['all_samples']
        
        # Check if local_idx is within bounds
        if local_idx >= all_samples.shape[0]:
            raise IndexError(f"Source index {source_idx} (local {local_idx}) out of bounds for batch {batch_idx}")
        
        samples = all_samples[local_idx]
    
    batch_info = {
        'batch_idx': batch_idx,
        'local_idx': local_idx,
        'batch_file': str(batch_file)
    }
    
    return samples, batch_info


def identify_good_bad_sources(result_filepath, n_good=50, n_bad=50, 
                              chi2_percentile=90, zscore_threshold=3.0):
    """
    Identify good and bad fit sources based on chi2 and z-score metrics.
    
    Parameters:
    -----------
    result_filepath : str
        Path to PAE results .npz file
    n_good : int
        Number of good sources to select
    n_bad : int
        Number of bad sources to select
    chi2_percentile : float
        Percentile threshold for high chi2 (for bad fits)
    zscore_threshold : float
        Threshold for |z-score| (for bad fits)
    
    Returns:
    --------
    good_indices : array
        Indices of good fit sources
    bad_indices : array
        Indices of bad fit sources
    stats : dict
        Dictionary with selection statistics
    """
    # Load results
    res = np.load(result_filepath)
    
    # Compute chi-squared (if available in results)
    # Otherwise estimate from residuals
    if 'chi2' in res.files:
        chi2_values = res['chi2']
    else:
        # Estimate chi2 from z-score and other metrics
        pae_bias = res['z_med'] - res['ztrue']
        pae_unc = 0.5 * (res['err_low'] + res['err_high'])
        pae_zscore = pae_bias / pae_unc
        # Use z-score as proxy for fit quality
        chi2_values = pae_zscore**2
    
    # Compute z-scores
    pae_bias = res['z_med'] - res['ztrue']
    pae_unc = 0.5 * (res['err_low'] + res['err_high'])
    pae_zscore = pae_bias / pae_unc
    
    # Identify bad fits
    chi2_thresh = np.percentile(chi2_values, chi2_percentile)
    high_chi2_mask = chi2_values > chi2_thresh
    high_zscore_mask = np.abs(pae_zscore) > zscore_threshold
    bad_mask = high_chi2_mask | high_zscore_mask
    
    # Identify good fits
    chi2_good_thresh = np.percentile(chi2_values, 25)  # Lower quartile
    zscore_good_thresh = 1.0
    low_chi2_mask = chi2_values < chi2_good_thresh
    low_zscore_mask = np.abs(pae_zscore) < zscore_good_thresh
    good_mask = low_chi2_mask & low_zscore_mask
    
    # Get indices
    bad_indices = np.where(bad_mask)[0]
    good_indices = np.where(good_mask)[0]
    
    # Handle insufficient bad sources: take worst N by |zscore|
    if len(bad_indices) < n_bad:
        print(f"  ⚠ Only {len(bad_indices)} sources meet bad fit criteria (|zscore| > {zscore_threshold})")
        print(f"    Taking worst {n_bad} redshift offenders by |zscore| instead...")
        
        # Sort by absolute zscore (descending) and take worst N
        abs_zscore_order = np.argsort(np.abs(pae_zscore))[::-1]
        bad_indices = abs_zscore_order[:min(n_bad, len(pae_zscore))]
    elif len(bad_indices) > n_bad:
        # Random selection if more sources than requested
        np.random.seed(42)
        bad_indices = np.random.choice(bad_indices, n_bad, replace=False)
    
    # Handle insufficient good sources: take best N by |zscore|
    if len(good_indices) < n_good:
        print(f"  ⚠ Only {len(good_indices)} sources meet good fit criteria")
        print(f"    Taking best {n_good} sources by |zscore| instead...")
        
        # Sort by absolute zscore (ascending) and take best N
        abs_zscore_order = np.argsort(np.abs(pae_zscore))
        good_indices = abs_zscore_order[:min(n_good, len(pae_zscore))]
    elif len(good_indices) > n_good:
        # Random selection if more sources than requested
        np.random.seed(43)
        good_indices = np.random.choice(good_indices, n_good, replace=False)
    
    # Sort for consistent ordering
    bad_indices = np.sort(bad_indices)
    good_indices = np.sort(good_indices)
    
    stats = {
        'n_total': len(chi2_values),
        'n_good_available': np.sum(good_mask),
        'n_bad_available': np.sum(bad_mask),
        'n_good_selected': len(good_indices),
        'n_bad_selected': len(bad_indices),
        'chi2_threshold': chi2_thresh,
        'chi2_good_threshold': chi2_good_thresh,
        'zscore_threshold': zscore_threshold,
        'median_chi2_good': np.median(chi2_values[good_indices]) if len(good_indices) > 0 else np.nan,
        'median_chi2_bad': np.median(chi2_values[bad_indices]) if len(bad_indices) > 0 else np.nan,
        'median_zscore_good': np.median(np.abs(pae_zscore[good_indices])) if len(good_indices) > 0 else np.nan,
        'median_zscore_bad': np.median(np.abs(pae_zscore[bad_indices])) if len(bad_indices) > 0 else np.nan,
    }
    
    return good_indices, bad_indices, stats


def plot_source_reconstruction(PAE_obj, spherex_dat, samples, source_idx,
                               save_dir, source_type='good', fix_z=False,
                               figsize=(7, 4), corner_figsize=(7, 7),
                               umin=-4.0, umax=4.0, verbose=False, dpi=150,
                               batch_info=None):
    """
    Generate reconstruction plots for a single source.
    
    Parameters:
    -----------
    PAE_obj : PAE model object
        Trained PAE model
    spherex_dat : SPHERExData
        Data container with observed spectra
    samples : array
        MCMC samples for this specific source
    source_idx : int
        Index of source to plot
    save_dir : Path
        Directory to save figures
    source_type : str
        'good' or 'bad' for subdirectory naming
    fix_z : bool
        Whether redshift was fixed during sampling
    figsize : tuple
        Figure size for spectral reconstruction
    corner_figsize : tuple
        Figure size for corner plot
    umin, umax : float
        Limits for latent space corner plots
    verbose : bool
        Print debug information
    batch_info : dict, optional
        Information about batch loading
    
    Returns:
    --------
    success : bool
        Whether plotting succeeded
    """
    try:
        # Entry log for plotting
        if verbose:
            bstr = f" [batch {batch_info['batch_idx']}, local {batch_info['local_idx']}]" if batch_info else ""
            print(f"  -> Starting plot for source {source_idx}{bstr} (type={source_type})")

        # Extract data for this source
        src_idxs = spherex_dat.src_idxs
        actual_idx = src_idxs[source_idx]
        
        norms_array = np.array(spherex_dat.norms)[:,0]
        norm = norms_array[actual_idx]
        
        spec_obs = spherex_dat.all_spec_obs[actual_idx]
        weight = spherex_dat.weights[actual_idx]
        ztrue = spherex_dat.redshift[actual_idx]  # Spectroscopic redshift from real data
        flux_unc = spherex_dat.all_flux_unc[actual_idx]
        srcid_obs = spherex_dat.srcid_obs[actual_idx]
        
        # For real data, we don't have noiseless spectra
        # Only spectroscopic redshift (ztrue) is used for comparison
        spec_true = None
        
        if verbose:
            print(f"  Using spectroscopic redshift: z_spec = {ztrue:.4f}")
        
        # Process posterior samples
        if fix_z:
            redshift_fix = ztrue
        else:
            redshift_fix = None

        if verbose:
            print("  Computing posterior and reconstructions...")

        # samples are already provided for this specific source
        recon_x, logL, redshift_post = proc_spec_post(
            PAE_obj, samples, spec_obs, weight, 
            combine_chains=True, burn_in=1000, thin_fac=1, 
            redshift_fix=redshift_fix
        )
        
        # Rescale reconstructions
        recon_x *= norm
        if verbose:
            print("  Reconstruction computed; preparing to save figures...")
        
        # Compute statistics
        spec_pcts = [np.percentile(recon_x, pct, axis=0) for pct in [5, 16, 68, 95]]
        spec_68pct_range = [spec_pcts[1], spec_pcts[2]]
        spec_95pct_range = [spec_pcts[0], spec_pcts[3]]
        spec_med = np.mean(recon_x, axis=0)
        
        # Determine y-axis limits based on observed spectrum
        ylim = [-30, max(100, 1.5*np.max(spec_obs*norm))]
        
        # Create subdirectory (good_fits/ or bad_fits/)
        subdir = save_dir / f"{source_type}_fits"
        subdir.mkdir(parents=True, exist_ok=True)
        
        # 1. Spectral reconstruction plot
        fig_recon = plot_post_spec_recon(
            PAE_obj.wave_obs, norm*spec_obs, 
            MAP_spec=None, spec_truth=None, flux_unc=norm*flux_unc,
            spec_recon_med=spec_med, 
            spec_68pct_interval=spec_68pct_range, 
            spec_95pct_interval=spec_95pct_range,
            redshift_post=redshift_post, redshift_true=ztrue, 
            recon_indiv=None, ylim=ylim,
            bbox_to_anchor=[0.1, 1.3], figsize=figsize, legend_fs=11, 
            alph=0.2/np.sqrt(len(PAE_obj.wave_obs)/102),
            post_color='C3', color='k', ztrue_color='k'
        )
        
        recon_filename = subdir / f"source_{source_idx:05d}_reconstruction.png"
        if verbose:
            print(f"    Saving reconstruction plot -> {recon_filename}")
        fig_recon.savefig(recon_filename, bbox_inches='tight')
        plt.close(fig_recon)
        if verbose:
            print("    Saved reconstruction plot")
        
        # 2. Corner plot
        nlatent = PAE_obj.nlatent
        param_names = [f'$u_{i+1}$' for i in range(nlatent)]
        if not fix_z:
            param_names.append('$z_{gal}$')
            ztrue_plot = ztrue
        else:
            ztrue_plot = None
        
        all_samp_plot = samples.reshape(-1, len(param_names))
        fig_corner = make_corner_plot(
            all_samp_plot, param_names=param_names, 
            figsize=corner_figsize, redshift_true=ztrue_plot, 
            smooth=0.5, nbin=20, title_fontsize=9, fix_z=fix_z, 
            umin=umin, umax=umax, levels=[0.68, 0.95], dz=0.03, title_fmt='.2f'
        )
        
        corner_filename = subdir / f"source_{source_idx:05d}_corner.png"
        if verbose:
            print(f"    Saving corner plot -> {corner_filename}")
        fig_corner.savefig(corner_filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig_corner)
        if verbose:
            print("    Saved corner plot")
        
        # 3. Redshift posterior plot (if not fixed)
        if not fix_z:
            fig_z = plot_redshift_posterior(
                redshift_post, redshift_use=ztrue, 
                include_pcts=True, include_mean=True, include_median=False,
                figsize=(4, 3), nbins=40, bbox_to_anchor=[1.0, 1.2],
                dz_pad=0.01, color='C3', fillcolor='C3', ztrue_color='k'
            )
            
            zposterior_filename = subdir / f"source_{source_idx:05d}_zposterior.png"
            if verbose:
                print(f"    Saving redshift posterior -> {zposterior_filename}")
            fig_z.savefig(zposterior_filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig_z)
            if verbose:
                print("    Saved redshift posterior")
        
        if verbose:
            batch_str = f" [batch {batch_info['batch_idx']}, local {batch_info['local_idx']}]" if batch_info else ""
            print(f"  ✓ Source {source_idx}{batch_str} ({source_type}): z_true={ztrue:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error plotting source {source_idx}: {str(e)}")
        return False


def load_model_config_from_samples(sample_filepath):
    """
    Try to load model configuration from sample file metadata.
    Falls back to checking result filename or using defaults.
    
    Parameters:
    -----------
    sample_filepath : str
        Path to samples file
    
    Returns:
    --------
    dict : Model configuration or None if not found
    """
    try:
        with np.load(sample_filepath, allow_pickle=True) as data:
            # Check if metadata was saved with samples
            if 'model_config' in data.files:
                config = data['model_config'].item()
                print("  ✓ Loaded model config from sample file metadata")
                return config
    except:
        pass
    
    # Fallback: try to infer from filename
    # Sample files typically named: PAE_samples_combined_<datestr>.npz or similar
    filename = Path(sample_filepath).stem
    
    # Common pattern: could contain nlatent info in parent directory or filename
    # For now, return None and use defaults
    return None


def generate_source_plots(result_filepath, sample_filepath, data_filepath,
                          pae_config, save_dir, n_good=50, n_bad=50,
                          chi2_percentile=90, zscore_threshold=3.0,
                          fix_z=False, verbose=True, batch_size=800):
    """
    Main function to generate all source reconstruction plots.
    
    Parameters:
    -----------
    result_filepath : str
        Path to PAE results file
    sample_filepath : str
        Path to PAE samples file
    data_filepath : str
        Path to observed data file (not currently used, kept for compatibility)
    pae_config : dict
        Configuration for PAE model (nlatent, sig_level_norm, run_name, etc.)
        These should match the model used during the redshift run!
    save_dir : Path
        Root directory for saving figures
    n_good, n_bad : int
        Number of good/bad sources to plot
    chi2_percentile : float
        Percentile for chi2 threshold
    zscore_threshold : float
        Threshold for z-score
    fix_z : bool
        Whether redshift was fixed
    verbose : bool
        Print progress
    
    Returns:
    --------
    stats : dict
        Summary statistics
    """
    print(f"\n{'='*70}")
    print("Generating per-source reconstruction plots")
    print(f"{'='*70}")
    
    # Step 1: Identify good and bad sources
    print("\n[1/4] Identifying good and bad fit sources...")
    good_indices, bad_indices, selection_stats = identify_good_bad_sources(
        result_filepath, n_good=n_good, n_bad=n_bad,
        chi2_percentile=chi2_percentile, zscore_threshold=zscore_threshold
    )
    
    print(f"  Total sources: {selection_stats['n_total']:,}")
    print(f"  Good fits available: {selection_stats['n_good_available']:,}")
    print(f"  Bad fits available: {selection_stats['n_bad_available']:,}")
    print(f"  Good fits selected: {selection_stats['n_good_selected']}")
    print(f"  Bad fits selected: {selection_stats['n_bad_selected']}")
    print(f"  Chi2 threshold (bad): {selection_stats['chi2_threshold']:.2f}")
    print(f"  |Z-score| threshold: {selection_stats['zscore_threshold']:.2f}")
    
    # Try to load model config from samples (for future compatibility)
    saved_config = load_model_config_from_samples(sample_filepath)
    if saved_config:
        print("\n  Note: Using saved model configuration from sample file")
        # Override with saved config if available
        pae_config.update(saved_config)
    
    # Step 2: Load PAE model and data using set_up_pae_wrapper
    # This matches how the model was originally set up during training
    print("\n[2/4] Loading PAE model and data...")
    print(f"  Model: {pae_config['run_name']}")
    print(f"  Filter set: {pae_config['filter_set_name']}")
    print(f"  nlatent: {pae_config['nlatent']}")
    print(f"  sig_level_norm: {pae_config['sig_level_norm']}")
    
    # Determine number of bands from filter set name
    if '306' in pae_config['filter_set_name']:
        nbands_obs = 306
    elif '408' in pae_config['filter_set_name']:
        nbands_obs = 408
    else:
        nbands_obs = 102  # default
    
    print(f"  Initializing PAE model with {nbands_obs} bands...")
    PAE_obj, wave_obs, dat_obs, property_cat_df_restframe, \
        property_cat_df_obs, params = set_up_pae_wrapper(
            nlatent=pae_config['nlatent'],
            sig_level_norm=pae_config['sig_level_norm'],
            sel_str=pae_config.get('sel_str', 'all'),
            abs_norm=True,
            run_name=pae_config['run_name'],
            redshift_in_flow=False,
            with_ext_phot=pae_config.get('with_ext_phot', False),
            inference_dtype=jnp.float32,
            load_rf_dat=False,
            load_obs_dat=True,
            filter_set_name=pae_config['filter_set_name'],
            nbands_obs=nbands_obs
        )
    print(f"  ✓ PAE model initialized successfully")
    print(f"  ✓ Model and data loaded successfully")
    print(f"  Number of observed bands: {len(wave_obs)}")
    
    # Step 3: Prepare for batch sample loading
    print("\n[3/4] Preparing batch sample loading...")
    
    # Determine if we have collated samples or batch files
    result_dir = Path(result_filepath).parent
    
    # Check if sample_filepath is actually a collated file
    use_collated = False
    if sample_filepath and Path(sample_filepath).exists():
        try:
            with np.load(sample_filepath) as test_data:
                all_samples = test_data['all_samples']
                use_collated = True
                print(f"  ✓ Using collated samples: {all_samples.shape}")
        except:
            pass
    
    if not use_collated:
        # Count batch files to estimate total sources
        batch_files = list(result_dir.glob('PAE_samples_batch*_*.npz'))
        print(f"  ✓ Found {len(batch_files)} batch sample files")
        print(f"  Will load samples on-demand from batch files (batch_size={batch_size})")
        all_samples = None  # Signal to load per-source
    
    # Create SPHERExData object from the loaded data
    spherex_dat = SPHERExData.from_prep(
        dat_obs,
        property_cat_df_restframe,
        property_cat_df_obs,
        phot_snr_min=pae_config.get('phot_snr_min', 50),
        phot_snr_max=pae_config.get('phot_snr_max', 300),
        zmin=pae_config.get('zmin', None),
        zmax=pae_config.get('zmax', None)
    )
    print(f"  Data sources: {len(spherex_dat.src_idxs)}")
    
    # Verify that we have enough sources
    n_sources_needed = max(max(good_indices) if len(good_indices) > 0 else 0,
                           max(bad_indices) if len(bad_indices) > 0 else 0) + 1
    if len(spherex_dat.src_idxs) < n_sources_needed:
        print(f"\n  WARNING: Data has {len(spherex_dat.src_idxs)} sources but need {n_sources_needed}")
        print(f"  This may happen if the run used different filters/SNR cuts")
        print(f"  Some sources may fail to plot")
    
    # Step 4: Generate plots
    print("\n[4/4] Generating plots...")
    
    n_success_good = 0
    n_success_bad = 0
    
    # Plot good fits
    if len(good_indices) > 0:
        print(f"\n  Plotting {len(good_indices)} good fits...")
        for i, idx in enumerate(good_indices):
            if verbose:
                print(f"    [{i+1}/{len(good_indices)}] ", end='')
            
            # Load samples for this source
            if all_samples is None:
                # Load from batch file
                try:
                    samples, batch_info = load_batch_samples(result_dir, idx, batch_size)
                except Exception as e:
                    print(f"  ✗ Could not load samples for source {idx}: {e}")
                    continue
            else:
                # Use collated samples
                samples = all_samples[idx]
                batch_info = None
            
            success = plot_source_reconstruction(
                PAE_obj, spherex_dat, samples, idx,
                save_dir, source_type='good', fix_z=fix_z,
                verbose=verbose, batch_info=batch_info
            )
            if success:
                n_success_good += 1
    
    # Plot bad fits
    if len(bad_indices) > 0:
        print(f"\n  Plotting {len(bad_indices)} bad fits...")
        for i, idx in enumerate(bad_indices):
            if verbose:
                print(f"    [{i+1}/{len(bad_indices)}] ", end='')
            
            # Load samples for this source
            if all_samples is None:
                # Load from batch file
                try:
                    samples, batch_info = load_batch_samples(result_dir, idx, batch_size)
                except Exception as e:
                    print(f"  ✗ Could not load samples for source {idx}: {e}")
                    continue
            else:
                # Use collated samples
                samples = all_samples[idx]
                batch_info = None
            
            success = plot_source_reconstruction(
                PAE_obj, spherex_dat, samples, idx,
                save_dir, source_type='bad', fix_z=fix_z,
                verbose=verbose, batch_info=batch_info
            )
            if success:
                n_success_bad += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Good fits: {n_success_good}/{len(good_indices)} plotted successfully")
    print(f"Bad fits: {n_success_bad}/{len(bad_indices)} plotted successfully")
    print(f"\nFigures saved to:")
    print(f"  {save_dir / 'good_fits'}")
    print(f"  {save_dir / 'bad_fits'}")
    print(f"{'='*70}\n")
    
    return {
        **selection_stats,
        'n_plotted_good': n_success_good,
        'n_plotted_bad': n_success_bad,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate per-source reconstruction plots for PAE results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input file specification (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--datestr', type=str,
                            help='Date string to auto-detect files')
    input_group.add_argument('--result-file', type=str,
                            help='Path to PAE results .npz file')
    
    # Additional file paths (required if not using datestr)
    parser.add_argument('--sample-file', type=str,
                       help='Path to PAE samples .npz file (required with --result-file)')
    parser.add_argument('--data-file', type=str,
                       help='[Optional] Path to observed data .npz file - usually auto-loaded from PAE config')
    
    # PAE model configuration (must match redshift run!)
    parser.add_argument('--run-name', type=str,
                       default='jax_conv1_nlatent=5_siglevelnorm=0.01_newAllen_all_091325',
                       help='Model run name')
    parser.add_argument('--filter-set', type=str, default='SPHEREx_filter_306',
                       help='Filter set name (default: SPHEREx_filter_306 for 306-band)')
    parser.add_argument('--nlatent', type=int, default=5,
                       help='Number of latent dimensions (default: 5)')
    parser.add_argument('--sig-level-norm', type=float, default=0.01,
                       help='Sigma level normalization (default: 0.01)')
    parser.add_argument('--sel-str', type=str, default='all',
                       help='Selection string (default: all)')
    parser.add_argument('--phot-snr-min', type=float, default=50,
                       help='Minimum photometric SNR (default: 50)')
    parser.add_argument('--phot-snr-max', type=float, default=300,
                       help='Maximum photometric SNR (default: 300)')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for figures (auto if using datestr)')
    
    # Selection criteria
    parser.add_argument('--n-good', type=int, default=50,
                       help='Number of good fit sources to plot (default: 50)')
    parser.add_argument('--n-bad', type=int, default=50,
                       help='Number of bad fit sources to plot (default: 50)')
    parser.add_argument('--chi2-percentile', type=float, default=90,
                       help='Percentile for chi2 threshold (default: 90)')
    parser.add_argument('--zscore-threshold', type=float, default=3.0,
                       help='Z-score threshold for bad fits (default: 3.0)')
    parser.add_argument('--batch-size', type=int, default=800,
                       help='Batch size used in redshift run (default: 800)')
    
    # Other options
    parser.add_argument('--fix-z', action='store_true',
                       help='Redshift was fixed during sampling')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')
    
    args = parser.parse_args()
    
    # Auto-detect files if using datestr
    if args.datestr:
        scratch_path = Path(scratch_basepath) / 'data' / 'pae_sample_results' / 'MCLMC'
        
        # Check both batched and mock directories
        batched_dir = scratch_path / 'batched' / args.datestr
        mock_dir = scratch_path / f'mock_{args.datestr}'
        
        if batched_dir.exists():
            result_dir = batched_dir
        elif mock_dir.exists():
            result_dir = mock_dir
        else:
            print(f"Error: Could not find results directory for datestr '{args.datestr}'")
            print(f"  Checked: {batched_dir}")
            print(f"  Checked: {mock_dir}")
            sys.exit(1)
        
        # Find result file
        result_files = list(result_dir.glob('PAE_results_combined_*.npz'))
        if not result_files:
            print(f"Error: No result file found in {result_dir}")
            sys.exit(1)
        result_filepath = str(result_files[0])
        
        # Find sample file (collated or batch files)
        sample_files = list(result_dir.glob('PAE_samples_combined_*.npz'))
        if sample_files:
            sample_filepath = str(sample_files[0])
            print(f"Found collated samples: {Path(sample_filepath).name}")
        else:
            # Check for batch files
            batch_files = list(result_dir.glob('PAE_samples_batch*_*.npz'))
            if not batch_files:
                print(f"Error: No sample files found in {result_dir}")
                print(f"  Looked for: PAE_samples_combined_*.npz or PAE_samples_batch*_*.npz")
                sys.exit(1)
            # Use None to signal batch loading
            sample_filepath = None
            print(f"Found {len(batch_files)} batch sample files (will load on-demand)")
        
        # Try to load run configuration from saved params
        config_file = result_dir / 'run_params.npz'
        config_loaded = False
        
        if config_file.exists():
            print(f"\nLoading configuration from: {config_file.name}")
            try:
                saved_config = np.load(config_file, allow_pickle=True)
                
                # Extract PAE config from saved params
                pae_config = {
                    'run_name': str(saved_config['run_name']),
                    'filter_set_name': str(saved_config['filter_set_name']),
                    'nlatent': int(saved_config['nlatent']),
                    'sig_level_norm': float(saved_config['sig_level_norm']),
                }
                
                # Override batch_size from saved config (command line default is 800)
                if 'batch_size' in saved_config.files:
                    args.batch_size = int(saved_config['batch_size'])
                
                # Override fix_z from saved config if not set on command line
                # args.fix_z defaults to False from action='store_true'
                if 'fix_z' in saved_config.files and not args.fix_z:
                    args.fix_z = bool(saved_config['fix_z'])
                
                config_loaded = True
                print("  ✓ Loaded PAE configuration from run_params.npz:")
                print(f"    run_name: {pae_config['run_name']}")
                print(f"    filter_set: {pae_config['filter_set_name']}")
                print(f"    nlatent: {pae_config['nlatent']}")
                print(f"    sig_level_norm: {pae_config['sig_level_norm']}")
                print(f"    batch_size: {args.batch_size}")
                print(f"    fix_z: {args.fix_z}")
                
            except Exception as e:
                print(f"  ⚠ Error loading run_params.npz: {e}")
                print("  Falling back to command-line arguments")
                config_loaded = False
        else:
            print(f"\n⚠ No run_params.npz found - using command-line arguments")
            print("  (This may cause issues if parameters don't match the original run)")
            config_loaded = False
        
        # For data file, we need to look in the original data location
        # This might need to be specified based on the actual run configuration
        # For now, we'll use a placeholder that needs to be configured
        data_filepath = None  # Will be loaded from config if needed
        
        # Set output directory
        if not args.output_dir:
            fig_base = Path(scratch_basepath) / 'figures' / 'redshift_validation' / args.datestr
            args.output_dir = str(fig_base)
            
    else:
        # Using explicit file paths
        if not args.sample_file:
            print("Error: --sample-file is required when using --result-file")
            sys.exit(1)
        
        result_filepath = args.result_file
        sample_filepath = args.sample_file
        data_filepath = args.data_file  # Optional
        config_loaded = False  # Using explicit paths, no config loaded
        
        if not args.output_dir:
            args.output_dir = str(Path(result_filepath).parent.parent.parent / 'figures' / 'redshift_validation')
    
    # Verify files exist
    if not Path(result_filepath).exists():
        print(f"Error: Result file not found: {result_filepath}")
        sys.exit(1)
    if sample_filepath and not Path(sample_filepath).exists():
        print(f"Error: Sample file not found: {sample_filepath}")
        sys.exit(1)
    
    # Create output directory
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Build PAE configuration
    # If config was loaded from run_params.npz, pae_config has 4 keys
    # We always need to add the additional params
    if not (args.datestr and config_loaded):
        # Build from command-line arguments (no config loaded)
        # These must match the settings used during the original redshift run!
        pae_config = {
            'run_name': args.run_name,
            'filter_set_name': args.filter_set,
            'nlatent': args.nlatent,
            'sig_level_norm': args.sig_level_norm,
        }
    
    # Always add these params (either to loaded config or command-line config)
    pae_config['sel_str'] = args.sel_str
    pae_config['phot_snr_min'] = args.phot_snr_min
    pae_config['phot_snr_max'] = args.phot_snr_max
    pae_config['with_ext_phot'] = False
    
    print(f"\n{'='*70}")
    print("PAE Model Configuration")
    print(f"{'='*70}")
    print(f"run_name:        {pae_config['run_name']}")
    print(f"filter_set:      {pae_config['filter_set_name']}")
    print(f"nlatent:         {pae_config['nlatent']}")
    print(f"sig_level_norm:  {pae_config['sig_level_norm']}")
    print(f"sel_str:         {pae_config['sel_str']}")
    print(f"\nIMPORTANT: These must match the model used in your redshift run!")
    print(f"For multicore_test_16k_wf_123025, use: --filter-set SPHEREx_filter_306")
    print(f"{'='*70}")
    
    # Generate plots
    stats = generate_source_plots(
        result_filepath, sample_filepath, data_filepath,
        pae_config, save_dir,
        n_good=args.n_good, n_bad=args.n_bad,
        chi2_percentile=args.chi2_percentile,
        zscore_threshold=args.zscore_threshold,
        fix_z=args.fix_z,
        verbose=args.verbose,
        batch_size=args.batch_size
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
