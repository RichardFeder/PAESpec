#!/usr/bin/env python
"""
Generate summary plots and diagnostics for PAE redshift estimation results.

This script can be run standalone or automatically after redshift estimation
to produce comprehensive diagnostic plots and statistics.

Usage:
    # Single result file
    python scripts/generate_redshift_plots.py --result-file path/to/PAE_results_combined_XXX.npz
    
    # Automatic detection from datestr
    python scripts/generate_redshift_plots.py --datestr multicore_test_16k_123025
    
    # With custom output directory
    python scripts/generate_redshift_plots.py --result-file file.npz --output-dir ./custom_figures
    
    # Skip showing plots (only save)
    python scripts/generate_redshift_plots.py --result-file file.npz --no-show
"""

import sys
import os
import subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

from config import scratch_basepath
from diagnostics.diagnostics_jax import compute_redshift_stats
from utils.utils_jax import make_plotstr_count


# =============================================================================
# MCLMC TUNING PARAMETER DIAGNOSTICS
# =============================================================================

def compute_mclmc_tuning_stats(tuned_L, tuned_step_size):
    """
    Compute diagnostics for MCLMC tuning parameters across chains.
    
    The relative dispersion of L and step_size across chains for a single source
    can indicate whether the sampler found consistent geometry. High dispersion
    may indicate pathological posteriors or convergence issues.
    
    Parameters:
    -----------
    tuned_L : ndarray, shape (n_sources, n_chains)
        Tuned trajectory length L for each chain of each source
    tuned_step_size : ndarray, shape (n_sources, n_chains)
        Tuned step size for each chain of each source
    
    Returns:
    --------
    dict with keys:
        'L_mean': mean L across chains (n_sources,)
        'L_std': std L across chains (n_sources,)
        'L_cv': coefficient of variation = std/mean (n_sources,)
        'step_mean': mean step_size across chains (n_sources,)
        'step_std': std step_size across chains (n_sources,)
        'step_cv': coefficient of variation for step_size (n_sources,)
        'L_step_ratio': L / step_size mean across chains (n_sources,)
        'L_step_ratio_cv': CV of L/step_size (n_sources,)
    """
    # Handle cases where data might be None or have wrong shape
    if tuned_L is None or tuned_step_size is None:
        return None
    
    tuned_L = np.atleast_2d(tuned_L)
    tuned_step_size = np.atleast_2d(tuned_step_size)
    
    # Mean and std across chains (axis=1)
    L_mean = np.nanmean(tuned_L, axis=1)
    L_std = np.nanstd(tuned_L, axis=1)
    L_cv = L_std / (L_mean + 1e-10)  # Coefficient of variation
    
    step_mean = np.nanmean(tuned_step_size, axis=1)
    step_std = np.nanstd(tuned_step_size, axis=1)
    step_cv = step_std / (step_mean + 1e-10)
    
    # L/step_size ratio (related to effective number of leapfrog steps)
    L_step_ratio = tuned_L / (tuned_step_size + 1e-10)
    L_step_ratio_mean = np.nanmean(L_step_ratio, axis=1)
    L_step_ratio_std = np.nanstd(L_step_ratio, axis=1)
    L_step_ratio_cv = L_step_ratio_std / (L_step_ratio_mean + 1e-10)
    
    return {
        'L_mean': L_mean,
        'L_std': L_std,
        'L_cv': L_cv,
        'step_mean': step_mean,
        'step_std': step_std,
        'step_cv': step_cv,
        'L_step_ratio': L_step_ratio_mean,
        'L_step_ratio_cv': L_step_ratio_cv,
        'n_chains': tuned_L.shape[1] if tuned_L.ndim > 1 else 1
    }


def plot_mclmc_diagnostics(result_filepath, save_dir=None, show_plots=False, use_hexbin=True):
    """
    Generate diagnostic plots for MCLMC tuning parameters.
    
    This creates standalone figures analyzing:
    1. Distribution of tuned L and step_size
    2. Correlation between tuning parameter dispersion and redshift errors
    3. Comparison of tuning stability for good vs bad fits
    
    Parameters:
    -----------
    result_filepath : str
        Path to the PAE results .npz file
    save_dir : str, optional
        Directory to save figures (will create 'mclmc/' subdirectory)
    show_plots : bool
        Whether to display plots interactively
        
    Returns:
    --------
    dict : Dictionary containing tuning statistics
    """
    # Load results with allow_pickle to handle object arrays
    try:
        res = np.load(result_filepath, allow_pickle=True)
    except Exception as e:
        print(f"ERROR: Failed to load result file: {e}")
        return None
    
    result_name = Path(result_filepath).stem
    
    # Create subdirectory for MCLMC plots
    if save_dir:
        mclmc_dir = Path(save_dir) / 'mclmc'
        mclmc_dir.mkdir(parents=True, exist_ok=True)
    else:
        mclmc_dir = None
    
    try:
        n_sources = len(res['ztrue'])
    except Exception as e:
        print(f"ERROR: Cannot determine number of sources: {e}")
        return None
    
    # Check if tuning parameters exist
    if 'tuned_L' not in res or 'tuned_step_size' not in res:
        print("WARNING: MCLMC tuning parameters not found in results file.")
        print("  This may be an older result file before tuning parameter saving was added.")
        return None
    
    try:
        tuned_L = res['tuned_L']
        tuned_step_size = res['tuned_step_size']
    except Exception as e:
        print(f"ERROR: Failed to extract tuning parameters: {e}")
        return None
    
    # Handle case where tuning params are None/empty
    if tuned_L is None or tuned_step_size is None:
        print("WARNING: MCLMC tuning parameters are None in results file.")
        return None
    
    if tuned_L.size == 0 or tuned_step_size.size == 0:
        print("WARNING: MCLMC tuning parameters are empty.")
        return None
    
    print(f"\n{'='*70}")
    print("MCLMC TUNING PARAMETER DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"tuned_L shape: {tuned_L.shape}")
    print(f"tuned_step_size shape: {tuned_step_size.shape}")
    
    # Compute tuning statistics
    print("Computing tuning statistics...")
    try:
        tuning_stats = compute_mclmc_tuning_stats(tuned_L, tuned_step_size)
    except Exception as e:
        print(f"ERROR: Failed to compute tuning statistics: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    if tuning_stats is None:
        print("WARNING: Tuning statistics computation returned None")
        return None
    
    print(f"Number of chains per source: {tuning_stats['n_chains']}")
    print(f"Mean L: {np.nanmedian(tuning_stats['L_mean']):.4f}")
    print(f"Median L CV (dispersion): {np.nanmedian(tuning_stats['L_cv']):.4f}")
    print(f"Mean step_size: {np.nanmedian(tuning_stats['step_mean']):.6f}")
    print(f"Median step_size CV: {np.nanmedian(tuning_stats['step_cv']):.4f}")
    
    # Extract redshift-related quantities for correlation
    print("Extracting redshift quality metrics...")
    try:
        pae_bias = res['z_med'] - res['ztrue']
        pae_unc = 0.5 * (res['err_low'] + res['err_high'])
        pae_zscore = np.abs(pae_bias / pae_unc)
        pae_dzoneplusz = pae_unc / (1 + res['z_med'])
        
        # Use dict access with defaults to avoid KeyError
        R_hat = res['R_hat'] if 'R_hat' in res else np.full(n_sources, np.nan)
        chi2 = res['chi2'] if 'chi2' in res else np.full(n_sources, np.nan)
    except Exception as e:
        print(f"ERROR: Failed to extract redshift metrics: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ====================================================================
    # FIGURE MCLMC-1: Tuning Parameter Distributions
    # ====================================================================
    print("\nGenerating Figure MCLMC-1: Tuning parameter distributions...")
    fig1, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Panel 1: Distribution of mean L
    ax = axes[0, 0]
    L_mean_valid = tuning_stats['L_mean'][np.isfinite(tuning_stats['L_mean'])]
    ax.hist(L_mean_valid, bins=50, color='steelblue', alpha=0.7, edgecolor='navy')
    ax.axvline(np.median(L_mean_valid), color='red', linestyle='--', linewidth=2, 
               label=f'Median: {np.median(L_mean_valid):.3f}')
    ax.set_xlabel('Mean tuned L (across chains)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Trajectory Length Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 2: Distribution of L coefficient of variation
    ax = axes[0, 1]
    L_cv_valid = tuning_stats['L_cv'][np.isfinite(tuning_stats['L_cv'])]
    ax.hist(L_cv_valid, bins=50, color='darkorange', alpha=0.7, edgecolor='saddlebrown')
    ax.axvline(np.median(L_cv_valid), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(L_cv_valid):.3f}')
    ax.set_xlabel('L coefficient of variation (σ/μ)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('L Dispersion Across Chains', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 3: Distribution of mean step_size
    ax = axes[0, 2]
    step_mean_valid = tuning_stats['step_mean'][np.isfinite(tuning_stats['step_mean'])]
    ax.hist(step_mean_valid, bins=50, color='seagreen', alpha=0.7, edgecolor='darkgreen')
    ax.axvline(np.median(step_mean_valid), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(step_mean_valid):.4f}')
    ax.set_xlabel('Mean tuned step_size', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Step Size Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 4: Distribution of step_size CV
    ax = axes[1, 0]
    step_cv_valid = tuning_stats['step_cv'][np.isfinite(tuning_stats['step_cv'])]
    ax.hist(step_cv_valid, bins=50, color='mediumpurple', alpha=0.7, edgecolor='indigo')
    ax.axvline(np.median(step_cv_valid), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(step_cv_valid):.3f}')
    ax.set_xlabel('Step size coefficient of variation', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Step Size Dispersion Across Chains', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 5: L vs step_size
    ax = axes[1, 1]
    valid_mask = np.isfinite(tuning_stats['L_mean']) & np.isfinite(tuning_stats['step_mean'])
    if use_hexbin:
        hb = ax.hexbin(tuning_stats['step_mean'][valid_mask], tuning_stats['L_mean'][valid_mask],
                       bins='log', cmap='viridis', mincnt=1, gridsize=50)
        plt.colorbar(hb, ax=ax, label='log10(count)')
    else:
        ax.scatter(tuning_stats['step_mean'][valid_mask], tuning_stats['L_mean'][valid_mask],
                   alpha=0.3, s=5, c='navy')
    ax.set_xlabel('Mean step_size', fontsize=11)
    ax.set_ylabel('Mean L', fontsize=11)
    ax.set_title('L vs Step Size', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Panel 6: L/step_size ratio (effective leapfrog steps)
    ax = axes[1, 2]
    ratio_valid = tuning_stats['L_step_ratio'][np.isfinite(tuning_stats['L_step_ratio'])]
    ax.hist(ratio_valid, bins=50, color='coral', alpha=0.7, edgecolor='darkred')
    ax.axvline(np.median(ratio_valid), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(ratio_valid):.1f}')
    ax.set_xlabel('L / step_size (effective leapfrog steps)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Trajectory Discretization', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    fig1.suptitle(f'{result_name}: MCLMC Tuning Parameters\n(N={n_sources:,})', fontsize=14)
    plt.tight_layout()
    
    if mclmc_dir:
        fig1_path = mclmc_dir / f'{result_name}_mclmc_tuning_distributions.png'
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: mclmc/{fig1_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig1)
    
    # ====================================================================
    # FIGURE MCLMC-2: Tuning Dispersion vs Redshift Quality
    # ====================================================================
    print("Generating Figure MCLMC-2: Tuning dispersion vs redshift quality...")
    fig2, axes = plt.subplots(2, 3, figsize=(10, 8))
    
    # Valid mask for correlations
    valid = (np.isfinite(tuning_stats['L_cv']) & np.isfinite(pae_zscore) & 
             np.isfinite(R_hat) & np.isfinite(chi2))
    
    # Panel 1: L_cv vs |z-score|
    ax = axes[0, 0]
    if use_hexbin:
        hb = ax.hexbin(tuning_stats['L_cv'][valid], pae_zscore[valid],
                       bins='log', cmap='viridis', mincnt=1, gridsize=50,
                       extent=[0, 1, 0, 10])
        plt.colorbar(hb, ax=ax, label='log10(count)')
    else:
        ax.scatter(tuning_stats['L_cv'][valid], pae_zscore[valid],
                   alpha=0.2, s=5, c='steelblue')
    ax.set_xlabel('L coefficient of variation', fontsize=11)
    ax.set_ylabel('|z-score|', fontsize=11)
    ax.set_title('L Dispersion vs Redshift Error', fontsize=12)
    ax.set_ylim(0, 10)
    ax.axhline(3, color='red', linestyle='--', alpha=0.5, label='|z-score|=3')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 2: L_cv vs R-hat
    ax = axes[0, 1]
    if use_hexbin:
        hb = ax.hexbin(tuning_stats['L_cv'][valid], R_hat[valid],
                       bins='log', cmap='viridis', mincnt=1, gridsize=50,
                       extent=[0, 1, 0.9, 2.0])
        plt.colorbar(hb, ax=ax, label='log10(count)')
    else:
        ax.scatter(tuning_stats['L_cv'][valid], R_hat[valid],
                   alpha=0.2, s=5, c='darkorange')
    ax.set_xlabel('L coefficient of variation', fontsize=11)
    ax.set_ylabel('R-hat', fontsize=11)
    ax.set_title('L Dispersion vs Convergence', fontsize=12)
    ax.set_ylim(0.9, 2.0)
    ax.axhline(1.1, color='red', linestyle='--', alpha=0.5, label='R-hat=1.1')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 3: L_cv vs reduced chi2
    ax = axes[0, 2]
    if use_hexbin:
        hb = ax.hexbin(tuning_stats['L_cv'][valid], chi2[valid],
                       bins='log', cmap='viridis', mincnt=1, gridsize=50,
                       extent=[0, 1, 0, 10])
        plt.colorbar(hb, ax=ax, label='log10(count)')
    else:
        ax.scatter(tuning_stats['L_cv'][valid], chi2[valid],
                   alpha=0.2, s=5, c='seagreen')
    ax.set_xlabel('L coefficient of variation', fontsize=11)
    ax.set_ylabel('Reduced χ²', fontsize=11)
    ax.set_title('L Dispersion vs Fit Quality', fontsize=12)
    ax.set_ylim(0, 10)
    ax.axhline(2, color='red', linestyle='--', alpha=0.5, label='χ²=2')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 4: step_cv vs |z-score|
    ax = axes[1, 0]
    if use_hexbin:
        hb = ax.hexbin(tuning_stats['step_cv'][valid], pae_zscore[valid],
                       bins='log', cmap='viridis', mincnt=1, gridsize=50,
                       extent=[0, 1, 0, 10])
        plt.colorbar(hb, ax=ax, label='log10(count)')
    else:
        ax.scatter(tuning_stats['step_cv'][valid], pae_zscore[valid],
                   alpha=0.2, s=5, c='mediumpurple')
    ax.set_xlabel('Step size coefficient of variation', fontsize=11)
    ax.set_ylabel('|z-score|', fontsize=11)
    ax.set_title('Step Size Dispersion vs Redshift Error', fontsize=12)
    ax.set_ylim(0, 10)
    ax.axhline(3, color='red', linestyle='--', alpha=0.5, label='|z-score|=3')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 5: L_cv distribution for good vs bad fits
    ax = axes[1, 1]
    good_fit = valid & (np.abs(pae_zscore) < 3) & (R_hat < 1.5)
    bad_fit = valid & ((np.abs(pae_zscore) >= 3) | (R_hat >= 1.5))
    
    n_good = np.sum(good_fit)
    n_bad = np.sum(bad_fit)
    
    bins_cv = np.linspace(0, 1, 30)
    ax.hist(tuning_stats['L_cv'][good_fit], bins=bins_cv, alpha=0.6, color='green',
            label=f'Good fits (n={n_good})', density=True)
    ax.hist(tuning_stats['L_cv'][bad_fit], bins=bins_cv, alpha=0.6, color='red',
            label=f'Bad fits (n={n_bad})', density=True)
    ax.set_xlabel('L coefficient of variation', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('L Dispersion: Good vs Bad Fits', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 6: z-score vs chi2 (key quality diagnostic)
    ax = axes[1, 2]
    valid_zchi = valid & np.isfinite(pae_zscore) & np.isfinite(chi2)
    n_valid_zchi = np.sum(valid_zchi)
    
    if n_valid_zchi > 0:
        if use_hexbin:
            hb = ax.hexbin(chi2[valid_zchi], pae_zscore[valid_zchi],
                           bins='log', cmap='viridis', mincnt=1, gridsize=50,
                           extent=[0, 10, 0, 10])
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label('log10(count)', fontsize=9)
        else:
            # Limit points if too many to avoid hanging
            if n_valid_zchi > 50000:
                print(f"  Note: Downsampling {n_valid_zchi} points to 50000 for panel 6")
                downsample_idx = np.random.choice(np.where(valid_zchi)[0], 50000, replace=False)
                downsample_mask = np.zeros(len(valid_zchi), dtype=bool)
                downsample_mask[downsample_idx] = True
                valid_zchi = downsample_mask
            scatter = ax.scatter(chi2[valid_zchi], pae_zscore[valid_zchi],
                                c=tuning_stats['L_cv'][valid_zchi],
                                cmap='viridis', alpha=0.3, s=10, vmin=0, vmax=1)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('L CV', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)

    ax.axhline(3, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='|z-score|=3')
    ax.axvline(2, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='χ²=2')
    ax.set_xlabel('Reduced χ²', fontsize=11)
    ax.set_ylabel('|z-score|', fontsize=11)
    ax.set_title('Redshift Error vs Fit Quality', fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    fig2.suptitle(f'{result_name}: MCLMC Tuning vs Redshift Quality', fontsize=14)
    plt.tight_layout()
    
    if mclmc_dir:
        fig2_path = mclmc_dir / f'{result_name}_mclmc_tuning_vs_quality.png'
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: mclmc/{fig2_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig2)
    
    # ====================================================================
    # FIGURE MCLMC-3: Tuning by Redshift Bin
    # ====================================================================
    print("Generating Figure MCLMC-3: Tuning parameters by redshift...")
    fig3, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    z_bins = np.linspace(0, 3, 21)
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    ztrue = res['ztrue']
    
    # Compute median L_cv per redshift bin
    L_cv_by_z = []
    L_mean_by_z = []
    step_mean_by_z = []
    
    for i in range(len(z_bins) - 1):
        in_bin = (ztrue >= z_bins[i]) & (ztrue < z_bins[i+1]) & np.isfinite(tuning_stats['L_cv'])
        if np.sum(in_bin) > 10:
            L_cv_by_z.append(np.median(tuning_stats['L_cv'][in_bin]))
            L_mean_by_z.append(np.median(tuning_stats['L_mean'][in_bin]))
            step_mean_by_z.append(np.median(tuning_stats['step_mean'][in_bin]))
        else:
            L_cv_by_z.append(np.nan)
            L_mean_by_z.append(np.nan)
            step_mean_by_z.append(np.nan)
    
    ax = axes[0]
    ax.plot(z_centers, L_cv_by_z, 'o-', color='darkorange', linewidth=2, markersize=6)
    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_ylabel('Median L CV', fontsize=12)
    ax.set_title('L Dispersion vs Redshift', fontsize=13)
    ax.grid(alpha=0.3)
    
    ax = axes[1]
    ax.plot(z_centers, L_mean_by_z, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_ylabel('Median L', fontsize=12)
    ax.set_title('Trajectory Length vs Redshift', fontsize=13)
    ax.grid(alpha=0.3)
    
    ax = axes[2]
    ax.plot(z_centers, step_mean_by_z, 'o-', color='seagreen', linewidth=2, markersize=6)
    ax.set_xlabel('Redshift', fontsize=12)
    ax.set_ylabel('Median step_size', fontsize=12)
    ax.set_title('Step Size vs Redshift', fontsize=13)
    ax.grid(alpha=0.3)
    
    fig3.suptitle(f'{result_name}: MCLMC Tuning by Redshift', fontsize=14)
    plt.tight_layout()
    
    if mclmc_dir:
        fig3_path = mclmc_dir / f'{result_name}_mclmc_tuning_by_redshift.png'
        fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: mclmc/{fig3_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig3)
    
    # Print correlation statistics
    print(f"\n{'='*70}")
    print("CORRELATION SUMMARY:")
    print(f"{'='*70}")
    
    valid_corr = np.isfinite(tuning_stats['L_cv']) & np.isfinite(pae_zscore)
    if np.sum(valid_corr) > 10:
        corr_L_zscore = np.corrcoef(tuning_stats['L_cv'][valid_corr], 
                                     pae_zscore[valid_corr])[0, 1]
        print(f"Correlation (L_cv, |z-score|): {corr_L_zscore:.4f}")
    
    valid_corr = np.isfinite(tuning_stats['L_cv']) & np.isfinite(R_hat)
    if np.sum(valid_corr) > 10:
        corr_L_rhat = np.corrcoef(tuning_stats['L_cv'][valid_corr], 
                                   R_hat[valid_corr])[0, 1]
        print(f"Correlation (L_cv, R-hat): {corr_L_rhat:.4f}")
    
    valid_corr = np.isfinite(tuning_stats['L_cv']) & np.isfinite(chi2)
    if np.sum(valid_corr) > 10:
        corr_L_chi2 = np.corrcoef(tuning_stats['L_cv'][valid_corr], 
                                   chi2[valid_corr])[0, 1]
        print(f"Correlation (L_cv, chi2): {corr_L_chi2:.4f}")
    
    print(f"\nGood fits (|z-score|<3 & R-hat<1.1): n={n_good}")
    print(f"  Median L_cv: {np.nanmedian(tuning_stats['L_cv'][good_fit]):.4f}")
    print(f"Bad fits: n={n_bad}")
    print(f"  Median L_cv: {np.nanmedian(tuning_stats['L_cv'][bad_fit]):.4f}")
    
    # ====================================================================
    # FIGURE MCLMC-4: Z-score vs Chi2 Detailed Analysis
    # ====================================================================
    print("Generating Figure MCLMC-4: Z-score vs chi2 detailed analysis...")
    fig4, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Define uncertainty bins for subplots
    unc_bins = [
        (0.0, 0.003, r'$\sigma_{z/(1+z)} < 0.003$'),
        (0.003, 0.01, r'$0.003 \leq \sigma_{z/(1+z)} < 0.01$'),
        (0.01, 0.03, r'$0.01 \leq \sigma_{z/(1+z)} < 0.03$'),
        (0.03, 0.1, r'$0.03 \leq \sigma_{z/(1+z)} < 0.1$'),
        (0.1, 0.2, r'$0.1 \leq \sigma_{z/(1+z)} < 0.2$'),
        (0.0, 0.2, r'All: $\sigma_{z/(1+z)} < 0.2$')  # Combined
    ]
    
    for idx, (unc_min, unc_max, label) in enumerate(unc_bins):
        ax = axes.flatten()[idx]
        
        # Create mask for this uncertainty bin
        if unc_max == 0.2 and unc_min == 0.0:
            # Special case: all sources
            unc_mask = (pae_dzoneplusz < unc_max) & np.isfinite(pae_zscore) & np.isfinite(chi2)
        else:
            unc_mask = ((pae_dzoneplusz >= unc_min) & (pae_dzoneplusz < unc_max) & 
                       np.isfinite(pae_zscore) & np.isfinite(chi2))
        
        n_in_bin = np.sum(unc_mask)
        
        if n_in_bin > 10:
            if use_hexbin:
                hb = ax.hexbin(chi2[unc_mask], pae_zscore[unc_mask],
                               bins='log', cmap='viridis', mincnt=1, gridsize=40,
                               extent=[0, 10, 0, 10])
                plt.colorbar(hb, ax=ax, label='log10(count)')
            else:
                ax.scatter(chi2[unc_mask], pae_zscore[unc_mask],
                          alpha=0.3, s=5, c='navy')

            # Add threshold lines
            ax.axhline(3, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axvline(2, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            
            # Count outliers
            n_high_zscore = np.sum((pae_zscore[unc_mask] > 3))
            n_high_chi2 = np.sum((chi2[unc_mask] > 2))
            n_both = np.sum((pae_zscore[unc_mask] > 3) & (chi2[unc_mask] > 2))
            
            # Add statistics text box
            stats_text = (
                f'N = {n_in_bin}\n'
                f'|z|>3: {n_high_zscore} ({100*n_high_zscore/n_in_bin:.1f}%)\n'
                f'χ²>2: {n_high_chi2} ({100*n_high_chi2/n_in_bin:.1f}%)\n'
                f'Both: {n_both} ({100*n_both/n_in_bin:.1f}%)'
            )
            ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        else:
            ax.text(0.5, 0.5, f'Insufficient data\n(n={n_in_bin})',
                   transform=ax.transAxes, fontsize=12, ha='center', va='center')
        
        ax.set_xlabel('Reduced χ²', fontsize=11)
        ax.set_ylabel('|z-score|', fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.grid(alpha=0.3)
    
    fig4.suptitle(f'{result_name}: Z-score vs Chi² by Uncertainty Bin', fontsize=14)
    plt.tight_layout()
    
    if mclmc_dir:
        fig4_path = mclmc_dir / f'{result_name}_zscore_vs_chi2_by_uncertainty.png'
        fig4.savefig(fig4_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: mclmc/{fig4_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig4)
    
    # Return tuning statistics for use in filtering
    tuning_stats['good_fit_mask'] = good_fit
    tuning_stats['bad_fit_mask'] = bad_fit
    
    return tuning_stats


def plot_preinit_convergence_diagnostics(result_filepath, save_dir=None, show_plots=False, use_hexbin=True):
    """
    Generate diagnostic plots for pre-reinitialization log-posteriors and convergence.
    
    This creates plots to assess whether pre-reinitialization log-posteriors correlate
    with redshift errors, and whether chains have converged by comparing pre-reinit
    and final log-posteriors.
    
    Parameters:
    -----------
    result_filepath : str
        Path to the PAE results .npz file
    save_dir : str, optional
        Directory to save figures (will create 'convergence/' subdirectory)
    show_plots : bool
        Whether to display plots interactively
        
    Returns:
    --------
    dict : Dictionary containing convergence statistics
    """
    # Load results with allow_pickle to handle object arrays
    try:
        res = np.load(result_filepath, allow_pickle=True)
    except Exception as e:
        print(f"ERROR: Failed to load result file: {e}")
        return None
    
    result_name = Path(result_filepath).stem
    
    # Create subdirectory for convergence plots
    if save_dir:
        conv_dir = Path(save_dir) / 'convergence'
        conv_dir.mkdir(parents=True, exist_ok=True)
    else:
        conv_dir = None
    
    try:
        n_sources = len(res['ztrue'])
    except Exception as e:
        print(f"ERROR: Cannot determine number of sources: {e}")
        return None
    
    # Check if pre-reinitialization log-posteriors exist
    if 'preinit_final_logL' not in res:
        print("WARNING: Pre-reinitialization log-posteriors not found in results file.")
        print("  This may be an older result file or init_reinit was not enabled.")
        return None
    
    try:
        preinit_logL = res['preinit_final_logL']
        all_log_L = res['all_log_L']  # Post-burn-in mean log-posteriors per chain
    except Exception as e:
        print(f"ERROR: Failed to extract log-posteriors: {e}")
        return None
    
    # Handle case where preinit log-posteriors are None/empty
    if preinit_logL is None or preinit_logL.size == 0:
        print("WARNING: Pre-reinitialization log-posteriors are None or empty.")
        return None
    
    print(f"\n{'='*70}")
    print("PRE-REINITIALIZATION CONVERGENCE DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"preinit_logL shape: {preinit_logL.shape}")
    print(f"all_log_L shape: {all_log_L.shape}")
    
    # Extract redshift-related quantities
    print("Extracting redshift quality metrics...")
    try:
        pae_bias = res['z_med'] - res['ztrue']
        pae_unc = 0.5 * (res['err_low'] + res['err_high'])
        pae_zscore = np.abs(pae_bias / pae_unc)
        pae_dzoneplusz = pae_unc / (1 + res['z_med'])
        
        # Use dict access with defaults to avoid KeyError
        R_hat = res['R_hat'] if 'R_hat' in res else np.full(n_sources, np.nan)
        chi2 = res['chi2'] if 'chi2' in res else np.full(n_sources, np.nan)
    except Exception as e:
        print(f"ERROR: Failed to extract redshift metrics: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Compute convergence metrics
    # Best chain's pre-reinit log-posterior (maximum across chains)
    best_preinit_logL = np.nanmax(preinit_logL, axis=1)
    
    # Best chain's final log-posterior (maximum across chains)
    best_final_logL = np.nanmax(all_log_L, axis=1)
    
    # Difference: final - preinit (positive = improvement)
    logL_improvement = best_final_logL - best_preinit_logL
    
    # Dispersion in pre-reinit log-posteriors (indicator of initial convergence)
    preinit_logL_std = np.nanstd(preinit_logL, axis=1)
    
    # Dispersion in final log-posteriors
    final_logL_std = np.nanstd(all_log_L, axis=1)
    
    print(f"Median best pre-reinit log-posterior: {np.nanmedian(best_preinit_logL):.2f}")
    print(f"Median best final log-posterior: {np.nanmedian(best_final_logL):.2f}")
    print(f"Median log-posterior improvement: {np.nanmedian(logL_improvement):.2f}")
    print(f"Median pre-reinit log-L std: {np.nanmedian(preinit_logL_std):.4f}")
    print(f"Median final log-L std: {np.nanmedian(final_logL_std):.4f}")
    
    # ====================================================================
    # FIGURE CONV-1: Pre-reinit log-posterior vs Z-score
    # ====================================================================
    print("\nGenerating Figure CONV-1: Pre-reinit log-posterior vs z-score...")
    fig1, axes = plt.subplots(2, 3, figsize=(12, 7))
    
    # Define uncertainty bins for subplots
    unc_bins = [
        (0.0, 0.003, r'$\sigma_{z/(1+z)} < 0.003$'),
        (0.003, 0.01, r'$0.003 \leq \sigma_{z/(1+z)} < 0.01$'),
        (0.01, 0.03, r'$0.01 \leq \sigma_{z/(1+z)} < 0.03$'),
        (0.03, 0.1, r'$0.03 \leq \sigma_{z/(1+z)} < 0.1$'),
        (0.1, 0.2, r'$0.1 \leq \sigma_{z/(1+z)} < 0.2$'),
        (0.0, 0.2, r'All: $\sigma_{z/(1+z)} < 0.2$')  # Combined
    ]
    
    for idx, (unc_min, unc_max, label) in enumerate(unc_bins):
        ax = axes.flatten()[idx]
        
        # Create mask for this uncertainty bin
        if unc_max == 0.2 and unc_min == 0.0:
            unc_mask = (pae_dzoneplusz < unc_max) & np.isfinite(pae_zscore) & np.isfinite(best_preinit_logL)
        else:
            unc_mask = ((pae_dzoneplusz >= unc_min) & (pae_dzoneplusz < unc_max) & 
                       np.isfinite(pae_zscore) & np.isfinite(best_preinit_logL))
        
        n_in_bin = np.sum(unc_mask)
        
        if n_in_bin > 10:
            if use_hexbin:
                hb = ax.hexbin(best_preinit_logL[unc_mask], pae_zscore[unc_mask],
                               bins='log', cmap='viridis', mincnt=1, gridsize=40,
                               extent=[-200, 0, 0, 10])
                cbar = plt.colorbar(hb, ax=ax)
                cbar.set_label('log10(count)', fontsize=9)
            else:
                # Scatter plot colored by chi2
                scatter = ax.scatter(best_preinit_logL[unc_mask], pae_zscore[unc_mask],
                                   c=chi2[unc_mask], alpha=0.4, s=10, cmap='viridis',
                                   vmin=0, vmax=5)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(r'$\chi^2_{red}$', fontsize=9)
            
            # Add threshold lines
            ax.axhline(3, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='|z-score|=3')
            
            # Compute correlation
            valid_corr = np.isfinite(best_preinit_logL[unc_mask]) & np.isfinite(pae_zscore[unc_mask])
            if np.sum(valid_corr) > 10:
                corr = np.corrcoef(best_preinit_logL[unc_mask][valid_corr], 
                                 pae_zscore[unc_mask][valid_corr])[0, 1]
                
                # Count outliers
                n_high_zscore = np.sum(pae_zscore[unc_mask] > 3)
                
                # Add statistics text box
                stats_text = (
                    f'N = {n_in_bin}\n'
                    f'r = {corr:.3f}\n'
                    f'|z|>3: {n_high_zscore} ({100*n_high_zscore/n_in_bin:.1f}%)'
                )
                ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        else:
            ax.text(0.5, 0.5, f'Insufficient data\n(n={n_in_bin})',
                   transform=ax.transAxes, fontsize=12, ha='center', va='center')
        
        ax.set_xlabel('Best Chain Pre-reinit log-posterior', fontsize=11)
        ax.set_ylabel('|z-score|', fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.set_xlim(-200, 0)
        ax.set_ylim(0, 10)
        ax.grid(alpha=0.3)
    
    fig1.suptitle(f'{result_name}: Pre-reinitialization Log-Posterior vs Z-score', fontsize=14)
    plt.tight_layout()
    
    if conv_dir:
        fig1_path = conv_dir / f'{result_name}_preinit_logL_vs_zscore.png'
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: convergence/{fig1_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig1)
    
    # ====================================================================
    # FIGURE CONV-2: Log-posterior improvement vs Z-score (convergence test)
    # ====================================================================
    print("Generating Figure CONV-2: Log-posterior improvement vs z-score...")
    fig2, axes = plt.subplots(2, 3, figsize=(12, 7))
    
    for idx, (unc_min, unc_max, label) in enumerate(unc_bins):
        ax = axes.flatten()[idx]
        
        # Create mask for this uncertainty bin
        if unc_max == 0.2 and unc_min == 0.0:
            unc_mask = (pae_dzoneplusz < unc_max) & np.isfinite(pae_zscore) & np.isfinite(logL_improvement)
        else:
            unc_mask = ((pae_dzoneplusz >= unc_min) & (pae_dzoneplusz < unc_max) & 
                       np.isfinite(pae_zscore) & np.isfinite(logL_improvement))
        
        n_in_bin = np.sum(unc_mask)
        
        if n_in_bin > 10:
            # Filter to reasonable improvement range (> -10)
            improvement_filter = logL_improvement[unc_mask] > -10

            if use_hexbin:
                hb = ax.hexbin(logL_improvement[unc_mask][improvement_filter],
                               pae_zscore[unc_mask][improvement_filter],
                               bins='log', cmap='viridis', mincnt=1, gridsize=40)
                cbar = plt.colorbar(hb, ax=ax)
                cbar.set_label('log10(count)', fontsize=9)
            else:
                # Scatter plot colored by R-hat
                scatter = ax.scatter(logL_improvement[unc_mask][improvement_filter],
                                   pae_zscore[unc_mask][improvement_filter],
                                   c=R_hat[unc_mask][improvement_filter], alpha=0.4, s=10, cmap='coolwarm',
                                   vmin=0.95, vmax=1.2)
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(r'$\hat{R}$', fontsize=9)
            
            # Add threshold lines
            ax.axhline(3, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='No improvement')
            
            # Compute correlation (on filtered data)
            valid_corr = (np.isfinite(logL_improvement[unc_mask]) & 
                         np.isfinite(pae_zscore[unc_mask]) & improvement_filter)
            if np.sum(valid_corr) > 10:
                corr = np.corrcoef(logL_improvement[unc_mask][valid_corr], 
                                 pae_zscore[unc_mask][valid_corr])[0, 1]
                
                # Count sources with negative improvement (got worse)
                n_worse = np.sum(logL_improvement[unc_mask][improvement_filter] < 0)
                n_high_zscore = np.sum(pae_zscore[unc_mask][improvement_filter] > 3)
                n_worse_and_bad_z = np.sum((logL_improvement[unc_mask][improvement_filter] < 0) & 
                                          (pae_zscore[unc_mask][improvement_filter] > 3))
                
                # Add statistics text box
                n_filtered = np.sum(improvement_filter)
                stats_text = (
                    f'N = {n_filtered}\n'
                    f'r = {corr:.3f}\n'
                    f'ΔlogL<0: {n_worse} ({100*n_worse/n_filtered:.1f}%)\n'
                    f'|z|>3: {n_high_zscore} ({100*n_high_zscore/n_filtered:.1f}%)\n'
                    f'Both: {n_worse_and_bad_z} ({100*n_worse_and_bad_z/n_filtered:.1f}%)'
                )
                ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        else:
            ax.text(0.5, 0.5, f'Insufficient data\n(n={n_in_bin})',
                   transform=ax.transAxes, fontsize=12, ha='center', va='center')
        
        ax.set_xlabel('Log-posterior Improvement (Final - Pre-reinit)', fontsize=11)
        ax.set_ylabel('|z-score|', fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.set_xlim(-10, None)
        ax.set_ylim(0, 10)
        ax.grid(alpha=0.3)
    
    fig2.suptitle(f'{result_name}: Log-Posterior Improvement vs Z-score (Convergence)', fontsize=14)
    plt.tight_layout()
    
    if conv_dir:
        fig2_path = conv_dir / f'{result_name}_logL_improvement_vs_zscore.png'
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: convergence/{fig2_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig2)
    
    # ====================================================================
    # FIGURE CONV-3: Pre-reinit dispersion diagnostics
    # ====================================================================
    print("Generating Figure CONV-3: Pre-reinit chain dispersion diagnostics...")
    fig3, axes = plt.subplots(2, 2, figsize=(9, 8))
    
    # Panel 1: Distribution of pre-reinit log-L std
    ax = axes[0, 0]
    valid = np.isfinite(preinit_logL_std) & (preinit_logL_std < 500)
    ax.hist(preinit_logL_std[valid], bins=50, color='steelblue', alpha=0.7, edgecolor='navy')
    ax.axvline(np.median(preinit_logL_std[valid]), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(preinit_logL_std[valid]):.3f}')
    ax.set_xlabel('Pre-reinit log-L std (across chains)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Pre-reinit Chain Dispersion Distribution', fontsize=12)
    ax.set_xlim(0, 500)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 2: Pre-reinit dispersion vs z-score
    ax = axes[0, 1]
    valid = np.isfinite(preinit_logL_std) & np.isfinite(pae_zscore) & (pae_dzoneplusz < 0.2) & (preinit_logL_std < 500)
    if use_hexbin:
        hb = ax.hexbin(preinit_logL_std[valid], pae_zscore[valid],
                       bins='log', cmap='viridis', mincnt=1, gridsize=50,
                       extent=[0, 500, 0, 10])
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label('log10(count)', fontsize=9)
    else:
        scatter = ax.scatter(preinit_logL_std[valid], pae_zscore[valid],
                            c=chi2[valid], alpha=0.3, s=5, cmap='viridis', vmin=0, vmax=5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r'$\chi^2_{red}$', fontsize=9)
    ax.axhline(3, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Pre-reinit log-L std', fontsize=11)
    ax.set_ylabel('|z-score|', fontsize=11)
    ax.set_title(r'Chain Dispersion vs Z-score ($\sigma_{z/(1+z)} < 0.2$)', fontsize=12)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 10)
    ax.grid(alpha=0.3)
    
    # Compute correlation
    if np.sum(valid) > 10:
        corr = np.corrcoef(preinit_logL_std[valid], pae_zscore[valid])[0, 1]
        ax.text(0.97, 0.97, f'r = {corr:.3f}\nN = {np.sum(valid)}',
               transform=ax.transAxes, fontsize=9, va='top', ha='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Panel 3: Pre-reinit vs final dispersion
    ax = axes[1, 0]
    valid = np.isfinite(preinit_logL_std) & np.isfinite(final_logL_std) & (preinit_logL_std < 500) & (final_logL_std < 500)
    if use_hexbin:
        hb = ax.hexbin(preinit_logL_std[valid], final_logL_std[valid],
                       bins='log', cmap='viridis', mincnt=1, gridsize=50,
                       extent=[0, 500, 0, 500])
        plt.colorbar(hb, ax=ax, label='log10(count)')
    else:
        ax.scatter(preinit_logL_std[valid], final_logL_std[valid], alpha=0.3, s=5, c='navy')
    
    # Add 1:1 line
    ax.plot([0, 500], [0, 500], 'r--', linewidth=2, alpha=0.5, label='1:1 line')
    
    ax.set_xlabel('Pre-reinit log-L std', fontsize=11)
    ax.set_ylabel('Final log-L std', fontsize=11)
    ax.set_title('Pre-reinit vs Final Chain Dispersion', fontsize=12)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel 4: Improvement vs pre-reinit dispersion
    ax = axes[1, 1]
    valid = (np.isfinite(preinit_logL_std) & np.isfinite(logL_improvement) & 
             (pae_dzoneplusz < 0.2) & (preinit_logL_std < 500) & (logL_improvement > -10))
    if use_hexbin:
        hb = ax.hexbin(preinit_logL_std[valid], logL_improvement[valid],
                       bins='log', cmap='viridis', mincnt=1, gridsize=50)
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label('log10(count)', fontsize=9)
    else:
        scatter = ax.scatter(preinit_logL_std[valid], logL_improvement[valid],
                            c=pae_zscore[valid], alpha=0.3, s=5, cmap='coolwarm', vmin=0, vmax=5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('|z-score|', fontsize=9)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Pre-reinit log-L std', fontsize=11)
    ax.set_ylabel('Log-posterior Improvement', fontsize=11)
    ax.set_title(r'Improvement vs Pre-reinit Dispersion ($\sigma_{z/(1+z)} < 0.2$)', fontsize=12)
    ax.set_xlim(0, 500)
    ax.set_ylim(-10, None)
    ax.grid(alpha=0.3)
    
    # Compute correlation
    if np.sum(valid) > 10:
        corr = np.corrcoef(preinit_logL_std[valid], logL_improvement[valid])[0, 1]
        ax.text(0.97, 0.03, f'r = {corr:.3f}\nN = {np.sum(valid)}',
               transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    fig3.suptitle(f'{result_name}: Chain Dispersion Diagnostics', fontsize=14)
    plt.tight_layout()
    
    if conv_dir:
        fig3_path = conv_dir / f'{result_name}_chain_dispersion_diagnostics.png'
        fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: convergence/{fig3_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig3)
    
    # Return convergence statistics
    convergence_stats = {
        'best_preinit_logL': best_preinit_logL,
        'best_final_logL': best_final_logL,
        'logL_improvement': logL_improvement,
        'preinit_logL_std': preinit_logL_std,
        'final_logL_std': final_logL_std,
        'median_improvement': np.nanmedian(logL_improvement),
        'n_sources': n_sources
    }
    
    return convergence_stats


def plot_production_convergence_diagnostics(result_filepath, save_dir=None, show_plots=False, snr_min=None, use_hexbin=True, snr_array=None, band4_neg_flag=None):
    """
    Generate comprehensive convergence diagnostic plots for production runs.
    
    Creates 4 diagnostic plots:
    1. Distribution of chi2 values from posterior mean (2 × log-likelihood)
    2. PAE total chi2 vs Template Fitting total chi2 comparison
    3. Distribution of R-hat values from MCMC chains
    4. Distribution of redshift chain autocorrelation lengths
    
    Parameters:
    -----------
    result_filepath : str
        Path to the PAE results .npz file
    save_dir : str, optional
        Directory to save figures (will create 'convergence/' subdirectory)
    show_plots : bool
        Whether to display plots interactively
    snr_min : float or None
        Minimum total SNR for inclusion. None = no filter.
    snr_array : np.ndarray or None
        Pre-computed per-source SNR values (e.g. snr_quad from parquet).
        When provided, used directly instead of the weights-based proxy.
        
    Returns:
    --------
    dict : Dictionary containing convergence statistics
    """
    # Load results
    try:
        res = np.load(result_filepath, allow_pickle=True)
    except Exception as e:
        print(f"ERROR: Failed to load result file: {e}")
        return None
    
    result_name = Path(result_filepath).stem
    
    # Create subdirectory for convergence plots
    if save_dir:
        conv_dir = Path(save_dir) / 'convergence'
        conv_dir.mkdir(parents=True, exist_ok=True)
    else:
        conv_dir = None
    
    try:
        n_sources = len(res['ztrue'])
    except Exception as e:
        print(f"ERROR: Cannot determine number of sources: {e}")
        return None

    # Build SNR mask
    if snr_array is not None and snr_min is not None:
        snr_mask = np.array(snr_array) >= snr_min
        n_pass = np.sum(snr_mask)
        print(f"  SNR filter (snr_quad ≥ {snr_min}): {n_pass}/{n_sources} sources pass ({100*n_pass/n_sources:.1f}%)")
    else:
        if snr_min is not None:
            print(f"  ⚠ snr_min={snr_min} requested but snr_quad array not available — SNR cut skipped.")
        snr_mask = np.ones(n_sources, dtype=bool)
    if band4_neg_flag is not None:
        b4_keep = ~np.array(band4_neg_flag, dtype=bool)
        n_b4_pass = np.sum(b4_keep)
        print(f"  Band-4 neg-flux flag: {n_b4_pass}/{n_sources} sources keep ({100*n_b4_pass/n_sources:.1f}%)")
        snr_mask &= b4_keep

    print(f"\nGenerating production convergence diagnostics for {np.sum(snr_mask):,} sources (of {n_sources:,} total)...")

    # Initialize variables for statistics
    chi2_valid = np.array([])
    R_hat_valid = np.array([])
    frac_converged = 0.0
    
    # ==================================================================
    # PLOT 1: Distribution of chi2 from posterior mean
    # ==================================================================
    print("  Creating chi2 distribution plot...")
    
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    
    # Get mean log-likelihood per source
    if 'all_log_L' in res:
        all_log_L = res['all_log_L']
        # Shape could be (n_sources, n_chains, n_samples) or (n_sources, n_chains) or (n_sources,)
        if len(all_log_L.shape) == 3:
            mean_log_L = np.mean(all_log_L, axis=(1, 2))  # Average over chains and samples
        elif len(all_log_L.shape) == 2:
            mean_log_L = np.mean(all_log_L, axis=1)  # Average over chains
        else:
            mean_log_L = all_log_L  # Already scalars
        
        # Convert to chi2: chi2 = -2 * log(L)
        chi2_from_logL = -2.0 * mean_log_L
        
        # Filter out inf/nan (and apply SNR mask)
        chi2_valid = chi2_from_logL[np.isfinite(chi2_from_logL) & snr_mask]
        

        chi2bins = np.linspace(50, 500, 50)
        if len(chi2_valid) > 0:
            # Plot histogram
            ax1.hist(chi2_valid, bins=chi2bins, edgecolor='k', alpha=0.7, color='steelblue')
            ax1.axvline(np.median(chi2_valid), color='red', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(chi2_valid):.1f}')
            
            ax1.set_xlabel(r'$\chi^2$ from Posterior Mean', fontsize=13)
            ax1.set_ylabel('Count', fontsize=13)
            ax1.set_title(r'Distribution of $\chi^2 = -2 \log \mathcal{L}$ (Posterior Mean)', fontsize=14)
            ax1.legend(fontsize=11)
            ax1.grid(alpha=0.3)
            
            # Add statistics in text box
            stats_text = f'N = {len(chi2_valid):,}\nMedian = {np.median(chi2_valid):.1f}\nMean = {np.mean(chi2_valid):.1f}\nStd = {np.std(chi2_valid):.1f}'
            ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        else:
            ax1.text(0.5, 0.5, 'No valid chi2 values available', transform=ax1.transAxes,
                    ha='center', va='center', fontsize=14)
    else:
        ax1.text(0.5, 0.5, 'all_log_L not found in results', transform=ax1.transAxes,
                ha='center', va='center', fontsize=14)
    
    if conv_dir:
        fig1_path = conv_dir / f'{result_name}_chi2_distribution.png'
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: convergence/{fig1_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig1)
    
    # ==================================================================
    # PLOT 2: PAE total chi2 vs Template Fitting total chi2
    # ==================================================================
    print("  Creating PAE vs TF chi2 comparison plot...")
    
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    
    # Get PAE total (absolute) chi2 — chi2_full is the sum over all bands, not divided by ndof
    chi2_pae = res.get('chi2_full', None)
    
    # Get TF total chi2 — minchi2_gals is the raw minimum chi2 from template fitting
    chi2_tf = res.get('minchi2_gals', None)
    
    if chi2_pae is not None and chi2_tf is not None:
        # Filter valid values and apply SNR mask; cap extreme outliers at 99.5th percentile
        mask = np.isfinite(chi2_pae) & np.isfinite(chi2_tf) & snr_mask
        chi2_pae_valid = chi2_pae[mask]
        chi2_tf_valid = chi2_tf[mask]
        
        if len(chi2_pae_valid) > 0:
            cap = max(np.percentile(chi2_tf_valid, 99.5), np.percentile(chi2_pae_valid, 99.5))
            plot_mask = (chi2_tf_valid <= cap) & (chi2_pae_valid <= cap)
            chi2_pae_plot = chi2_pae_valid[plot_mask]
            chi2_tf_plot = chi2_tf_valid[plot_mask]

            if use_hexbin:
                hb = ax2.hexbin(chi2_tf_plot, chi2_pae_plot,
                                bins='log', cmap='viridis', mincnt=1, gridsize=60,
                                extent=(0, cap, 0, cap))
                plt.colorbar(hb, ax=ax2, label='log$_{10}$(count)')
            else:
                ax2.scatter(chi2_tf_plot, chi2_pae_plot, alpha=0.3, s=10, color='steelblue')

            # Add 1:1 line
            ax2.plot([0, cap], [0, cap], 'k--', linewidth=1.5, label='1:1', zorder=10)
            
            ax2.set_xlabel(r'Template Fitting $\chi^2$', fontsize=13)
            ax2.set_ylabel(r'PAE $\chi^2$', fontsize=13)
            ax2.set_title('PAE vs Template Fitting Total Chi-Squared', fontsize=14)
            ax2.legend(fontsize=11)
            ax2.grid(alpha=0.3)
            ax2.set_xlim(0, cap)
            ax2.set_ylim(0, cap)
            
            # Add statistics (using all unclipped valid values for accuracy)
            corr = np.corrcoef(chi2_tf_valid, chi2_pae_valid)[0, 1]
            med_tf = np.median(chi2_tf_valid)
            med_pae = np.median(chi2_pae_valid)
            # frac_pae_better = np.mean(chi2_pae_valid < chi2_tf_valid)
            # stats_text = (f'$N = {len(chi2_pae_valid):,}$\n'
            #               f'Corr $= {corr:.3f}$\n'
            #               f'Median TF $= {med_tf:.1f}$\n'
            #               f'Median PAE $= {med_pae:.1f}$\n'
            #               f'PAE better: ${frac_pae_better:.1%}$')
            # ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, fontsize=9,
            #         verticalalignment='bottom', horizontalalignment='right',
            #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        else:
            ax2.text(0.5, 0.5, 'No valid chi2 pairs available', transform=ax2.transAxes,
                    ha='center', va='center', fontsize=14)
    else:
        missing = []
        if chi2_pae is None:
            missing.append('PAE chi2_full')
        if chi2_tf is None:
            missing.append('minchi2_gals')
        ax2.text(0.5, 0.5, f'Missing: {", ".join(missing)}', transform=ax2.transAxes,
                ha='center', va='center', fontsize=14)
    
    if conv_dir:
        fig2_path = conv_dir / f'{result_name}_chi2_pae_vs_tf.png'
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: convergence/{fig2_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig2)
    
    # ==================================================================
    # PLOT 3: R-hat distribution
    # ==================================================================
    print("  Creating R-hat distribution plot...")
    
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    
    if 'R_hat' in res:
        R_hat = res['R_hat']
        R_hat_valid = R_hat[np.isfinite(R_hat) & snr_mask]

        rhatbins = np.linspace(0.9, 3.0, 40)
        
        if len(R_hat_valid) > 0:
            ax3.hist(R_hat_valid, bins=rhatbins, edgecolor='k', alpha=0.7, color='forestgreen')
            ax3.axvline(1.1, color='red', linestyle='--', linewidth=2,
                       label=r'Convergence threshold ($\hat{R} = 1.1$)')
            ax3.axvline(np.median(R_hat_valid), color='orange', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(R_hat_valid):.3f}')
            
            ax3.set_xlabel(r'Gelman-Rubin $\hat{R}$', fontsize=13)
            ax3.set_ylabel('Count', fontsize=13)
            ax3.set_title(r'Distribution of $\hat{R}$ Convergence Statistic', fontsize=14)
            ax3.legend(fontsize=11)
            ax3.grid(alpha=0.3)
            
            # Statistics
            frac_converged = np.sum(R_hat_valid < 1.1) / len(R_hat_valid)
            stats_text = f'N = {len(R_hat_valid):,}\\nMedian = {np.median(R_hat_valid):.3f}\\nConverged (<1.1) = {100*frac_converged:.1f}%'
            ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        else:
            ax3.text(0.5, 0.5, 'No valid R-hat values available', transform=ax3.transAxes,
                    ha='center', va='center', fontsize=14)
    else:
        ax3.text(0.5, 0.5, 'R_hat not found in results', transform=ax3.transAxes,
                ha='center', va='center', fontsize=14)
    
    if conv_dir:
        fig3_path = conv_dir / f'{result_name}_rhat_distribution.png'
        fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: convergence/{fig3_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig3)
    
    # ==================================================================
    # PLOT 4: Redshift chain autocorrelation lengths
    # ==================================================================
    print("  Creating autocorrelation length distribution plot...")
    
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    
    # Use pre-computed autocorrelation lengths from results if available
    if 'autocorr_length' in res:
        autocorr_lengths = res['autocorr_length']
        
        # Filter out NaN values (sources where autocorrelation couldn't be computed), and apply SNR mask
        valid_mask = ~np.isnan(autocorr_lengths) & snr_mask
        autocorr_lengths = autocorr_lengths[valid_mask]
        
        if len(autocorr_lengths) > 0:
            ax4.hist(autocorr_lengths, bins=50, edgecolor='k', alpha=0.7, color='mediumpurple')
            ax4.axvline(np.median(autocorr_lengths), color='red', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(autocorr_lengths):.1f}')
            
            ax4.set_xlabel('Autocorrelation Length (steps)', fontsize=13)
            ax4.set_ylabel('Count', fontsize=13)
            ax4.set_title('Distribution of Redshift Chain Autocorrelation Lengths', fontsize=14)
            ax4.legend(fontsize=11)
            ax4.grid(alpha=0.3)
            
            # Statistics
            stats_text = f'N = {len(autocorr_lengths):,}\nMedian = {np.median(autocorr_lengths):.1f}\nMean = {np.mean(autocorr_lengths):.1f}'
            ax4.text(0.98, 0.97, stats_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        else:
            ax4.text(0.5, 0.5, 'No valid autocorrelation lengths computed', transform=ax4.transAxes,
                    ha='center', va='center', fontsize=14)
    
    # Fallback: compute from all_samples if autocorr_length not in results
    elif 'all_samples' in res:
        try:
            from scipy import signal
            
            all_samples = res['all_samples']
            # Shape: (n_sources, n_chains, n_steps, n_dim)
            # Redshift is at index -1
            
            if len(all_samples.shape) == 4:
                n_sources_samp, n_chains, n_steps, n_dim = all_samples.shape
                autocorr_lengths = []
                
                # Compute for subset of sources (computing for all can be slow)
                n_compute = min(n_sources_samp, 1000)
                print(f"    Computing autocorrelation for {n_compute}/{n_sources_samp} sources (fallback method)...")
                
                for i in range(n_compute):
                    # Get redshift samples for this source (all chains combined)
                    z_samples = all_samples[i, :, :, -1].ravel()  # Redshift at index -1
                    
                    # Compute autocorrelation
                    acf = np.correlate(z_samples - z_samples.mean(), z_samples - z_samples.mean(), mode='full')
                    acf = acf[len(acf)//2:] / acf[0]  # Normalize
                    
                    # Find first crossing below 0.5
                    below_half = np.where(acf < 0.5)[0]
                    if len(below_half) > 0:
                        autocorr_lengths.append(below_half[0])
                    else:
                        autocorr_lengths.append(len(acf))  # Never crossed
                
                autocorr_lengths = np.array(autocorr_lengths)
                
                if len(autocorr_lengths) > 0:
                    ax4.hist(autocorr_lengths, bins=50, edgecolor='k', alpha=0.7, color='mediumpurple')
                    ax4.axvline(np.median(autocorr_lengths), color='red', linestyle='--', linewidth=2,
                               label=f'Median: {np.median(autocorr_lengths):.1f}')
                    
                    ax4.set_xlabel('Autocorrelation Length (steps)', fontsize=13)
                    ax4.set_ylabel('Count', fontsize=13)
                    ax4.set_title('Distribution of Redshift Chain Autocorrelation Lengths', fontsize=14)
                    ax4.legend(fontsize=11)
                    ax4.grid(alpha=0.3)
                    
                    # Statistics
                    stats_text = f'N = {len(autocorr_lengths):,}\nMedian = {np.median(autocorr_lengths):.1f}\nMean = {np.mean(autocorr_lengths):.1f}'
                    ax4.text(0.98, 0.97, stats_text, transform=ax4.transAxes, fontsize=10,
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                else:
                    ax4.text(0.5, 0.5, 'Could not compute autocorrelations', transform=ax4.transAxes,
                            ha='center', va='center', fontsize=14)
            else:
                ax4.text(0.5, 0.5, f'Unexpected all_samples shape: {all_samples.shape}', transform=ax4.transAxes,
                        ha='center', va='center', fontsize=14)
        except Exception as e:
            ax4.text(0.5, 0.5, f'Error computing autocorrelations: {e}', transform=ax4.transAxes,
                    ha='center', va='center', fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'autocorr_length not found in results\\n(may need updated pipeline)', transform=ax4.transAxes,
                ha='center', va='center', fontsize=14)
    
    if conv_dir:
        fig4_path = conv_dir / f'{result_name}_autocorrelation_distribution.png'
        fig4.savefig(fig4_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved: convergence/{fig4_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig4)
    
    print("  ✓ Production convergence diagnostics complete!")
    
    # Return statistics dictionary
    stats = {
        'n_sources': n_sources,
    }
    
    if 'all_log_L' in res and len(chi2_valid) > 0:
        stats['chi2_median'] = np.median(chi2_valid)
        stats['chi2_mean'] = np.mean(chi2_valid)
    
    if 'R_hat' in res and len(R_hat_valid) > 0:
        stats['rhat_median'] = np.median(R_hat_valid)
        stats['rhat_frac_converged'] = frac_converged
    
    return stats


def plot_normalization_diagnostics(result_filepath, save_dir=None, show_plots=False, use_hexbin=True):
    """
    Generate diagnostic plots for photometric normalization factors.
    
    Creates overview plots and per-uncertainty-bin plots showing:
    - Distribution of normalization values
    - Correlations with redshift errors
    
    Parameters:
    -----------
    result_filepath : str
        Path to the PAE results .npz file
    save_dir : str, optional
        Directory to save figures (will create 'normalization/' subdirectory)
    show_plots : bool
        Whether to display plots interactively
        
    Returns:
    --------
    dict : Dictionary containing normalization statistics
    """
    # Load results
    try:
        res = np.load(result_filepath, allow_pickle=True)
    except Exception as e:
        print(f"ERROR: Failed to load result file: {e}")
        return None
    
    result_name = Path(result_filepath).stem
    
    # Check if phot_norms is available
    if 'phot_norms' not in res:
        print(f"⚠ WARNING: 'phot_norms' not found in results file")
        print(f"   Available keys: {list(res.keys())}")
        print(f"   Skipping normalization diagnostics")
        return None
    
    phot_norms = res['phot_norms']
    if phot_norms.ndim > 1:
        phot_norms = phot_norms.squeeze()
    
    # Get redshift data
    ztrue = res['ztrue']
    z_mean = res['z_mean']
    err_low = res['err_low']
    err_high = res['err_high']
    
    # Compute metrics
    dz = z_mean - ztrue
    dz_norm = dz / (1 + ztrue)
    sigma_z = 0.5 * (err_low + err_high)
    zscore = dz / sigma_z
    abs_zscore = np.abs(zscore)
    sigma_z_norm = sigma_z / (1 + z_mean)  # For binning
    
    n_sources = len(phot_norms)
    
    # Remove NaN/inf
    valid_mask = np.isfinite(phot_norms) & np.isfinite(dz_norm) & np.isfinite(abs_zscore)
    n_valid = np.sum(valid_mask)
    
    if n_valid < n_sources:
        print(f"⚠ WARNING: {n_sources - n_valid} sources have NaN/inf, excluding from plots")
    
    phot_norms_valid = phot_norms[valid_mask]
    dz_norm_valid = dz_norm[valid_mask]
    abs_zscore_valid = abs_zscore[valid_mask]
    ztrue_valid = ztrue[valid_mask]

    print('min/max of ztrue valid:', np.min(ztrue_valid), np.max(ztrue_valid))
    sigma_z_norm_valid = sigma_z_norm[valid_mask]
    
    # Create subdirectory for normalization plots
    if save_dir:
        norm_dir = Path(save_dir) / 'normalization'
        norm_dir.mkdir(parents=True, exist_ok=True)
    else:
        norm_dir = None
    
    # Compute overall statistics
    from scipy.stats import spearmanr, pearsonr
    norm_stats = {
        'median': np.median(phot_norms_valid),
        'mean': np.mean(phot_norms_valid),
        'std': np.std(phot_norms_valid),
        'min': np.min(phot_norms_valid),
        'max': np.max(phot_norms_valid),
        'p16': np.percentile(phot_norms_valid, 16),
        'p84': np.percentile(phot_norms_valid, 84),
    }
    
    corr_zscore_pearson, pval_zscore_pearson = pearsonr(phot_norms_valid, abs_zscore_valid)
    corr_zscore_spearman, pval_zscore_spearman = spearmanr(phot_norms_valid, abs_zscore_valid)
    
    norm_stats.update({
        'corr_zscore_pearson': corr_zscore_pearson,
        'corr_zscore_spearman': corr_zscore_spearman
    })
    
    # ====================================================================
    # FIGURE 1: OVERALL NORMALIZATION DIAGNOSTICS
    # ====================================================================
    print("\nGenerating normalization overview plot...")
    fig1, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot 1: Histogram
    ax1 = axes[0]
    bins = np.linspace(-500, 1000, 50)
    counts, bins, patches = ax1.hist(phot_norms_valid, bins=bins, alpha=0.7, 
                                      color='steelblue', edgecolor='black')
    ax1.axvline(norm_stats['median'], color='red', linestyle='--', linewidth=2, 
                label=f"Median: {norm_stats['median']:.2e}")
    ax1.axvline(norm_stats['mean'], color='orange', linestyle='--', linewidth=2, 
                label=f"Mean: {norm_stats['mean']:.2e}")
    ax1.set_xlabel('Photometric Normalization', fontsize=11)
    ax1.set_ylabel('Number of Sources', fontsize=11)
    ax1.set_title('Distribution of Normalization', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    stats_text = f"N = {n_valid:,}\nStd = {norm_stats['std']:.2e}"
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

    ax1.set_xlim(-500, 1000)  # Restrict normalization axis

    
    # Plot 2: Normalization vs |z-score|
    ax2 = axes[1]
    if use_hexbin:
        hb = ax2.hexbin(phot_norms_valid, abs_zscore_valid,
                        bins='log', cmap='viridis', mincnt=1, gridsize=50)
        plt.colorbar(hb, ax=ax2, label='log10(count)')
    else:
        sc2 = ax2.scatter(phot_norms_valid, abs_zscore_valid,
                          c=ztrue_valid, cmap='viridis', alpha=0.4, s=5, vmin=0, vmax=np.max(ztrue_valid))
        plt.colorbar(sc2, ax=ax2, label='True z')
    ax2.set_xlabel('Normalization', fontsize=11)
    ax2.set_ylabel('|z-score|', fontsize=11)
    ax2.set_title(f'Norm vs Z-score (ρ={corr_zscore_spearman:.3f})', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlim(-500, 1000)  # Restrict normalization axis
    ax2.set_yscale('log')
    ax2.set_ylim(1, 1e3)
    ax2.grid(alpha=0.3)
    cbar2 = plt.colorbar(sc2, ax=ax2, label='True z')
    ax2.axhline(1, color='red', linestyle='--', linewidth=1, alpha=0.4, label='|z-score| = 1')
    ax2.axhline(3, color='orange', linestyle='--', linewidth=1, alpha=0.4, label='|z-score| = 3')
    ax2.legend(fontsize=8, loc='upper right')
    
    fig1.suptitle(f'{result_name}: Normalization Diagnostics (N={n_valid:,})', 
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if norm_dir:
        fig1_path = norm_dir / f'{result_name}_norm_overview.png'
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: normalization/{fig1_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig1)
    
    # ====================================================================
    # FIGURES BY UNCERTAINTY BIN
    # ====================================================================
    print("Generating normalization plots by uncertainty bin...")
    
    # Define uncertainty bins (same as main plots)
    unc_bins = [
        (0.0, 0.003, r'$\sigma_{z/(1+z)} < 0.003$'),
        (0.003, 0.01, r'$0.003 \leq \sigma_{z/(1+z)} < 0.01$'),
        (0.01, 0.03, r'$0.01 \leq \sigma_{z/(1+z)} < 0.03$'),
        (0.03, 0.1, r'$0.03 \leq \sigma_{z/(1+z)} < 0.1$'),
        (0.1, 0.2, r'$0.1 \leq \sigma_{z/(1+z)} < 0.2$')
    ]
    
    # Create one figure per bin

    bins = np.linspace(-500, 1000, 50)

    for bin_idx, (unc_min, unc_max, label) in enumerate(unc_bins):
        bin_mask = (sigma_z_norm_valid >= unc_min) & (sigma_z_norm_valid < unc_max)
        n_bin = np.sum(bin_mask)
        
        if n_bin < 10:
            print(f"  Bin {bin_idx}: Only {n_bin} sources, skipping")
            continue
        
        norms_bin = phot_norms_valid[bin_mask]
        dz_norm_bin = dz_norm_valid[bin_mask]
        abs_zscore_bin = abs_zscore_valid[bin_mask]
        ztrue_bin = ztrue_valid[bin_mask]
        
        # Compute bin statistics
        corr_bin, pval_bin = spearmanr(norms_bin, abs_zscore_bin)
        
        # Create 1x2 figure for this bin
        fig_bin, axes_bin = plt.subplots(1, 2, figsize=(10, 4))
        
        # Histogram
        ax = axes_bin[0]
        ax.hist(norms_bin, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.median(norms_bin), color='red', linestyle='--', linewidth=2, 
                   label=f"Median: {np.median(norms_bin):.2e}")
        ax.set_xlabel('Normalization', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Distribution (N={n_bin:,})', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_yscale('log')
        ax.set_xlim(-500, 1000)  # Restrict normalization axis
        # Correlation plot
        ax = axes_bin[1]
        if use_hexbin:
            hb = ax.hexbin(norms_bin, abs_zscore_bin,
                           bins='log', cmap='viridis', mincnt=1, gridsize=40)
            plt.colorbar(hb, ax=ax, label='log10(count)')
        else:
            sc = ax.scatter(norms_bin, abs_zscore_bin,
                           c=ztrue_bin, cmap='viridis', alpha=0.4, s=5, vmin=0, vmax=np.max(ztrue_valid))
            plt.colorbar(sc, ax=ax, label='True z')
        ax.axhline(1, color='red', linestyle='--', linewidth=1, alpha=0.4, label='|z-score| = 1')
        ax.axhline(3, color='orange', linestyle='--', linewidth=1, alpha=0.4, label='|z-score| = 3')
        ax.legend(fontsize=8, loc='upper right')
        
        fig_bin.suptitle(f'{result_name}: {label}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        if norm_dir:
            fig_path = norm_dir / f'{result_name}_norm_bin{bin_idx}.png'
            fig_bin.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: normalization/{fig_path.name}")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig_bin)
    
    # Print summary
    print(f"\n{'='*70}")
    print("NORMALIZATION STATISTICS:")
    print(f"{'='*70}")
    print(f"  Valid sources:      {n_valid:,} / {n_sources:,}")
    print(f"  Median:             {norm_stats['median']:.3e}")
    print(f"  Mean:               {norm_stats['mean']:.3e}")
    print(f"  Std:                {norm_stats['std']:.3e}")
    print(f"  [Min, Max]:         [{norm_stats['min']:.3e}, {norm_stats['max']:.3e}]")
    print(f"  [P16, P84]:         [{norm_stats['p16']:.3e}, {norm_stats['p84']:.3e}]")
    print(f"\nCorrelations with |z-score|:")
    print(f"  Pearson r:          {corr_zscore_pearson:+.4f} (p={pval_zscore_pearson:.2e})")
    print(f"  Spearman ρ:         {corr_zscore_spearman:+.4f} (p={pval_zscore_spearman:.2e})")
    print(f"{'='*70}\n")
    
    return norm_stats


def plot_normalization_abs_diagnostics(res, result_name, save_dir=None, show_plots=False):
    """
    Generate diagnostic plots for photometric normalization using ABSOLUTE VALUES.
    This focuses on sources near zero normalization (both positive and negative).
    
    Shows:
    1. Histogram of |normalization|
    2. |z-score| vs |normalization| correlation
    3. Per-uncertainty-bin analysis
    
    Parameters:
    -----------
    res : numpy file handle
        Loaded .npz file containing PAE results
    result_name : str
        Name for plot titles and filenames
    save_dir : str, optional
        Directory to save figures
    show_plots : bool
        Whether to display plots interactively
    
    Returns:
    --------
    dict : Statistics about absolute normalization values
    """
    print("\n" + "="*70)
    print("GENERATING ABSOLUTE NORMALIZATION DIAGNOSTICS")
    print("="*70)
    
    # Extract required data (same as plot_normalization_diagnostics)
    try:
        phot_norms = res['phot_norms']
        if phot_norms.ndim > 1:
            phot_norms = phot_norms.squeeze()
        ztrue = res['ztrue']
        z_mean = res['z_mean']
        err_low = res['err_low']
        err_high = res['err_high']
    except KeyError as e:
        print(f"ERROR: Missing required field {e}")
        return None
    
    # Compute metrics (same as plot_normalization_diagnostics)
    dz = z_mean - ztrue
    dz_norm = dz / (1 + ztrue)
    sigma_z = 0.5 * (err_low + err_high)
    zscore = dz / sigma_z
    abs_zscore = np.abs(zscore)
    sigma_z_norm = sigma_z / (1 + z_mean)
    
    n_sources = len(phot_norms)
    
    # Filter out invalid data
    valid_mask = np.isfinite(phot_norms) & np.isfinite(dz_norm) & np.isfinite(abs_zscore)
    
    n_valid = np.sum(valid_mask)
    print(f"Valid sources: {n_valid:,} / {n_sources:,} ({100*n_valid/n_sources:.1f}%)")
    
    if n_valid < 10:
        print("ERROR: Insufficient valid sources for analysis")
        return None
    
    # Apply mask and compute absolute values
    phot_norms_valid = phot_norms[valid_mask]
    abs_phot_norms_valid = np.abs(phot_norms_valid)
    
    dz_valid = dz[valid_mask]
    dz_norm_valid = dz_norm[valid_mask]
    abs_zscore_valid = abs_zscore[valid_mask]
    sigma_z_norm_valid = sigma_z_norm[valid_mask]
    ztrue_valid = ztrue[valid_mask]
    
    # Create subdirectory for normalization plots
    if save_dir:
        norm_dir = Path(save_dir) / 'normalization'
        norm_dir.mkdir(parents=True, exist_ok=True)
    else:
        norm_dir = None
    
    # Compute statistics on absolute values
    from scipy.stats import spearmanr, pearsonr
    abs_norm_stats = {
        'median': np.median(abs_phot_norms_valid),
        'mean': np.mean(abs_phot_norms_valid),
        'std': np.std(abs_phot_norms_valid),
        'min': np.min(abs_phot_norms_valid),
        'max': np.max(abs_phot_norms_valid),
        'p16': np.percentile(abs_phot_norms_valid, 16),
        'p84': np.percentile(abs_phot_norms_valid, 84),
    }
    
    # Correlations with absolute values
    corr_zscore_pearson, pval_zscore_pearson = pearsonr(abs_phot_norms_valid, abs_zscore_valid)
    corr_zscore_spearman, pval_zscore_spearman = spearmanr(abs_phot_norms_valid, abs_zscore_valid)
    
    abs_norm_stats.update({
        'corr_zscore_pearson': corr_zscore_pearson,
        'corr_zscore_spearman': corr_zscore_spearman
    })
    
    # Also track how many sources have negative normalization
    n_negative = np.sum(phot_norms_valid < 0)
    pct_negative = 100 * n_negative / n_valid
    print(f"Sources with negative normalization: {n_negative:,} ({pct_negative:.2f}%)")
    
    # ====================================================================
    # FIGURE 1: OVERALL ABSOLUTE NORMALIZATION DIAGNOSTICS
    # ====================================================================
    print("\nGenerating absolute normalization overview plot...")
    fig1, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot 1: Histogram of |normalization|
    ax1 = axes[0]
    bins = np.linspace(0, 1000, 50)
    counts, bins, patches = ax1.hist(abs_phot_norms_valid, bins=bins, alpha=0.7, 
                                      color='steelblue', edgecolor='black')
    ax1.axvline(abs_norm_stats['median'], color='red', linestyle='--', linewidth=2, 
                label=f"Median: {abs_norm_stats['median']:.2e}")
    ax1.axvline(abs_norm_stats['mean'], color='orange', linestyle='--', linewidth=2, 
                label=f"Mean: {abs_norm_stats['mean']:.2e}")
    ax1.set_xlabel('|Photometric Normalization|', fontsize=11)
    ax1.set_ylabel('Number of Sources', fontsize=11)
    ax1.set_title('Distribution of |Normalization|', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    
    stats_text = f"N = {n_valid:,}\nStd = {abs_norm_stats['std']:.2e}\n{pct_negative:.1f}% negative"
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    ax1.set_xlim(0, 1000)
    
    # Plot 2: |Normalization| vs |z-score|
    ax2 = axes[1]
    sc2 = ax2.scatter(abs_phot_norms_valid, abs_zscore_valid, 
                      c=ztrue_valid, cmap='viridis', alpha=0.4, s=5, vmin=0, vmax=np.max(ztrue_valid))
    ax2.set_xlabel('|Normalization|', fontsize=11)
    ax2.set_ylabel('|z-score|', fontsize=11)
    ax2.set_title(f'|Norm| vs |Z-score| (ρ={corr_zscore_spearman:.3f})', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 1000)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlim(1, 1e4)
    ax2.set_ylim(1, 1e3)
    ax2.grid(alpha=0.3)
    cbar2 = plt.colorbar(sc2, ax=ax2, label='True z')
    ax2.axhline(1, color='red', linestyle='--', linewidth=1, alpha=0.4, label='|z-score| = 1')
    ax2.axhline(3, color='orange', linestyle='--', linewidth=1, alpha=0.4, label='|z-score| = 3')
    ax2.legend(fontsize=8, loc='upper right')
    
    fig1.suptitle(f'{result_name}: Absolute Normalization Diagnostics (N={n_valid:,})', 
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if norm_dir:
        fig1_path = norm_dir / f'{result_name}_norm_abs_overview.png'
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: normalization/{fig1_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig1)
    
    # ====================================================================
    # FIGURES BY UNCERTAINTY BIN
    # ====================================================================
    print("Generating absolute normalization plots by uncertainty bin...")
    
    # Define uncertainty bins (same as main plots)
    unc_bins = [
        (0.0, 0.003, r'$\sigma_{z/(1+z)} < 0.003$'),
        (0.003, 0.01, r'$0.003 \leq \sigma_{z/(1+z)} < 0.01$'),
        (0.01, 0.03, r'$0.01 \leq \sigma_{z/(1+z)} < 0.03$'),
        (0.03, 0.1, r'$0.03 \leq \sigma_{z/(1+z)} < 0.1$'),
        (0.1, 0.2, r'$0.1 \leq \sigma_{z/(1+z)} < 0.2$')
    ]
    
    bins = np.linspace(0, 1000, 50)
    
    for bin_idx, (unc_min, unc_max, label) in enumerate(unc_bins):
        bin_mask = (sigma_z_norm_valid >= unc_min) & (sigma_z_norm_valid < unc_max)
        n_bin = np.sum(bin_mask)
        
        if n_bin < 10:
            print(f"  Bin {bin_idx}: Only {n_bin} sources, skipping")
            continue
        
        abs_norms_bin = abs_phot_norms_valid[bin_mask]
        norms_bin = phot_norms_valid[bin_mask]  # Keep for negative count
        abs_zscore_bin = abs_zscore_valid[bin_mask]
        ztrue_bin = ztrue_valid[bin_mask]
        
        # Compute bin statistics
        corr_bin, pval_bin = spearmanr(abs_norms_bin, abs_zscore_bin)
        n_neg_bin = np.sum(norms_bin < 0)
        pct_neg_bin = 100 * n_neg_bin / n_bin
        
        # Create 1x2 figure for this bin
        fig_bin, axes_bin = plt.subplots(1, 2, figsize=(10, 4))
        
        # Histogram
        ax = axes_bin[0]
        ax.hist(abs_norms_bin, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(np.median(abs_norms_bin), color='red', linestyle='--', linewidth=2, 
                   label=f"Median: {np.median(abs_norms_bin):.2e}")
        ax.set_xlabel('|Normalization|', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(f'Distribution (N={n_bin:,}, {pct_neg_bin:.1f}% neg)', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_yscale('log')
        ax.set_xlim(0, 1000)
        
        # Correlation plot
        ax = axes_bin[1]
        if use_hexbin:
            hb = ax.hexbin(abs_norms_bin, abs_zscore_bin,
                           bins='log', cmap='viridis', mincnt=1, gridsize=40)
            plt.colorbar(hb, ax=ax, label='log10(count)')
        else:
            sc = ax.scatter(abs_norms_bin, abs_zscore_bin,
                           c=ztrue_bin, cmap='viridis', alpha=0.4, s=5, vmin=0, vmax=np.max(ztrue_valid))
            plt.colorbar(sc, ax=ax, label='True z')
        ax.set_xlabel('|Normalization|', fontsize=11)
        ax.set_ylabel('|z-score|', fontsize=11)
        ax.set_title(f'|Norm| vs |Z-score| (ρ={corr_bin:.3f}, p={pval_bin:.1e})', fontsize=12)
        ax.set_xlim(1, 1e4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1, 1e2)
        ax.grid(alpha=0.3)
        ax.axhline(1, color='red', linestyle='--', linewidth=1, alpha=0.4, label='|z-score| = 1')
        ax.axhline(3, color='orange', linestyle='--', linewidth=1, alpha=0.4, label='|z-score| = 3')
        ax.legend(fontsize=8, loc='upper right')
        
        fig_bin.suptitle(f'{result_name}: {label}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        if norm_dir:
            fig_path = norm_dir / f'{result_name}_norm_abs_bin{bin_idx}.png'
            fig_bin.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: normalization/{fig_path.name}")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig_bin)
    
    # Print summary
    print(f"\n{'='*70}")
    print("ABSOLUTE NORMALIZATION STATISTICS:")
    print(f"{'='*70}")
    print(f"  Valid sources:      {n_valid:,} / {n_sources:,}")
    print(f"  Negative norms:     {n_negative:,} ({pct_negative:.2f}%)")
    print(f"  Median |norm|:      {abs_norm_stats['median']:.3e}")
    print(f"  Mean |norm|:        {abs_norm_stats['mean']:.3e}")
    print(f"  Std |norm|:         {abs_norm_stats['std']:.3e}")
    print(f"  [Min, Max] |norm|:  [{abs_norm_stats['min']:.3e}, {abs_norm_stats['max']:.3e}]")
    print(f"  [P16, P84] |norm|:  [{abs_norm_stats['p16']:.3e}, {abs_norm_stats['p84']:.3e}]")
    print(f"\nCorrelations with |z-score|:")
    print(f"  Pearson r:          {corr_zscore_pearson:+.4f} (p={pval_zscore_pearson:.2e})")
    print(f"  Spearman ρ:         {corr_zscore_spearman:+.4f} (p={pval_zscore_spearman:.2e})")
    print(f"{'='*70}\n")
    
    return abs_norm_stats


def plot_pae_summary(result_filepath, save_dir=None, show_plots=False, zscore_range=(-10, 10),
                     rhat_max=None, chi2_max=None, chain_std_max=None, quality_tier_max=None,
                     tuning_cv_min=None, frac_sampled_min=None, snr_min=None, use_hexbin=True,
                     snr_array=None, band4_neg_flag=None):
    """
    Generate comprehensive summary plots for PAE redshift inference results.
    
    Parameters:
    -----------
    result_filepath : str
        Path to the PAE results .npz file
    save_dir : str, optional
        Directory to save figures. If None, figures are not saved.
    show_plots : bool
        Whether to display plots interactively
    zscore_range : tuple
        Range for z-score plots (min, max)
    rhat_max : float or None
        Maximum R-hat for inclusion. None = no filter.
    chi2_max : float or None
        Maximum reduced chi² for inclusion. None = no filter.
    chain_std_max : float or None
        Maximum chain z std dev (normalized) for inclusion. None = no filter.
    quality_tier_max : int or None
        Maximum quality tier for inclusion (0-3). None = no filter.
    tuning_cv_max : float or None
        Maximum MCLMC tuning L coefficient of variation. None = no filter.
        Lower values = more consistent tuning across chains.
    frac_sampled_min : float or None
        Minimum spectral completeness (frac_sampled_102) for inclusion. None = no filter.
        E.g., 0.7 = only include sources with >=70% of bands sampled.
    snr_min : float or None
        Minimum total SNR (sqrt(sum(weights))) for inclusion. None = no filter.
        E.g., 20 = only include sources with total SNR >= 20.
    
    Returns:
    --------
    dict : Dictionary containing computed statistics
    """
    # Load results with allow_pickle
    try:
        res = np.load(result_filepath, allow_pickle=True)
        n_sources = len(res['ztrue'])
    except Exception as e:
        print(f"ERROR: Failed to load result file: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Extract filename for titles
    result_name = Path(result_filepath).stem
    
    # Build quality mask if filters are specified
    quality_mask = np.ones(n_sources, dtype=bool)
    filter_summary = []
    
    if rhat_max is not None and 'R_hat' in res:
        rhat_filter = res['R_hat'] < rhat_max
        n_pass = np.sum(rhat_filter)
        filter_summary.append(f"R-hat < {rhat_max}: {n_pass}/{n_sources} ({100*n_pass/n_sources:.1f}%)")
        quality_mask &= rhat_filter
    
    if chi2_max is not None and 'chi2' in res:
        chi2_filter = res['chi2'] < chi2_max
        n_pass = np.sum(chi2_filter)
        filter_summary.append(f"χ²_red < {chi2_max}: {n_pass}/{n_sources} ({100*n_pass/n_sources:.1f}%)")
        quality_mask &= chi2_filter
    
    if chain_std_max is not None and 'chain_z_std_norm' in res:
        chain_filter = res['chain_z_std_norm'] < chain_std_max
        n_pass = np.sum(chain_filter)
        filter_summary.append(f"Chain σ_z/(1+z) < {chain_std_max}: {n_pass}/{n_sources} ({100*n_pass/n_sources:.1f}%)")
        quality_mask &= chain_filter
    
    if quality_tier_max is not None and 'quality_tier' in res:
        tier_filter = res['quality_tier'] <= quality_tier_max
        n_pass = np.sum(tier_filter)
        filter_summary.append(f"Quality tier ≤ {quality_tier_max}: {n_pass}/{n_sources} ({100*n_pass/n_sources:.1f}%)")
        quality_mask &= tier_filter
    
    # Add MCLMC tuning CV filter
    if tuning_cv_min is not None and 'tuned_L' in res:
        tuning_stats = compute_mclmc_tuning_stats(res['tuned_L'], res['tuned_step_size'])
        if tuning_stats is not None:
            tuning_filter = tuning_stats['L_cv'] > tuning_cv_min
            n_pass = np.sum(tuning_filter)
            filter_summary.append(f"L_cv > {tuning_cv_min}: {n_pass}/{n_sources} ({100*n_pass/n_sources:.1f}%)")
            quality_mask &= tuning_filter
    
    # Add spectral completeness filter
    if frac_sampled_min is not None and 'frac_sampled_102' in res:
        frac_filter = res['frac_sampled_102'] >= frac_sampled_min
        n_pass = np.sum(frac_filter)
        filter_summary.append(f"frac_sampled_102 ≥ {frac_sampled_min}: {n_pass}/{n_sources} ({100*n_pass/n_sources:.1f}%)")
        quality_mask &= frac_filter

    # Add SNR filter
    if snr_array is not None and snr_min is not None:
        snr_filter = np.array(snr_array) >= snr_min
        n_pass = np.sum(snr_filter)
        filter_summary.append(f"snr_quad ≥ {snr_min}: {n_pass}/{n_sources} ({100*n_pass/n_sources:.1f}%)")
        quality_mask &= snr_filter
    elif snr_min is not None:
        print(f"  ⚠ snr_min={snr_min} requested but snr_quad array not available — SNR cut skipped.")

    # Add band-4 negative-flux flag filter
    if band4_neg_flag is not None:
        b4_filter = ~np.array(band4_neg_flag, dtype=bool)
        n_pass = np.sum(b4_filter)
        filter_summary.append(f"band4_neg_flux_ok: {n_pass}/{n_sources} ({100*n_pass/n_sources:.1f}%)")
        quality_mask &= b4_filter

    n_passing = np.sum(quality_mask)
    
    print(f"\n{'='*70}")
    print(f"Generating summary plots for: {result_name}")
    print(f"Number of sources: {n_sources:,}")
    
    if filter_summary:
        print(f"\n{'='*70}")
        print(f"QUALITY FILTERING SUMMARY")
        print(f"{'='*70}")
        for line in filter_summary:
            print(f"  {line}")
        print(f"\nFinal: {n_passing}/{n_sources} sources pass all filters ({100*n_passing/n_sources:.1f}%)")
    
    print(f"{'='*70}")
    
    # Compute derived quantities
    pae_bias = res['z_med'] - res['ztrue']
    pae_unc = 0.5 * (res['err_low'] + res['err_high'])
    pae_zscore = pae_bias / pae_unc
    pae_dzoneplusz = pae_unc / (1 + res['z_med'])

    tf_bias = res['z_TF'] - res['ztrue']
    tf_unc = res['z_TF_err']
    # Guard against zero TF uncertainties (missing TF data returns zeros)
    tf_unc_safe = np.where(tf_unc > 0, tf_unc, np.nan)
    tf_zscore = tf_bias / tf_unc_safe
    tf_dzoneplusz = tf_unc_safe / (1 + res['z_TF'])

    # Require finite values in the key redshift quantities.  Sources with NaN
    # z_med, ztrue, or unc would otherwise appear as garbage points at (0,0)
    # or NaN-coordinate hexbin artifacts.  Fold these into quality_mask so
    # every downstream per-figure mask inherits the finite check automatically.
    _finite_pae = (np.isfinite(res['z_med']) & np.isfinite(res['ztrue'])
                   & np.isfinite(pae_unc) & (pae_unc > 0))
    _finite_tf  = (np.isfinite(res['z_TF'])  & (res['z_TF'] != 0)
                   & np.isfinite(res['ztrue']))
    # quality_mask gates PAE panels; we also track a TF-specific finite mask.
    # For PAE: require finite PAE quantities.
    # For TF panels: combined later as tf_mask & _finite_tf.
    _n_nonfinite_pae = int(np.sum(quality_mask & ~_finite_pae))
    _n_nonfinite_tf  = int(np.sum(quality_mask & ~_finite_tf))
    if _n_nonfinite_pae > 0:
        print(f"  ⚠ Dropping {_n_nonfinite_pae:,} sources with non-finite PAE z/unc values from plots.")
    if _n_nonfinite_tf > 0:
        print(f"  ⚠ Dropping {_n_nonfinite_tf:,} sources with non-finite/zero TF z values from TF plots.")
    quality_mask = quality_mask & _finite_pae
    quality_mask_tf = quality_mask & _finite_tf  # used for TF-specific panels

    _hexbin_mincnt = 1
    
    # Define uncertainty bins
    unc_bins = [
        (0.0, 0.003, r'$\sigma_{z/(1+z)} < 0.003$'),
        (0.003, 0.01, r'$0.003 \leq \sigma_{z/(1+z)} < 0.01$'),
        (0.01, 0.03, r'$0.01 \leq \sigma_{z/(1+z)} < 0.03$'),
        (0.03, 0.1, r'$0.03 \leq \sigma_{z/(1+z)} < 0.1$'),
        (0.1, 0.2, r'$0.1 \leq \sigma_{z/(1+z)} < 0.2$')
    ]
    
    # ====================================================================
    # FIGURE 1: Z-SCORE DISTRIBUTIONS BY UNCERTAINTY BIN
    # ====================================================================
    print("\nGenerating Figure 1: Z-score distributions by uncertainty bin...")
    fig1, axes = plt.subplots(2, 3, figsize=(10, 7))
    axes = axes.flatten()
    
    bins_zscore = np.linspace(zscore_range[0], zscore_range[1], 30)
    
    stats_summary = {}
    
    for idx, (unc_min, unc_max, label) in enumerate(unc_bins):
        ax = axes[idx]
        
        # Create masks for this uncertainty bin (combined with quality mask)
        pae_mask = (pae_dzoneplusz >= unc_min) & (pae_dzoneplusz < unc_max) & quality_mask
        tf_mask = (tf_dzoneplusz >= unc_min) & (tf_dzoneplusz < unc_max) & quality_mask_tf

        n_pae = np.sum(pae_mask)
        n_tf = np.sum(tf_mask)
        
        # Track maximum histogram height for y-limit setting
        max_count = 0
        
        if n_pae > 0:
            pae_zs_bin = pae_zscore[pae_mask]
            pae_zs_bin = pae_zs_bin[np.isfinite(pae_zs_bin)]
            
            counts_pae, _, _ = ax.hist(pae_zs_bin, bins=bins_zscore, alpha=0.7, color='blue', 
                   histtype='step', linewidth=2)
            max_count = max(max_count, np.max(counts_pae))
            ax.axvline(np.nanmedian(pae_zs_bin), color='blue', linestyle='--', linewidth=2)
            
            # Compute statistics
            pae_median = np.nanmedian(pae_zs_bin)
            pae_std = np.nanstd(pae_zs_bin)
            pae_nmad = 1.4826 * np.nanmedian(np.abs(pae_zs_bin - np.nanmedian(pae_zs_bin)))
        else:
            pae_median = pae_std = pae_nmad = np.nan
        
        if n_tf > 0:
            tf_zs_bin = tf_zscore[tf_mask]
            tf_zs_bin = tf_zs_bin[np.isfinite(tf_zs_bin)]
            
            counts_tf, _, _ = ax.hist(tf_zs_bin, bins=bins_zscore, alpha=0.7, color='black',
                   histtype='step', linewidth=2)
            max_count = max(max_count, np.max(counts_tf))
            ax.axvline(np.nanmedian(tf_zs_bin), color='black', linestyle='--', linewidth=2)
            
            # Compute statistics
            tf_median = np.nanmedian(tf_zs_bin)
            tf_std = np.nanstd(tf_zs_bin)
            tf_nmad = 1.4826 * np.nanmedian(np.abs(tf_zs_bin - np.nanmedian(tf_zs_bin)))
        else:
            tf_median = tf_std = tf_nmad = np.nan
        
        # Store statistics
        stats_summary[f'bin_{idx}'] = {
            'unc_range': (unc_min, unc_max),
            'n_pae': n_pae,
            'n_tf': n_tf,
            'pae_median': pae_median,
            'pae_std': pae_std,
            'pae_nmad': pae_nmad,
            'tf_median': tf_median,
            'tf_std': tf_std,
            'tf_nmad': tf_nmad
        }
        
        ax.axvline(0, color='red', linestyle='-', alpha=0.3, linewidth=1.5)
        ax.set_xlabel('Z-score', fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel('Count', fontsize=11)
        ax.set_title(label, fontsize=13)
        ax.grid(alpha=0.3)
        ax.set_xlim(zscore_range)
        
        # Set y-limit to 130% of max count to give space for legend
        if max_count > 0:
            ax.set_ylim(0, max_count * 1.3)
        
        # Create unified legend with counts and statistics
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2, 
                   label=f'PAE (n={n_pae}): μ={pae_median:.2f}, σ={pae_nmad:.2f}'),
            Line2D([0], [0], color='black', linewidth=2,
                   label=f'TF (n={n_tf}): μ={tf_median:.2f}, σ={tf_nmad:.2f}')
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc=2, framealpha=0.9)
    
    # Sixth panel: CDF of absolute z-scores
    ax = axes[5]
    pae_mask_cdf = (pae_dzoneplusz < 0.2) & quality_mask
    tf_mask_cdf = (tf_dzoneplusz < 0.2) & quality_mask_tf
    
    pae_abs_zscore = np.abs(pae_zscore[pae_mask_cdf])
    pae_abs_zscore = pae_abs_zscore[np.isfinite(pae_abs_zscore)]
    
    tf_abs_zscore = np.abs(tf_zscore[tf_mask_cdf])
    tf_abs_zscore = tf_abs_zscore[np.isfinite(tf_abs_zscore)]
    
    pae_sorted = np.sort(pae_abs_zscore)
    tf_sorted = np.sort(tf_abs_zscore)
    
    pae_cdf = np.arange(1, len(pae_sorted) + 1) / len(pae_sorted)
    tf_cdf = np.arange(1, len(tf_sorted) + 1) / len(tf_sorted)
    
    ax.plot(pae_sorted, pae_cdf, color='blue', linewidth=2)
    ax.plot(tf_sorted, tf_cdf, color='black', linewidth=2)
    
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(0.68, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('|Z-score|', fontsize=11)
    ax.set_ylabel('Cumulative Fraction', fontsize=11)
    ax.set_title(r'$\sigma_{z/(1+z)} < 0.2$', fontsize=13)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    
    frac_pae_within1 = np.sum(pae_abs_zscore < 1) / len(pae_abs_zscore) if len(pae_abs_zscore) > 0 else 0
    frac_tf_within1 = np.sum(tf_abs_zscore < 1) / len(tf_abs_zscore) if len(tf_abs_zscore) > 0 else 0
    
    legend_elements = [
        Line2D([0], [0], color='blue', linewidth=2, 
               label=f'PAE (n={len(pae_sorted)}): {frac_pae_within1:.1%} < 1'),
        Line2D([0], [0], color='black', linewidth=2,
               label=f'TF (n={len(tf_sorted)}): {frac_tf_within1:.1%} < 1'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
               label='|z-score| = 1'),
        Line2D([0], [0], color='gray', linestyle=':', linewidth=1.5,
               label='68% line')
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc=4, framealpha=0.9)
    
    fig1.suptitle(f'{result_name}\n(N={n_sources:,})', fontsize=14, fontweight='normal')
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    if save_dir:
        zscore_dir = Path(save_dir) / 'zscore'
        zscore_dir.mkdir(parents=True, exist_ok=True)
        fig1_path = zscore_dir / f'{result_name}_zscore_by_uncertainty.png'
        fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig1_path.relative_to(save_dir)}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig1)
    
    # ====================================================================
    # FIGURE 2: REDSHIFT DISTRIBUTIONS BY UNCERTAINTY BIN
    # ====================================================================
    print("Generating Figure 2: Redshift distributions by uncertainty bin...")
    fig2, axes2 = plt.subplots(2, 3, figsize=(10, 7))
    axes2 = axes2.flatten()
    z_bins = np.linspace(0.0, 3.0, 40)

    for idx, (unc_min, unc_max, label) in enumerate(unc_bins):
        ax = axes2[idx]
        pae_mask = (pae_dzoneplusz >= unc_min) & (pae_dzoneplusz < unc_max) & quality_mask
        tf_mask = (tf_dzoneplusz >= unc_min) & (tf_dzoneplusz < unc_max) & quality_mask_tf

        n_pae = np.sum(pae_mask)
        n_tf = np.sum(tf_mask)

        if n_pae > 0:
            pae_zs = res['z_med'][pae_mask]
            pae_zs = pae_zs[np.isfinite(pae_zs)]
            ax.hist(pae_zs, bins=z_bins, histtype='step', color='blue', linewidth=2,
                    label=f'PAE (n={n_pae})')

        if n_tf > 0:
            tf_zs = res['z_TF'][tf_mask]
            tf_zs = tf_zs[np.isfinite(tf_zs)]
            ax.hist(tf_zs, bins=z_bins, histtype='step', color='black', linewidth=2,
                    label=f'TF (n={n_tf})')

        ax.set_xlim(0.0, 3.0)
        ax.set_xlabel('Redshift', fontsize=11)
        if idx % 3 == 0:
            ax.set_ylabel('Count', fontsize=11)
        ax.set_title(label, fontsize=13)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc=1, framealpha=0.9)

    fig2.suptitle(f'{result_name}', fontsize=14, fontweight='normal')
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    if save_dir:
        scatter_dir = Path(save_dir) / 'scatter'
        scatter_dir.mkdir(parents=True, exist_ok=True)
        fig2_path = scatter_dir / f'{result_name}_redshift_by_uncertainty.png'
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig2_path.relative_to(save_dir)}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig2)
    
    # ====================================================================
    # FIGURE 3: REDSHIFT COMPARISON PLOTS
    # ====================================================================
    print("Generating Figure 3: Redshift comparison plots...")
    zmin, zmax = 0.0, 3.0
    alpha = 0.02
    zsp = np.linspace(zmin, zmax, 100)
    
    fig3, axes3 = plt.subplots(2, 2, figsize=(9, 8))
    
    # Overall comparison - PAE
    dz_max = 0.2
    pae_mask_plot = (pae_dzoneplusz < dz_max) & quality_mask
    n_pae_plot = np.sum(pae_mask_plot)
    
    # Compute statistics for PAE
    _, _, pae_bias_stat, pae_NMAD, pae_cond_outl, pae_outl_rate, pae_outl_rate_15pct = compute_redshift_stats(
        res['z_med'][pae_mask_plot], 
        res['ztrue'][pae_mask_plot], 
        sigma_z_select=pae_unc[pae_mask_plot],
        nsig_outlier=3
    )
    pae_med_sigz = np.median(pae_dzoneplusz[pae_mask_plot])
    pae_plotstr = make_plotstr_count(n_pae_plot, pae_NMAD, pae_med_sigz, pae_bias_stat, pae_outl_rate * 100)
    
    if use_hexbin:
        hb = axes3[0, 0].hexbin(res['ztrue'][pae_mask_plot], res['z_med'][pae_mask_plot],
                                bins='log', cmap='plasma', mincnt=_hexbin_mincnt, gridsize=60,
                                extent=(zmin, zmax, zmin, zmax))
        if np.any(np.asarray(hb.get_array()) > 0):
            plt.colorbar(hb, ax=axes3[0, 0], label='log10(count)')
    else:
        axes3[0, 0].errorbar(res['ztrue'][pae_mask_plot], res['z_med'][pae_mask_plot],
                            yerr=[np.abs(res['err_low'])[pae_mask_plot],
                                  np.abs(res['err_high'])[pae_mask_plot]],
                            fmt='o', color='blue', alpha=alpha, markersize=2, capsize=1)
        from scipy.stats import gaussian_kde
        if n_pae_plot > 50:
            xy = np.vstack([res['ztrue'][pae_mask_plot], res['z_med'][pae_mask_plot]])
            valid_mask = np.isfinite(xy[0]) & np.isfinite(xy[1])
            if np.sum(valid_mask) > 50:
                xy_valid = xy[:, valid_mask]
                z_density = gaussian_kde(xy_valid)(xy_valid)
                axes3[0, 0].tricontour(xy_valid[0], xy_valid[1], z_density,
                                      levels=5, colors='dodgerblue', linewidths=1.5, alpha=0.8, zorder=30)
    axes3[0, 0].plot(zsp, zsp, 'k--', linewidth=1.5, zorder=10)
    axes3[0, 0].text(1.5, 0.1, pae_plotstr,
                    fontsize=9, verticalalignment='bottom', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    axes3[0, 0].set_xlabel('Spec-z', fontsize=11)
    axes3[0, 0].set_ylabel(r'$\hat{z}$ (PAE)', fontsize=11)
    axes3[0, 0].set_title(f'PAE: ' + r'$\sigma_{z/(1+z)} < $' + f'{dz_max} (N={n_pae_plot:,})', 
                         fontsize=13)
    axes3[0, 0].set_xlim(zmin, zmax)
    axes3[0, 0].set_ylim(zmin, zmax)
    axes3[0, 0].grid(alpha=0.3)
    
    # Overall comparison - TF
    tf_mask_plot = (tf_dzoneplusz < dz_max) & quality_mask_tf
    n_tf_plot = np.sum(tf_mask_plot)
    
    # Compute statistics for TF
    _, _, tf_bias_stat, tf_NMAD, tf_cond_outl, tf_outl_rate, tf_outl_rate_15pct = compute_redshift_stats(
        res['z_TF'][tf_mask_plot], 
        res['ztrue'][tf_mask_plot], 
        sigma_z_select=tf_unc[tf_mask_plot],
        nsig_outlier=3
    )
    tf_med_sigz = np.median(tf_dzoneplusz[tf_mask_plot])
    tf_plotstr = make_plotstr_count(n_tf_plot, tf_NMAD, tf_med_sigz, tf_bias_stat, tf_outl_rate * 100)
    
    if use_hexbin:
        hb = axes3[0, 1].hexbin(res['ztrue'][tf_mask_plot], res['z_TF'][tf_mask_plot],
                                bins='log', cmap='plasma', mincnt=_hexbin_mincnt, gridsize=60,
                                extent=(zmin, zmax, zmin, zmax))
        if np.any(np.asarray(hb.get_array()) > 0):
            plt.colorbar(hb, ax=axes3[0, 1], label='log10(count)')
    else:
        axes3[0, 1].errorbar(res['ztrue'][tf_mask_plot], res['z_TF'][tf_mask_plot],
                            yerr=res['z_TF_err'][tf_mask_plot],
                            fmt='o', color='black', alpha=alpha, markersize=2, capsize=1)
        if n_tf_plot > 50:
            xy_tf = np.vstack([res['ztrue'][tf_mask_plot], res['z_TF'][tf_mask_plot]])
            valid_mask_tf = np.isfinite(xy_tf[0]) & np.isfinite(xy_tf[1])
            if np.sum(valid_mask_tf) > 50:
                xy_tf_valid = xy_tf[:, valid_mask_tf]
                z_density_tf = gaussian_kde(xy_tf_valid)(xy_tf_valid)
                axes3[0, 1].tricontour(xy_tf_valid[0], xy_tf_valid[1], z_density_tf,
                                      levels=5, colors='orangered', linewidths=1.5, alpha=0.8, zorder=30)
    axes3[0, 1].plot(zsp, zsp, 'k--', linewidth=1.5, zorder=10)
    axes3[0, 1].text(1.5, 0.1, tf_plotstr, 
                    fontsize=9, verticalalignment='bottom', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    axes3[0, 1].set_xlabel('Spec-z', fontsize=11)
    axes3[0, 1].set_ylabel(r'$\hat{z}$ (TF)', fontsize=11)
    axes3[0, 1].set_title(f'TF: ' + r'$\sigma_{z/(1+z)} < $' + f'{dz_max} (N={n_tf_plot:,})', 
                         fontsize=13)
    axes3[0, 1].set_xlim(zmin, zmax)
    axes3[0, 1].set_ylim(zmin, zmax)
    axes3[0, 1].grid(alpha=0.3)
    
    # Bias histograms
    bins_bias = np.linspace(-0.5, 0.5, 50)
    
    axes3[1, 0].hist(pae_bias, bins=bins_bias, histtype='step', label='PAE', color='blue', linewidth=2)
    axes3[1, 0].axvline(np.nanmedian(pae_bias), color='blue', linestyle='--', linewidth=2)
    axes3[1, 0].hist(tf_bias, bins=bins_bias, histtype='step', label='TF', color='black', linewidth=2)
    axes3[1, 0].axvline(np.nanmedian(tf_bias), color='black', linestyle='--', linewidth=2)
    axes3[1, 0].set_xlabel(r'$\hat{z} - z_{true}$', fontsize=13)
    axes3[1, 0].set_ylabel('Count', fontsize=13)
    axes3[1, 0].set_title('Bias Distribution (All)', fontsize=13)
    axes3[1, 0].legend(fontsize=11)
    axes3[1, 0].grid(alpha=0.3)
    
    # Uncertainty histogram
    bins_unc = np.logspace(-4, 0, 30)
    
    axes3[1, 1].hist(pae_dzoneplusz, bins=bins_unc, histtype='step', color='blue', 
                    label='PAE', linewidth=2)
    axes3[1, 1].hist(tf_dzoneplusz, bins=bins_unc, histtype='step', color='black', 
                    label='TF', linewidth=2)
    axes3[1, 1].set_xscale('log')
    axes3[1, 1].set_xlabel(r'$\sigma_{z/(1+z)}$', fontsize=13)
    axes3[1, 1].set_ylabel('Count', fontsize=13)
    axes3[1, 1].set_title('Uncertainty Distribution', fontsize=13)
    axes3[1, 1].legend(fontsize=11)
    axes3[1, 1].grid(alpha=0.3)
    
    fig3.suptitle(f'{result_name}', fontsize=14, fontweight='normal')
    plt.tight_layout()
    
    if save_dir:
        fig3_path = Path(save_dir) / f'{result_name}_redshift_comparison.png'
        fig3.savefig(fig3_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig3_path.name}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig3)
    
    # ====================================================================
    # FIGURE 4: DETAILED COMPARISONS BY UNCERTAINTY THRESHOLD
    # ====================================================================
    print("Generating Figure 4: Detailed comparisons by uncertainty threshold...")
    
    # Define uncertainty thresholds to loop through
    dz_thresholds = [0.003, 0.01, 0.03, 0.1, 0.2]
    zscoremin, zscoremax = -8, 8
    # Adaptive x-axis ranges for normalized bias plots (coarser bins need larger range)
    bias_norm_ranges = [
        (-0.1, 0.1),    # d=0: σ_z < 0.003 (finest)
        (-0.15, 0.15),    # d=1: 0.003 ≤ σ_z < 0.01
        (-0.2, 0.2),    # d=2: 0.01 ≤ σ_z < 0.03
        (-0.5, 0.5),    # d=3: 0.03 ≤ σ_z < 0.1 (second coarsest)
        (-1.0, 1.0),    # d=4: 0.1 ≤ σ_z < 0.2 (coarsest)
    ]
    bins_zscore = np.linspace(zscoremin, zscoremax, 30)
    
    for d, dz_thresh in enumerate(dz_thresholds):
        # Create masks for this threshold

        if d==0:
            print(f"\n  Processing dz/(1+z) < {dz_thresh} ...")

            pae_mask = (pae_dzoneplusz < dz_thresh) & quality_mask
            tf_mask = (tf_dzoneplusz < dz_thresh) & quality_mask_tf
        else:
            print(f"  Processing dz/(1+z) < {dz_thresh} ...")

            pae_mask = (pae_dzoneplusz >= dz_thresholds[d-1]) & (pae_dzoneplusz < dz_thresh) & quality_mask
            tf_mask = (tf_dzoneplusz >= dz_thresholds[d-1]) & (tf_dzoneplusz < dz_thresh) & quality_mask_tf
            
        n_pae = np.sum(pae_mask)
        n_tf = np.sum(tf_mask)
        
        # ====================================================================
        # Sub-figure 4a: 3-panel scatter plots (PAE vs spec-z, TF vs spec-z, PAE vs TF)
        # ====================================================================
        fig4a, axes4a = plt.subplots(1, 3, figsize=(11, 4))
        
        if dz_thresh < 0.1:
            alpha_scatter = 0.05
        else:
            alpha_scatter = 0.02

        zsp = np.linspace(zmin, zmax, 100)
        
        # Compute statistics for PAE (this threshold bin)
        if n_pae > 10:  # Need sufficient sources for stats
            _, _, pae_bias_d, pae_NMAD_d, _, pae_outl_d, pae_outl_d_15pct = compute_redshift_stats(
                res['z_med'][pae_mask], 
                res['ztrue'][pae_mask], 
                sigma_z_select=pae_unc[pae_mask],
                nsig_outlier=3
            )
            pae_med_sigz_d = np.median(pae_dzoneplusz[pae_mask])
            pae_plotstr_d = make_plotstr_count(n_pae, pae_NMAD_d, pae_med_sigz_d, pae_bias_d, pae_outl_d * 100)
        else:
            pae_plotstr_d = f'N={n_pae}\n(insufficient for stats)'
        
        # Panel 1: PAE vs spec-z
        if use_hexbin:
            hb = axes4a[0].hexbin(res['ztrue'][pae_mask], res['z_med'][pae_mask],
                                  bins='log', cmap='plasma', mincnt=1, gridsize=50,
                                  extent=(zmin, zmax, zmin, zmax))
            if np.any(np.asarray(hb.get_array()) > 0):
                plt.colorbar(hb, ax=axes4a[0], label='log10(count)')
        else:
            axes4a[0].errorbar(res['ztrue'][pae_mask], res['z_med'][pae_mask],
                              yerr=[np.abs(res['err_low'])[pae_mask], np.abs(res['err_high'])[pae_mask]],
                              fmt='o', color='k', alpha=alpha_scatter, markersize=3, capsize=2)
            from scipy.stats import gaussian_kde
            if n_pae > 50:
                xy_pae = np.vstack([res['ztrue'][pae_mask], res['z_med'][pae_mask]])
                valid_mask_pae = np.isfinite(xy_pae[0]) & np.isfinite(xy_pae[1])
                if np.sum(valid_mask_pae) > 50:
                    xy_pae_valid = xy_pae[:, valid_mask_pae]
                    z_density_pae = gaussian_kde(xy_pae_valid)(xy_pae_valid)
                    axes4a[0].tricontour(xy_pae_valid[0], xy_pae_valid[1], z_density_pae,
                                        levels=5, colors='dodgerblue', linewidths=1.5, alpha=0.8, zorder=30)
        axes4a[0].plot(zsp, zsp, 'k--', linewidth=1.5, zorder=10)
        axes4a[0].text(1.5, 0.1, pae_plotstr_d,
                      fontsize=8, verticalalignment='bottom', 
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        axes4a[0].set_xlim(zmin, zmax)
        axes4a[0].set_ylim(zmin, zmax)
        axes4a[0].set_xlabel('spec-z', fontsize=14)
        axes4a[0].set_ylabel(r'$\hat{z}$ (PAE)', fontsize=14)
        if d == 0:
            title_pae = r'$\sigma_{z/(1+z)}^{PAE} < $' + f'{dz_thresh}: {n_pae}'
        else:
            title_pae = f'{dz_thresholds[d-1]:.3f}' + r'$ \leq \sigma_{z/(1+z)}^{PAE} < $' + f'{dz_thresh}: {n_pae}'
        axes4a[0].set_title(title_pae, fontsize=12)
        axes4a[0].grid(alpha=0.3)
        
        # Compute statistics for TF (this threshold bin)
        if n_tf > 10:  # Need sufficient sources for stats
            _, _, tf_bias_d, tf_NMAD_d, _, tf_outl_d, tf_outl_d_15pct = compute_redshift_stats(
                res['z_TF'][tf_mask], 
                res['ztrue'][tf_mask], 
                sigma_z_select=tf_unc[tf_mask],
                nsig_outlier=3
            )
            tf_med_sigz_d = np.median(tf_dzoneplusz[tf_mask])
            tf_plotstr_d = make_plotstr_count(n_tf, tf_NMAD_d, tf_med_sigz_d, tf_bias_d, tf_outl_d * 100)
        else:
            tf_plotstr_d = f'N={n_tf}\n(insufficient for stats)'
        
        # Panel 2: TF vs spec-z
        if use_hexbin:
            hb = axes4a[1].hexbin(res['ztrue'][tf_mask], res['z_TF'][tf_mask],
                                  bins='log', cmap='plasma', mincnt=1, gridsize=50,
                                  extent=(zmin, zmax, zmin, zmax))
            if np.any(np.asarray(hb.get_array()) > 0):
                plt.colorbar(hb, ax=axes4a[1], label='log10(count)')
        else:
            axes4a[1].errorbar(res['ztrue'][tf_mask], res['z_TF'][tf_mask],
                              yerr=res['z_TF_err'][tf_mask],
                              fmt='o', color='k', alpha=alpha_scatter, markersize=3, capsize=2)
            if n_tf > 50:
                xy_tf_4a = np.vstack([res['ztrue'][tf_mask], res['z_TF'][tf_mask]])
                valid_mask_tf_4a = np.isfinite(xy_tf_4a[0]) & np.isfinite(xy_tf_4a[1])
                if np.sum(valid_mask_tf_4a) > 50:
                    xy_tf_4a_valid = xy_tf_4a[:, valid_mask_tf_4a]
                    z_density_tf_4a = gaussian_kde(xy_tf_4a_valid)(xy_tf_4a_valid)
                    axes4a[1].tricontour(xy_tf_4a_valid[0], xy_tf_4a_valid[1], z_density_tf_4a,
                                        levels=5, colors='orangered', linewidths=1.5, alpha=0.8, zorder=30)
        axes4a[1].plot(zsp, zsp, 'k--', linewidth=1.5, zorder=10)
        axes4a[1].text(1.5, 0.1, tf_plotstr_d,
                      fontsize=8, verticalalignment='bottom',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        axes4a[1].set_xlim(zmin, zmax)
        axes4a[1].set_ylim(zmin, zmax)
        axes4a[1].set_xlabel('spec-z', fontsize=14)
        axes4a[1].set_ylabel(r'$\hat{z}$ (template fitting)', fontsize=14)
        if d == 0:
            title_tf = r'$\sigma_{z/(1+z)}^{TF} < $' + f'{dz_thresh}: {n_tf}'
        else:
            title_tf = f'{dz_thresholds[d-1]:.3f}' + r'$ \leq \sigma_{z/(1+z)}^{TF} < $' + f'{dz_thresh}: {n_tf}'
        axes4a[1].set_title(title_tf, fontsize=12)
        axes4a[1].grid(alpha=0.3)
        
        # Panel 3: PAE vs TF
        if use_hexbin:
            hb = axes4a[2].hexbin(res['z_TF'][pae_mask], res['z_med'][pae_mask],
                                  bins='log', cmap='plasma', mincnt=1, gridsize=50,
                                  extent=(zmin, zmax, zmin, zmax))
            if np.any(np.asarray(hb.get_array()) > 0):
                plt.colorbar(hb, ax=axes4a[2], label='log10(count)')
        else:
            axes4a[2].errorbar(res['z_TF'][pae_mask], res['z_med'][pae_mask],
                              yerr=[np.abs(res['err_low'])[pae_mask], np.abs(res['err_high'])[pae_mask]],
                              fmt='o', color='k', alpha=alpha_scatter, markersize=3, capsize=2)
            if n_pae > 50:
                xy_pae_tf = np.vstack([res['z_TF'][pae_mask], res['z_med'][pae_mask]])
                valid_mask_pae_tf = np.isfinite(xy_pae_tf[0]) & np.isfinite(xy_pae_tf[1])
                if np.sum(valid_mask_pae_tf) > 50:
                    xy_pae_tf_valid = xy_pae_tf[:, valid_mask_pae_tf]
                    z_density_pae_tf = gaussian_kde(xy_pae_tf_valid)(xy_pae_tf_valid)
                    axes4a[2].tricontour(xy_pae_tf_valid[0], xy_pae_tf_valid[1], z_density_pae_tf,
                                        levels=5, colors='magenta', linewidths=1.5, alpha=0.8, zorder=30)
        axes4a[2].plot(zsp, zsp, 'k--', linewidth=1.5, zorder=10)
        axes4a[2].set_xlim(zmin, zmax)
        axes4a[2].set_ylim(zmin, zmax)
        axes4a[2].set_xlabel(r'$\hat{z}$ (template fitting)', fontsize=14)
        axes4a[2].set_ylabel(r'$\hat{z}$ (PAE)', fontsize=14)
        axes4a[2].set_title('PAE vs TF', fontsize=12)
        axes4a[2].grid(alpha=0.3)
        
        fig4a.suptitle(f'{result_name} (N={n_sources:,})', fontsize=16, y=1.0)
        plt.tight_layout()
        
        if save_dir:
            scatter_dir = Path(save_dir) / 'scatter'
            scatter_dir.mkdir(parents=True, exist_ok=True)
            if d==0:
                fig4a_path = scatter_dir / f'{result_name}_scatter_dz{dz_thresh:.3f}.png'
            else:
                fig4a_path = scatter_dir / f'{result_name}_scatter_{dz_thresholds[d-1]:.3f}_dz_{dz_thresh:.3f}.png'
            fig4a.savefig(fig4a_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {fig4a_path.relative_to(save_dir)}")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig4a)

        # ====================================================================
        # Sub-figure 4a (zoom): zoomed to z < 1.5 for 3% and better bins
        # ====================================================================
        if dz_thresh <= 0.03:
            zmax_zoom = 1.5
            zsp_zoom = np.linspace(0, zmax_zoom, 100)
            fig4a_zoom, axes4a_zoom = plt.subplots(1, 3, figsize=(11, 4))

            # Panel 1: PAE vs spec-z (zoomed)
            pae_mask_zoom = pae_mask & (res['ztrue'] < zmax_zoom)
            n_pae_zoom = np.sum(pae_mask_zoom)
            if use_hexbin and n_pae_zoom > 0:
                hb = axes4a_zoom[0].hexbin(res['ztrue'][pae_mask_zoom], res['z_med'][pae_mask_zoom],
                                           bins='log', cmap='plasma', mincnt=1, gridsize=50,
                                           extent=(0, zmax_zoom, 0, zmax_zoom))
                if np.any(np.asarray(hb.get_array()) > 0):
                    plt.colorbar(hb, ax=axes4a_zoom[0], label='log10(count)')
            axes4a_zoom[0].plot(zsp_zoom, zsp_zoom, 'k--', linewidth=1.5, zorder=10)
            axes4a_zoom[0].text(zmax_zoom * 0.5, 0.05, pae_plotstr_d,
                                fontsize=8, verticalalignment='bottom',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
            axes4a_zoom[0].set_xlim(0, zmax_zoom)
            axes4a_zoom[0].set_ylim(0, zmax_zoom)
            axes4a_zoom[0].set_xlabel('spec-z', fontsize=14)
            axes4a_zoom[0].set_ylabel(r'$\hat{z}$ (PAE)', fontsize=14)
            axes4a_zoom[0].set_title(title_pae, fontsize=12)
            axes4a_zoom[0].grid(alpha=0.3)

            # Panel 2: TF vs spec-z (zoomed)
            tf_mask_zoom = tf_mask & (res['ztrue'] < zmax_zoom)
            n_tf_zoom = np.sum(tf_mask_zoom)
            if use_hexbin and n_tf_zoom > 0:
                hb = axes4a_zoom[1].hexbin(res['ztrue'][tf_mask_zoom], res['z_TF'][tf_mask_zoom],
                                           bins='log', cmap='plasma', mincnt=1, gridsize=50,
                                           extent=(0, zmax_zoom, 0, zmax_zoom))
                if np.any(np.asarray(hb.get_array()) > 0):
                    plt.colorbar(hb, ax=axes4a_zoom[1], label='log10(count)')
            axes4a_zoom[1].plot(zsp_zoom, zsp_zoom, 'k--', linewidth=1.5, zorder=10)
            axes4a_zoom[1].text(zmax_zoom * 0.5, 0.05, tf_plotstr_d,
                                fontsize=8, verticalalignment='bottom',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
            axes4a_zoom[1].set_xlim(0, zmax_zoom)
            axes4a_zoom[1].set_ylim(0, zmax_zoom)
            axes4a_zoom[1].set_xlabel('spec-z', fontsize=14)
            axes4a_zoom[1].set_ylabel(r'$\hat{z}$ (template fitting)', fontsize=14)
            axes4a_zoom[1].set_title(title_tf, fontsize=12)
            axes4a_zoom[1].grid(alpha=0.3)

            # Panel 3: PAE vs TF (zoomed)
            pae_tf_mask_zoom = pae_mask & (res['z_TF'] < zmax_zoom)
            n_pae_tf_zoom = np.sum(pae_tf_mask_zoom)
            if use_hexbin and n_pae_tf_zoom > 0:
                hb = axes4a_zoom[2].hexbin(res['z_TF'][pae_tf_mask_zoom], res['z_med'][pae_tf_mask_zoom],
                                           bins='log', cmap='plasma', mincnt=1, gridsize=50,
                                           extent=(0, zmax_zoom, 0, zmax_zoom))
                if np.any(np.asarray(hb.get_array()) > 0):
                    plt.colorbar(hb, ax=axes4a_zoom[2], label='log10(count)')
            axes4a_zoom[2].plot(zsp_zoom, zsp_zoom, 'k--', linewidth=1.5, zorder=10)
            axes4a_zoom[2].set_xlim(0, zmax_zoom)
            axes4a_zoom[2].set_ylim(0, zmax_zoom)
            axes4a_zoom[2].set_xlabel(r'$\hat{z}$ (template fitting)', fontsize=14)
            axes4a_zoom[2].set_ylabel(r'$\hat{z}$ (PAE)', fontsize=14)
            axes4a_zoom[2].set_title('PAE vs TF', fontsize=12)
            axes4a_zoom[2].grid(alpha=0.3)

            fig4a_zoom.suptitle(f'{result_name} — z < {zmax_zoom} (N={n_sources:,})', fontsize=16, y=1.0)
            plt.tight_layout()

            if save_dir:
                scatter_dir = Path(save_dir) / 'scatter'
                scatter_dir.mkdir(parents=True, exist_ok=True)
                if d == 0:
                    fig4a_zoom_path = scatter_dir / f'{result_name}_scatter_dz{dz_thresh:.3f}_zoom{zmax_zoom}.png'
                else:
                    fig4a_zoom_path = scatter_dir / f'{result_name}_scatter_{dz_thresholds[d-1]:.3f}_dz_{dz_thresh:.3f}_zoom{zmax_zoom}.png'
                fig4a_zoom.savefig(fig4a_zoom_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved: {fig4a_zoom_path.relative_to(save_dir)}")

            if show_plots:
                plt.show()
            else:
                plt.close(fig4a_zoom)

        # ====================================================================
        # Sub-figure 4b: Normalized bias histogram
        # ====================================================================
        fig4b, ax4b = plt.subplots(figsize=(6, 4))
        
        # Get adaptive x-axis range for this uncertainty bin
        bias_min, bias_max = bias_norm_ranges[d]
        bins_bias_norm = np.linspace(bias_min, bias_max, 40)
        
        # Compute normalized bias
        pae_bias_norm = pae_bias[pae_mask] / (1 + res['ztrue'][pae_mask])
        tf_bias_norm = tf_bias[tf_mask] / (1 + res['ztrue'][tf_mask])
        
        ax4b.hist(pae_bias_norm, bins=bins_bias_norm, histtype='step', label='PAE', color='b', linewidth=2)
        ax4b.axvline(np.nanmedian(pae_bias_norm), color='b', linestyle='--', linewidth=2, label='Median (PAE)')
        ax4b.hist(tf_bias_norm, bins=bins_bias_norm, histtype='step', label='TF', color='k', linewidth=2)
        ax4b.axvline(np.nanmedian(tf_bias_norm), color='k', linestyle='--', linewidth=2, label='Median (TF)')
        
        ax4b.set_xlabel(r'$\Delta z / (1+z_{spec})$', fontsize=14)
        ax4b.set_ylabel('Count', fontsize=14)
        if d == 0:
            title_str = r'$\sigma_{z/(1+z)} < $' + f'{dz_thresh}'
        else:
            title_str = f'{dz_thresholds[d-1]:.3f}' + r'$ \leq \sigma_{z/(1+z)} < $' + f'{dz_thresh}'
        ax4b.set_title(title_str, fontsize=13)
        ax4b.grid(alpha=0.2)
        ax4b.legend(loc=1, fontsize=11)
        ax4b.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        if save_dir:
            bias_dir = Path(save_dir) / 'bias'
            bias_dir.mkdir(parents=True, exist_ok=True)
            fig4b_path = bias_dir / f'{result_name}_bias_normalized_dz{dz_thresh:.3f}.png'
            fig4b.savefig(fig4b_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {fig4b_path.relative_to(save_dir)}")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig4b)
        
        # ====================================================================
        # Sub-figure 4c: Z-score histogram
        # ====================================================================
        fig4c, ax4c = plt.subplots(figsize=(6, 4))
        
        pae_zscore_masked = pae_zscore[pae_mask]
        tf_zscore_masked = tf_zscore[tf_mask]
        
        ax4c.hist(pae_zscore_masked, bins=bins_zscore, histtype='step', label='PAE', color='b', linewidth=2)
        pae_median_zscore = np.nanmedian(pae_zscore_masked)
        ax4c.axvline(pae_median_zscore, color='b', linestyle='--', linewidth=2, label=f'Median (PAE): {pae_median_zscore:.2f}')
        
        ax4c.hist(tf_zscore_masked, bins=bins_zscore, histtype='step', label='TF', color='k', linewidth=2)
        tf_median_zscore = np.nanmedian(tf_zscore_masked)
        ax4c.axvline(tf_median_zscore, color='k', linestyle='--', linewidth=2, label=f'Median (TF): {tf_median_zscore:.2f}')
        
        ax4c.set_xlabel('Redshift z-score', fontsize=14)
        ax4c.set_ylabel('Count', fontsize=14)
        if d == 0:
            title_str = r'$\sigma_{z/(1+z)} < $' + f'{dz_thresh}'
        else:
            title_str = f'{dz_thresholds[d-1]:.3f}' + r'$ \leq \sigma_{z/(1+z)} < $' + f'{dz_thresh}'
        ax4c.set_title(title_str, fontsize=13)
        ax4c.grid(alpha=0.2)
        ax4c.legend(loc=1, fontsize=10)
        ax4c.tick_params(labelsize=12)
        
        plt.tight_layout()
        
        if save_dir:
            fig4c_path = Path(save_dir)  / 'zscore' / f'{result_name}_zscore_dz{dz_thresh:.3f}.png'

            # fig4c_path = Path(save_dir) / f'{result_name}_zscore_dz{dz_thresh:.3f}.png'
            fig4c.savefig(fig4c_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {fig4c_path.name}")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig4c)
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS:")
    print(f"{'='*70}")
    for key, stats in stats_summary.items():
        if isinstance(stats, dict) and 'unc_range' in stats:
            print(f"\n{key} ({stats['unc_range'][0]:.4f} - {stats['unc_range'][1]:.4f}):")
            print(f"  PAE: n={stats['n_pae']}, median={stats['pae_median']:.3f}, NMAD={stats['pae_nmad']:.3f}")
            print(f"  TF:  n={stats['n_tf']}, median={stats['tf_median']:.3f}, NMAD={stats['tf_nmad']:.3f}")
    
    # ====================================================================
    # FIGURE 5: BINNED DIAGNOSTIC PLOTS (BIAS & NMAD vs PROPERTIES)
    # ====================================================================
    print("\nGenerating Figure 5: Binned diagnostic plots (bias & NMAD vs properties)...")
    
    # Calculate total SNR if weights are available
    if 'weights' in res:
        # SNR = flux / (1/sqrt(weight)) for each band, then sum in quadrature
        weights = res['weights']
        # Assume obs fluxes can be reconstructed or are stored
        # For now, use a proxy: median weight as inverse variance
        total_snr = np.sqrt(np.sum(weights, axis=1))  # Simplified SNR proxy
    else:
        total_snr = None
    
    # Get number of nonzero measurements
    if 'n_measurements_nonzero' in res:
        n_measurements = res['n_measurements_nonzero']
    elif 'weights' in res:
        n_measurements = np.sum(res['weights'] > 0, axis=1)
    else:
        n_measurements = None
    
    fig5, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel 1: Mean bias vs spec-z
    ax = axes[0, 0]
    z_bins = np.linspace(0, 3, 21)
    z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
    
    bias_by_z_pae = []
    bias_by_z_tf = []
    for i in range(len(z_bins) - 1):
        mask_pae = (res['ztrue'] >= z_bins[i]) & (res['ztrue'] < z_bins[i+1]) & quality_mask & np.isfinite(pae_bias)
        mask_tf = (res['ztrue'] >= z_bins[i]) & (res['ztrue'] < z_bins[i+1]) & np.isfinite(tf_bias)
        
        if np.sum(mask_pae) > 5:
            bias_by_z_pae.append(np.nanmedian(pae_bias[mask_pae]))
        else:
            bias_by_z_pae.append(np.nan)
            
        if np.sum(mask_tf) > 5:
            bias_by_z_tf.append(np.nanmedian(tf_bias[mask_tf]))
        else:
            bias_by_z_tf.append(np.nan)
    
    ax.plot(z_centers, bias_by_z_pae, 'o-', color='blue', linewidth=2, markersize=6, label='PAE')
    ax.plot(z_centers, bias_by_z_tf, 's-', color='black', linewidth=2, markersize=5, label='TF')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spectroscopic Redshift', fontsize=12)
    ax.set_ylabel('Median Bias (Δz)', fontsize=12)
    ax.set_title('Redshift Bias vs Spec-z', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Panel 2: NMAD vs spec-z
    ax = axes[0, 1]
    nmad_by_z_pae = []
    nmad_by_z_tf = []
    for i in range(len(z_bins) - 1):
        mask_pae = (res['ztrue'] >= z_bins[i]) & (res['ztrue'] < z_bins[i+1]) & quality_mask & np.isfinite(pae_bias)
        mask_tf = (res['ztrue'] >= z_bins[i]) & (res['ztrue'] < z_bins[i+1]) & np.isfinite(tf_bias)
        
        if np.sum(mask_pae) > 5:
            nmad_pae = 1.4826 * np.nanmedian(np.abs(pae_bias[mask_pae] - np.nanmedian(pae_bias[mask_pae])))
            nmad_by_z_pae.append(nmad_pae)
        else:
            nmad_by_z_pae.append(np.nan)
            
        if np.sum(mask_tf) > 5:
            nmad_tf = 1.4826 * np.nanmedian(np.abs(tf_bias[mask_tf] - np.nanmedian(tf_bias[mask_tf])))
            nmad_by_z_tf.append(nmad_tf)
        else:
            nmad_by_z_tf.append(np.nan)
    
    ax.plot(z_centers, nmad_by_z_pae, 'o-', color='blue', linewidth=2, markersize=6, label='PAE')
    ax.plot(z_centers, nmad_by_z_tf, 's-', color='black', linewidth=2, markersize=5, label='TF')
    ax.set_xlabel('Spectroscopic Redshift', fontsize=12)
    ax.set_ylabel('NMAD(Δz)', fontsize=12)
    ax.set_title('Redshift Scatter vs Spec-z', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_yscale('log')
    
    # Panel 3: Outlier fraction vs spec-z
    ax = axes[0, 2]
    outlier_by_z_pae = []
    outlier_by_z_tf = []
    for i in range(len(z_bins) - 1):
        mask_pae = (res['ztrue'] >= z_bins[i]) & (res['ztrue'] < z_bins[i+1]) & quality_mask & np.isfinite(pae_zscore)
        mask_tf = (res['ztrue'] >= z_bins[i]) & (res['ztrue'] < z_bins[i+1]) & np.isfinite(tf_zscore)
        
        if np.sum(mask_pae) > 5:
            outlier_by_z_pae.append(100 * np.sum(np.abs(pae_zscore[mask_pae]) > 3) / np.sum(mask_pae))
        else:
            outlier_by_z_pae.append(np.nan)
            
        if np.sum(mask_tf) > 5:
            outlier_by_z_tf.append(100 * np.sum(np.abs(tf_zscore[mask_tf]) > 3) / np.sum(mask_tf))
        else:
            outlier_by_z_tf.append(np.nan)
    
    ax.plot(z_centers, outlier_by_z_pae, 'o-', color='blue', linewidth=2, markersize=6, label='PAE')
    ax.plot(z_centers, outlier_by_z_tf, 's-', color='black', linewidth=2, markersize=5, label='TF')
    ax.set_xlabel('Spectroscopic Redshift', fontsize=12)
    ax.set_ylabel('Outlier Fraction (%)', fontsize=12)
    ax.set_title('Outliers (|z-score|>3) vs Spec-z', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Panel 4: Bias vs number of measurements
    ax = axes[1, 0]
    if n_measurements is not None:
        n_meas_bins = np.array([0, 20, 40, 60, 80, 100, 120])
        n_meas_centers = 0.5 * (n_meas_bins[:-1] + n_meas_bins[1:])
        
        bias_by_nmeas_pae = []
        bias_by_nmeas_tf = []
        for i in range(len(n_meas_bins) - 1):
            mask_pae = (n_measurements >= n_meas_bins[i]) & (n_measurements < n_meas_bins[i+1]) & quality_mask & np.isfinite(pae_bias)
            mask_tf = (n_measurements >= n_meas_bins[i]) & (n_measurements < n_meas_bins[i+1]) & np.isfinite(tf_bias)
            
            if np.sum(mask_pae) > 5:
                bias_by_nmeas_pae.append(np.nanmedian(pae_bias[mask_pae]))
            else:
                bias_by_nmeas_pae.append(np.nan)
                
            if np.sum(mask_tf) > 5:
                bias_by_nmeas_tf.append(np.nanmedian(tf_bias[mask_tf]))
            else:
                bias_by_nmeas_tf.append(np.nan)
        
        ax.plot(n_meas_centers, bias_by_nmeas_pae, 'o-', color='blue', linewidth=2, markersize=6, label='PAE')
        ax.plot(n_meas_centers, bias_by_nmeas_tf, 's-', color='black', linewidth=2, markersize=5, label='TF')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Number of Measurements', fontsize=12)
        ax.set_ylabel('Median Bias (Δz)', fontsize=12)
        ax.set_title('Bias vs N_measurements', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'N_measurements not available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
    
    # Panel 5: NMAD vs number of measurements
    ax = axes[1, 1]
    if n_measurements is not None:
        nmad_by_nmeas_pae = []
        nmad_by_nmeas_tf = []
        for i in range(len(n_meas_bins) - 1):
            mask_pae = (n_measurements >= n_meas_bins[i]) & (n_measurements < n_meas_bins[i+1]) & quality_mask & np.isfinite(pae_bias)
            mask_tf = (n_measurements >= n_meas_bins[i]) & (n_measurements < n_meas_bins[i+1]) & np.isfinite(tf_bias)
            
            if np.sum(mask_pae) > 5:
                nmad_pae = 1.4826 * np.nanmedian(np.abs(pae_bias[mask_pae] - np.nanmedian(pae_bias[mask_pae])))
                nmad_by_nmeas_pae.append(nmad_pae)
            else:
                nmad_by_nmeas_pae.append(np.nan)
                
            if np.sum(mask_tf) > 5:
                nmad_tf = 1.4826 * np.nanmedian(np.abs(tf_bias[mask_tf] - np.nanmedian(tf_bias[mask_tf])))
                nmad_by_nmeas_tf.append(nmad_tf)
            else:
                nmad_by_nmeas_tf.append(np.nan)
        
        ax.plot(n_meas_centers, nmad_by_nmeas_pae, 'o-', color='blue', linewidth=2, markersize=6, label='PAE')
        ax.plot(n_meas_centers, nmad_by_nmeas_tf, 's-', color='black', linewidth=2, markersize=5, label='TF')
        ax.set_xlabel('Number of Measurements', fontsize=12)
        ax.set_ylabel('NMAD(Δz)', fontsize=12)
        ax.set_title('Scatter vs N_measurements', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'N_measurements not available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
    
    # Panel 6: Bias/NMAD vs reduced chi2 (alternative diagnostic)
    ax = axes[1, 2]
    if 'chi2' in res:
        chi2 = res['chi2']
        chi2_bins = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
        chi2_centers = 0.5 * (chi2_bins[:-1] + chi2_bins[1:])
        
        nmad_by_chi2_pae = []
        for i in range(len(chi2_bins) - 1):
            mask_pae = (chi2 >= chi2_bins[i]) & (chi2 < chi2_bins[i+1]) & quality_mask & np.isfinite(pae_bias)
            
            if np.sum(mask_pae) > 5:
                nmad_pae = 1.4826 * np.nanmedian(np.abs(pae_bias[mask_pae] - np.nanmedian(pae_bias[mask_pae])))
                nmad_by_chi2_pae.append(nmad_pae)
            else:
                nmad_by_chi2_pae.append(np.nan)
        
        ax.plot(chi2_centers, nmad_by_chi2_pae, 'o-', color='blue', linewidth=2, markersize=6, label='PAE')
        ax.set_xlabel('Reduced χ²', fontsize=12)
        ax.set_ylabel('NMAD(Δz)', fontsize=12)
        ax.set_title('Scatter vs Fit Quality', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'chi2 not available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
    
    fig5.suptitle(f'{result_name}: Binned Diagnostic Plots', fontsize=15, y=0.995)
    plt.tight_layout()
    
    if save_dir:
        fig5_path = Path(save_dir) / f'{result_name}_binned_diagnostics.png'
        fig5.savefig(fig5_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig5_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig5)
    
    return stats_summary


def plot_nz_distributions(result_filepath, save_dir=None, show_plots=False, sigma_cut=0.2):
    """
    Generate N(z) distribution plots for PAE and TF methods with corresponding spec-z distributions.
    
    Parameters:
    -----------
    result_filepath : str
        Path to the PAE results .npz file
    save_dir : str, optional
        Directory to save figures. If None, figures are not saved.
    show_plots : bool
        Whether to display plots interactively
    sigma_cut : float
        Maximum normalized uncertainty σ_z/(1+z) for inclusion
    
    Returns:
    --------
    dict : Dictionary containing computed statistics
    """
    # Load results
    try:
        res = np.load(result_filepath, allow_pickle=True)
        n_sources = len(res['ztrue'])
    except Exception as e:
        print(f"ERROR: Failed to load result file: {e}")
        return None
    
    result_name = Path(result_filepath).stem
    
    # Extract data
    z_true = res['ztrue']
    z_pae = res['z_med']
    z_tf = res['z_TF']
    
    # Compute uncertainties
    pae_unc = 0.5 * (res['err_low'] + res['err_high'])
    tf_unc = res['z_TF_err']
    
    # Compute normalized uncertainties
    pae_sigma_norm = pae_unc / (1 + z_pae)
    tf_sigma_norm = tf_unc / (1 + z_tf)
    
    # Apply sigma cut and validity filters
    pae_valid = np.isfinite(z_pae) & np.isfinite(pae_sigma_norm) & (pae_sigma_norm < sigma_cut)
    tf_valid = np.isfinite(z_tf) & np.isfinite(tf_sigma_norm) & (tf_sigma_norm < sigma_cut)
    
    n_pae_valid = np.sum(pae_valid)
    n_tf_valid = np.sum(tf_valid)
    
    print(f"\nN(z) Distribution Plot:")
    print(f"  PAE sources with σ_z/(1+z) < {sigma_cut}: {n_pae_valid}/{n_sources} ({100*n_pae_valid/n_sources:.1f}%)")
    print(f"  TF sources with σ_z/(1+z) < {sigma_cut}: {n_tf_valid}/{n_sources} ({100*n_tf_valid/n_sources:.1f}%)")
    
    if n_pae_valid == 0 and n_tf_valid == 0:
        print("  No sources pass the sigma cut for either method. Skipping N(z) plot.")
        return None
    
    # Set up redshift bins
    z_values_for_range = [z_true]  # Always include true redshifts
    if n_pae_valid > 0:
        z_values_for_range.append(z_pae[pae_valid])
    if n_tf_valid > 0:
        z_values_for_range.append(z_tf[tf_valid])
    
    z_min = np.nanmin([np.nanmin(arr) for arr in z_values_for_range])
    z_max = np.nanmax([np.nanmax(arr) for arr in z_values_for_range])
    z_bins = np.linspace(max(0, z_min-0.1), z_max+0.1, 50)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Panel 1: PAE results
    if n_pae_valid > 0:
        # PAE redshift distribution
        counts_pae, _, _ = ax1.hist(z_pae[pae_valid], bins=z_bins, alpha=0.7, 
                                   color='blue', label=f'PAE estimates (N={n_pae_valid:,})',
                                   density=True)
        
        # Corresponding spec-z distribution for PAE-selected sources
        counts_spec_pae, _, _ = ax1.hist(z_true[pae_valid], bins=z_bins, alpha=0.7,
                                        color='red', label=f'Spec-z for PAE selection',
                                        density=True, histtype='step', linewidth=2)
        
        # Statistics
        pae_median = np.median(z_pae[pae_valid])
        spec_pae_median = np.median(z_true[pae_valid])
        
        ax1.axvline(pae_median, color='blue', linestyle='--', alpha=0.8, 
                   label=f'PAE median: {pae_median:.3f}')
        ax1.axvline(spec_pae_median, color='red', linestyle='--', alpha=0.8,
                   label=f'Spec-z median: {spec_pae_median:.3f}')
    else:
        ax1.text(0.5, 0.5, f'No PAE sources with\nσ_z/(1+z) < {sigma_cut}', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=14)
    
    ax1.set_xlabel('Redshift', fontsize=14)
    ax1.set_ylabel('Normalized N(z)', fontsize=14)
    ax1.set_title(f'PAE Method\n(σ_z/(1+z) < {sigma_cut})', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: TF results
    if n_tf_valid > 0:
        # TF redshift distribution
        counts_tf, _, _ = ax2.hist(z_tf[tf_valid], bins=z_bins, alpha=0.7,
                                  color='green', label=f'TF estimates (N={n_tf_valid:,})',
                                  density=True)
        
        # Corresponding spec-z distribution for TF-selected sources
        counts_spec_tf, _, _ = ax2.hist(z_true[tf_valid], bins=z_bins, alpha=0.7,
                                       color='red', label=f'Spec-z for TF selection',
                                       density=True, histtype='step', linewidth=2)
        
        # Statistics
        tf_median = np.median(z_tf[tf_valid])
        spec_tf_median = np.median(z_true[tf_valid])
        
        ax2.axvline(tf_median, color='green', linestyle='--', alpha=0.8,
                   label=f'TF median: {tf_median:.3f}')
        ax2.axvline(spec_tf_median, color='red', linestyle='--', alpha=0.8,
                   label=f'Spec-z median: {spec_tf_median:.3f}')
    else:
        ax2.text(0.5, 0.5, f'No TF sources with\nσ_z/(1+z) < {sigma_cut}', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=14)
    
    ax2.set_xlabel('Redshift', fontsize=14)
    ax2.set_ylabel('Normalized N(z)', fontsize=14)
    ax2.set_title(f'Template Fitting Method\n(σ_z/(1+z) < {sigma_cut})', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'{result_name}: Redshift Distributions', fontsize=16)
    plt.tight_layout()
    
    if save_dir:
        fig_path = Path(save_dir) / f'{result_name}_nz_distributions.png'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {fig_path.name}")
    
    if show_plots:
        plt.show()
    else:
        plt.close(fig)
    
    # Return statistics
    stats = {
        'n_pae_valid': n_pae_valid,
        'n_tf_valid': n_tf_valid,
        'sigma_cut': sigma_cut
    }
    
    if n_pae_valid > 0:
        stats.update({
            'pae_median': pae_median,
            'spec_pae_median': spec_pae_median,
            'pae_mean': np.mean(z_pae[pae_valid]),
            'spec_pae_mean': np.mean(z_true[pae_valid])
        })
    
    if n_tf_valid > 0:
        stats.update({
            'tf_median': tf_median,
            'spec_tf_median': spec_tf_median,
            'tf_mean': np.mean(z_tf[tf_valid]),
            'spec_tf_mean': np.mean(z_true[tf_valid])
        })
    
    return stats


# ==============================================================================
# STANDALONE CHI2 COMPARISON FIGURE
# ==============================================================================

def plot_chi2_comparison(result_filepath, save_dir=None, show_plots=False,
                         snr_min=None, use_hexbin=True, snr_array=None,
                         frac_sampled_min=None, chi2min=50, chi2max=3000,
                         band4_neg_flag=None):
    """Two-panel chi² comparison: histogram of PAE vs TF chi², and PAE vs TF scatter.

    Panel 1 – overlaid histograms of chi2_full (PAE) and minchi2_gals (TF) on a
               log x-axis, restricted to 10 ≤ chi² ≤ 1000, after applying quality cuts.
    Panel 2 – log-log hexbin (or scatter) of PAE chi² vs TF chi², with a 1:1 line,
               also restricted to 10 ≤ chi² ≤ 1000.

    Parameters
    ----------
    result_filepath : str
        Path to the combined PAE results .npz file.
    save_dir : str, optional
        Directory in which to save the output figure.
    show_plots : bool
        If True, display the figure interactively.
    snr_min : float, optional
        Minimum SNR threshold (same proxy as the rest of the pipeline).
    use_hexbin : bool
        If True use log-scaled hexbin in Panel 2; otherwise scatter.
    snr_array : np.ndarray, optional
        Pre-computed SNR values for each source.
    frac_sampled_min : float, optional
        Minimum spectral completeness (frac_sampled_102) for inclusion.
    """
    print("\n  Creating chi² comparison figure (PAE vs TF)...")

    res = np.load(str(result_filepath), allow_pickle=True)

    if 'chi2_full' not in res or 'minchi2_gals' not in res:
        missing = [k for k in ('chi2_full', 'minchi2_gals') if k not in res]
        print(f"  ⚠ Skipping chi2 comparison — missing keys: {missing}")
        return None

    chi2_pae = np.array(res['chi2_full'], dtype=float)
    chi2_tf  = np.array(res['minchi2_gals'], dtype=float)

    # Build quality mask
    quality_mask = np.ones(len(chi2_pae), dtype=bool)
    
    # SNR mask — prefer explicit snr_array (parquet broadband), fall back to weights proxy
    if snr_array is not None and snr_min is not None:
        snr_mask = np.array(snr_array) >= snr_min
        quality_mask &= snr_mask
    elif snr_min is not None and 'weights' in res:
        weights = np.array(res['weights'])
        snr_proxy = np.sqrt(np.sum(weights, axis=1)) if weights.ndim == 2 else np.sqrt(weights)
        quality_mask &= (snr_proxy >= snr_min)
    
    # frac_sampled_102 mask
    if frac_sampled_min is not None and 'frac_sampled_102' in res:
        frac_mask = res['frac_sampled_102'] >= frac_sampled_min
        quality_mask &= frac_mask

    # Band-4 negative-flux flag
    if band4_neg_flag is not None:
        quality_mask &= ~np.array(band4_neg_flag, dtype=bool)

    valid = np.isfinite(chi2_pae) & np.isfinite(chi2_tf) & quality_mask & (chi2_pae > 0) & (chi2_tf > 0)
    chi2_pae_v = chi2_pae[valid]
    chi2_tf_v  = chi2_tf[valid]

    # Restrict to 10 ≤ chi² ≤ 1000 for both panels
    plot_mask   = (chi2_pae_v >= chi2min) & (chi2_pae_v <= chi2max) & (chi2_tf_v >= chi2min) & (chi2_tf_v <= chi2max)
    chi2_pae_p  = chi2_pae_v[plot_mask]
    chi2_tf_p   = chi2_tf_v[plot_mask]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.subplots_adjust(wspace=0.35)

    # ── Panel 1: overlaid histograms (log x-axis) ────────────────
    ax1 = axes[0]
    bins = np.geomspace(chi2min, chi2max, 60)
    ax1.hist(chi2_tf_p,  bins=bins, color='steelblue',
             label='Template fitting', histtype='step', linewidth=1.5)
    ax1.hist(chi2_pae_p, bins=bins, color='darkorange',
             label='PAE', histtype='step', linewidth=1.5)
    ax1.set_xscale('log')
    ax1.set_xlim(chi2min, chi2max)
    ax1.axvline(np.median(chi2_tf_p),  color='steelblue',  linestyle='--',
                linewidth=1.5, label=f'TF median = {np.median(chi2_tf_p):.1f}')
    ax1.axvline(np.median(chi2_pae_p), color='darkorange',  linestyle='--',
                linewidth=1.5, label=f'PAE median = {np.median(chi2_pae_p):.1f}')
    ax1.set_xlabel(r'$\chi^2$', fontsize=13)
    ax1.set_ylabel('$N_{source}$', fontsize=13)
    ax1.legend(fontsize=10, loc=2, bbox_to_anchor=[0.0, 1.4])
    ax1.grid(alpha=0.3)
    ax1.text(0.97, 0.97, f'$N = {plot_mask.sum():,}$', transform=ax1.transAxes,
             fontsize=10, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # ── Panel 2: PAE vs TF hexbin / scatter (log-log, 10-1000) ────────────
    ax2 = axes[1]
    log_tf  = np.log10(chi2_tf_p)
    log_pae = np.log10(chi2_pae_p)
    # Fix range to chi2min-chi2max (log10: log10(chi2min)-log10(chi2max))
    xmin, xmax = np.log10(chi2min), np.log10(chi2max)
    ymin, ymax = xmin, xmax

    if use_hexbin:
        hb = ax2.hexbin(log_tf, log_pae,
                        bins='log', cmap='viridis', mincnt=1, gridsize=60,
                        extent=(xmin, xmax, ymin, ymax))
        plt.colorbar(hb, ax=ax2, label='log$_{10}$(count)')
    else:
        ax2.scatter(log_tf, log_pae, alpha=0.2, s=4, color='steelblue', rasterized=True)

    # 1:1 line in log space
    ax2.plot([xmin, xmax], [ymin, ymax], 'k--', linewidth=1.5, label='1:1', zorder=10)

    # Format tick labels as actual chi2 values
    tick_vals = [10, 30, 100, 300, 1000]
    tick_locs = [np.log10(v) for v in tick_vals]
    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels([str(v) for v in tick_vals])
    ax2.set_yticks(tick_locs)
    ax2.set_yticklabels([str(v) for v in tick_vals])

    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlabel(r'Template fitting $\chi^2$', fontsize=13)
    ax2.set_ylabel(r'PAE $\chi^2$', fontsize=13)
    ax2.set_title(r'PAE vs Template Fitting $\chi^2$', fontsize=14)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(alpha=0.3)

    if save_dir:
        chi2_dir = Path(save_dir) / 'convergence'
        chi2_dir.mkdir(parents=True, exist_ok=True)
        result_name = Path(result_filepath).stem
        out_path = chi2_dir / f'{result_name}_chi2_pae_vs_tf_standalone.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"    ✓ Saved: convergence/{out_path.name}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig)

    return {'n_valid': valid.sum(), 
            'n_plotted': len(chi2_pae_p),
            'median_chi2_tf': float(np.median(chi2_tf_p)),
            'median_chi2_tf_all': float(np.median(chi2_tf_v)),
            'median_chi2_pae': float(np.median(chi2_pae_p)),
            'median_chi2_pae_all': float(np.median(chi2_pae_v))}


# =============================================================================
# RUN COMPARISON
# =============================================================================

def plot_run_comparison(new_result_file, baseline_datestr, save_dir,
                        show_plots=False, frac_sampled_min=0.7,
                        snr_min=None, snr_array_new=None,
                        snr_array_baseline=None,
                        band4_neg_flag_new=None):
    """
    Generate comparison plots between a new run and a baseline run.

    Sources are matched by src_id (inner join) so the comparison is fair even
    when the two runs have different source counts.  Stats are computed on the
    intersection only.

    Produces three figures saved to <save_dir>/comparison/:
      C1  CDF of sigma_z/(1+z) – cumulative fraction vs threshold, both runs
      C2  NMAD and outlier-rate bar charts broken down by sigma_z/(1+z) bin
      C3  Delta-z/(1+z) bias histograms overlaid
    """
    from config import scratch_basepath

    new_result_file = Path(new_result_file)
    baseline_dir = (Path(scratch_basepath) / 'data' / 'pae_sample_results'
                    / 'MCLMC' / 'batched' / baseline_datestr)
    baseline_file = baseline_dir / f'PAE_results_combined_{baseline_datestr}.npz'

    print(f"\n{'='*70}")
    print("RUN COMPARISON")
    print(f"{'='*70}")
    print(f"  New run:      {new_result_file.stem}")
    print(f"  Baseline run: {baseline_datestr}")

    if not baseline_file.exists():
        print(f"  ⚠ Baseline result file not found: {baseline_file}")
        print(f"    Skipping comparison plots.")
        return None

    new_res  = np.load(str(new_result_file),  allow_pickle=True)
    base_res = np.load(str(baseline_file), allow_pickle=True)

    new_ids  = np.array(new_res['src_id'],  dtype=np.int64)
    base_ids = np.array(base_res['src_id'], dtype=np.int64)

    # Inner join on src_id
    common_ids = np.intersect1d(new_ids, base_ids)
    n_common   = len(common_ids)
    print(f"  New run sources:      {len(new_ids):,}")
    print(f"  Baseline sources:     {len(base_ids):,}")
    print(f"  Matched (inner join): {n_common:,}")

    if n_common < 10:
        print(f"  ⚠ Too few matched sources ({n_common}). Skipping comparison.")
        return None

    # Build index arrays (new_idx / base_idx) → positions of common_ids in each file
    new_sort  = np.argsort(new_ids)
    base_sort = np.argsort(base_ids)
    new_idx   = new_sort[np.searchsorted(new_ids[new_sort],   common_ids)]
    base_idx  = base_sort[np.searchsorted(base_ids[base_sort], common_ids)]

    # Quality mask derived from the NEW run only
    quality_mask = np.ones(n_common, dtype=bool)
    if frac_sampled_min is not None and 'frac_sampled_102' in new_res:
        frac = np.array(new_res['frac_sampled_102'])[new_idx]
        quality_mask &= frac >= frac_sampled_min
    if snr_min is not None:
        # Prefer the new run's SNR array; fall back to the baseline's if new is unavailable.
        # If neither is available, skip the cut rather than using a proxy.
        _snr_for_cut = snr_array_new if snr_array_new is not None else snr_array_baseline
        if _snr_for_cut is not None:
            snr_matched = np.array(_snr_for_cut)[new_idx if snr_array_new is not None else base_idx]
            quality_mask &= np.isfinite(snr_matched) & (snr_matched >= snr_min)
            _snr_src = 'new run' if snr_array_new is not None else 'baseline (fallback)'
            print(f"  SNR cut ≥ {snr_min} ({_snr_src}): {quality_mask.sum():,} / {n_common:,} pass")
        else:
            print(f"  ⚠ snr_min={snr_min} requested but no SNR array available for either run — SNR cut skipped.")

    if band4_neg_flag_new is not None:
        b4_flag_matched = np.array(band4_neg_flag_new)[new_idx]
        n_before = quality_mask.sum()
        quality_mask &= ~b4_flag_matched.astype(bool)
        print(f"  Band-4 neg-flux flag (new run): {quality_mask.sum():,} / {n_before:,} pass")

    n_quality = int(quality_mask.sum())
    print(f"  After quality cuts:   {n_quality:,}")

    if n_quality < 10:
        print(f"  ⚠ Too few sources after quality cuts. Skipping comparison.")
        return None

    def _get(res, idx, key):
        """Return res[key][idx][quality_mask], or None if key absent / wrong shape."""
        if key not in res:
            return None
        arr = np.array(res[key])
        if arr.ndim == 0 or len(arr) != len(np.array(res['src_id'])):
            return None
        return arr[idx][quality_mask]

    new_zmed    = _get(new_res,  new_idx,  'z_med')
    new_ztrue   = _get(new_res,  new_idx,  'ztrue')
    new_err_lo  = _get(new_res,  new_idx,  'err_low')
    new_err_hi  = _get(new_res,  new_idx,  'err_high')
    new_zTF     = _get(new_res,  new_idx,  'z_TF')
    new_zTF_err = _get(new_res,  new_idx,  'z_TF_err')

    base_zmed    = _get(base_res, base_idx, 'z_med')
    base_ztrue   = _get(base_res, base_idx, 'ztrue')
    base_err_lo  = _get(base_res, base_idx, 'err_low')
    base_err_hi  = _get(base_res, base_idx, 'err_high')
    base_zTF     = _get(base_res, base_idx, 'z_TF')
    base_zTF_err = _get(base_res, base_idx, 'z_TF_err')

    if new_zmed is None or new_ztrue is None or base_zmed is None or base_ztrue is None:
        print("  ⚠ Missing z_med or ztrue arrays. Skipping comparison.")
        return None

    # PAE uncertainty → sigma_z/(1+z)
    new_pae_unc  = (0.5 * (np.abs(new_err_lo)  + np.abs(new_err_hi))
                    if (new_err_lo  is not None and new_err_hi  is not None) else None)
    base_pae_unc = (0.5 * (np.abs(base_err_lo) + np.abs(base_err_hi))
                    if (base_err_lo is not None and base_err_hi is not None) else None)

    new_pae_dz  = new_pae_unc  / (1 + new_zmed)  if new_pae_unc  is not None else None
    base_pae_dz = base_pae_unc / (1 + base_zmed) if base_pae_unc is not None else None

    # TF uncertainty (guard zeros)
    def _safe_tf_unc(arr):
        if arr is None:
            return None
        return np.where(arr > 0, arr, np.nan)

    new_tf_unc_safe  = _safe_tf_unc(new_zTF_err)
    base_tf_unc_safe = _safe_tf_unc(base_zTF_err)

    new_tf_dz  = (new_tf_unc_safe  / (1 + new_zTF)
                  if (new_tf_unc_safe is not None and new_zTF  is not None) else None)
    base_tf_dz = (base_tf_unc_safe / (1 + base_zTF)
                  if (base_tf_unc_safe is not None and base_zTF is not None) else None)

    new_name  = new_result_file.stem.replace('PAE_results_combined_', '')
    base_name = baseline_datestr

    # TF is a single reference: only requires the baseline to have valid TF values.
    # The new run may also have TF values (the same pipeline), but since no data
    # exclusion is applied to TF independently, we treat TF as one reference series
    # taken from whichever run has it (prefer baseline; fall back to new run).
    has_tf = (base_zTF is not None and np.any(base_zTF != 0))
    if not has_tf and new_zTF is not None and np.any(new_zTF != 0):
        # Fall back to new run's TF if baseline has none
        tf_z       = new_zTF
        tf_ztrue   = new_ztrue
        tf_dz      = new_tf_dz
        tf_src     = f'TF ({new_name})'
        has_tf     = True
    elif has_tf:
        tf_z       = base_zTF
        tf_ztrue   = base_ztrue
        tf_dz      = base_tf_dz
        tf_src     = f'TF ({base_name})'
    else:
        tf_z = tf_ztrue = tf_dz = tf_src = None

    comp_dir = Path(save_dir) / 'comparison'
    comp_dir.mkdir(parents=True, exist_ok=True)

    def _cdf(arr):
        a = arr[np.isfinite(arr)]
        return np.array([np.mean(a < t) for t in thresholds])

    thresholds = np.linspace(0, 0.1, 500)

    # ------------------------------------------------------------------
    # Figure C1: CDF of sigma_z/(1+z) — all three series on one panel
    # ------------------------------------------------------------------
    fig_c1, ax = plt.subplots(figsize=(7, 5))
    if new_pae_dz is not None:
        ax.plot(thresholds, _cdf(new_pae_dz),  color='dodgerblue', lw=2,
                label=f'PAE new ({new_name})')
    if base_pae_dz is not None:
        ax.plot(thresholds, _cdf(base_pae_dz), color='royalblue', lw=2, ls='--',
                label=f'PAE baseline ({base_name})')
    if has_tf and tf_dz is not None:
        ax.plot(thresholds, _cdf(tf_dz), color='orangered', lw=2, ls=':',
                label=tf_src)
    for xv in (0.003, 0.01):
        ax.axvline(xv, color='gray', lw=1, ls=':', alpha=0.7)
    ax.set(xlabel=r'$\sigma_{z/(1+z)}$', ylabel='Cumulative fraction',
           title='CDF of photometric uncertainty', xlim=(0, 0.06), ylim=(0, 1))
    ax.legend(fontsize=9);  ax.grid(alpha=0.3)
    fig_c1.suptitle(f'Uncertainty CDF comparison  —  N={n_quality:,} matched sources',
                    fontsize=13)
    plt.tight_layout()
    c1_path = comp_dir / f'{new_name}_vs_{base_name}_cdf.png'
    fig_c1.savefig(str(c1_path), dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {c1_path.relative_to(save_dir)}")
    if not show_plots:
        plt.close(fig_c1)

    # ------------------------------------------------------------------
    # Figure C2: NMAD and outlier rate by sigma_z/(1+z) bin
    # Three grouped bars per bin: PAE-new, PAE-baseline, TF-ref
    # ------------------------------------------------------------------
    dz_edges  = [0.003, 0.01, 0.03, np.inf]
    dz_labels = ['<0.003', '0.003–0.01', '0.01–0.03', '>0.03']
    n_bins    = len(dz_edges)

    def _bin_stats(zml, ztrue, dz_arr):
        nmads, outls, ns = [], [], []
        for i, dz_hi in enumerate(dz_edges):
            dz_lo = 0.0 if i == 0 else dz_edges[i - 1]
            mask = (dz_arr >= dz_lo) & (dz_arr < dz_hi) if not np.isinf(dz_hi) \
                   else (dz_arr >= dz_lo)
            mask &= np.isfinite(dz_arr) & np.isfinite(zml) & np.isfinite(ztrue)
            if mask.sum() >= 10:
                _, _, _, nm, _, outl, _ = compute_redshift_stats(
                    zml[mask], ztrue[mask], nsig_outlier=3)
                nmads.append(nm);  outls.append(outl * 100);  ns.append(int(mask.sum()))
            else:
                nmads.append(np.nan);  outls.append(np.nan);  ns.append(0)
        return nmads, outls, ns

    _nan_bins = lambda: ([np.nan]*n_bins, [np.nan]*n_bins, [0]*n_bins)

    new_pae_nmads,  new_pae_outls,  new_pae_ns  = (
        _bin_stats(new_zmed,  new_ztrue,  new_pae_dz)  if new_pae_dz  is not None else _nan_bins())
    base_pae_nmads, base_pae_outls, base_pae_ns = (
        _bin_stats(base_zmed, base_ztrue, base_pae_dz) if base_pae_dz is not None else _nan_bins())
    tf_nmads, tf_outls, tf_ns = (
        _bin_stats(tf_z, tf_ztrue, tf_dz) if (has_tf and tf_dz is not None) else _nan_bins())

    x      = np.arange(n_bins)
    n_bars = 3 if has_tf else 2
    width  = 0.25 if has_tf else 0.35
    offsets = np.linspace(-(n_bars - 1) * width / 2, (n_bars - 1) * width / 2, n_bars)

    fig_c2, (ax_nmad, ax_outl) = plt.subplots(2, 1, figsize=(10, 9))

    for ax, metric_sets, ylabel, title in [
        (ax_nmad, [new_pae_nmads, base_pae_nmads, tf_nmads],   'NMAD',           'NMAD by σ bin'),
        (ax_outl, [new_pae_outls, base_pae_outls, tf_outls],   'Outlier rate (%)', 'Outlier rate by σ bin'),
    ]:
        labels_colors = [
            (f'PAE new ({new_name})',       'dodgerblue'),
            (f'PAE baseline ({base_name})', 'royalblue'),
            (f'{tf_src}',                   'orangered'),
        ]
        for k, (vals, (lbl, col)) in enumerate(zip(metric_sets, labels_colors)):
            if k == 2 and not has_tf:
                continue
            ax.bar(x + offsets[k], vals, width, label=lbl, color=col, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(dz_labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12);  ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9);  ax.grid(alpha=0.3, axis='y')

    # Annotate source counts on the outlier-rate panel
    for k, (ns, col) in enumerate([(new_pae_ns, 'dodgerblue'),
                                    (base_pae_ns, 'royalblue'),
                                    (tf_ns if has_tf else [0]*n_bins, 'orangered')]):
        if k == 2 and not has_tf:
            continue
        for i, n in enumerate(ns):
            if n > 0:
                ax_outl.text(x[i] + offsets[k], 0.3, f'N={n:,}',
                             ha='center', va='bottom', fontsize=6, rotation=90, color=col)

    fig_c2.suptitle(f'Metric comparison by σ bin  —  N={n_quality:,} matched sources',
                    fontsize=13)
    plt.tight_layout()
    c2_path = comp_dir / f'{new_name}_vs_{base_name}_metrics_by_bin.png'
    fig_c2.savefig(str(c2_path), dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {c2_path.relative_to(save_dir)}")
    if not show_plots:
        plt.close(fig_c2)

    # ------------------------------------------------------------------
    # Figure C3: Delta-z/(1+z) bias histograms — all three on one panel
    # ------------------------------------------------------------------
    new_pae_bias  = (new_zmed  - new_ztrue)  / (1 + new_ztrue)
    base_pae_bias = (base_zmed - base_ztrue) / (1 + base_ztrue)

    bins_bias = np.linspace(-0.05, 0.05, 80)
    fig_c3, ax = plt.subplots(figsize=(7, 5))

    vn = np.isfinite(new_pae_bias)
    vb = np.isfinite(base_pae_bias)
    ax.hist(new_pae_bias[vn],  bins=bins_bias, histtype='step', color='dodgerblue', lw=2,
            density=True, label=f'PAE new  (med={np.nanmedian(new_pae_bias):.4f})')
    ax.hist(base_pae_bias[vb], bins=bins_bias, histtype='step', color='royalblue', lw=2, ls='--',
            density=True, label=f'PAE base (med={np.nanmedian(base_pae_bias):.4f})')
    if has_tf and tf_z is not None and tf_ztrue is not None:
        tf_bias = (tf_z - tf_ztrue) / (1 + tf_ztrue)
        vt = np.isfinite(tf_bias) & (tf_z != 0)
        if vt.any():
            ax.hist(tf_bias[vt], bins=bins_bias, histtype='step', color='orangered', lw=2, ls=':',
                    density=True, label=f'{tf_src} (med={np.nanmedian(tf_bias[vt]):.4f})')
    ax.axvline(0, color='gray', lw=1, ls=':')
    ax.set(xlabel=r'$\Delta z/(1+z)$', ylabel='Density', title='Bias distribution')
    ax.legend(fontsize=9);  ax.grid(alpha=0.3)

    fig_c3.suptitle(f'Bias comparison  —  N={n_quality:,} matched sources', fontsize=13)
    plt.tight_layout()
    c3_path = comp_dir / f'{new_name}_vs_{base_name}_bias.png'
    fig_c3.savefig(str(c3_path), dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {c3_path.relative_to(save_dir)}")
    if not show_plots:
        plt.close(fig_c3)

    # ------------------------------------------------------------------
    # Figures C4 / C5: Hexbin redshift recovery plots
    # C4 — overall (all σ bins combined), one column per series
    # C5 — one row per σ bin of the *new* PAE run, same column layout
    # ------------------------------------------------------------------
    zmin_h, zmax_h = 0.0, 3.0
    zsp_h = np.linspace(zmin_h, zmax_h, 100)
    _gs   = 60   # hexbin gridsize
    _ext  = (zmin_h, zmax_h, zmin_h, zmax_h)

    # Series definitions: (z_ml, z_true, label, colour)
    _series = [
        (new_zmed,  new_ztrue,  f'PAE new\n({new_name})',       'plasma'),
        (base_zmed, base_ztrue, f'PAE baseline\n({base_name})', 'plasma'),
    ]
    if has_tf and tf_z is not None and tf_ztrue is not None:
        _series.append((tf_z, tf_ztrue, tf_src, 'inferno'))

    n_cols_h = len(_series)

    def _hexbin_panel(ax, x, y, title, cmap='plasma'):
        """Draw a hexbin recovery panel; return the hexbin artist."""
        fin = np.isfinite(x) & np.isfinite(y)
        hb = ax.hexbin(x[fin], y[fin], bins='log', cmap=cmap, mincnt=1,
                       gridsize=_gs, extent=_ext)
        ax.plot(zsp_h, zsp_h, 'w--', lw=1.2, zorder=5)
        ax.set(xlim=(zmin_h, zmax_h), ylim=(zmin_h, zmax_h),
               xlabel='spec-z', title=title)
        if np.any(np.asarray(hb.get_array()) > 0):
            plt.colorbar(hb, ax=ax, label='log₁₀(N)', pad=0.02)
        return hb, int(fin.sum())

    def _stat_text(zml, ztrue):
        fin = np.isfinite(zml) & np.isfinite(ztrue)
        if fin.sum() < 10:
            return f'N={fin.sum()}'
        _, _, bias, nmad, _, outl, _ = compute_redshift_stats(zml[fin], ztrue[fin], nsig_outlier=3)
        return f'N={fin.sum():,}\nNMAD={nmad:.4f}\nbias={bias:.4f}\noutl={outl*100:.1f}%'

    # ---- C4: Overall hexbins ----------------------------------------
    fig_c4, axes_c4 = plt.subplots(1, n_cols_h, figsize=(5 * n_cols_h, 5))
    if n_cols_h == 1:
        axes_c4 = [axes_c4]

    for ax, (zml, ztr, lbl, cmap) in zip(axes_c4, _series):
        _, n_fin = _hexbin_panel(ax, zml, ztr, lbl, cmap=cmap)
        ax.set_ylabel(r'$\hat{z}$')
        ax.text(0.03, 0.97, _stat_text(zml, ztr),
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

    fig_c4.suptitle(
        f'Redshift recovery comparison  —  N={n_quality:,} matched sources', fontsize=13)
    plt.tight_layout()
    c4_path = comp_dir / f'{new_name}_vs_{base_name}_hexbin_overall.png'
    fig_c4.savefig(str(c4_path), dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved: {c4_path.relative_to(save_dir)}")
    if not show_plots:
        plt.close(fig_c4)

    # ---- C5: Hexbins binned by σ_z/(1+z) of the *new* PAE run -------
    # Use the same dz_edges / dz_labels defined for C2.
    # Rows = σ bins, columns = series.
    hz_rows = []
    hz_row_labels = []
    for i, dz_hi in enumerate(dz_edges):
        dz_lo = 0.0 if i == 0 else dz_edges[i - 1]
        if np.isinf(dz_hi):
            bin_mask = new_pae_dz >= dz_lo if new_pae_dz is not None else None
        else:
            bin_mask = ((new_pae_dz >= dz_lo) & (new_pae_dz < dz_hi)) if new_pae_dz is not None else None
        if bin_mask is not None and bin_mask.sum() >= 5:
            hz_rows.append(bin_mask)
            hz_row_labels.append(dz_labels[i])

    if hz_rows:
        n_rows_h = len(hz_rows)
        fig_c5, axes_c5 = plt.subplots(n_rows_h, n_cols_h,
                                        figsize=(5 * n_cols_h, 4.5 * n_rows_h),
                                        squeeze=False)
        for r, (row_mask, row_lbl) in enumerate(zip(hz_rows, hz_row_labels)):
            # The row mask is defined on the new-PAE matched array.
            # For baseline/TF we show the same matched subset (by row_mask index).
            series_masked = [
                (new_zmed[row_mask],  new_ztrue[row_mask],  f'PAE new  σ∈{row_lbl}', 'plasma'),
                (base_zmed[row_mask], base_ztrue[row_mask], f'PAE base σ∈{row_lbl}', 'plasma'),
            ]
            if has_tf and tf_z is not None and tf_ztrue is not None:
                series_masked.append(
                    (tf_z[row_mask], tf_ztrue[row_mask], f'{tf_src} σ∈{row_lbl}', 'inferno'))

            for c, (ax, (zml, ztr, lbl, cmap)) in enumerate(
                    zip(axes_c5[r], series_masked)):
                _hexbin_panel(ax, zml, ztr, lbl if r == 0 else '', cmap=cmap)
                if c == 0:
                    ax.set_ylabel(f'σ∈{row_lbl}\n' + r'$\hat{z}$', fontsize=9)
                else:
                    ax.set_ylabel(r'$\hat{z}$')
                ax.text(0.03, 0.97, _stat_text(zml, ztr),
                        transform=ax.transAxes, fontsize=7, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

        fig_c5.suptitle(
            f'Redshift recovery by σ_z/(1+z) bin  —  N={n_quality:,} matched sources',
            fontsize=13)
        plt.tight_layout()
        c5_path = comp_dir / f'{new_name}_vs_{base_name}_hexbin_by_sigbin.png'
        fig_c5.savefig(str(c5_path), dpi=200, bbox_inches='tight')
        print(f"  ✓ Saved: {c5_path.relative_to(save_dir)}")
        if not show_plots:
            plt.close(fig_c5)

    # ------------------------------------------------------------------
    # Figure C6: Direct run-to-run comparison  (z_new vs z_baseline)
    # Left panel : hexbin(z_base, z_new) with 1:1 line
    # Right panel: residual (z_new − z_base)/(1+z_base) vs z_base
    # Optional 3rd/4th panels: same for TF (if available)
    # ------------------------------------------------------------------
    fin_both = np.isfinite(new_zmed) & np.isfinite(base_zmed)
    if fin_both.sum() >= 10:
        n_c6_cols = 4 if has_tf and tf_z is not None else 2
        fig_c6, axes_c6 = plt.subplots(1, n_c6_cols, figsize=(5.5 * n_c6_cols, 5))
        if n_c6_cols == 2:
            axes_c6 = list(axes_c6)

        # --- PAE scatter panel ---
        ax = axes_c6[0]
        fb = fin_both
        hb6 = ax.hexbin(base_zmed[fb], new_zmed[fb], bins='log', cmap='plasma',
                        mincnt=1, gridsize=_gs, extent=_ext)
        ax.plot(zsp_h, zsp_h, 'w--', lw=1.2, zorder=5)
        ax.set(xlabel=f'z PAE baseline ({base_name})',
               ylabel=f'z PAE new ({new_name})',
               title='PAE new vs PAE baseline',
               xlim=(zmin_h, zmax_h), ylim=(zmin_h, zmax_h))
        if np.any(np.asarray(hb6.get_array()) > 0):
            plt.colorbar(hb6, ax=ax, label='log₁₀(N)', pad=0.02)
        resid_pae = (new_zmed[fb] - base_zmed[fb]) / (1 + base_zmed[fb])
        nmad_pae  = 1.4826 * np.nanmedian(np.abs(resid_pae - np.nanmedian(resid_pae)))
        ax.text(0.03, 0.97,
                f'N={fb.sum():,}\nNMAD(Δz)={nmad_pae:.4f}\nmedΔz={np.nanmedian(resid_pae):.4f}',
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))

        # --- PAE residual panel ---
        ax = axes_c6[1]
        _resid_ext = (zmin_h, zmax_h, -0.1, 0.1)
        hb6r = ax.hexbin(base_zmed[fb], resid_pae, bins='log', cmap='plasma',
                         mincnt=1, gridsize=_gs, extent=_resid_ext)
        ax.axhline(0, color='w', lw=1.2, ls='--', zorder=5)
        ax.set(xlabel=f'z PAE baseline ({base_name})',
               ylabel=r'$(z_\mathrm{new} - z_\mathrm{base})/(1+z_\mathrm{base})$',
               title='PAE run-to-run residual',
               xlim=(zmin_h, zmax_h), ylim=(-0.1, 0.1))
        if np.any(np.asarray(hb6r.get_array()) > 0):
            plt.colorbar(hb6r, ax=ax, label='log₁₀(N)', pad=0.02)

        # --- TF panels (optional) ---
        if has_tf and tf_z is not None:
            # TF z vs baseline PAE z
            fin_tf6 = np.isfinite(tf_z) & np.isfinite(base_zmed) & (tf_z != 0)
            ax = axes_c6[2]
            if fin_tf6.sum() >= 10:
                hb6t = ax.hexbin(base_zmed[fin_tf6], tf_z[fin_tf6], bins='log', cmap='inferno',
                                 mincnt=1, gridsize=_gs, extent=_ext)
                ax.plot(zsp_h, zsp_h, 'w--', lw=1.2, zorder=5)
                if np.any(np.asarray(hb6t.get_array()) > 0):
                    plt.colorbar(hb6t, ax=ax, label='log₁₀(N)', pad=0.02)
                resid_tf = (tf_z[fin_tf6] - base_zmed[fin_tf6]) / (1 + base_zmed[fin_tf6])
                nmad_tf  = 1.4826 * np.nanmedian(np.abs(resid_tf - np.nanmedian(resid_tf)))
                ax.text(0.03, 0.97,
                        f'N={fin_tf6.sum():,}\nNMAD(Δz)={nmad_tf:.4f}\nmedΔz={np.nanmedian(resid_tf):.4f}',
                        transform=ax.transAxes, fontsize=7, va='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
            ax.set(xlabel=f'z PAE baseline ({base_name})',
                   ylabel=f'z TF ({tf_src})',
                   title='TF vs PAE baseline',
                   xlim=(zmin_h, zmax_h), ylim=(zmin_h, zmax_h))

            # TF residual vs baseline PAE
            ax = axes_c6[3]
            if fin_tf6.sum() >= 10:
                hb6tr = ax.hexbin(base_zmed[fin_tf6], resid_tf, bins='log', cmap='inferno',
                                  mincnt=1, gridsize=_gs, extent=_resid_ext)
                ax.axhline(0, color='w', lw=1.2, ls='--', zorder=5)
                if np.any(np.asarray(hb6tr.get_array()) > 0):
                    plt.colorbar(hb6tr, ax=ax, label='log₁₀(N)', pad=0.02)
            ax.set(xlabel=f'z PAE baseline ({base_name})',
                   ylabel=r'$(z_\mathrm{TF} - z_\mathrm{base})/(1+z_\mathrm{base})$',
                   title='TF vs PAE baseline residual',
                   xlim=(zmin_h, zmax_h), ylim=(-0.1, 0.1))

        fig_c6.suptitle(
            f'Run-to-run comparison  —  N={n_quality:,} matched sources', fontsize=13)
        plt.tight_layout()
        c6_path = comp_dir / f'{new_name}_vs_{base_name}_run_comparison.png'
        fig_c6.savefig(str(c6_path), dpi=200, bbox_inches='tight')
        print(f"  ✓ Saved: {c6_path.relative_to(save_dir)}")
        if not show_plots:
            plt.close(fig_c6)

    print(f"\n  ✓ Comparison plots saved to: {comp_dir}")
    return {'n_common': n_common, 'n_quality': n_quality}


def main(args):
    """Main function."""
    
    # Determine result file path
    if args.result_file:
        result_file = Path(args.result_file)
    elif args.datestr:
        # Auto-detect from datestr
        base_dir = Path(scratch_basepath) / 'data' / 'pae_sample_results' / 'MCLMC' / 'batched' / args.datestr
        pattern = f'PAE_results_combined_{args.datestr}.npz'
        result_file = base_dir / pattern
        
        if not result_file.exists():
            print(f"ERROR: Could not find result file: {result_file}")
            print(f"Tried: {result_file}")
            return 1
    else:
        print("ERROR: Must specify either --result-file or --datestr")
        return 1
    
    if not result_file.exists():
        print(f"ERROR: Result file does not exist: {result_file}")
        return 1
    
    # Determine output directory
    if args.output_dir:
        save_dir = Path(args.output_dir)
    elif args.datestr:
        save_dir = Path(scratch_basepath) / 'figures' / 'redshift_validation' / args.datestr
    else:
        save_dir = result_file.parent / 'figures'
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("REDSHIFT RESULT PLOTTING")
    print(f"{'='*70}")
    print(f"Result file: {result_file}")
    print(f"Output directory: {save_dir}")
    print(f"Show plots: {args.show}")

    # ------------------------------------------------------------------
    # Compute / load broadband SNR from parquet (snr_quad)
    # ------------------------------------------------------------------
    snr_quad_arr = None
    band4_neg_flag_arr = None
    _parquet_path = '/pscratch/sd/r/rmfeder/data/l3_data/full_validation_sz_0-1000.0_z_0-1000.0.parquet'
    _snr_cache_path = save_dir / 'snr_diagnostics' / 'snr_cache.npz'

    print(f"\n{'='*70}")
    print("BROADBAND SNR CACHE")
    print(f"{'='*70}")

    if _snr_cache_path.exists():
        print(f"  Loading SNR cache: {_snr_cache_path}")
        _cache = np.load(str(_snr_cache_path), allow_pickle=True)
        _cache_snr  = np.array(_cache['snr_quad'])
        # Validate cache alignment: if the cache contains src_id, re-index by it
        # rather than assuming positional correspondence with the result file.
        if 'src_id' in _cache:
            _cache_ids   = np.array(_cache['src_id'], dtype=np.int64)
            _result_ids  = np.array(np.load(str(result_file), allow_pickle=True)['src_id'],
                                    dtype=np.int64)
            if len(_cache_ids) == len(_result_ids) and np.array_equal(_cache_ids, _result_ids):
                # Fast path: identical order
                snr_quad_arr = _cache_snr
                print(f"  Loaded snr_quad for {len(snr_quad_arr):,} sources from cache (order verified).")
                if 'band4_neg_flux_flag' in _cache:
                    band4_neg_flag_arr = np.array(_cache['band4_neg_flux_flag'], dtype=bool)
                    print(f"  Loaded band4_neg_flux_flag: {band4_neg_flag_arr.sum():,} flagged sources.")
            else:
                # Re-index: build a lookup from cache and map to result order
                _id_to_snr = dict(zip(_cache_ids, _cache_snr))
                snr_quad_arr = np.array([_id_to_snr.get(int(s), np.nan)
                                         for s in _result_ids])
                _n_found = np.sum(np.isfinite(snr_quad_arr))
                _n_missing = len(_result_ids) - _n_found
                print(f"  Cache re-indexed by src_id: {_n_found:,} matched, "
                      f"{_n_missing:,} missing (set to NaN).")
                if _n_missing > 0:
                    print(f"  ⚠ {_n_missing:,} sources not in SNR cache "
                          f"(result file has more/different sources than when cache was built). "
                          f"Delete {_snr_cache_path} to recompute.")
                if 'band4_neg_flux_flag' in _cache:
                    _id_to_b4 = dict(zip(_cache_ids, np.array(_cache['band4_neg_flux_flag'], dtype=bool)))
                    band4_neg_flag_arr = np.array([_id_to_b4.get(int(s), False) for s in _result_ids], dtype=bool)
                    print(f"  Loaded band4_neg_flux_flag (re-indexed): {band4_neg_flag_arr.sum():,} flagged sources.")
        else:
            # Old cache format without src_id — trust positional alignment
            _result_n = len(np.load(str(result_file), allow_pickle=True)['src_id'])
            if len(_cache_snr) != _result_n:
                print(f"  ⚠ SNR cache length ({len(_cache_snr):,}) != result file length "
                      f"({_result_n:,}). Cache is stale — ignoring it.")
                print(f"    Delete {_snr_cache_path} to rebuild.")
                snr_quad_arr = None
            else:
                snr_quad_arr = _cache_snr
                print(f"  Loaded snr_quad for {len(snr_quad_arr):,} sources from cache "
                      f"(no src_id in cache — positional alignment assumed).")
                if 'band4_neg_flux_flag' in _cache:
                    band4_neg_flag_arr = np.array(_cache['band4_neg_flux_flag'], dtype=bool)
                    print(f"  Loaded band4_neg_flux_flag: {band4_neg_flag_arr.sum():,} flagged sources.")
    else:
        try:
            import pyarrow.parquet as pq
            print(f"  Computing snr_quad + band-4 neg-flux flag from parquet (this may take a few minutes)...")
            _res_tmp = np.load(str(result_file), allow_pickle=True)
            _src_ids = np.array(_res_tmp['src_id'])

            # Load SPHEREx central wavelengths from the filter set used for this run.
            # We read filter_set_name from the saved run_params.npz (written by redshift_job_batched.py).
            # This is always the observed-frame channel grid (e.g. 306 channels for SPHEREx_filter_306).
            # Band 4 is defined as channels with central wavelength >= 2.42 µm.
            _run_params_file = (Path(scratch_basepath) / 'data' / 'pae_sample_results' / 'MCLMC'
                                / 'batched' / args.datestr / 'run_params.npz')
            _b4_idx = None
            if _run_params_file.exists():
                try:
                    from models.pae_jax import load_filter_central_wavelengths
                    _rp = np.load(str(_run_params_file), allow_pickle=True)
                    _fset = str(_rp['filter_set_name'])
                    _wl, _ = load_filter_central_wavelengths(_fset, filtfiles=None)
                    _b4_idx = np.where(_wl >= 2.42)[0]
                    print(f"  Band-4 channels from filter set '{_fset}': "
                          f"{len(_b4_idx)} channels ≥ 2.42 µm (indices {_b4_idx[0]}–{_b4_idx[-1]})")
                except Exception as _filt_e:
                    print(f"  ⚠ Could not load filter central wavelengths: {_filt_e}")
            if _b4_idx is None:
                print(f"  ⚠ run_params.npz not found or filter load failed — band-4 flag will not be computed.")
                _b4_idx = np.array([], dtype=int)

            # Inline fast computation of snr_quad + band4_neg_flux_flag
            _id_col   = 'SPHERExRefID'
            _flux_col = 'flux_dered_fiducial'
            _ferr_col = 'flux_err_dered_fiducial'
            _pf = pq.ParquetFile(_parquet_path)
            _src_id_set = set(int(s) for s in _src_ids)
            _id_to_quad   = {}
            _id_to_b4flag = {}
            for _batch in _pf.iter_batches(batch_size=50_000,
                                            columns=[_id_col, _flux_col, _ferr_col]):
                _ids_b = np.array(_batch.column(_id_col).to_pylist(), dtype=np.int64)
                _hit = np.array([int(s) in _src_id_set for s in _ids_b])
                if not np.any(_hit):
                    continue
                _widx = np.where(_hit)[0]
                _flux_s = np.array([_batch.column(_flux_col)[i].as_py() for i in _widx], dtype=np.float64)
                _ferr_s = np.array([_batch.column(_ferr_col)[i].as_py() for i in _widx], dtype=np.float64)
                for _i, _sid in enumerate(_ids_b[_hit]):
                    _f, _s = _flux_s[_i], _ferr_s[_i]
                    # SNR (quadrature sum over positive-flux, finite-error channels)
                    _ok = (_f > 0) & (_s > 0) & (_s < 5e4)
                    _id_to_quad[_sid] = float(np.sqrt(np.sum((_f[_ok]/_s[_ok])**2))) if np.any(_ok) else np.nan
                    # Band-4 negative-flux flag: flag if >50% of band-4 channels
                    # have negative flux AND the mean band-4 flux is negative.
                    if len(_b4_idx) > 0 and len(_f) > max(_b4_idx):
                        _b4_f    = _f[_b4_idx]
                        _neg_frac = np.sum(_b4_f < 0) / len(_b4_f)
                        _b4_mean  = np.mean(_b4_f)
                        _id_to_b4flag[_sid] = bool(_neg_frac > 0.5 and _b4_mean < 0)
                    else:
                        _id_to_b4flag[_sid] = False

            snr_quad_arr = np.array([_id_to_quad.get(int(s), np.nan) for s in _src_ids])
            band4_neg_flag_arr = np.array([_id_to_b4flag.get(int(s), False) for s in _src_ids], dtype=bool)
            _n_b4_flagged = int(np.sum(band4_neg_flag_arr))
            print(f"  Band-4 neg-flux flag: {_n_b4_flagged:,} / {len(_src_ids):,} sources flagged "
                  f"({100*_n_b4_flagged/len(_src_ids):.2f}%)")
            _snr_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(_snr_cache_path), snr_quad=snr_quad_arr, src_id=_src_ids,
                     band4_neg_flux_flag=band4_neg_flag_arr)
            print(f"  Saved SNR + band-4 flag cache → {_snr_cache_path}")
        except Exception as _e:
            print(f"  ⚠ Could not compute snr_quad from parquet: {_e}")
            print(f"    SNR cut will be skipped (set snr_min=None to suppress this warning).")

    # Print source count summary across all cuts
    print(f"\n{'='*70}")
    print("SOURCE COUNT SUMMARY")
    print(f"{'='*70}")
    try:
        _res_cnt = np.load(str(result_file), allow_pickle=True)
        _n_total = len(_res_cnt['src_id'])
        print(f"  Total sources in results file:         {_n_total:,}")

        if 'frac_sampled_102' in _res_cnt and args.frac_sampled_min is not None:
            _frac_mask = _res_cnt['frac_sampled_102'] >= args.frac_sampled_min
            print(f"  After frac_sampled_102 \u2265 {args.frac_sampled_min}:      {_frac_mask.sum():,}"
                  f"  ({100*_frac_mask.mean():.1f}%,  cut: {(~_frac_mask).sum():,})")
        else:
            _frac_mask = np.ones(_n_total, dtype=bool)

        if snr_quad_arr is not None and args.snr_min is not None:
            _snr_mask = np.isfinite(snr_quad_arr) & (snr_quad_arr >= args.snr_min)
            print(f"  After snr_quad \u2265 {args.snr_min}:                  {_snr_mask.sum():,}"
                  f"  ({100*_snr_mask.mean():.1f}%,  cut: {(~_snr_mask).sum():,})")
            _both_mask = _frac_mask & _snr_mask
            print(f"  After BOTH cuts:                       {_both_mask.sum():,}"
                  f"  ({100*_both_mask.mean():.1f}%,  cut: {(~_both_mask).sum():,})")
        print(f"{'='*70}\n")
    except Exception as _e:
        print(f"  ⚠ Could not compute source counts: {_e}")

    # Generate plots
    try:
        stats = plot_pae_summary(
            str(result_file),
            save_dir=str(save_dir),
            show_plots=args.show,
            zscore_range=args.zscore_range,
            rhat_max=args.rhat_max,
            chi2_max=args.chi2_max,
            chain_std_max=args.chain_std_max,
            quality_tier_max=args.quality_tier_max,
            tuning_cv_min=args.tuning_cv_min,
            frac_sampled_min=args.frac_sampled_min,
            snr_min=args.snr_min,
            use_hexbin=args.hexbin,
            snr_array=snr_quad_arr,
            band4_neg_flag=band4_neg_flag_arr,
        )
        
        # Generate production convergence diagnostics
        print("\n" + "="*70)
        print("Generating production convergence diagnostics...")
        print("="*70)
        
        try:
            conv_stats = plot_production_convergence_diagnostics(
                str(result_file),
                save_dir=str(save_dir),
                show_plots=args.show,
                snr_min=args.snr_min,
                use_hexbin=args.hexbin,
                snr_array=snr_quad_arr,
                band4_neg_flag=band4_neg_flag_arr,
            )
            if conv_stats is None:
                print("⚠ Production convergence diagnostics skipped (see warnings above)")
        except Exception as e:
            print(f"⚠ Production convergence diagnostics failed with error: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing with other plots...")

        # Generate standalone chi2 comparison figure
        print("\n" + "="*70)
        print("Generating chi² comparison figure (PAE vs TF)...")
        print("="*70)

        try:
            chi2_stats = plot_chi2_comparison(
                str(result_file),
                save_dir=str(save_dir),
                show_plots=args.show,
                snr_min=args.snr_min,
                use_hexbin=args.hexbin,
                snr_array=snr_quad_arr,
                frac_sampled_min=args.frac_sampled_min,
                band4_neg_flag=band4_neg_flag_arr,
            )
            if chi2_stats is None:
                print("⚠ chi² comparison skipped (see warnings above)")
        except Exception as e:
            print(f"⚠ chi² comparison failed with error: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing with other plots...")
        
        # Generate N(z) distribution plots
        print("\n" + "="*70)
        print("Generating N(z) distribution plots...")
        print("="*70)
        
        try:
            nz_stats = plot_nz_distributions(
                str(result_file),
                save_dir=str(save_dir),
                show_plots=args.show
            )
            if nz_stats is None:
                print("⚠ N(z) distribution plots skipped (see warnings above)")
        except Exception as e:
            print(f"⚠ N(z) distribution plots failed with error: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing with other plots...")
        
        # Generate MCLMC tuning diagnostics if requested
        if args.mclmc_diagnostics:
            print("\n" + "="*70)
            print("Generating MCLMC tuning diagnostics...")
            print("="*70)
            
            # Quick check if tuning parameters exist before calling
            try:
                test_res = np.load(str(result_file), allow_pickle=True)
                has_tuning = 'tuned_L' in test_res and 'tuned_step_size' in test_res
                test_res.close() if hasattr(test_res, 'close') else None
            except:
                has_tuning = False
            
            if not has_tuning:
                print("Skipping MCLMC diagnostics: tuning parameters not in results file")
                print("  (This is expected for older result files)")
            else:
                try:
                    tuning_stats = plot_mclmc_diagnostics(
                        str(result_file),
                        save_dir=str(save_dir),
                        show_plots=args.show
                    )
                    if tuning_stats is None:
                        print("⚠ MCLMC diagnostics skipped (see warnings above)")
                except Exception as e:
                    print(f"⚠ MCLMC diagnostics failed with error: {e}")
                    import traceback
                    traceback.print_exc()
                    print("  Continuing with other plots...")
        
        # Generate pre-reinitialization convergence diagnostics
        print("\n" + "="*70)
        print("Generating pre-reinitialization convergence diagnostics...")
        print("="*70)
        
        # Quick check if pre-reinit log-posteriors exist before calling
        try:
            test_res = np.load(str(result_file), allow_pickle=True)
            has_preinit = 'preinit_final_logL' in test_res and test_res['preinit_final_logL'] is not None
            test_res.close() if hasattr(test_res, 'close') else None
        except:
            has_preinit = False
        
        if not has_preinit:
            print("Skipping convergence diagnostics: pre-reinit log-posteriors not in results file")
            print("  (This is expected if init_reinit was not enabled)")
        else:
            try:
                conv_stats = plot_preinit_convergence_diagnostics(
                    str(result_file),
                    save_dir=str(save_dir),
                    show_plots=args.show
                )
                if conv_stats is None:
                    print("⚠ Convergence diagnostics skipped (see warnings above)")
            except Exception as e:
                print(f"⚠ Convergence diagnostics failed with error: {e}")
                import traceback
                traceback.print_exc()
                print("  Continuing with other plots...")
        
        # Generate normalization diagnostics
        print("\n" + "="*70)
        print("Generating normalization diagnostics...")
        print("="*70)
        
        try:
            norm_stats = plot_normalization_diagnostics(
                str(result_file),
                save_dir=str(save_dir),
                show_plots=args.show
            )
            if norm_stats is None:
                print("⚠ Normalization diagnostics skipped (see warnings above)")
        except Exception as e:
            print(f"⚠ Normalization diagnostics failed with error: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing...")
        
        # Generate absolute normalization diagnostics
        print("\n" + "="*70)
        print("Generating absolute normalization diagnostics...")
        print("="*70)
        
        try:
            # Load the results file
            res = np.load(str(result_file), allow_pickle=True)
            result_name = result_file.stem
            
            abs_norm_stats = plot_normalization_abs_diagnostics(
                res,
                result_name,
                save_dir=str(save_dir),
                show_plots=args.show
            )
            if abs_norm_stats is None:
                print("⚠ Absolute normalization diagnostics skipped (see warnings above)")
        except Exception as e:
            print(f"⚠ Absolute normalization diagnostics failed with error: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing...")
        
        print(f"\n{'='*70}")
        print("✓ ALL PLOTS GENERATED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"Figures saved to: {save_dir}")
        print(f"{'='*70}\n")

        # ── SNR diagnostics ────────────────────────────────────────────────
        if args.snr_diagnostics and args.datestr:
            print("\n" + "="*70)
            print("Generating SNR diagnostics (plot_snr_diagnostics.py)...")
            print("="*70)
            snr_script = Path(__file__).resolve().parent / 'plot_snr_diagnostics.py'
            snr_outdir = save_dir / 'snr_diagnostics'
            snr_cache_path = snr_outdir / 'snr_cache.npz'
            snr_outdir.mkdir(parents=True, exist_ok=True)
            snr_cmd = [
                sys.executable, str(snr_script),
                '--datestr', args.datestr,
                '--outdir', str(snr_outdir),
                '--snr-metric', 'snr_quad',
            ]
            if snr_cache_path.exists():
                snr_cmd += ['--snr-cache', str(snr_cache_path)]
            else:
                snr_cmd += ['--obs-snr', '--save-snr-cache', str(snr_cache_path)]
            if not args.hexbin:
                snr_cmd.append('--no-hexbin')
            try:
                result = subprocess.run(snr_cmd, check=True)
                print("✓ SNR diagnostics complete")
            except subprocess.CalledProcessError as e:
                print(f"⚠ SNR diagnostics failed (exit code {e.returncode}) — continuing")
            except Exception as e:
                print(f"⚠ SNR diagnostics error: {e} — continuing")
        elif args.snr_diagnostics and not args.datestr:
            print("⚠ SNR diagnostics skipped: requires --datestr (not available with --result-file)")

        # ------------------------------------------------------------------
        # Run comparison (optional)
        # ------------------------------------------------------------------
        if args.compare_datestr:
            print("\n" + "="*70)
            print("Generating comparison plots vs baseline run...")
            print("="*70)
            # Try to load the baseline run's cached SNR so we can apply a
            # consistent cut even when the new run's parquet SNR is unavailable.
            _baseline_snr_cache = (Path(scratch_basepath) / 'figures' / 'redshift_validation'
                                   / args.compare_datestr / 'snr_diagnostics' / 'snr_cache.npz')
            _snr_baseline = None
            if _baseline_snr_cache.exists():
                try:
                    _bc = np.load(str(_baseline_snr_cache), allow_pickle=True)
                    _snr_baseline = np.array(_bc['snr_quad'])
                    print(f"  Loaded baseline SNR cache ({len(_snr_baseline):,} sources)")
                except Exception as _be:
                    print(f"  ⚠ Could not load baseline SNR cache: {_be}")
            try:
                plot_run_comparison(
                    new_result_file=str(result_file),
                    baseline_datestr=args.compare_datestr,
                    save_dir=str(save_dir),
                    show_plots=args.show,
                    frac_sampled_min=args.frac_sampled_min,
                    snr_min=args.snr_min,
                    snr_array_new=snr_quad_arr,
                    snr_array_baseline=_snr_baseline,
                    band4_neg_flag_new=band4_neg_flag_arr,
                )
            except Exception as e:
                print(f"⚠ Comparison plots failed: {e}")
                import traceback
                traceback.print_exc()
                print("  Continuing...")

        return 0
        
    except Exception as e:
        print(f"\n✗ ERROR generating plots: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate summary plots for PAE redshift estimation results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--result-file', type=str,
                      help='Path to PAE results .npz file')
    group.add_argument('--datestr', type=str,
                      help='Date string to auto-locate result file')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for figures (auto-generated if not specified)')
    
    # Display
    parser.add_argument('--show', action='store_true',
                       help='Show plots interactively (default: save only)')
    parser.add_argument('--no-show', dest='show', action='store_false',
                       help='Do not show plots (only save)')
    parser.set_defaults(show=False)
    
    # Plot options
    parser.add_argument('--zscore-range', type=float, nargs=2, default=(-10, 10),
                       help='Range for z-score plots')
    
    # Quality filters
    parser.add_argument('--rhat_max', type=float, default=None,
                       help='Maximum R-hat for inclusion (None = no filter)')
    parser.add_argument('--chi2_max', type=float, default=None,
                       help='Maximum reduced chi² for inclusion (None = no filter)')
    parser.add_argument('--chain_std_max', type=float, default=None,
                       help='Maximum chain z std dev (normalized) for inclusion (None = no filter)')
    parser.add_argument('--quality_tier_max', type=int, default=None,
                       help='Maximum quality tier for inclusion (0-3, None = no filter)')
    parser.add_argument('--tuning_cv_min', type=float, default=0.02,
                       help='Minimum MCLMC tuning L coefficient of variation (None = no filter)')
    parser.add_argument('--frac_sampled_min', type=float, default=0.7,
                       help='Minimum spectral completeness (frac_sampled_102) for inclusion (e.g., 0.7 = >=70%% bands, None = no filter)')
    parser.add_argument('--snr_min', type=float, default=20.0,
                       help='Minimum total SNR (sqrt(sum(weights))) for inclusion (default: 20, None = no filter)')

    # Hexbin
    parser.add_argument('--hexbin', action='store_true', default=True,
                       help='Use log-scaled hexbin plots instead of scatter (default: True)')
    parser.add_argument('--no-hexbin', dest='hexbin', action='store_false',
                       help='Use scatter plots instead of hexbins')

    # MCLMC diagnostics
    parser.add_argument('--mclmc-diagnostics', action='store_true', default=True,
                       help='Generate MCLMC tuning parameter diagnostic plots')
    parser.add_argument('--no-mclmc-diagnostics', dest='mclmc_diagnostics', action='store_false',
                       help='Skip MCLMC tuning parameter diagnostic plots')

    # SNR diagnostics
    parser.add_argument('--snr-diagnostics', action='store_true', default=True,
                       help='Run plot_snr_diagnostics.py after main plots (requires --datestr)')
    parser.add_argument('--no-snr-diagnostics', dest='snr_diagnostics', action='store_false',
                       help='Skip SNR diagnostic plots')

    # Comparison run
    parser.add_argument('--compare-datestr', type=str,
                        default='multinode_validation_run_022126',
                        help='Baseline run datestr to compare against (default: '
                             'multinode_validation_run_022126). Set empty string to skip.')
    
    args = parser.parse_args()
    # Treat empty string as "no comparison"
    if args.compare_datestr == '':
        args.compare_datestr = None
    
    sys.exit(main(args))
