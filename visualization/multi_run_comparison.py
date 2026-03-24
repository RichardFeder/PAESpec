#!/usr/bin/env python3
"""
Functions for comparing multiple PAE runs and template fitting results.

This module provides utilities to:
1. Compute redshift statistics (errors, z-scores, PIT) for each run
2. Generate comparison plots across multiple configurations
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from utils.utils_jax import compute_pit_values_pae, compute_pit_values_tf
from utils.utils_jax import compute_pit_values_pae_bias_corrected, compute_pit_values_tf_bias_corrected


@dataclass
class RedshiftStatistics:
    """Container for redshift statistics from a single run."""
    
    # Method identifier
    method_name: str
    
    # True redshifts
    z_true: np.ndarray
    
    # Estimated redshifts and uncertainties
    z_est: np.ndarray
    sigmaz: np.ndarray
    
    # Derived statistics
    fractional_sigma: np.ndarray  # sigma_z / (1 + z_est)
    dz_norm: np.ndarray  # (z_est - z_true) / (1 + z_true)
    z_score: np.ndarray  # (z_est - z_true) / sigma_z
    
    # PIT values (probability integral transform)
    pit_values: Optional[np.ndarray] = None
    
    # Additional data for computing PIT
    samples: Optional[np.ndarray] = None  # For PAE
    zpdf: Optional[np.ndarray] = None  # For TF
    zpdf_grid: Optional[np.ndarray] = None  # For TF
    
    # Quality metrics
    rhat: Optional[np.ndarray] = None
    chi2: Optional[np.ndarray] = None
    
    # Plotting style
    color: str = 'b'
    linestyle: str = '-'
    marker: str = 'o'
    
    # PAE-specific metadata
    sample_log_amplitude: bool = False


def compute_pae_statistics(z_true: np.ndarray,
                           z_est: np.ndarray,
                           sigmaz: np.ndarray,
                           samples: np.ndarray,
                           method_name: str = 'PAE',
                           rhat: Optional[np.ndarray] = None,
                           chi2: Optional[np.ndarray] = None,
                           color: str = 'b',
                           linestyle: str = '-',
                           marker: str = 'o',
                           sample_log_amplitude: bool = False) -> RedshiftStatistics:
    """
    Compute redshift statistics for a PAE run.
    
    Parameters
    ----------
    z_true : array
        True redshifts
    z_est : array
        Estimated redshifts (medians)
    sigmaz : array
        Redshift uncertainties
    samples : array
        Posterior samples, shape (n_sources, n_samples, n_dim) or (n_sources, n_chains, n_steps, n_dim)
    method_name : str
        Identifier for this run (e.g., 'PAE prior=3', 'PAE prior=1')
    rhat : array, optional
        R-hat convergence diagnostic
    chi2 : array, optional
        Chi-squared values
    color : str
        Plotting color
    linestyle : str
        Line style for plots
    marker : str
        Marker style for plots
    sample_log_amplitude : bool
        Whether log-amplitude was sampled (affects PIT computation)
        
    Returns
    -------
    RedshiftStatistics
        Container with all computed statistics
    """
    
    # Compute derived statistics
    fractional_sigma = sigmaz / (1 + z_est)
    dz_norm = (z_est - z_true) / (1 + z_true)
    z_score = (z_est - z_true) / sigmaz
    
    # Create statistics object
    stats = RedshiftStatistics(
        method_name=method_name,
        z_true=z_true,
        z_est=z_est,
        sigmaz=sigmaz,
        fractional_sigma=fractional_sigma,
        dz_norm=dz_norm,
        z_score=z_score,
        samples=samples,
        rhat=rhat,
        chi2=chi2,
        color=color,
        linestyle=linestyle,
        marker=marker
    )
    
    return stats


def compute_tf_statistics(z_true: np.ndarray,
                          z_est: np.ndarray,
                          sigmaz: np.ndarray,
                          zpdf: Optional[np.ndarray] = None,
                          zpdf_grid: Optional[np.ndarray] = None,
                          method_name: str = 'TF',
                          chi2: Optional[np.ndarray] = None,
                          color: str = 'k',
                          linestyle: str = '-',
                          marker: str = 's') -> RedshiftStatistics:
    """
    Compute redshift statistics for template fitting results.
    
    Parameters
    ----------
    z_true : array
        True redshifts
    z_est : array
        Estimated redshifts
    sigmaz : array
        Redshift uncertainties
    zpdf : array, optional
        Redshift PDFs, shape (n_sources, n_z_bins)
    zpdf_grid : array, optional
        Redshift grid for PDFs
    method_name : str
        Identifier for this method
    chi2 : array, optional
        Chi-squared values
    color : str
        Plotting color
    linestyle : str
        Line style for plots
    marker : str
        Marker style for plots
        
    Returns
    -------
    RedshiftStatistics
        Container with all computed statistics
    """
    
    # Compute derived statistics
    fractional_sigma = sigmaz / (1 + z_est)
    dz_norm = (z_est - z_true) / (1 + z_true)
    z_score = (z_est - z_true) / sigmaz
    
    # Create statistics object
    stats = RedshiftStatistics(
        method_name=method_name,
        z_true=z_true,
        z_est=z_est,
        sigmaz=sigmaz,
        fractional_sigma=fractional_sigma,
        dz_norm=dz_norm,
        z_score=z_score,
        zpdf=zpdf,
        zpdf_grid=zpdf_grid,
        chi2=chi2,
        color=color,
        linestyle=linestyle,
        marker=marker
    )
    
    return stats


def compute_pit_for_all_methods(statistics_list: List[RedshiftStatistics],
                                 bias_correct: bool = False) -> List[RedshiftStatistics]:
    """
    Compute PIT values for all methods in the list.
    
    Parameters
    ----------
    statistics_list : list of RedshiftStatistics
        List of statistics objects (will be modified in place)
    bias_correct : bool
        Whether to apply bias correction to PIT values
        
    Returns
    -------
    list of RedshiftStatistics
        Same list with PIT values computed
    """
    
    for stats in statistics_list:
        print(f"Computing PIT for {stats.method_name}...")
        
        if stats.samples is not None:
            # PAE-like method with posterior samples
            # Check if we need bias correction
            sample_log_amplitude = stats.samples.shape[-1] > 1  # Heuristic
            
            if bias_correct:
                pit_values, bias = compute_pit_values_pae_bias_corrected(
                    stats.z_true, stats.samples, stats.z_est,
                    sample_log_amplitude=sample_log_amplitude
                )
                print(f"  {stats.method_name} mean bias: {bias:.4f}")
            else:
                pit_values = compute_pit_values_pae(
                    stats.z_true, stats.samples,
                    sample_log_amplitude=sample_log_amplitude
                )
            
            stats.pit_values = pit_values
            
        elif stats.zpdf is not None:
            # TF-like method with PDF
            if bias_correct:
                pit_values, bias = compute_pit_values_tf_bias_corrected(
                    stats.z_true, stats.zpdf_grid, stats.zpdf, stats.z_est
                )
                print(f"  {stats.method_name} mean bias: {bias:.4f}")
            else:
                pit_values = compute_pit_values_tf(
                    stats.z_true, stats.zpdf_grid, stats.zpdf
                )
            
            stats.pit_values = pit_values
        else:
            print(f"  Warning: Cannot compute PIT for {stats.method_name} (no samples or PDF)")
    
    return statistics_list


def plot_multi_method_coverage_comparison(
    statistics_list: List[RedshiftStatistics],
    sigz_bins: np.ndarray,
    figsize: Tuple[int, int] = (10, 11),
    z_score_xlim: List[float] = [-6, 6],
    fracz_widths: Optional[List[float]] = None,
    label_fs: int = 12,
    title_fs: int = 14,
    nbins: int = 30,
    height_ratios: List[float] = [1, 1, 1.6, 1.6],
    hspace: float = 0.3,
    wspace: float = 0.05,
    alpha: float = 0.6,
    s: float = 2,
    legend_loc: int = 2,
    legend_fs: int = 12,
    legend_ncol: int = 3,
    bbox_to_anchor: List[float] = [-0.1, 1.8],
    bias_correct_pit: bool = False
) -> plt.Figure:
    """
    Generate coverage comparison grid for multiple methods.
    
    This extends plot_coverage_comparison_grid to handle arbitrary numbers of methods.
    
    Parameters
    ----------
    statistics_list : list of RedshiftStatistics
        List of statistics for each method to compare
    sigz_bins : array
        Edges of fractional uncertainty bins for columns
    figsize : tuple
        Figure size
    z_score_xlim : list
        x-axis limits for z-score plots
    fracz_widths : list, optional
        Width of fractional error plots for each bin
    label_fs : int
        Label font size
    title_fs : int
        Title font size
    nbins : int
        Number of histogram bins
    height_ratios : list
        Relative heights of the three rows
    hspace : float
        Vertical spacing between subplots
    wspace : float
        Horizontal spacing between subplots
    alpha : float
        Transparency for scatter points
    s : float
        Marker size for QQ plots
    legend_loc : str
        Legend location
    legend_fs : int
        Legend font size
    bias_correct_pit : bool, optional
        If True, compute PIT with bias correction applied separately for each uncertainty bin.
        This requires that samples or zpdf be available in the RedshiftStatistics objects.
        Default is False (use pre-computed PIT values).
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure
    """
    
    import matplotlib as mpl
    from scipy.stats import kstest as _kstest


    ncols = len(sigz_bins) - 1
    fig, axes = plt.subplots(
        nrows=4, ncols=ncols, figsize=figsize,
        gridspec_kw={'hspace': hspace, 'wspace': wspace, 'height_ratios': height_ratios}
    )
    
    # Ensure axes is 2D
    if ncols == 1:
        axes = axes[:, np.newaxis]
    
    # Loop over fractional uncertainty bins (columns)
    for i in range(ncols):
        low_sigz, high_sigz = sigz_bins[i], sigz_bins[i + 1]
        
        # Collect data for this bin from all methods
        bin_data = []
        for stats in statistics_list:
            mask = (stats.fractional_sigma >= low_sigz) & (stats.fractional_sigma < high_sigz)
            
            if np.sum(mask) < 2:
                continue
            
            # Always use pre-computed PIT values for row 3 (uncorrected)
            pit_uncorrected = stats.pit_values[mask] if stats.pit_values is not None else None
            
            bin_data.append({
                'method': stats.method_name,
                'dz_norm': stats.dz_norm[mask],
                'z_score': stats.z_score[mask],
                'pit_uncorrected': pit_uncorrected,
                'color': stats.color,
                'linestyle': stats.linestyle,
                'marker': stats.marker
            })
        
        # Handle empty bins
        if len(bin_data) == 0:
            for row in range(4):
                axes[row, i].set_title(f'Bin {i+1}\n(n=0)')
                axes[row, i].axis('off')
            continue
        
        # --- Row 1: Fractional Redshift Error Histograms ---
        ax = axes[0, i]
        
        if i == 0:
            title = f'$\\sigma_{{z/(1+z)}}<{high_sigz}$'
        else:
            title = f'{low_sigz}$<\\sigma_{{z/(1+z)}}<{high_sigz}$'
        ax.set_title(title, fontsize=title_fs)

        
        if fracz_widths is not None:
            ax.set_xlim(-fracz_widths[i], fracz_widths[i])
            dzbins = np.linspace(-fracz_widths[i], fracz_widths[i], nbins)
        else:
            print('using nbins=', nbins)
            dzbins = nbins
        
        for d, data in enumerate(bin_data):
            ax.hist(
                data['dz_norm'], bins=dzbins, histtype='step',
                color=data['color'], linestyle=data['linestyle'],
                label=data['method'], density=False, linewidth=1.5
            )
            ax.axvline(
                np.median(data['dz_norm']),
                color=data['color'], linestyle='-', linewidth=1)
        
        ax.set_yticks([])
        ax.set_xlabel('$\\Delta z/(1+z)$', fontsize=label_fs)
        
        if i == 0:
            ax.legend(loc=legend_loc, ncol=legend_ncol, bbox_to_anchor=bbox_to_anchor,
                       fontsize=legend_fs, framealpha=0.8)
        
        # ax.grid(alpha=0.3)
        # --- Row 2: Z-score Histograms ---
        ax = axes[1, i]
        
        zscore_bins = np.linspace(z_score_xlim[0], z_score_xlim[1], nbins)
        
        for d, data in enumerate(bin_data):
            ax.hist(
                data['z_score'], bins=zscore_bins, histtype='step',
                color=data['color'], linestyle=data['linestyle'],
                label=data['method'], density=True, linewidth=1.5
            )
            ax.axvline(
                np.median(data['z_score']),
                color=data['color'], linestyle='-', linewidth=1)
        
        ax.set_xlim(z_score_xlim)
        ax.set_xlabel('Redshift z-score', fontsize=label_fs)
        # ax.grid(alpha=0.3, axis='both')
        if i > 0:
            ax.set_yticks([])

        # --- Row 3: PIT without bias correction ---
        ax = axes[2, i]
        
        # Add text label in first panel
        if i == 0:
            ax.text(0.05, 0.9, 'No bias correction', 
                   transform=ax.transAxes, fontsize=12, verticalalignment='top')
        
        ks_text_idx = 0  # counter for stacking KS annotations
        for d, data in enumerate(bin_data):
            if data['pit_uncorrected'] is None:
                continue
            
            # Filter out NaN/inf values from PIT
            pit_valid = data['pit_uncorrected'][np.isfinite(data['pit_uncorrected'])]
            
            if len(pit_valid) == 0:
                print(f"  WARNING: No valid PIT values for {data['method']} in bin {i}")
                continue
            
            n = len(pit_valid)
            pit_sorted = np.sort(pit_valid)
            uniform_quantiles = np.linspace(1/(2*n), 1-1/(2*n), n)
            
            ax.plot(
                uniform_quantiles, pit_sorted,
                color=data['color'],
                linestyle='-', linewidth=2,
                label=data['method']
            )

            # KS statistic vs uniform(0,1)
            ks_stat = _kstest(pit_valid, 'uniform').statistic
            ax.text(
                0.97, 0.03 + ks_text_idx * 0.13,
                f'KS={ks_stat:.3f}',
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=9, color=data['color'],
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
            )
            ks_text_idx += 1
        
        ax.plot([0, 1], [0, 1], 'k--', label='Ideal', linewidth=1.5, alpha=0.8, zorder=100)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Theoretical Quantiles', fontsize=label_fs)
        
        if i == 0:
            ax.set_ylabel('Empirical Quantiles', fontsize=label_fs)
        
        # ax.grid(alpha=0.3, axis='both')
        if i > 0:
            ax.set_yticks([])
        
        # --- Row 4: PIT with bias correction ---
        ax = axes[3, i]
        
        # Add text label in first panel
        if i == 0:
            ax.text(0.05, 0.9, 'With bias correction', 
                   transform=ax.transAxes, fontsize=12, verticalalignment='top')
        
        # Compute bias-corrected PIT for this bin if not already computed
        bin_data_corrected = []
        for stats in statistics_list:
            mask = (stats.fractional_sigma >= low_sigz) & (stats.fractional_sigma < high_sigz)
            
            if np.sum(mask) < 2:
                continue
            
            # Compute bias-corrected PIT per-bin
            if stats.samples is not None:
                # PAE-like method with samples
                pit_corrected, bias_bin = compute_pit_values_pae_bias_corrected(
                    stats.z_true[mask],
                    stats.samples[mask],
                    stats.z_est[mask],
                    sample_log_amplitude=stats.sample_log_amplitude
                )
            elif stats.zpdf is not None:
                # TF-like method with PDF
                pit_corrected, bias_bin = compute_pit_values_tf_bias_corrected(
                    stats.z_true[mask],
                    stats.zpdf_grid,
                    stats.zpdf[mask],
                    stats.z_est[mask]
                )
            else:
                continue
            
            bin_data_corrected.append({
                'method': stats.method_name,
                'pit': pit_corrected,
                'color': stats.color,
                'linestyle': stats.linestyle
            })
        
        ks_text_idx = 0  # counter for stacking KS annotations
        for d, data in enumerate(bin_data_corrected):
            if data['pit'] is None:
                continue
            
            # Filter out NaN/inf values from PIT
            pit_valid = data['pit'][np.isfinite(data['pit'])]
            
            if len(pit_valid) == 0:
                print(f"  WARNING: No valid bias-corrected PIT values for {data['method']} in bin {i}")
                continue
            
            n = len(pit_valid)
            pit_sorted = np.sort(pit_valid)
            uniform_quantiles = np.linspace(1/(2*n), 1-1/(2*n), n)
            
            ax.plot(
                uniform_quantiles, pit_sorted,
                color=data['color'],
                linestyle='-', linewidth=2,
                label=data['method']
            )

            # KS statistic vs uniform(0,1)
            ks_stat = _kstest(pit_valid, 'uniform').statistic
            ax.text(
                0.97, 0.03 + ks_text_idx * 0.13,
                f'KS={ks_stat:.3f}',
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=9, color=data['color'],
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
            )
            ks_text_idx += 1
        
        ax.plot([0, 1], [0, 1], 'k--', label='Ideal', linewidth=1.5, alpha=0.8, zorder=100)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Theoretical Quantiles', fontsize=label_fs)
        
        if i == 0:
            ax.set_ylabel('Empirical Quantiles', fontsize=label_fs)
        
        # ax.grid(alpha=0.3, axis='both')
        if i > 0:
            ax.set_yticks([])
    
    plt.subplots_adjust(wspace=0.4, hspace=0.35)
    # plt.tight_layout()
    return fig


def filter_by_quality(statistics_list: List[RedshiftStatistics],
                      rhat_max: Optional[float] = None,
                      chi2_max: Optional[float] = None) -> List[RedshiftStatistics]:
    """
    Apply quality filters to all statistics in list.
    
    Parameters
    ----------
    statistics_list : list of RedshiftStatistics
        Statistics to filter (will create new objects)
    rhat_max : float, optional
        Maximum R-hat threshold
    chi2_max : float, optional
        Maximum chi-squared threshold
        
    Returns
    -------
    list of RedshiftStatistics
        New list with filtered statistics
    """
    
    filtered_list = []

    for stats in statistics_list:
        n_total = len(stats.z_true)

        # Template fitting (no R-hat) is kept completely unfiltered,
        # matching the behaviour of generate_mock_plots.py which keeps
        # TF arrays UNFILTERED (only PAE quality cuts are applied).
        if stats.rhat is None:
            mask = np.ones(n_total, dtype=bool)
            print(f"{stats.method_name}: no R-hat available, keeping all {n_total} sources (unfiltered)")
        else:
            mask = np.ones(n_total, dtype=bool)
            # finite guards (match generate_mock_plots.py)
            mask &= np.isfinite(stats.z_true) & np.isfinite(stats.z_est)
            mask &= np.isfinite(stats.rhat)
            if rhat_max is not None:
                rhat_fail = ~(stats.rhat < rhat_max)
                n_rhat_fail = int(np.sum(rhat_fail & mask))
                mask &= ~rhat_fail
                print(f"{stats.method_name}: {n_rhat_fail} sources filtered by R-hat")
            if chi2_max is not None and stats.chi2 is not None:
                chi2_fail = ~(np.isfinite(stats.chi2) & (stats.chi2 < chi2_max))
                n_chi2_fail = int(np.sum(chi2_fail & mask))
                mask &= ~chi2_fail
                print(f"{stats.method_name}: {n_chi2_fail} sources filtered by chi2")
            n_kept = int(np.sum(mask))
            print(f"{stats.method_name}: keeping {n_kept}/{n_total} sources ({100*n_kept/n_total:.1f}%)")
        
        # Create filtered copy
        filtered_stats = RedshiftStatistics(
            method_name=stats.method_name,
            z_true=stats.z_true[mask],
            z_est=stats.z_est[mask],
            sigmaz=stats.sigmaz[mask],
            fractional_sigma=stats.fractional_sigma[mask],
            dz_norm=stats.dz_norm[mask],
            z_score=stats.z_score[mask],
            pit_values=stats.pit_values[mask] if stats.pit_values is not None else None,
            samples=stats.samples[mask] if stats.samples is not None else None,
            zpdf=stats.zpdf[mask] if stats.zpdf is not None else None,
            zpdf_grid=stats.zpdf_grid,
            rhat=stats.rhat[mask] if stats.rhat is not None else None,
            chi2=stats.chi2[mask] if stats.chi2 is not None else None,
            color=stats.color,
            linestyle=stats.linestyle,
            marker=stats.marker
        )
        
        filtered_list.append(filtered_stats)
    
    return filtered_list
