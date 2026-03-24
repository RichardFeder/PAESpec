#!/usr/bin/env python3
"""
Compare coverage statistics across multiple PAE mock runs.

This script:
1. Loads multiple PAE redshift results from different configurations
2. Optionally loads template fitting (TF) results for baseline comparison
3. Computes redshift statistics (errors, z-scores, PIT) for each run
4. Generates comparison plots showing coverage across all methods

Usage:
    # Compare two PAE runs with different priors:
    python scripts/compare_pae_runs.py \\
        --run-ids mock_prior1_010625 mock_prior3_010625 \\
        --run-labels "PAE prior=1" "PAE prior=3" \\
        --run-colors blue red \\
        --include-tf \\
        --output-dir ./figures/pae_comparison

    # Compare using direct paths:
    python scripts/compare_pae_runs.py \\
        --run-paths /path/to/run1/PAE_results_combined.npz /path/to/run2/PAE_results_combined.npz \\
        --run-labels "Config A" "Config B" \\
        --output-dir ./figures/comparison

    # With quality filtering:
    python scripts/compare_pae_runs.py \\
        --run-ids mock_validation_010625 mock_validation_011525 \\
        --run-labels "Run 1" "Run 2" \\
        --rhat-max 1.8 \\
        --chi2-max 3.0 \\
        --output-dir ./figures/comparison
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_proc.data_file_utils import load_tf_results
from visualization.multi_run_comparison import (
    compute_pae_statistics,
    compute_tf_statistics,
    compute_pit_for_all_methods,
    plot_multi_method_coverage_comparison,
    filter_by_quality,
    RedshiftStatistics
)
from diagnostics.diagnostics_jax import compute_redshift_stats
from config import scratch_basepath


def load_pae_run(run_id=None, results_path=None, samples_path=None):
    """
    Load PAE results and samples for a single run.
    
    Parameters
    ----------
    run_id : str, optional
        Date string identifier for the run
    results_path : str, optional
        Direct path to results file
    samples_path : str, optional
        Direct path to samples file
        
    Returns
    -------
    tuple : (pae_results, pae_samples, results_path, samples_path)
    """
    
    if results_path is None:
        if run_id is None:
            raise ValueError("Must provide either run_id or results_path")
        
        base_dir = Path(scratch_basepath) / 'data' / 'pae_sample_results' / 'MCLMC' / 'batched' / run_id
        results_path = base_dir / f'PAE_results_combined_{run_id}.npz'
        samples_path = base_dir / f'PAE_samples_combined_{run_id}.npz'
    
    results_path = Path(results_path)
    if samples_path is not None:
        samples_path = Path(samples_path)
    else:
        # Try to infer samples path from results path
        samples_path = results_path.parent / results_path.name.replace('results', 'samples')
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    print(f"  Loading results: {results_path}")
    pae_results = dict(np.load(str(results_path), allow_pickle=True))
    
    if samples_path.exists():
        print(f"  Loading samples: {samples_path}")
        pae_samples = dict(np.load(str(samples_path), allow_pickle=True))
    else:
        print(f"  Combined samples file not found: {samples_path}")
        print(f"  Loading samples from individual batch files...")
        
        # Try to find batch sample files
        base_dir = results_path.parent
        batch_files = sorted(list(base_dir.glob('PAE_samples_batch*.npz')))
        
        # Also check in task subdirectories
        if len(batch_files) == 0:
            batch_files = sorted(list(base_dir.glob('task*/PAE_samples_batch*.npz')))
        
        if len(batch_files) == 0:
            raise FileNotFoundError(
                f"No sample files found. Searched:\n"
                f"  - {base_dir}/PAE_samples_batch*.npz\n"
                f"  - {base_dir}/task*/PAE_samples_batch*.npz"
            )
        
        print(f"  Found {len(batch_files)} batch sample files")
        
        # Load and merge batch files
        all_samples = []
        for f in batch_files:
            data = dict(np.load(str(f), allow_pickle=True))
            all_samples.append(data)
        
        # Merge samples
        sample_keys = set().union(*[d.keys() for d in all_samples])
        pae_samples = {}
        
        for k in sample_keys:
            parts = [d.get(k, None) for d in all_samples]
            parts_nonnull = [p for p in parts if p is not None]
            
            if len(parts_nonnull) == 0:
                continue
            
            try:
                pae_samples[k] = np.concatenate(parts_nonnull, axis=0)
            except Exception:
                try:
                    pae_samples[k] = np.vstack(parts_nonnull)
                except Exception:
                    print(f"  Warning: Could not merge samples for key '{k}'")
                    continue
        
        print(f"  Merged {len(pae_samples)} sample arrays")
    
    return pae_results, pae_samples, str(results_path), str(samples_path)


def extract_pae_data(pae_results, pae_samples, nsrc=None, recompute_from_samples=False):
    """
    Extract redshift estimates and samples from PAE results.
    
    Parameters
    ----------
    pae_results : dict
        PAE results dictionary
    pae_samples : dict
        PAE samples dictionary
    nsrc : int, optional
        Number of sources to use (default: all)
    recompute_from_samples : bool
        Force recomputation of z_med from samples (for old results files)
        
    Returns
    -------
    tuple : (z_true, z_med, sigmaz, samples, rhat, chi2, sample_log_amplitude)
    """
    
    z_true = pae_results['ztrue']
    z_med = pae_results['z_med']
    err_low = pae_results['err_low']
    err_high = pae_results['err_high']
    sigmaz = 0.5 * (err_low + err_high)
    rhat = pae_results.get('R_hat', None)
    chi2 = pae_results.get('chi2', None)
    
    # Get samples
    if 'all_samples' in pae_samples:
        samples = pae_samples['all_samples']
        
        # Remove burn-in
        burnin = 1000
        if samples.ndim == 4:
            # Shape: (n_sources, n_chains, n_steps, n_dim)
            samples = samples[:, :, burnin:, :]
        elif samples.ndim == 3:
            # Shape: (n_sources, n_samples, n_dim)
            samples = samples[:, burnin:, :]
    else:
        samples = None
    
    # Check if log-amplitude was sampled
    sample_log_amplitude = pae_samples.get('sample_log_amplitude', False)
    if isinstance(sample_log_amplitude, np.ndarray):
        sample_log_amplitude = bool(sample_log_amplitude.flat[0])
    
    # Check if we need to recompute z_med (for old results with amplitude bug)
    if sample_log_amplitude and 'log_amp_median' not in pae_results:
        print("  ⚠ WARNING: Old results file detected (before amplitude fix)")
        print("  → Recomputing redshift statistics from samples...")
        recompute_from_samples = True
    
    if recompute_from_samples and samples is not None:
        # Recompute z_med and uncertainties from samples
        z_idx = -1  # Redshift is always at the end
        n_sources = samples.shape[0]
        z_med = np.zeros(n_sources)
        err_low = np.zeros(n_sources)
        err_high = np.zeros(n_sources)
        
        for i in range(n_sources):
            if samples.ndim == 4:
                zsamps = samples[i, :, :, z_idx].ravel()
            else:
                zsamps = samples[i, :, z_idx]
            
            z_med[i] = np.median(zsamps)
            # try:
            #     from diagnostics.diagnostics_jax import hpd_interval2
            #     lower, upper = hpd_interval2(zsamps, alpha=0.68)
            #     err_low[i] = z_med[i] - lower
            #     err_high[i] = upper - z_med[i]
            # except:
            lower, upper = np.percentile(zsamps, [16, 84])
            err_low[i] = z_med[i] - lower
            err_high[i] = upper - z_med[i]
        
        sigmaz = 0.5 * (err_low + err_high)
        print(f"  ✓ Recomputed statistics for {n_sources} sources")
    
    # Apply source limit
    if nsrc is not None:
        nsrc = min(nsrc, len(z_true))
        z_true = z_true[:nsrc]
        z_med = z_med[:nsrc]
        sigmaz = sigmaz[:nsrc]
        if samples is not None:
            samples = samples[:nsrc]
        if rhat is not None:
            rhat = rhat[:nsrc]
        if chi2 is not None:
            chi2 = chi2[:nsrc]
    
    return z_true, z_med, sigmaz, samples, rhat, chi2, sample_log_amplitude


def main():
    parser = argparse.ArgumentParser(
        description='Compare coverage statistics across multiple PAE runs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--run-ids', type=str, nargs='+',
                            help='Date string identifiers for PAE runs to compare')
    input_group.add_argument('--run-paths', type=str, nargs='+',
                            help='Direct paths to PAE results .npz files')
    
    parser.add_argument('--run-labels', type=str, nargs='+', default=None,
                       help='Labels for each run (must match number of runs)')
    parser.add_argument('--run-colors', type=str, nargs='+', default=None,
                       help='Colors for each run (e.g., "blue" "red" "green")')
    parser.add_argument('--run-markers', type=str, nargs='+', default=None,
                       help='Markers for each run (e.g., "o" "s" "^")')
    
    # Template fitting comparison
    parser.add_argument('--include-tf', action='store_true',
                       help='Include template fitting results in comparison')
    parser.add_argument('--tf-label', type=str, default='TF',
                       help='Label for template fitting method')
    parser.add_argument('--tf-color', type=str, default='black',
                       help='Color for template fitting')
    
    # Output options
    parser.add_argument('--output-dir', type=str,
                       default='/pscratch/sd/r/rmfeder/figures/pae_comparison/',
                       help='Directory to save output figures')
    parser.add_argument('--output-prefix', type=str, default='pae_comparison',
                       help='Prefix for output filenames')
    
    # Data selection
    parser.add_argument('--nsrc', type=int, default=None,
                       help='Number of sources to use (default: all)')
    
    # Quality filtering (defaults match run_redshift_job_mock.sh / generate_mock_plots.py)
    parser.add_argument('--rhat-max', type=float, default=100.0,
                       help='Maximum R-hat threshold for quality filtering (default: 100, i.e. no R-hat cut)')
    parser.add_argument('--chi2-max', type=float, default=2.0,
                       help='Maximum chi2 threshold for quality filtering (default: 5.0)')
    parser.add_argument('--no-filtering', action='store_true',
                       help='Disable quality filtering')
    
    # Plot options
    parser.add_argument('--sigz-bins', type=float, nargs='+',
                       default=[0.0, 0.01, 0.05, 0.2],
                       help='Fractional uncertainty bins for coverage plot')
    parser.add_argument('--fracz-widths', type=float, nargs='+',
                       default=[0.05, 0.2, 0.5],
                       help='Width of fractional error plots for each bin')
    parser.add_argument('--figsize', type=int, nargs=2, default=[14, 10],
                       help='Figure size (width height)')
    parser.add_argument('--bias-correct-pit', action='store_true',
                       help='Apply bias correction to PIT values')
    parser.add_argument('--show', action='store_true',
                       help='Display plots interactively')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")
    
    # Determine runs to load
    if args.run_ids:
        run_identifiers = args.run_ids
        use_ids = True
    else:
        run_identifiers = args.run_paths
        use_ids = False
    
    n_runs = len(run_identifiers)
    
    # Set up labels
    if args.run_labels:
        if len(args.run_labels) != n_runs:
            parser.error(f"Number of labels ({len(args.run_labels)}) must match number of runs ({n_runs})")
        run_labels = args.run_labels
    else:
        run_labels = [f"Run {i+1}" for i in range(n_runs)]
    
    # Set up colors
    default_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    if args.run_colors:
        if len(args.run_colors) != n_runs:
            parser.error(f"Number of colors ({len(args.run_colors)}) must match number of runs ({n_runs})")
        run_colors = args.run_colors
    else:
        run_colors = default_colors[:n_runs]
    
    # Set up markers
    default_markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X']
    if args.run_markers:
        if len(args.run_markers) != n_runs:
            parser.error(f"Number of markers ({len(args.run_markers)}) must match number of runs ({n_runs})")
        run_markers = args.run_markers
    else:
        run_markers = default_markers[:n_runs]
    
    # ========================================================================
    # 1. Load all PAE runs and compute statistics serially
    # ========================================================================
    print("="*70)
    print("LOADING PAE RUNS AND COMPUTING STATISTICS")
    print("="*70)
    
    statistics_list = []
    
    for i, (identifier, label, color, marker) in enumerate(zip(run_identifiers, run_labels, run_colors, run_markers)):
        print(f"\n[{i+1}/{n_runs}] Processing: {label}")
        print(f"  Identifier: {identifier}")
        
        try:
            # Load PAE results and samples
            if use_ids:
                pae_results, pae_samples, _, _ = load_pae_run(run_id=identifier)
            else:
                pae_results, pae_samples, _, _ = load_pae_run(results_path=identifier)
            
            # Extract data
            z_true, z_med, sigmaz, samples, rhat, chi2, sample_log_amplitude = extract_pae_data(
                pae_results, pae_samples, nsrc=args.nsrc
            )
            
            print(f"  Sources: {len(z_true)}")
            print(f"  z range: [{z_true.min():.3f}, {z_true.max():.3f}]")
            print(f"  Sample log-amplitude: {sample_log_amplitude}")
            if samples is not None:
                print(f"  Samples shape: {samples.shape}")
            
            # Create statistics object
            stats = compute_pae_statistics(
                z_true=z_true,
                z_est=z_med,
                sigmaz=sigmaz,
                samples=samples,
                method_name=label,
                rhat=rhat,
                chi2=chi2,
                color=color,
                marker=marker,
                sample_log_amplitude=sample_log_amplitude
            )
            
            print(f"  ✓ Statistics computed")
            
            # Compute PIT values immediately for this run
            print(f"  Computing PIT values...")
            if stats.samples is not None:
                # Always compute uncorrected PIT for row 3 of the plot
                from utils.utils_jax import compute_pit_values_pae
                pit_values = compute_pit_values_pae(
                    stats.z_true, stats.samples,
                    sample_log_amplitude=sample_log_amplitude
                )
                stats.pit_values = pit_values
                print(f"  ✓ PIT computed ({len(pit_values)} values)")
                
                if args.bias_correct_pit:
                    # For bias correction in row 4, keep samples for per-bin computation
                    print(f"    Bias correction enabled - keeping samples for per-bin bias-corrected PIT")
                    # DON'T clear samples - needed for per-bin PIT computation in row 4
                else:
                    # Clear samples from memory to save space if not needed
                    stats.samples = None
            else:
                print(f"  ⚠ No samples available, cannot compute PIT")
            
            statistics_list.append(stats)
            
            print(f"  ✓ Run complete\n")
            
        except Exception as e:
            print(f"  ✗ Failed to process {identifier}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(statistics_list) == 0:
        print("\nERROR: No PAE runs processed successfully!")
        return 1
    
    print(f"\n✓ Processed {len(statistics_list)} PAE run(s)")
    
    # ========================================================================
    # 2. Optionally load template fitting results
    # ========================================================================
    if args.include_tf:
        print("\n" + "="*70)
        print("LOADING TEMPLATE FITTING RESULTS")
        print("="*70)
        
        try:
            tf_results = load_tf_results(load_zpdf=True)
            zout_tf, dzout_tf, dz_oneplusz_tf, chisq_tf, match_z_tf, zpdf_tf = tf_results
            
            # Use the same sources as first PAE run
            nsrc_to_use = len(statistics_list[0].z_true)
            z_true_tf = statistics_list[0].z_true  # Should be same for all runs

            zout_tf = zout_tf[:nsrc_to_use]
            dzout_tf = dzout_tf[:nsrc_to_use]
            chisq_tf_sel = chisq_tf[:nsrc_to_use] if chisq_tf is not None else None

            # Apply the same reduced chi2 cut to TF as to PAE.
            # TF chi2 is total (not reduced); n_dof = 102 bands - 3 parameters.
            n_dof_tf = 102 - 3
            if chisq_tf_sel is not None and not args.no_filtering and args.chi2_max is not None:
                chisq_tf_reduced = chisq_tf_sel / n_dof_tf
                tf_chi2_mask = np.isfinite(chisq_tf_reduced) & (chisq_tf_reduced < args.chi2_max)
                n_tf_kept = int(np.sum(tf_chi2_mask))
                print(f"  TF chi2 cut (reduced < {args.chi2_max}, n_dof={n_dof_tf}): "
                      f"keeping {n_tf_kept}/{nsrc_to_use} sources")
            else:
                tf_chi2_mask = np.ones(nsrc_to_use, dtype=bool)
                print(f"  TF chi2 cut: not applied")

            z_true_tf = z_true_tf[tf_chi2_mask]
            zout_tf = zout_tf[tf_chi2_mask]
            dzout_tf = dzout_tf[tf_chi2_mask]
            chisq_tf_sel = chisq_tf_sel[tf_chi2_mask] if chisq_tf_sel is not None else None

            if zpdf_tf is not None:
                print(f"  TF zPDF shape before processing: {zpdf_tf.shape}")
                zpdf_tf = zpdf_tf[:nsrc_to_use, 3:]  # Skip first 3 columns (metadata)
                zpdf_tf = zpdf_tf[tf_chi2_mask]
                print(f"  TF zPDF shape after slicing: {zpdf_tf.shape}")

                # Create redshift grid matching the zPDF columns
                # The TF zPDF has 1501 redshift bins from 0 to 3.0
                n_zbins = zpdf_tf.shape[1]
                tf_zpdf_grid = np.linspace(0, 3.0, n_zbins)
                print(f"  TF zPDF grid: n={len(tf_zpdf_grid)}, range=[{tf_zpdf_grid[0]:.3f}, {tf_zpdf_grid[-1]:.3f}], dz={tf_zpdf_grid[1]-tf_zpdf_grid[0]:.5f}")

                # Normalize zPDF
                zpdf_tf /= np.sum(zpdf_tf, axis=1)[:, np.newaxis]
            else:
                tf_zpdf_grid = None

            print(f"  Sources: {len(zout_tf)}")
            print(f"  z range: [{zout_tf.min():.3f}, {zout_tf.max():.3f}]")

            # Create TF statistics
            tf_stats = compute_tf_statistics(
                z_true=z_true_tf,
                z_est=zout_tf,
                sigmaz=dzout_tf,
                zpdf=zpdf_tf,
                zpdf_grid=tf_zpdf_grid,
                method_name=args.tf_label,
                chi2=chisq_tf_sel,
                color=args.tf_color,
                marker='s'
            )
            
            print(f"  ✓ TF statistics computed")
            
            # Compute PIT for TF
            print(f"  Computing PIT values for TF...")
            if tf_stats.zpdf is not None:
                # Always compute uncorrected PIT for row 3 of the plot
                from utils.utils_jax import compute_pit_values_tf
                pit_values = compute_pit_values_tf(
                    tf_stats.z_true, tf_stats.zpdf_grid, tf_stats.zpdf
                )
                tf_stats.pit_values = pit_values
                print(f"  ✓ PIT computed ({len(pit_values)} values)")
                
                if args.bias_correct_pit:
                    # Keep zpdf and zpdf_grid for per-bin bias-corrected PIT in row 4
                    print(f"    Bias correction enabled - keeping zpdf for per-bin bias-corrected PIT")
                else:
                    # Can clear zpdf if not needed for bias correction
                    pass
            else:
                print(f"  ⚠ No zPDF available, cannot compute PIT")
            
            statistics_list.append(tf_stats)
            
        except Exception as e:
            print(f"  ✗ Failed to load TF results: {e}")
            import traceback
            traceback.print_exc()
    
    # ========================================================================
    # 3. Apply quality filtering
    # ========================================================================
    if not args.no_filtering and (args.rhat_max is not None or args.chi2_max is not None):
        print("\n" + "="*70)
        print("APPLYING QUALITY FILTERING")
        print("="*70)
        
        if args.rhat_max:
            print(f"  R-hat threshold: < {args.rhat_max}")
        if args.chi2_max:
            print(f"  Chi2 threshold: < {args.chi2_max}")
        print()
        
        statistics_list = filter_by_quality(
            statistics_list,
            rhat_max=args.rhat_max,
            chi2_max=args.chi2_max
        )
    
    # ========================================================================
    # 4. Generate comparison plot
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING COVERAGE COMPARISON PLOT")
    print("="*70)
    
    sigz_bins = np.array(args.sigz_bins)
    print(f"  Uncertainty bins: {sigz_bins}")
    print(f"  Fractional error widths: {args.fracz_widths}")
    
    fig = plot_multi_method_coverage_comparison(
        statistics_list=statistics_list,
        sigz_bins=sigz_bins,
        figsize=tuple(args.figsize),
        fracz_widths=args.fracz_widths,
        bias_correct_pit=args.bias_correct_pit
    )
    
    # Save figure
    bias_suffix = '_biascorrpit' if args.bias_correct_pit else ''
    output_path = output_dir / f'{args.output_prefix}_coverage_grid{bias_suffix}.pdf'
    fig.savefig(output_path, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    
    # ========================================================================
    # 5. Print summary statistics
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for stats in statistics_list:
        print(f"\n{stats.method_name}:")
        print(f"  Sources: {len(stats.z_true)}")
        
        # Check for NaN/inf in derived statistics
        n_bad_dz = np.sum(~np.isfinite(stats.dz_norm))
        n_bad_zscore = np.sum(~np.isfinite(stats.z_score))
        
        if n_bad_dz > 0:
            print(f"  ⚠ {n_bad_dz} sources with NaN/inf in Δz/(1+z)")
        if n_bad_zscore > 0:
            print(f"  ⚠ {n_bad_zscore} sources with NaN/inf in z-score")
        
        # Filter out NaN/inf values for statistics computation
        sigz_oneplusz = stats.sigmaz / (1.0 + stats.z_est)
        valid_mask = (np.isfinite(stats.z_est) & np.isfinite(stats.z_true) & 
                      np.isfinite(stats.sigmaz) & np.isfinite(sigz_oneplusz))
        
        z_est_valid = stats.z_est[valid_mask]
        z_true_valid = stats.z_true[valid_mask]
        sigmaz_valid = stats.sigmaz[valid_mask]
        sigz_oneplusz_valid = sigz_oneplusz[valid_mask]
        
        n_valid = np.sum(valid_mask)
        print(f"  Valid sources (finite values): {n_valid}/{len(stats.z_true)}")
        
        if n_valid == 0:
            print(f"  ⚠ No valid sources for statistics computation")
            continue
        
        # Standard redshift error statistics (all valid sources)
        print("\n  Standard Redshift Statistics (All Valid Sources):")
        try:
            arg_bias, arg_std, bias, NMAD, cond_outl, outl_rate, outl_rate_15pct = compute_redshift_stats(
                z_est_valid, z_true_valid, 
                sigma_z_select=sigz_oneplusz_valid, 
                nsig_outlier=3, 
                outlier_pct=15
            )
            
            mean_sigz = np.mean(sigz_oneplusz_valid)
            print(f"    N_sources: {n_valid}")
            print(f"    Mean σ_z/(1+z): {mean_sigz:.5f}")
            print(f"    Bias [median(Δz/(1+z))]: {bias:.5f}")
            print(f"    NMAD: {NMAD:.5f}")
            print(f"    Outlier rate (3σ): {outl_rate:.4f} ({int(outl_rate*100)}%)")
            print(f"    Outlier rate (15% fractional): {outl_rate_15pct:.4f} ({int(outl_rate_15pct*100)}%)")
        except Exception as e:
            print(f"    ⚠ Error computing redshift stats: {e}")
        
        # Statistics for sigma_z/(1+z) < 0.02
        print("\n  Statistics for σ_z/(1+z) < 0.02:")
        try:
            mask_002 = valid_mask & (sigz_oneplusz < 0.02)
            n_002 = np.sum(mask_002)
            
            if n_002 > 0:
                z_est_002 = stats.z_est[mask_002]
                z_true_002 = stats.z_true[mask_002]
                sigz_oneplusz_002 = sigz_oneplusz[mask_002]
                
                arg_bias, arg_std, bias, NMAD, cond_outl, outl_rate, outl_rate_15pct = compute_redshift_stats(
                    z_est_002, z_true_002,
                    sigma_z_select=sigz_oneplusz_002,
                    nsig_outlier=3,
                    outlier_pct=15
                )
                
                mean_sigz_002 = np.mean(sigz_oneplusz_002)
                print(f"    N_sources: {n_002} ({100*n_002/n_valid:.1f}% of valid)")
                print(f"    Mean σ_z/(1+z): {mean_sigz_002:.5f}")
                print(f"    Bias [median(Δz/(1+z))]: {bias:.5f}")
                print(f"    NMAD: {NMAD:.5f}")
                print(f"    Outlier rate (3σ): {outl_rate:.4f} ({int(outl_rate*100)}%)")
                print(f"    Outlier rate (15% fractional): {outl_rate_15pct:.4f} ({int(outl_rate_15pct*100)}%)")
            else:
                print(f"    No sources with σ_z/(1+z) < 0.02")
        except Exception as e:
            print(f"    ⚠ Error: {e}")
        
        # Statistics for sigma_z/(1+z) < 0.2
        print("\n  Statistics for σ_z/(1+z) < 0.2:")
        try:
            mask_02 = valid_mask & (sigz_oneplusz < 0.2)
            n_02 = np.sum(mask_02)
            
            if n_02 > 0:
                z_est_02 = stats.z_est[mask_02]
                z_true_02 = stats.z_true[mask_02]
                sigz_oneplusz_02 = sigz_oneplusz[mask_02]
                
                arg_bias, arg_std, bias, NMAD, cond_outl, outl_rate, outl_rate_15pct = compute_redshift_stats(
                    z_est_02, z_true_02,
                    sigma_z_select=sigz_oneplusz_02,
                    nsig_outlier=3,
                    outlier_pct=15
                )
                
                mean_sigz_02 = np.mean(sigz_oneplusz_02)
                print(f"    N_sources: {n_02} ({100*n_02/n_valid:.1f}% of valid)")
                print(f"    Mean σ_z/(1+z): {mean_sigz_02:.5f}")
                print(f"    Bias [median(Δz/(1+z))]: {bias:.5f}")
                print(f"    NMAD: {NMAD:.5f}")
                print(f"    Outlier rate (3σ): {outl_rate:.4f} ({int(outl_rate*100)}%)")
                print(f"    Outlier rate (15% fractional): {outl_rate_15pct:.4f} ({int(outl_rate_15pct*100)}%)")
            else:
                print(f"    No sources with σ_z/(1+z) < 0.2")
        except Exception as e:
            print(f"    ⚠ Error: {e}")
        
        print(f"\n  Coverage Statistics:")
        print(f"    Median |Δz/(1+z)|: {np.nanmedian(np.abs(stats.dz_norm)):.4f}")
        print(f"    Median z-score: {np.nanmedian(stats.z_score):.4f}")
        print(f"    σ(z-score): {np.nanstd(stats.z_score):.4f}")
        
        if stats.pit_values is not None:
            # Compute uniformity metrics
            n_bad_pit = np.sum(~np.isfinite(stats.pit_values))
            if n_bad_pit > 0:
                print(f"    ⚠ {n_bad_pit} sources with NaN/inf in PIT values")
            
            valid_pit = stats.pit_values[np.isfinite(stats.pit_values)]
            if len(valid_pit) > 0:
                ks_stat = np.max(np.abs(np.sort(valid_pit) - np.linspace(0, 1, len(valid_pit))))
                print(f"    PIT KS statistic: {ks_stat:.4f} (based on {len(valid_pit)} valid sources)")
            else:
                print(f"    ⚠ No valid PIT values")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
