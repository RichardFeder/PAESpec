"""
Comprehensive Redshift Analysis Script

This script generates a comprehensive suite of analysis plots for PAE redshift
estimation results, comparing against template fitting. It automates the plots
typically generated in paper_plots.ipynb with systematic file naming.

Usage:
    python comprehensive_redshift_analysis.py \
        --pae_results <path> \
        --pae_samples <path> \
        --run_name <name> \
        [--output_dir figures/analysis/] \
        [--burn_in 1000]
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_proc import load_tf_results, load_combined_pae_results
from visualization.result_plotting_fns import (
    prepare_data_for_plotting,
    compare_pae_tf_redshifts,
    plot_coverage_comparison_grid,
    compare_chi2_tf_pae,
    plot_zerr_tf_pae_with_binned_corr,
    compare_zscore_dists_tf_pae,
    plot_gr_statistic_chains,
    compare_PDFs_TF_PAE,
    plot_pit_histogram,
    plot_qq,
    compare_sigmaz_hdpi_secondmom,
    plot_reduced_chi2,
    plot_chi2_vs_redshift_error,
    plot_phot_snr_vs_redshift_error
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive redshift analysis plots"
    )
    
    parser.add_argument(
        '--pae_results',
        type=str,
        required=True,
        help='Path or glob pattern for PAE results file(s)'
    )
    
    parser.add_argument(
        '--pae_samples',
        type=str,
        required=True,
        help='Path or glob pattern for PAE samples file(s)'
    )
    
    parser.add_argument(
        '--run_name',
        type=str,
        required=True,
        help='Name for this run (used in figure filenames)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='figures/analysis/',
        help='Directory to save output figures'
    )
    
    parser.add_argument(
        '--burn_in',
        type=int,
        default=1000,
        help='Number of burn-in steps to trim from chains'
    )
    
    parser.add_argument(
        '--nsrc',
        type=int,
        default=None,
        help='Number of sources to analyze (default: all)'
    )
    
    parser.add_argument(
        '--tf_load_zpdf',
        action='store_true',
        default=True,
        help='Load template fitting PDFs (required for some plots)'
    )
    
    parser.add_argument(
        '--plot_groups',
        type=str,
        nargs='+',
        default=['all'],
        choices=['all', 'basic', 'coverage', 'diagnostics', 'pdf', 'convergence'],
        help='Which plot groups to generate'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved figures'
    )
    
    return parser.parse_args()


def save_figure(fig, output_dir, filename, dpi=300):
    """Save figure with consistent settings"""
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)


def compute_statistics(z_true, z_out, sigmaz=None):
    """Compute summary statistics"""
    dz = z_out - z_true
    dz_norm = dz / (1 + z_true)
    
    stats = {
        'bias': np.median(dz_norm),
        'nmad': 1.48 * np.median(np.abs(dz_norm - np.median(dz_norm))),
        'outlier_frac': np.sum(np.abs(dz_norm) > 0.15) / len(dz_norm),
    }
    
    if sigmaz is not None:
        zscore = dz / sigmaz
        stats['zscore_mean'] = np.mean(zscore)
        stats['zscore_std'] = np.std(zscore)
    
    return stats


def plot_basic_comparison(data, run_name, output_dir, dpi=300):
    """Generate basic comparison plots"""
    print("\n=== Generating Basic Comparison Plots ===")
    
    (z_true, z_out_pae, sigmaz_pae, pae_samples,
     z_out_tf, dzout_tf, tf_zpdfs, tf_zpdf_fine_z, redshift_Rhat) = data
    
    # 1. PAE vs TF comparison
    print("  Creating PAE vs TF comparison...")
    all_med_z = [z_out_tf, z_out_pae]
    all_err_low = [dzout_tf, sigmaz_pae]
    all_err_high = [dzout_tf, sigmaz_pae]
    
    fig = compare_pae_tf_redshifts(
        all_med_z, all_err_low, all_err_high, z_true,
        ylabels=['$\\hat{z}_{TF}$', '$\\hat{z}_{PAE}$'],
        figsize=(12, 5)
    )
    save_figure(fig, output_dir, f'{run_name}_comparison_pae_tf.png', dpi)
    
    # 2. Detailed comparison with correlation
    print("  Creating detailed comparison with binned correlation...")
    fig = plot_zerr_tf_pae_with_binned_corr(
        z_out_tf, z_true, z_out_pae, z_true, sigmaz_pae,
        figsize=(12, 5), n_bins=10
    )
    save_figure(fig, output_dir, f'{run_name}_comparison_detailed.png', dpi)
    
    # Compute and print statistics
    stats_pae = compute_statistics(z_true, z_out_pae, sigmaz_pae)
    stats_tf = compute_statistics(z_true, z_out_tf, dzout_tf)
    
    print(f"\n  PAE Statistics:")
    print(f"    Bias: {stats_pae['bias']:.4f}")
    print(f"    NMAD: {stats_pae['nmad']:.4f}")
    print(f"    Outlier fraction: {stats_pae['outlier_frac']:.3f}")
    if 'zscore_mean' in stats_pae:
        print(f"    Z-score mean: {stats_pae['zscore_mean']:.3f}")
        print(f"    Z-score std: {stats_pae['zscore_std']:.3f}")
    
    print(f"\n  TF Statistics:")
    print(f"    Bias: {stats_tf['bias']:.4f}")
    print(f"    NMAD: {stats_tf['nmad']:.4f}")
    print(f"    Outlier fraction: {stats_tf['outlier_frac']:.3f}")


def plot_coverage_analysis(data, run_name, output_dir, burn_in=1000, dpi=300):
    """Generate coverage analysis plots"""
    print("\n=== Generating Coverage Analysis Plots ===")
    
    (z_true, z_out_pae, sigmaz_pae, pae_samples,
     z_out_tf, dzout_tf, tf_zpdfs, tf_zpdf_fine_z, redshift_Rhat) = data
    
    if tf_zpdfs is None:
        print("  Warning: TF PDFs not loaded, skipping coverage plots")
        return
    
    # Trim burn-in for coverage analysis
    if len(pae_samples.shape) == 4 and burn_in > 0:
        pae_samples_trimmed = pae_samples[:, :, burn_in:, :]
    else:
        pae_samples_trimmed = pae_samples
    
    # Coverage comparison grid
    print("  Creating coverage comparison grid...")
    sigz_bins = np.array([0.0, 0.02, 0.05, 0.1, 0.2])
    fig = plot_coverage_comparison_grid(
        z_true, z_out_pae, sigmaz_pae, pae_samples_trimmed,
        z_out_tf, dzout_tf, tf_zpdfs, tf_zpdf_fine_z,
        sigz_bins=sigz_bins,
        figsize=(12, 10)
    )
    save_figure(fig, output_dir, f'{run_name}_coverage_grid.png', dpi)


def plot_diagnostics(data, run_name, output_dir, dpi=300):
    """Generate diagnostic plots"""
    print("\n=== Generating Diagnostic Plots ===")
    
    (z_true, z_out_pae, sigmaz_pae, pae_samples,
     z_out_tf, dzout_tf, tf_zpdfs, tf_zpdf_fine_z, redshift_Rhat) = data
    
    # 1. Chi-squared comparison (if available in results file)
    # Note: This requires loading chi2 values from results
    print("  Chi-squared comparison (if available)...")
    # This would need chi2 values from the results file
    
    # 2. Z-score comparison
    print("  Creating z-score comparison...")
    zscore_pae = (z_out_pae - z_true) / sigmaz_pae
    zscore_tf = (z_out_tf - z_true) / dzout_tf
    
    fig = compare_zscore_dists_tf_pae(
        zscore_pae, zscore_tf,
        figsize=(8, 4),
        zscore_lim=[-5, 5]
    )
    save_figure(fig, output_dir, f'{run_name}_zscore_comparison.png', dpi)
    
    # 3. Gelman-Rubin statistic histogram
    print("  Creating Gelman-Rubin statistic histogram...")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(redshift_Rhat, bins=50, edgecolor='k', alpha=0.7)
    ax.axvline(1.1, color='r', linestyle='--', label='R-hat = 1.1 threshold')
    ax.set_xlabel('Gelman-Rubin R-hat', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Chain Convergence Diagnostic', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, output_dir, f'{run_name}_rhat_histogram.png', dpi)
    
    print(f"  Mean R-hat: {np.nanmean(redshift_Rhat):.4f}")
    print(f"  Fraction R-hat > 1.1: {np.sum(redshift_Rhat > 1.1) / len(redshift_Rhat):.3f}")


def plot_pdf_examples(data, run_name, output_dir, n_examples=10, dpi=300):
    """Generate PDF comparison examples"""
    print("\n=== Generating PDF Comparison Examples ===")
    
    (z_true, z_out_pae, sigmaz_pae, pae_samples,
     z_out_tf, dzout_tf, tf_zpdfs, tf_zpdf_fine_z, redshift_Rhat) = data
    
    if tf_zpdfs is None:
        print("  Warning: TF PDFs not loaded, skipping PDF plots")
        return
    
    # Select interesting examples (mix of good and challenging cases)
    n_src = len(z_true)
    example_idxs = np.linspace(0, n_src-1, n_examples, dtype=int)
    
    print(f"  Creating PDF comparisons for {n_examples} examples...")
    
    # Extract redshift samples (last dimension)
    if len(pae_samples.shape) == 4:
        z_samples_pae = pae_samples[:, :, :, -1]  # (n_src, n_chain, n_step)
    else:
        z_samples_pae = pae_samples
    
    fig = compare_PDFs_TF_PAE(
        tf_zpdf_fine_z, tf_zpdfs[example_idxs],
        z_samples_pae[example_idxs],
        z_true[example_idxs],
        figsize=(12, 8),
        nrows=2,
        ncols=5
    )
    save_figure(fig, output_dir, f'{run_name}_pdf_examples.png', dpi)


def plot_pit_analysis(data, run_name, output_dir, burn_in=1000, dpi=300):
    """Generate PIT (Probability Integral Transform) analysis"""
    print("\n=== Generating PIT Analysis ===")
    
    (z_true, z_out_pae, sigmaz_pae, pae_samples,
     z_out_tf, dzout_tf, tf_zpdfs, tf_zpdf_fine_z, redshift_Rhat) = data
    
    # Extract redshift samples (use full chains for PIT)
    if len(pae_samples.shape) == 4:
        z_samples_pae = pae_samples[:, :, :, -1]  # (n_src, n_chain, n_step)
    else:
        z_samples_pae = pae_samples
    
    # Compute PIT values for PAE
    print("  Computing PIT values for PAE...")
    pit_pae = []
    for i in range(len(z_true)):
        z_posterior = z_samples_pae[i].flatten()
        pit = (z_posterior < z_true[i]).mean()
        pit_pae.append(pit)
    pit_pae = np.array(pit_pae)
    
    # PIT histogram
    print("  Creating PIT histogram...")
    fig = plot_pit_histogram(pit_pae, f'PAE PIT Distribution ({run_name})')
    save_figure(fig, output_dir, f'{run_name}_pit_histogram.png', dpi)
    
    # Q-Q plot
    print("  Creating Q-Q plot...")
    # For Q-Q, we just need PAE for now (could add TF if available)
    fig = plot_qq([pit_pae], f'PAE Q-Q Plot ({run_name})',
                  colors=['C1'], labels=['PAE'])
    save_figure(fig, output_dir, f'{run_name}_pit_qq.png', dpi)
    
    # Compute uniformity statistics
    ks_stat = np.max(np.abs(np.sort(pit_pae) - np.linspace(0, 1, len(pit_pae))))
    print(f"  Kolmogorov-Smirnov statistic: {ks_stat:.4f}")
    print(f"  (Values < 0.05 indicate good calibration)")


def generate_summary_report(data, run_name, output_dir):
    """Generate a text summary report"""
    print("\n=== Generating Summary Report ===")
    
    (z_true, z_out_pae, sigmaz_pae, pae_samples,
     z_out_tf, dzout_tf, tf_zpdfs, tf_zpdf_fine_z, redshift_Rhat) = data
    
    stats_pae = compute_statistics(z_true, z_out_pae, sigmaz_pae)
    stats_tf = compute_statistics(z_true, z_out_tf, dzout_tf)
    
    report_path = Path(output_dir) / f'{run_name}_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write(f"Redshift Analysis Summary: {run_name}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Number of sources: {len(z_true)}\n")
        f.write(f"Redshift range: [{z_true.min():.2f}, {z_true.max():.2f}]\n\n")
        
        f.write("PAE Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Bias (median Δz/(1+z)): {stats_pae['bias']:.4f}\n")
        f.write(f"  NMAD: {stats_pae['nmad']:.4f}\n")
        f.write(f"  Outlier fraction (|Δz/(1+z)| > 0.15): {stats_pae['outlier_frac']:.3f}\n")
        if 'zscore_mean' in stats_pae:
            f.write(f"  Z-score mean: {stats_pae['zscore_mean']:.3f}\n")
            f.write(f"  Z-score std: {stats_pae['zscore_std']:.3f}\n")
        f.write(f"  Mean R-hat: {np.nanmean(redshift_Rhat):.4f}\n")
        f.write(f"  Fraction R-hat > 1.1: {np.sum(redshift_Rhat > 1.1) / len(redshift_Rhat):.3f}\n\n")
        
        f.write("Template Fitting Results:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Bias (median Δz/(1+z)): {stats_tf['bias']:.4f}\n")
        f.write(f"  NMAD: {stats_tf['nmad']:.4f}\n")
        f.write(f"  Outlier fraction (|Δz/(1+z)| > 0.15): {stats_tf['outlier_frac']:.3f}\n\n")
        
        if len(pae_samples.shape) == 4:
            n_src, n_chain, n_step, n_param = pae_samples.shape
            f.write("Sampling Configuration:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Chains per galaxy: {n_chain}\n")
            f.write(f"  Steps per chain: {n_step}\n")
            f.write(f"  Parameters: {n_param}\n")
    
    print(f"  Saved: {run_name}_summary.txt")


def main():
    """Main execution function"""
    args = parse_args()
    
    print("=" * 60)
    print("Comprehensive Redshift Analysis")
    print("=" * 60)
    print(f"Run name: {args.run_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Burn-in: {args.burn_in} steps")
    
    # Load data
    print("\n=== Loading Data ===")
    
    # Load template fitting results
    print("Loading template fitting results...")
    zout_tf, dzout_tf, dz_oneplusz_tf, chisq_tf, match_z, zpdf_tf = load_tf_results(
        load_zpdf=args.tf_load_zpdf
    )
    tf_results = (zout_tf, dzout_tf, dz_oneplusz_tf, chisq_tf, match_z, zpdf_tf)
    tf_zpdf_fine_z = np.linspace(0, 3, 1501)
    
    # Load PAE results
    print(f"Loading PAE results from:")
    print(f"  Results: {args.pae_results}")
    print(f"  Samples: {args.pae_samples}")
    
    data = prepare_data_for_plotting(
        pae_save_fpath=args.pae_results,
        pae_sample_fpath=args.pae_samples,
        tf_results=tf_results,
        tf_zpdf_fine_z=tf_zpdf_fine_z,
        nsrc=args.nsrc if args.nsrc is not None else len(match_z),
        src_idxs=None
    )
    
    if data is None:
        raise RuntimeError("Failed to load data. Check file paths.")
    
    (z_true, z_out_pae, sigmaz_pae, pae_samples,
     z_out_tf, dzout_tf, tf_zpdfs, tf_zpdf_fine_z, redshift_Rhat) = data
    
    print(f"Loaded {len(z_true)} sources")
    
    # Determine which plots to generate
    plot_all = 'all' in args.plot_groups
    
    # Generate plots
    if plot_all or 'basic' in args.plot_groups:
        plot_basic_comparison(data, args.run_name, args.output_dir, args.dpi)
    
    if plot_all or 'coverage' in args.plot_groups:
        plot_coverage_analysis(data, args.run_name, args.output_dir, args.burn_in, args.dpi)
    
    if plot_all or 'diagnostics' in args.plot_groups:
        plot_diagnostics(data, args.run_name, args.output_dir, args.dpi)
    
    if plot_all or 'pdf' in args.plot_groups:
        plot_pdf_examples(data, args.run_name, args.output_dir, n_examples=10, dpi=args.dpi)
    
    if plot_all or 'convergence' in args.plot_groups:
        plot_pit_analysis(data, args.run_name, args.output_dir, args.burn_in, args.dpi)
    
    # Generate summary report
    generate_summary_report(data, args.run_name, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"All outputs saved to: {args.output_dir}")
    print()


if __name__ == '__main__':
    main()
