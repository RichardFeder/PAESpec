"""
Compare PAE and Template Fitting Redshift Results

This script streamlines the process of loading PAE redshift results from 
redshift_job.py output files and comparing them with template fitting results.
It produces comparison plots as in paper_plots.ipynb.

Supports both single files and multiple batch files via glob patterns.
Full chains are preserved in the loaded data - burn-in trimming only happens
when computing statistics for specific plots. This allows the same loaded data
to be used for both coverage analysis (needs burn-in trimming) and PIT evaluation
(needs full chains).

Usage:
    # Single file mode:
    python compare_redshift_results.py \
        --pae_results PAE_results_1000_srcs.npz \
        --pae_samples PAE_samples_1000_srcs.npz \
        [--tf_load_zpdf] [--burn_in 1000]
    
    # Batch mode (multiple files with glob patterns):
    python compare_redshift_results.py \
        --pae_results "PAE_results_*_start*.npz" \
        --pae_samples "PAE_samples_*_start*.npz" \
        --plot_coverage --tf_load_zpdf --burn_in 1000
    
Note: The --burn_in parameter controls trimming for coverage statistics,
      but full chains remain available in the returned data for PIT/other analyses.
"""

import argparse
import numpy as np
from pathlib import Path
import sys
from glob import glob

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_proc import load_tf_results, load_combined_pae_results
from visualization.result_plotting_fns import (
    prepare_data_for_plotting, 
    compare_pae_tf_redshifts,
    plot_coverage_comparison_grid
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compare PAE and Template Fitting redshift results"
    )
    
    parser.add_argument(
        '--pae_results',
        type=str,
        required=True,
        help='Path or glob pattern for PAE redshift results .npz file(s) (e.g., "PAE_results_*_start*.npz")'
    )
    
    parser.add_argument(
        '--pae_samples',
        type=str,
        required=True,
        help='Path or glob pattern for PAE samples .npz file(s) (e.g., "PAE_samples_*_start*.npz")'
    )
    
    parser.add_argument(
        '--tf_load_zpdf',
        action='store_true',
        help='Load template fitting PDFs (required for coverage plots)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='figures/comparison/',
        help='Directory to save output figures'
    )
    
    parser.add_argument(
        '--nsrc',
        type=int,
        default=None,
        help='Number of sources to plot (default: all)'
    )
    
    parser.add_argument(
        '--src_idxs',
        type=int,
        nargs='+',
        default=None,
        help='Specific source indices to plot'
    )
    
    parser.add_argument(
        '--plot_comparison',
        action='store_true',
        default=True,
        help='Plot PAE vs TF redshift comparison (default: True)'
    )
    
    parser.add_argument(
        '--plot_coverage',
        action='store_true',
        help='Plot coverage comparison grid (requires --tf_load_zpdf)'
    )
    
    parser.add_argument(
        '--sigz_bins',
        type=float,
        nargs='+',
        default=[0.0, 0.02, 0.05, 0.1, 0.2],
        help='Fractional uncertainty bins for coverage plot'
    )
    
    parser.add_argument(
        '--burn_in',
        type=int,
        default=1000,
        help='Number of burn-in steps to exclude from chains (default: 1000)'
    )
    
    return parser.parse_args()


def load_pae_and_tf_data(pae_results_path, pae_samples_path, load_zpdf=False, 
                          nsrc=None, src_idxs=None, burn_in=1000, nz_fine=1501):
    """
    Load PAE results from redshift_job.py output and template fitting results.
    Automatically handles both single files and multiple batch files (via glob patterns).
    
    Args:
        pae_results_path: Path or glob pattern for PAE redshift results .npz file(s)
        pae_samples_path: Path or glob pattern for PAE samples .npz file(s)
        load_zpdf: Whether to load TF PDFs (needed for coverage plots)
        nsrc: Number of sources to load (None = all)
        src_idxs: Specific source indices (None = sequential from start)
        burn_in: Number of burn-in steps to exclude (default: 1000)
    
    Returns:
        Tuple of prepared data arrays ready for plotting
    """
    
    print(f"\nLoading PAE results from:")
    print(f"  Results pattern: {pae_results_path}")
    print(f"  Samples pattern: {pae_samples_path}")
    
    # Check if we're dealing with multiple files (glob pattern) or single file
    results_files = glob(pae_results_path)
    samples_files = glob(pae_samples_path)
    
    use_combined_loader = len(results_files) > 1 or len(samples_files) > 1
    
    if use_combined_loader:
        print(f"  Detected {len(results_files)} result files and {len(samples_files)} sample files")
        print("  Using combined batch loader...")
        
        # Load and combine all batch files
        pae_results, pae_samples_dict = load_combined_pae_results(
            results_pattern=pae_results_path,
            samples_pattern=pae_samples_path,
            verbose=True
        )
        
        # Save to temporary combined files for prepare_data_for_plotting
        # NOTE: Keep full chains here - burn-in will be trimmed later when computing statistics
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        temp_results = temp_dir / 'combined_results.npz'
        temp_samples = temp_dir / 'combined_samples.npz'
        
        np.savez(temp_results, **pae_results)
        np.savez(temp_samples, z_samples=pae_samples_dict['z_samples'])  # Save FULL chains
        
        pae_results_path = str(temp_results)
        pae_samples_path = str(temp_samples)
        
        print(f"  Combined {len(pae_results['source_ids'])} total sources")
        print(f"  NOTE: Full chains preserved (burn-in will be trimmed during analysis)")
    else:
        print("  Single file mode")
    
    # Load template fitting results
    print("\nLoading template fitting results...")
    zout_tf, dzout_tf, dz_oneplusz_tf, chisq_tf, match_z, zpdf_tf = load_tf_results(
        load_zpdf=load_zpdf
    )
    
    # Package TF results
    tf_results = (zout_tf, dzout_tf, dz_oneplusz_tf, chisq_tf, match_z, zpdf_tf)
    
    # Define TF PDF fine redshift grid (matching paper_plots.ipynb)
    tf_zpdf_fine_z = np.linspace(0, 3, nz_fine)
    
    # Use prepare_data_for_plotting to handle all data loading and indexing
    print(f"\nPreparing data for plotting (burn_in={burn_in} steps)...")
    data = prepare_data_for_plotting(
        pae_save_fpath=pae_results_path,
        pae_sample_fpath=pae_samples_path,
        tf_results=tf_results,
        tf_zpdf_fine_z=tf_zpdf_fine_z,
        nsrc=nsrc if nsrc is not None else len(match_z),
        src_idxs=src_idxs
    )
    
    if data is None:
        raise RuntimeError("Failed to load data. Check file paths.")
    
    (z_true, z_out_pae, sigmaz_pae, pae_samples, 
     z_out_tf, dzout_tf, tf_zpdfs, tf_zpdf_fine_z, redshift_Rhat) = data
    
    # NOTE: Keep full chains in the returned data
    # Burn-in trimming will be done per-analysis as needed (e.g., for coverage, not for PIT)
    
    print(f"\nLoaded {len(z_true)} sources")
    print(f"  PAE median redshift range: [{z_out_pae.min():.2f}, {z_out_pae.max():.2f}]")
    print(f"  TF median redshift range: [{z_out_tf.min():.2f}, {z_out_tf.max():.2f}]")
    print(f"  Mean R-hat: {np.nanmean(redshift_Rhat):.3f}")
    if len(pae_samples.shape) == 4:
        print(f"  Full chain shape: {pae_samples.shape} (sources, chains, steps, params)")
        print(f"  NOTE: Full chains preserved - burn-in={burn_in} will be applied per-analysis")
    
    return data


def plot_redshift_comparison(z_true, z_out_pae, err_low_pae, err_high_pae, 
                              z_out_tf, dzout_tf, output_dir):
    """
    Create PAE vs TF redshift comparison plot
    
    Args:
        z_true: True redshifts
        z_out_pae: PAE median redshifts
        err_low_pae: PAE lower errors
        err_high_pae: PAE upper errors
        z_out_tf: TF median redshifts
        dzout_tf: TF errors
        output_dir: Directory to save figure
    """
    
    print("\nGenerating PAE vs TF comparison plot...")
    
    # Prepare data in format expected by compare_pae_tf_redshifts
    all_med_z = [z_out_tf, z_out_pae]
    all_err_low = [dzout_tf, err_low_pae]
    all_err_high = [dzout_tf, err_high_pae]
    
    fig = compare_pae_tf_redshifts(
        all_med_z, all_err_low, all_err_high, z_true,
        ylabels=['$\\hat{z}_{TF}$', '$\\hat{z}_{PAE}$'],
        figsize=(12, 5)
    )
    
    # Save figure
    output_path = Path(output_dir) / 'pae_tf_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_path}")
    
    return fig


def plot_coverage_grid(z_true, z_out_pae, sigmaz_pae, pae_samples,
                       z_out_tf, dzout_tf, tf_zpdfs, tf_zpdf_fine_z,
                       sigz_bins, output_dir, burn_in=0):
    """
    Create coverage comparison grid plot
    
    Args:
        z_true: True redshifts
        z_out_pae: PAE median redshifts
        sigmaz_pae: PAE redshift uncertainties
        pae_samples: PAE posterior samples (full chains)
        z_out_tf: TF median redshifts  
        dzout_tf: TF uncertainties
        tf_zpdfs: TF redshift PDFs
        tf_zpdf_fine_z: TF PDF redshift grid
        sigz_bins: Fractional uncertainty bin edges
        output_dir: Directory to save figure
        burn_in: Number of burn-in steps to trim (default: 0)
    """
    
    print("\nGenerating coverage comparison grid...")
    
    # Trim burn-in if specified
    if burn_in > 0 and len(pae_samples.shape) == 4:
        print(f"  Trimming first {burn_in} steps from each chain for coverage analysis")
        pae_samples_trimmed = pae_samples[:, :, burn_in:, :]
    else:
        pae_samples_trimmed = pae_samples
    
    fig = plot_coverage_comparison_grid(
        z_true, z_out_pae, sigmaz_pae, pae_samples_trimmed,
        z_out_tf, dzout_tf, tf_zpdfs, tf_zpdf_fine_z,
        sigz_bins=np.array(sigz_bins),
        figsize=(12, 10)
    )
    
    # Save figure
    output_path = Path(output_dir) / 'coverage_comparison_grid.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved coverage grid to: {output_path}")
    
    return fig


def main():
    """Main execution function"""
    
    args = parse_args()
    
    # Load data
    data = load_pae_and_tf_data(
        args.pae_results,
        args.pae_samples,
        load_zpdf=args.tf_load_zpdf or args.plot_coverage,
        nsrc=args.nsrc,
        src_idxs=np.array(args.src_idxs) if args.src_idxs else None,
        burn_in=args.burn_in
    )
    
    (z_true, z_out_pae, sigmaz_pae, pae_samples,
     z_out_tf, dzout_tf, tf_zpdfs, tf_zpdf_fine_z, redshift_Rhat) = data
    
    # Compute PAE errors (using sigma as proxy for symmetric errors)
    err_low_pae = sigmaz_pae
    err_high_pae = sigmaz_pae
    
    # Generate requested plots
    if args.plot_comparison:
        plot_redshift_comparison(
            z_true, z_out_pae, err_low_pae, err_high_pae,
            z_out_tf, dzout_tf, args.output_dir
        )
    
    if args.plot_coverage:
        if tf_zpdfs is None:
            print("\nWarning: Coverage plot requires TF PDFs. Use --tf_load_zpdf")
        else:
            plot_coverage_grid(
                z_true, z_out_pae, sigmaz_pae, pae_samples,
                z_out_tf, dzout_tf, tf_zpdfs, tf_zpdf_fine_z,
                args.sigz_bins, args.output_dir, burn_in=args.burn_in
            )
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    main()
