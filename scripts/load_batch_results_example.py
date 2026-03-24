"""
Example script showing how to load and combine results from multiple batch jobs.

This is useful when you've run multiple SLURM array jobs with different START_IDX values,
resulting in multiple output files like:
    PAE_results_175_srcs_*_start0.npz
    PAE_results_175_srcs_*_start175.npz
    PAE_results_175_srcs_*_start350.npz
    ...
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_proc import load_combined_pae_results


def main():
    # Directory containing your batch results
    results_dir = Path("/global/homes/r/rmfeder/sed_vae/data/redshift_results")
    
    # Pattern matching all result files from different start indices
    # Example: "PAE_results_175_srcs_*_start*.npz" will match all files
    results_pattern = str(results_dir / "PAE_results_175_srcs_*_start*.npz")
    samples_pattern = str(results_dir / "PAE_samples_175_srcs_*_start*.npz")
    
    print("Loading combined PAE results...")
    print(f"Results pattern: {results_pattern}")
    print(f"Samples pattern: {samples_pattern}")
    
    # Load and combine all results
    results_dict, samples_dict = load_combined_pae_results(
        results_pattern=results_pattern,
        samples_pattern=samples_pattern,
        verbose=True
    )
    
    # Now you have combined arrays for all galaxies
    print("\n=== Combined Results ===")
    print(f"Total galaxies: {len(results_dict['source_ids'])}")
    print(f"Source IDs shape: {results_dict['source_ids'].shape}")
    print(f"MAP redshifts shape: {results_dict['z_map'].shape}")
    print(f"Full samples shape: {samples_dict['z_samples'].shape}")
    print(f"  -> {samples_dict['z_samples'].shape[0]} galaxies")
    print(f"  -> {samples_dict['z_samples'].shape[1]} chains")
    print(f"  -> {samples_dict['z_samples'].shape[2]} samples per chain (FULL chains, no burn-in removed)")
    
    # Example: Compute statistics across all galaxies
    print("\n=== Redshift Statistics ===")
    print(f"MAP redshift range: [{results_dict['z_map'].min():.3f}, {results_dict['z_map'].max():.3f}]")
    print(f"Median MAP redshift: {np.median(results_dict['z_map']):.3f}")
    
    # Example: Apply burn-in trimming ONLY when needed for specific analyses
    burn_in = 1000
    z_samples_trimmed = samples_dict['z_samples'][:, :, burn_in:]
    print(f"\n=== Coverage Analysis (with burn-in trimming) ===")
    print(f"After trimming {burn_in} burn-in samples:")
    print(f"  Samples shape: {z_samples_trimmed.shape}")
    
    # Example: Compute posterior means for each galaxy
    z_posterior_means = z_samples_trimmed.mean(axis=(1, 2))  # Mean over chains and samples
    print(f"  Posterior mean redshifts shape: {z_posterior_means.shape}")
    print(f"  Posterior mean range: [{z_posterior_means.min():.3f}, {z_posterior_means.max():.3f}]")
    
    # Example: For PIT analysis, use FULL chains (no burn-in trimming)
    print(f"\n=== PIT Analysis (using full chains) ===")
    print(f"Full chains shape: {samples_dict['z_samples'].shape}")
    print("For PIT, evaluate P(z < z_true | data) using all samples including burn-in")
    print("This ensures proper coverage of the full posterior distribution")
    
    # Example: Use with compare_redshift_results.py
    print("\n=== Next Steps ===")
    print("You can now use these combined results with comparison scripts:")
    print("\n1. Direct use with compare_redshift_results.py (recommended):")
    print("   python scripts/compare_redshift_results.py \\")
    print("       --pae_results 'PAE_results_*_start*.npz' \\")
    print("       --pae_samples 'PAE_samples_*_start*.npz' \\")
    print("       --burn_in 1000  # Applied per-analysis, not at load time")
    print("\n2. For custom analysis:")
    print("   - Full chains are preserved for PIT and other PDF-based analyses")
    print("   - Apply burn-in trimming only when computing summary statistics")
    print("   - Different analyses can use different burn-in values from same data")


if __name__ == "__main__":
    main()
