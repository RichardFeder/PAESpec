import sys
import os
# Add parent directory to path to import from reorganized packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import jax.numpy as jnp
import time
from models.pae_jax import initialize_PAE, load_spherex_data
from training.train_ae_jax import param_dict_gen
from sampling.sample_pae_batch_refactor import MCLMCSamplingConfig, sample_mclmc_wrapper
from data_proc.dataloader_jax import SPHERExData
from config import scratch_basepath


# ==================== Hyperparameters ====================
# Model configuration
filter_set_name = 'spherex_filters102/'
with_ext_phot = False
nlatent = 5
sig_level_norm = 0.01
sel_str = 'zlt22.5'
nf_alpha = 0.0

# Sampling configuration
num_steps = 2000
nsamp_init = 500
chi2_red_threshold = 1.5
gr_threshold = 1.5

# ============ REDSHIFT PRIOR CONFIGURATION ============
# Choose prior type: 0 = No prior, 1 = Gaussian, 2 = BPZ magnitude-dependent (not yet implemented)
redshift_prior_type = 1  # <-- SET THIS: 0 or 1 (BPZ type=2 coming soon)

# Gaussian prior parameters (used when redshift_prior_type = 1)
z0_prior = 0.65        # Center of Gaussian prior
sigma_prior = 0.6      # Width of Gaussian prior

# BPZ prior parameters (used when redshift_prior_type = 2)
alpha_prior = 2.0      # Power law index
beta_prior = 1.5       # Exponential cutoff parameter
m0_prior = 20.0        # Reference magnitude
kz_prior = 0.05        # Magnitude-redshift scaling
# ======================================================

# Performance optimizations
use_batched_logdensity = True  # Set to False to use old per-galaxy compilation

# Multi-core parallelization (distributes work across GPU cores)
use_multicore = False  # Set to True to use multiple GPU cores on a single node
n_devices_per_node = 4  # Number of GPU cores to use (each saturates ~600 chains or 150 sources × 4 chains)

# Data selection
fix_z = False
phot_snr_min = None
phot_snr_max = None

# File paths and naming
# run_name = 'jax_conv1_nlatent='+str(nlatent)+'_siglevelnorm='+str(sig_level_norm)+'_newAllen_all_PWnorm=1.2'
run_name='jax_conv1_nlatent='+str(nlatent)+'_siglevelnorm='+str(sig_level_norm)+'_newAllen_all_091325'

filename_flow = 'flow_model_iaf_092225'
datestr_test = '121525_batch'

# Batch processing defaults
default_ngal = 1000
default_batch_size = 100
default_start_idx = 0
# =========================================================


def run_redshifts(ngal, batch_size, start_idx=0):
    """
    Run redshift estimation on a batch of galaxies.
    
    Parameters
    ----------
    ngal : int
        Number of galaxies to process in this batch
    batch_size : int
        Batch size for processing
    start_idx : int
        Starting index for this batch (for processing data in chunks)
    """
    
    # Start timing
    t_start_total = time.time()
    
    # Generate descriptive strings for file naming
    filts_str = filter_set_name.replace('/', '').replace('_', '')

    if phot_snr_min is None:
        snrstr = 'allsnr'
    else:
        snrstr = f'snr{phot_snr_min}-{phot_snr_max}'
    
    # Generate file paths
    save_fpath = (scratch_basepath + 'data/pae_sample_results/MCLMC/zres/PAE_results_' + str(ngal) + '_srcs_' + 
                  filts_str + '_nlatent=' + str(nlatent) + '_' + snrstr + 
                  '_nfalpha=' + str(nf_alpha) + '_' + datestr_test + 
                  f'_start{start_idx}.npz')
    sample_fpath = (scratch_basepath + 'data/pae_sample_results/MCLMC/samples/PAE_samples_' + str(ngal) + '_srcs_' + 
                    filts_str + '_nlatent=' + str(nlatent) + '_' + snrstr + 
                    '_nfalpha=' + str(nf_alpha) + '_' + datestr_test + 
                    f'_start{start_idx}.npz')
    
    print(f"\n{'='*60}")
    print(f"Processing batch: {start_idx} to {start_idx + ngal}")
    print(f"Batch size: {batch_size}")
    print(f"Results will be saved to:\n  {save_fpath}")
    print(f"{'='*60}\n")
    
    # Initialize sampling configuration
    cfg = MCLMCSamplingConfig(
        num_steps=num_steps, 
        fix_z=fix_z, 
        nf_alpha=nf_alpha, 
        nsamp_init=nsamp_init,
        burn_in=500, 
        chi2_red_threshold=chi2_red_threshold, 
        gr_threshold=gr_threshold, 
        init_reinit=True,
        # Redshift prior configuration
        redshift_prior_type=redshift_prior_type,
        z0_prior=z0_prior,
        sigma_prior=sigma_prior,
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
        m0_prior=m0_prior,
        kz_prior=kz_prior,
        # Performance optimizations
        use_batched_logdensity=use_batched_logdensity,  # Single batched logdensity (recommended)
        use_multicore=use_multicore,  # Parallelize across GPU cores
        n_devices_per_node=n_devices_per_node  # Number of cores to use
    )
    
    # Print configuration for confirmation
    prior_names = {0: "No prior", 1: f"Gaussian (z0={z0_prior}, σ={sigma_prior})", 
                   2: f"BPZ magnitude-dependent (α={alpha_prior}, β={beta_prior})"}
    print(f"Redshift prior: {prior_names.get(redshift_prior_type, 'Unknown')}")
    print(f"Batched logdensity: {'Enabled' if use_batched_logdensity else 'Disabled (per-galaxy compilation)'}")
    print(f"Multi-core mode: {'Enabled' if use_multicore else 'Disabled (single core)'}")
    if use_multicore:
        print(f"  Using {n_devices_per_node} GPU cores (target: ~{150*n_devices_per_node} sources with 4 chains)")

    # Load SPHEREx data (observed and rest frame)
    t_start_data = time.time()
    dat_obs, property_cat_df_obs, property_cat_df_restframe, \
        central_wavelengths, wave_obs = load_spherex_data(
            sig_level_norm=sig_level_norm,
            sel_str=sel_str,
            abs_norm=True,
            with_ext_phot=with_ext_phot,
            load_rf_dat=False,  # Don't need rest frame data for sampling
            load_obs_dat=True,
            weight_soft=5e-4
        )
    
    # Create model parameters separately
    filters = [16, 32, 128, 256]
    n_hidden_encoder = [256, 64, 16]
    filter_sizes = [5, 5, 5, 5]
    params = param_dict_gen('jax', filter_sizes=filter_sizes, n_hidden_encoder=n_hidden_encoder,
                           lr=2e-4, filters=filters, nlatent=nlatent, epochs=100, nbands=500,
                           restframe=True, mean_sub_latents=False,
                           plot_interval=5, weight_decay=0., nepoch_flow=50)
    
    # Initialize PAE model (single load, not redundant)
    PAE_COSMOS = initialize_PAE(
        run_name, 
        filter_set_name=filter_set_name, 
        with_ext_phot=with_ext_phot,
        inference_dtype=jnp.float32,
        lam_min_rest=0.15, 
        lam_max_rest=5.0, 
        nlam_rest=500, 
        filename_flow=filename_flow
    )

    t_data_load = time.time() - t_start_data
    print(f"Data loading time: {t_data_load:.2f}s ({t_data_load/60:.2f} min)\n")

    # Prepare SPHEREx data wrapper for sampling
    spherex_dat = SPHERExData.from_prep(
        dat_obs,
        property_cat_df_obs,
        property_cat_df_restframe,  # Will be None since load_rf_dat=False
        phot_snr_min=phot_snr_min,
        phot_snr_max=phot_snr_max,
        zmin=None,
        zmax=None
    )

    # Define source indices for this batch
    src_idxs_sub = np.arange(start_idx, start_idx + ngal)
    
    # Run MCLMC sampling
    print("Starting MCLMC sampling...")
    t_start_sampling = time.time()
    
    all_samples, _, all_pm_log_L, all_pm_log_prior, all_pm_log_redshift, dchi2 = sample_mclmc_wrapper(
        PAE_COSMOS, 
        spherex_dat, 
        cfg, 
        ngal=ngal, 
        batch_size=batch_size,
        save_results=True,
        save_fpath=save_fpath,
        sample_fpath=sample_fpath,
        return_results=True, 
        do_cleanup=False, 
        src_idxs_sub=src_idxs_sub,
        property_cat_df=property_cat_df_obs
    )
    
    t_sampling = time.time() - t_start_sampling
    t_total = time.time() - t_start_total
    
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"Data loading:       {t_data_load:8.2f}s ({t_data_load/60:6.2f} min)")
    print(f"MCLMC sampling:     {t_sampling:8.2f}s ({t_sampling/60:6.2f} min)")
    print(f"Total runtime:      {t_total:8.2f}s ({t_total/60:6.2f} min)")
    print(f"{'='*60}")
    print(f"Processed {ngal} galaxies")
    print(f"Time per galaxy:    {t_sampling/ngal:8.2f}s")
    print(f"Batch size:         {batch_size}")
    print(f"Batched logdensity: {'Enabled' if use_batched_logdensity else 'Disabled'}")
    print(f"{'='*60}\n")
    
    print(f"Completed processing batch {start_idx} to {start_idx + ngal}")
    print(f"Results saved successfully!\n")
    
    return all_samples, all_pm_log_L, all_pm_log_prior, all_pm_log_redshift, dchi2


def main():
    """
    Main entry point with command-line argument parsing for batch processing.
    """
    parser = argparse.ArgumentParser(
        description='Run redshift estimation on SPHEREx data in batches',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--ngal', 
        type=int, 
        default=default_ngal,
        help='Number of galaxies to process in this batch'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=default_batch_size,
        help='Batch size for internal processing'
    )
    parser.add_argument(
        '--start-idx', 
        type=int, 
        default=default_start_idx,
        help='Starting index for this batch (useful for processing data in chunks)'
    )
    
    args = parser.parse_args()
    
    # Run redshift estimation
    run_redshifts(
        ngal=args.ngal, 
        batch_size=args.batch_size, 
        start_idx=args.start_idx
    )


if __name__ == '__main__':
    main()