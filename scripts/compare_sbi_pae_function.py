#!/usr/bin/env python3
"""
Function to compare SBI and PAE redshift estimates across different configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

import sys
sys.path.insert(0, '/global/homes/r/rmfeder/sed_vae/sp-ae-herex/')
from diagnostics.diagnostics_jax import hpd_interval2
from utils.utils_jax import compute_pit_values_sbi, compute_pit_values_pae, compute_empirical_coverage
from data_proc.data_file_utils import load_combined_pae_results
import jax
import jax.numpy as jnp


def hpd_interval2(samples, alpha=0.68):
    """
    Compute the highest posterior density (HPD) interval of a sample.

    Parameters
    ----------
    samples : array_like
        1D array of posterior samples (assumed continuous).
    alpha : float
        Credible interval level (e.g., 0.68 for 68% HPD).

    Returns
    -------
    hpd_min, hpd_max : float
        Bounds of the HPD interval.

    Raises
    ------
    ValueError
        If samples are invalid or alpha is out of range.
    """
    if not isinstance(samples, (jnp.ndarray, np.ndarray)):
        samples = jnp.asarray(samples)

    if samples.ndim != 1:
        raise ValueError("Samples must be a 1D array.")
    
    n = samples.shape[0]

    if n == 0:
        raise ValueError("Cannot compute HPD interval for an empty sample.")
    
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1 (exclusive).")

    # Sort the samples
    samples_sorted = jnp.sort(samples)

    # Calculate the number of elements in the interval
    # Use ceil to ensure at least one element if alpha is very small and n is small,
    # but floor is generally correct for the proportion. Let's stick to floor for consistency
    # with the conceptual definition of alpha * n elements.
    interval_size = int(jnp.floor(alpha * n))

    # If interval_size is 0, it means alpha is too small for the number of samples,
    # or n is too small. In this case, it's hard to define a meaningful HPD.
    # We can return a very narrow interval or raise an error.
    # For robustness, let's ensure interval_size is at least 1 if n > 0.
    if interval_size == 0 and n > 0:
        interval_size = 1 # Cover at least one sample if possible
        # Or, alternatively, raise an error:
        # raise ValueError(f"Not enough samples ({n}) to compute a {alpha*100}% HPD interval.")

    # Calculate the widths of all possible intervals of size 'interval_size'
    # The intervals are (samples_sorted[i], samples_sorted[i + interval_size - 1])
    # The number of such intervals is n - interval_size + 1
    # The width is samples_sorted[i + interval_size - 1] - samples_sorted[i]
    
    # Correct indexing for widths:
    # widths = samples_sorted[interval_size - 1:] - samples_sorted[:n - (interval_size - 1)]
    # This can be simplified to:
    widths = samples_sorted[interval_size - 1 : n] - samples_sorted[0 : n - (interval_size - 1)]

    # Find the index of the narrowest interval
    min_idx_start = jnp.argmin(widths)

    hpd_min = samples_sorted[min_idx_start]
    hpd_max = samples_sorted[min_idx_start + interval_size - 1]

    # Add a check, though with sorted samples, hpd_min <= hpd_max should always hold.
    # If it doesn't, it implies an issue with the sorted array or a JAX specific
    # numerical instability, which is highly unlikely for sorting.
    if hpd_min > hpd_max:
        # This case should theoretically not happen with correctly sorted samples.
        # If it does, it indicates a very fundamental problem or a misunderstanding
        # of the input data or JAX's behavior.
        # For extreme robustness, you might want to log a warning or adjust.
        # For now, let's raise a clearer error.
        raise RuntimeError(f"HPD interval lower bound ({hpd_min}) is greater than upper bound ({hpd_max}). "
                           "This indicates a potential issue with input data or JAX numerical stability.")

    return hpd_min, hpd_max

def proc_sbi_results(sbi_filepath, ngal_proc=2000, nprint=500, alpha=0.68, mag_max=None, min_mag=None, sidkey='sid', return_samples=False, compute_hpd=True):

    f = h5py.File(sbi_filepath, 'r')  # Open in read-only mode

    print(f.keys())

    src_id_sbi = f[sidkey]
    post_samples = f['samples']

    if mag_max is not None or min_mag is not None:
        mags = np.array(f['magnitude'])
        mask = np.ones_like(mags, dtype=bool)

        if mag_max is not None:
            mask *= (mags < mag_max)
    
        if min_mag is not None:
            mask *= (mags > min_mag)

        whichinmag = np.where(mask)[0]
    else:
        whichinmag = np.arange(len(src_id_sbi))

    # Determine actual number of sources to process
    n_available = len(whichinmag)
    n_to_process = min(ngal_proc, n_available)
    
    print(f'Available sources meeting criteria: {n_available}')
    print(f'Processing {n_to_process} sources (requested: {ngal_proc})')
    print('src ids has length', src_id_sbi.shape)
    print('post samples has shape', post_samples.shape)

    # Initialize arrays with the actual number we'll process
    dz_sbi = np.ones(n_to_process)
    med_sbi = np.ones(n_to_process)
    
    # Extract source IDs as numpy array before closing file
    selected_src_ids = np.array(src_id_sbi[whichinmag[:n_to_process]])
    
    # Extract samples for selected sources if requested
    # Important: Convert to numpy array to avoid HDF5 dataset reference issues
    if return_samples:
        selected_samples = np.array(post_samples[whichinmag[:n_to_process]])
        print(f'Extracted samples shape: {selected_samples.shape}')
        
        # When we have samples, compute medians directly from the array (much faster)
        med_sbi = np.median(selected_samples, axis=1)
        print(f'Computed medians from samples (shape: {med_sbi.shape})')
        
        # Only compute HPD if requested
        if compute_hpd:
            print(f'Computing HPD intervals for {n_to_process} sources...')
            for x in range(n_to_process):
                if x % nprint == 0:
                    print(f'  x = {x} / {n_to_process}')
                lower_sbi, upper_sbi = hpd_interval2(selected_samples[x], alpha=alpha)
                dz_sbi[x] = 0.5 * (upper_sbi - lower_sbi)
        else:
            print('Skipping HPD computation (compute_hpd=False)')
            dz_sbi = np.zeros(n_to_process)  # Set to zero when not computed
    else:
        # When not returning samples, process one at a time from HDF5
        print(f'Processing {n_to_process} sources from HDF5...')
        for x in range(n_to_process):
            if x % nprint == 0:
                print(f'  x = {x} / {n_to_process}')
        
            if compute_hpd:
                lower_sbi, upper_sbi = hpd_interval2(post_samples[whichinmag[x]], alpha=alpha)
                dz_sbi[x] = 0.5 * (upper_sbi - lower_sbi)

            # Compute median from samples
            med_sbi[x] = np.median(np.array(post_samples[whichinmag[x]]))

    # Close HDF5 file
    f.close()

    # Return arrays with the actual processed length
    if return_samples:
        return selected_src_ids, med_sbi, dz_sbi, selected_samples
    else:
        return selected_src_ids, med_sbi, dz_sbi

def compute_pit_and_coverage_comparison(config_data, coverage_levels=[0.68, 0.95]):
    """
    Compute PIT values and empirical coverage for SBI and PAE configurations.
    
    Parameters
    ----------
    config_data : dict
        Dictionary containing SBI and PAE configuration data.
        For SBI configs, must include 'sbi_samples' key.
        For PAE configs, must include 'pae_samples' key (if available).
    coverage_levels : list
        Coverage levels to compute (e.g., [0.68, 0.95]).
        
    Returns
    -------
    dict : Dictionary with PIT values and empirical coverage for each config.
    """
    pit_coverage_results = {}
    
    for config_name, config in config_data.items():
        print(f"\nComputing PIT for {config_name}...")
        
        z_true = config['truez']
        method = config['method']
        
        if method == 'SBI':
            if 'sbi_samples' in config:
                sbi_samples = config['sbi_samples']
                pit_values = compute_pit_values_sbi(z_true, sbi_samples)
                
                # Compute empirical coverage
                coverage = compute_empirical_coverage(pit_values, coverage_levels)
                
                pit_coverage_results[config_name] = {
                    'pit_values': pit_values,
                    'empirical_coverage': coverage,
                    'method': 'SBI'
                }
                
                # Print coverage summary
                print(f"  SBI Coverage:")
                for level, emp_cov in coverage.items():
                    print(f"    {level*100:.0f}% expected → {emp_cov*100:.1f}% empirical")
                    
            else:
                print(f"  WARNING: No SBI samples available for {config_name}")
                
        elif method == 'PAE':
            if 'pae_samples' in config:
                pae_samples = config['pae_samples']
                pit_values = compute_pit_values_pae(z_true, pae_samples)
                
                # Compute empirical coverage
                coverage = compute_empirical_coverage(pit_values, coverage_levels)
                
                pit_coverage_results[config_name] = {
                    'pit_values': pit_values,
                    'empirical_coverage': coverage,
                    'method': 'PAE'
                }
                
                # Print coverage summary
                print(f"  PAE Coverage:")
                for level, emp_cov in coverage.items():
                    print(f"    {level*100:.0f}% expected → {emp_cov*100:.1f}% empirical")
                    
            else:
                print(f"  WARNING: No PAE samples available for {config_name}")
        
    return pit_coverage_results


def get_matching_indices(src_ids_sbi, srcid_catgrid):

    matching_indices = []

    for idx in src_ids_sbi:
        matches = np.where(srcid_catgrid == idx)[0]  # Get all matching indices
        if matches.size > 0:  # Check if there are any matches
            matching_indices.append(matches[0])  # Append the first match
        else:
            matching_indices.append(-1)  # No match found

    matching_indices = np.array(matching_indices)
    
    return matching_indices

def load_sbi_post_results_with_samples(scratch_base = '/pscratch/sd/r/rmfeder/data/', ngal_proc=5000, 
                          filename_zlt23train=None, return_samples=True):
    """
    Load SBI and PAE results with full posterior samples for PIT computation.
    
    Parameters
    ----------
    scratch_base : str
        Base path for data files
    ngal_proc : int
        Number of galaxies to process
    filename_zlt23train : str, optional
        Path to SBI results file
    return_samples : bool
        Whether to return full posterior samples
        
    Returns
    -------
    tuple : (zl23train_bright, zl23train_faint, pae_bright, pae_faint)
        Each dict now includes sample data if return_samples=True
    """

    # load SBI results with samples
    if filename_zlt23train is None:
        filename_zlt23train = '/pscratch/sd/r/rmfeder/data/sbi_post/posterior_samples_big_encoder.hdf5'

    if return_samples:
        print('were in here, return samples is true')
        src_ids_sbi_zl23train_bright, med_sbi_est_zl23train_bright, dz_sbi_est_zl23train_bright, sbi_samples_bright = proc_sbi_results(filename_zlt23train, mag_max=20., sidkey='galaxy_id', return_samples=True, compute_hpd=False)
        src_ids_sbi_zl23train_faint, med_sbi_est_zl23train_faint, dz_sbi_est_zl23train_faint, sbi_samples_faint = proc_sbi_results(filename_zlt23train, mag_max=22.5, ngal_proc=10000, sidkey='galaxy_id', return_samples=True, compute_hpd=False)
    else:
        src_ids_sbi_zl23train_bright, med_sbi_est_zl23train_bright, dz_sbi_est_zl23train_bright = proc_sbi_results(filename_zlt23train, mag_max=20., sidkey='galaxy_id')
        src_ids_sbi_zl23train_faint, med_sbi_est_zl23train_faint, dz_sbi_est_zl23train_faint = proc_sbi_results(filename_zlt23train, mag_max=22.5, ngal_proc=10000, sidkey='galaxy_id')

    # load template fitting results
    catgrid_file_all = np.loadtxt('/pscratch/sd/r/rmfeder/data/select_catgrid_info/catgrid_info_COSMOS_zlt23.0_5xnoiserealiz') # all z < 23
    truez_all = catgrid_file_all[:,4]
    srcid_catgrid_all = catgrid_file_all[:,0]

    # cross match with truth catalog to get true redshifts    
    matching_indices_zl23train_bright = get_matching_indices(src_ids_sbi_zl23train_bright, srcid_catgrid_all)
    matching_indices_zl23train_faint = get_matching_indices(src_ids_sbi_zl23train_faint, srcid_catgrid_all)
    
    truez_zlt23train_bright = truez_all[matching_indices_zl23train_bright]
    truez_zlt23train_faint  = truez_all[matching_indices_zl23train_faint]

    # MCLMC results - bright
    if return_samples:
        # Use load_combined_pae_results to get both results and samples

        # mock_2ktest_bpz_nfalpha=1_sbi_compare_030226

        pae_results_bright, pae_samples_bright = load_combined_pae_results(
            scratch_base+'pae_sample_results/MCLMC/batched/mock_2ktest_bpz_nfalpha=1_sbi_compare_030226/task*/PAE_results_*_mock_2ktest_bpz_nfalpha=1_sbi_compare_030226.npz',
            samples_pattern=scratch_base+'pae_sample_results/MCLMC/batched/mock_2ktest_bpz_nfalpha=1_sbi_compare_030226/task*/PAE_samples_*_mock_2ktest_bpz_nfalpha=1_sbi_compare_030226.npz'
        )
        ztrue, zest = pae_results_bright['ztrue'], pae_results_bright['z_med']
        dz = 0.5*(pae_results_bright['err_low'] + pae_results_bright['err_high'])
    else:
        save_fpath_mclmc = scratch_base+'pae_sample_results/MCLMC/batched/mock_2ktest_bpz_nfalpha=1_sbi_compare_030226/PAE_results_combined_mock_2ktest_bpz_nfalpha=1_sbi_compare_030226.npz'
        zresult = np.load(save_fpath_mclmc)
        ztrue, zest = zresult['ztrue'], zresult['z_med']
        dz = 0.5*(zresult['err_low'] + zresult['err_high'])
        
    pae_bright = dict({'src_id':src_ids_sbi_zl23train_bright, 'zest':zest, 'truez':ztrue, 'dz_est':dz})

    # MCLMC results - faint
    if return_samples:
        pae_results_faint, pae_samples_faint = load_combined_pae_results(
            scratch_base+'pae_sample_results/MCLMC/batched/mock_20ktest_logz_bpz_nfalpha=1_030226/task*/PAE_results_*_mock_20ktest_logz_bpz_nfalpha=1_030226.npz',
            samples_pattern=scratch_base+'pae_sample_results/MCLMC/batched/mock_20ktest_logz_bpz_nfalpha=1_030226/task*/PAE_samples_*_mock_20ktest_logz_bpz_nfalpha=1_030226.npz'
        )
        print('pae_results_faint keys:', [k for k in pae_results_faint.keys()])
        ztrue_faint, zest_faint = pae_results_faint['ztrue'], pae_results_faint['z_med']
        dz_faint = 0.5*(pae_results_faint['err_low'] + pae_results_faint['err_high'])
    else:
        save_fpath_mclmc_faint = scratch_base+'pae_sample_results/MCLMC/batched/mock_20ktest_logz_bpz_nfalpha=1_030226/PAE_results_combined_mock_20ktest_logz_bpz_nfalpha=1_030226.npz'
        zresult_faint = np.load(save_fpath_mclmc_faint)
        print('zresult keys:', [k for k in zresult_faint.keys()])
        ztrue_faint, zest_faint = zresult_faint['ztrue'], zresult_faint['z_med']
        dz_faint = 0.5*(zresult_faint['err_low'] + zresult_faint['err_high'])

    print('len source indices and true z :', len(matching_indices_zl23train_faint), len(ztrue), len(zest))
    pae_faint = dict({'src_id':src_ids_sbi_zl23train_faint, 'zest':zest_faint[:ngal_proc], 'truez':ztrue_faint[:ngal_proc], 'dz_est':dz_faint[:ngal_proc]})

    # SBI results 
    zl23train_bright = dict({'src_id':src_ids_sbi_zl23train_bright, 'med_sbi_est':med_sbi_est_zl23train_bright, 'dz_sbi_est':dz_sbi_est_zl23train_bright, 'truez':truez_zlt23train_bright})
    zl23train_faint = dict({'src_id':src_ids_sbi_zl23train_faint, 'med_sbi_est':med_sbi_est_zl23train_faint, 'dz_sbi_est':dz_sbi_est_zl23train_faint, 'truez':truez_zlt23train_faint})
    
    # Add samples if requested
    if return_samples:
        zl23train_bright['sbi_samples'] = sbi_samples_bright
        zl23train_faint['sbi_samples'] = sbi_samples_faint
        print(f"\nAdded SBI samples to dictionaries:")
        print(f"  Bright samples shape: {sbi_samples_bright.shape}")
        print(f"  Faint samples shape: {sbi_samples_faint.shape}")
        
        # Add PAE samples from the loaded sample files
        if pae_samples_bright is not None and 'all_samples' in pae_samples_bright:
            pae_bright['pae_samples'] = pae_samples_bright['all_samples']
            print(f"  PAE bright samples shape: {pae_samples_bright['all_samples'].shape}")
        else:
            print(f"  WARNING: PAE bright samples not loaded")
            
        if pae_samples_faint is not None and 'all_samples' in pae_samples_faint:
            pae_faint['pae_samples'] = pae_samples_faint['all_samples'][:ngal_proc]
            print(f"  PAE faint samples shape: {pae_samples_faint['all_samples'][:ngal_proc].shape}")
        else:
            print(f"  WARNING: PAE faint samples not loaded")

    return zl23train_bright, zl23train_faint, pae_bright, pae_faint

def load_sbi_post_results(scratch_base = '/pscratch/sd/r/rmfeder/data/', ngal_proc=5000, 
                          filename_zlt23train=None, compute_hpd=True):
    """
    Original function for backward compatibility.
    """
    return load_sbi_post_results_with_samples(scratch_base, ngal_proc, filename_zlt23train, return_samples=False)


def compare_sbi_pae_configs(
    config_data,
    row_configs=None,
    figsize_scatter=(10, 7),
    figsize_hist=(4, 7),
    zmin=0.0,
    zmax=3.0,
    ngal=None,
    compute_stats=True,
    text_fontsize=13,
    title_fontsize=14,
    axis_fontsize=14,
    text_pos=(0.05, 0.95),
    stats_pos=(0.05, 0.05),
    alpha=0.2,
    point_size=2,
    save_path=None,
    show_plot=True,
    colors=['b', 'C1', 'k', 'b'],
    hist_bins=30,
    hist_range=(-0.3, 0.3),
    hist_alpha=0.6
):
    """
    Create two separate comparison figures: one for scatter plots and one for histograms.
    
    Parameters
    ----------
    config_data : dict or list
        If dict: Dictionary with keys corresponding to each configuration.
        If list: List where each element is a list of config names for that row.
        Each config dict should contain:
        - 'z_true' or 'truez': array of true redshifts
        - 'z_est', 'med_sbi_est', or 'zest': array of estimated redshifts  
        - 'title': string title for the subplot
        - 'method': 'SBI' or 'PAE' for appropriate key mapping
        - 'sigma_z' (optional): array of redshift uncertainties
    row_configs : list of lists, optional
        Specifies which configs go in which row
    figsize_scatter : tuple, optional
        Figure size for scatter plot figure (width, height)
    figsize_hist : tuple, optional
        Figure size for histogram figure (width, height)
        
    Returns
    -------
    fig_scatter : matplotlib.figure.Figure
        The scatter plot figure object
    fig_hist : matplotlib.figure.Figure
        The histogram figure object
    """
    
    # Determine row configuration
    if row_configs is None:
        config_keys = list(config_data.keys())
        # Default: first 3 in row 0, rest in row 1
        row_configs = [config_keys[:3], config_keys[3:]]
    
    # Create first figure with 2x3 subplots for scatter plots
    fig_scatter, axes_scatter = plt.subplots(2, 3, figsize=figsize_scatter, sharex=True, sharey=True)
    
    # Create second figure with 2x1 subplots for histograms
    fig_hist, axes_hist = plt.subplots(2, 1, figsize=figsize_hist)
    
    # Create line for perfect agreement
    linsp = np.linspace(zmin, zmax, 100)
    
    # Storage for histogram data
    row_errors = [[], []]  # Errors for each row
    row_labels = [[], []]  # Labels for each row
    row_colors = [[], []]  # Colors for each row
    
    # Storage for histogram data
    row_errors = [[], []]  # Errors for each row
    row_labels = [[], []]  # Labels for each row
    row_colors = [[], []]  # Colors for each row
    
    # Process each row
    for row_idx, row_config_names in enumerate(row_configs):
        for col_idx, config_name in enumerate(row_config_names):
            config = config_data[config_name]
            
            # Get the appropriate axis for scatter plot
            ax = axes_scatter[row_idx, col_idx]
            
            z_true = config['truez']

            if config['method'] == 'SBI':
                z_est = config['med_sbi_est']
                sigma_z = config['sigma_z']
                    
            elif config['method'] == 'PAE':
                z_est = config['zest']
                sigma_z = config['sigma_z']
     
            else:
                raise ValueError(f"Unknown method: {config['method']}. Must be 'SBI' or 'PAE'")
            
            # Limit number of galaxies if specified
            if ngal is not None:
                n_use = min(ngal, len(z_true))
                z_true = z_true[:n_use]
                z_est = z_est[:n_use]
                if sigma_z is not None:
                    sigma_z = sigma_z[:n_use]
            else:
                n_use = len(z_true)

            sigz_oneplusz = sigma_z / (1 + z_est)

            # Create subplot
            if config['method'] == 'PAE':
                textypos = 0.8
            else:
                textypos = text_pos[1]
                
            ax.text(text_pos[0], textypos, config['title'], fontsize=title_fontsize, 
                    transform=ax.transAxes)
            
            # Compute errors for histogram
            mask = (~np.isnan(z_est))
            dz = (z_est[mask] - z_true[mask]) / (1 + z_true[mask])
            row_errors[row_idx].append(dz)
            row_labels[row_idx].append(config['title'].replace('\n', ' '))
            
            # Determine color for this config
            config_idx = list(config_data.keys()).index(config_name)
            row_colors[row_idx].append(colors[config_idx] if config_idx < len(colors) else 'k')
            
            # Compute and display statistics if requested
            if compute_stats:
                try:
                    # Use your existing functions
                    arg_bias, arg_std, bias, NMAD, cond_outl, \
                    outl_rate, outl_rate_15pct = compute_redshift_stats(
                        z_est[mask], z_true[mask], sigma_z_select=sigz_oneplusz[mask]
                    )
                    
                    # Compute median sigma_z/(1+z) if uncertainty data is available
                    if sigma_z is not None:
                        sigz_oneplusz = sigma_z / (1 + z_est)
                        med_sigz = np.nanmedian(sigz_oneplusz)
                    else:
                        med_sigz = np.nan
                        
                    # Use your existing plot string function
                    plotstr = make_plotstr_count(n_use, NMAD, med_sigz, bias, outl_rate * 100)
                    
                    ax.text(stats_pos[0], stats_pos[1], plotstr, fontsize=text_fontsize, 
                            color='k', transform=ax.transAxes, bbox=dict(
                            boxstyle='round,pad=0.4',
                            facecolor='white',
                            alpha=0.8,
                            edgecolor='gray',
                            linewidth=0.5
                        ))
                                        
                except NameError:
                    print(f"Warning: compute_redshift_stats or make_plotstr_count not found. Skipping statistics for {config_name}")
                    pass
            
            # Plot scatter
            ax.scatter(z_true, z_est, s=point_size, color=row_colors[row_idx][col_idx], alpha=alpha)

            # Set labels
            if row_idx == 1:  # Bottom row
                ax.set_xlabel('True redshift', fontsize=axis_fontsize)

            if col_idx == 0:  # First column
                ax.set_ylabel('Estimated redshift', fontsize=axis_fontsize)
                
            ax.grid(alpha=0.3)
            ax.plot(linsp, linsp, color='grey', linestyle='dashed')
            ax.set_xlim(zmin, zmax)
            ax.set_ylim(zmin, zmax)

            if col_idx == 0:
                ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 0.5, 1.0, 1.5, 2.0])
            else:
                ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0], ['' for _ in range(5)])
        
        # Hide unused scatter plot axes in the row
        for col_idx in range(len(row_config_names), 3):
            axes_scatter[row_idx, col_idx].axis('off')
        
        # Plot histogram for this row
        hist_ax = axes_hist[row_idx]
        
        for i, (errors, label, color) in enumerate(zip(row_errors[row_idx], row_labels[row_idx], row_colors[row_idx])):
            if row_idx == 0:
                hist_range_row = [-0.06, 0.06]
            else:
                hist_range_row = [-0.5, 0.5]
            hist_ax.hist(errors, bins=np.linspace(hist_range_row[0], hist_range_row[1], 40),
                        label=label, color=color, histtype='step')
        
        hist_ax.axvline(0, color='grey', linestyle='dashed', linewidth=1.5)
        hist_ax.set_xlabel('$\\Delta z / (1 + z_{\\rm true})$', fontsize=axis_fontsize)
        hist_ax.set_ylabel('Counts', fontsize=axis_fontsize)
        hist_ax.set_yticks([], [])
        if row_idx==0:
            hist_ax.legend(fontsize=text_fontsize, loc=2, bbox_to_anchor=[0.0, 1.3])
    
    # Adjust spacing for scatter plot figure
    fig_scatter.subplots_adjust(hspace=0.05, wspace=0.05)
    
    # Adjust spacing for histogram figure
    fig_hist.subplots_adjust(hspace=0.3)
    
    if show_plot:
        plt.show()
    
    return fig_scatter, fig_hist


def prepare_config_data_from_load_results(zl23train_bright, zl23train_faint, pae_bright, pae_faint):

    """
    Convert the output from load_sbi_post_results into the format expected by compare_sbi_pae_configs.
    
    Parameters
    ----------
    zl20train_bright, zl23train_bright, zl23train_faint : dict
        SBI result dictionaries from load_sbi_post_results
    pae_bright : dict
        PAE result dictionary from load_sbi_post_results
        
    Returns
    -------
    config_data : dict
        Formatted data for compare_sbi_pae_configs
    """
    
    config_data = {
        'SBI_z<23_train_bright': {
            'truez': zl23train_bright['truez'], 
            'med_sbi_est': zl23train_bright['med_sbi_est'],
            'title': 'SBI\n$z_{AB}<23$ train\n$z_{AB}<20$ test',
            'method': 'SBI',
            'sigma_z':zl23train_bright['dz_sbi_est']

        },
        'PAE_bright': {
            'truez': pae_bright['truez'],
            'zest': pae_bright['zest'],
            'title': 'PAE ($z_{AB}<20$)',
            'method': 'PAE',
            'sigma_z':pae_bright['dz_est']

        },
        'SBI_z<23_train_faint': {
            'truez': zl23train_faint['truez'],
            'med_sbi_est': zl23train_faint['med_sbi_est'], 
            'title': 'SBI\n$z_{AB}<23$ train\n$z_{AB}<22.5$ test',
            'method': 'SBI',
            'sigma_z':zl23train_faint['dz_sbi_est']

        },
        'PAE_faint': {
            'truez': pae_faint['truez'],
            'zest': pae_faint['zest'],
            'title': 'PAE ($z_{AB}<22.5$)',
            'method': 'PAE',
            'sigma_z':pae_faint['dz_est']

        }
    }
    
    return config_data
