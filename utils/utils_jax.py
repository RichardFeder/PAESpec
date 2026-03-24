import jax.numpy as jnp
import jax
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import uniform
from statsmodels.distributions.empirical_distribution import ECDF


def make_plotstr(nbar, NMAD, mean_dz_oneplusz, bias, outl_pct, ndigit=4):
	
	plotstr = '$\\bar{n}=$'+str(int(nbar))+' deg$^{-2}$'
	plotstr += '\nNMAD='+str(np.round(NMAD, ndigit))
	plotstr += '\n$\\tilde{\\sigma}_{z/(1+z)}=$'+str(np.round(mean_dz_oneplusz, ndigit))
	plotstr += '\nbias='+str(np.round(bias, ndigit))
	plotstr += '\n$\\eta_{3\\sigma}=$'+str(np.round(outl_pct, 1))+'%'
	return plotstr

def make_plotstr_count(N, NMAD, mean_dz_oneplusz, bias, outl_pct, outl_pct_15=None, ndigit=4):
	"""Version of make_plotstr using galaxy count N instead of number density."""
	plotstr = 'N='+str(int(N))
	plotstr += '\nNMAD='+str(np.round(NMAD, ndigit))
	plotstr += '\n$\\tilde{\\sigma}_{z/(1+z)}=$'+str(np.round(mean_dz_oneplusz, ndigit))
	plotstr += '\nbias='+str(np.round(bias, ndigit))
	plotstr += '\n$\\eta_{3\\sigma}=$'+str(np.round(outl_pct, 1))+'%'
	if outl_pct_15 is not None:
		plotstr += '\n$\\eta_{15\\%}=$'+str(np.round(outl_pct_15, 1))+'%'
	return plotstr

def format_estimates(samples, quantiles=(0.16, 0.5, 0.84)):
    q16, q50, q84 = np.percentile(samples, [100*q for q in quantiles], axis=0)
    err_lower = q50 - q16
    err_upper = q84 - q50
    return q50, err_lower, err_upper

def calculate_reduced_chi2(log_L_values_per_gal, num_params, num_data_points_per_galaxy):
    """
    Calculates reduced chi-squared.
    log_L_values_per_gal is now 1D, representing the *best* log_L for each galaxy.
    """
    chi2 = -2 * log_L_values_per_gal
    
    degrees_of_freedom = num_data_points_per_galaxy - num_params
    degrees_of_freedom = jnp.where(degrees_of_freedom <= 0, 1, degrees_of_freedom)
    
    reduced_chi2 = chi2 / degrees_of_freedom
    return reduced_chi2

def compute_pit_values_pae(z_true, all_samples_data, ae_redshift_samples=None, sample_log_amplitude=False):
    """
    Computes PIT values for Probabilistic Autoencoder (PAE) samples.

    Args:
        z_true (np.array): Array of true redshifts.
        all_samples_data (np.array): Samples from the PAE.
        ae_redshift_samples (np.array, optional): Explicit redshift samples if using redshift_in_flow.
        sample_log_amplitude (bool): If True, last dim is log(amplitude), not redshift.

    Returns:
        np.array: Array of PIT values.
    """
    pit_values = np.zeros(len(z_true))
    
    # Determine redshift index
    # If sample_log_amplitude=True: [..., z, log_A], so z is at -2
    # Otherwise: [..., z], so z is at -1
    z_idx = -2 if sample_log_amplitude else -1
    
    for i in range(len(z_true)):
        # Extract redshift samples for the current source

        if ae_redshift_samples is not None:
            zsamp_rav = ae_redshift_samples[i].ravel()
        else:
            if all_samples_data.ndim==4:
                zsamp_rav= all_samples_data[i,:,:,z_idx].ravel()
            else:
                zsamp_rav= all_samples_data[i,:,z_idx]

        # Compute the Empirical Cumulative Distribution Function (ECDF)
        ecdf = ECDF(zsamp_rav)
        # Evaluate the ECDF at the true redshift
        pit_values[i] = ecdf(z_true[i])
    return pit_values


def compute_pit_values_sbi(z_true, sbi_samples):
    """
    Computes PIT values for SBI redshift samples.
    
    Args:
        z_true (np.array): Array of true redshifts.
        sbi_samples (np.array): Array of SBI redshift samples, shape (n_sources, n_samples).
        
    Returns:
        np.array: Array of PIT values.
    """
    pit_values = np.zeros(len(z_true))
    
    for i in range(len(z_true)):
        # Extract redshift samples for the current source
        zsamp = sbi_samples[i]
        
        # Remove any NaN or infinite values
        zsamp = zsamp[np.isfinite(zsamp)]
        
        if len(zsamp) == 0:
            pit_values[i] = np.nan
            continue
            
        # Compute the Empirical Cumulative Distribution Function (ECDF)
        ecdf = ECDF(zsamp)
        # Evaluate the ECDF at the true redshift
        pit_values[i] = ecdf(z_true[i])
        
    return pit_values


def compute_empirical_coverage(pit_values, coverage_levels=[0.68, 0.95]):
    """
    Compute empirical coverage from PIT values.
    
    For well-calibrated posteriors, PIT values should be uniformly distributed.
    Coverage is computed as the fraction of PIT values within the expected range.
    
    Args:
        pit_values (np.array): Array of PIT values in [0, 1].
        coverage_levels (list): List of coverage levels to compute (e.g., [0.68, 0.95]).
        
    Returns:
        dict: Dictionary with coverage levels as keys and empirical coverage as values.
    """
    # Remove NaN values
    pit_clean = pit_values[np.isfinite(pit_values)]
    
    if len(pit_clean) == 0:
        return {level: np.nan for level in coverage_levels}
    
    coverage_results = {}
    
    for level in coverage_levels:
        # For coverage level α, PIT values should fall in [(1-α)/2, (1+α)/2]
        lower_bound = (1 - level) / 2
        upper_bound = (1 + level) / 2
        
        # Count fraction of PIT values within the expected range
        within_bounds = np.sum((pit_clean >= lower_bound) & (pit_clean <= upper_bound))
        empirical_coverage = within_bounds / len(pit_clean)
        
        coverage_results[level] = empirical_coverage
    
    return coverage_results


def compute_pit_values_tf(z_true, finez_grid, zpdf_tf_data):
    """
    Computes PIT values for Template Fitting (TF) PDFs.

    Args:
        z_true (np.array): Array of true redshifts.
        finez_grid (np.array): Redshift grid for the PDFs.
        zpdf_tf_data (np.array): Array of TF PDFs (probability densities).

    Returns:
        np.array: Array of PIT values.
    """
    pit_values = np.zeros(len(z_true))
    dz_grid = finez_grid[1] - finez_grid[0]

    for i in range(len(z_true)):

        pdf_normalized = zpdf_tf_data[i] / np.sum(zpdf_tf_data[i])
        # Compute the cumulative distribution function (CDF)
        cdf = np.cumsum(pdf_normalized)

        # Interpolate the CDF to find the value at z_true[i]
        # interp1d creates a function that can be called to interpolate
        
        cdf_interp = interp1d(finez_grid, cdf, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))
        pit_values[i] = cdf_interp(z_true[i])

    # Ensure PIT values are within [0, 1] due to interpolation edge effects
    pit_values = np.clip(pit_values, 0.0, 1.0)
    return pit_values


def compute_pit_values_pae_bias_corrected(z_true, all_samples_data, z_estimated, ae_redshift_samples=None, sample_log_amplitude=False):
    """
    Computes bias-corrected PIT values for PAE by shifting samples by mean fractional bias.
    
    The bias correction works by:
    1. Computing mean fractional bias: <(z_est - z_true)/(1+z_true)> across all sources
    2. For each source, shifting samples by -mean_frac_bias * (1+z_true) to center the posterior
    3. Computing PIT from the shifted samples
    
    This fractional approach accounts for redshift-dependent uncertainties, where
    the same absolute bias has different significance at different redshifts.
    
    Args:
        z_true (np.array): Array of true redshifts.
        all_samples_data (np.array): Samples from the PAE.
        z_estimated (np.array): Array of estimated redshifts (e.g., median).
        ae_redshift_samples (np.array, optional): Pre-extracted redshift samples.
        sample_log_amplitude (bool): If True, last dim is log(amplitude), not redshift.
    
    Returns:
        tuple: (pit_values, mean_frac_bias) - PIT values after bias correction and the mean fractional bias applied
    """
    # Compute mean fractional bias: <(z_est - z_true)/(1+z_true)>
    # Only use finite values for bias computation
    finite_mask = np.isfinite(z_estimated) & np.isfinite(z_true)
    if np.sum(finite_mask) == 0:
        print("WARNING: No finite values for bias computation, returning uncorrected PIT")
        return compute_pit_values_pae(z_true, all_samples_data, ae_redshift_samples, sample_log_amplitude), 0.0
    
    fractional_bias = (z_estimated - z_true) / (1.0 + z_true)
    mean_frac_bias = np.median(fractional_bias[finite_mask])  # Use median for robustness
    print(f"  Bias correction: mean_frac_bias = {mean_frac_bias:.4f}")
    
    # Determine redshift index
    z_idx = -2 if sample_log_amplitude else -1
    
    pit_values = np.zeros(len(z_true))
    n_invalid = 0
    
    for i in range(len(z_true)):
        # Skip sources with invalid true redshift
        if not np.isfinite(z_true[i]):
            pit_values[i] = np.nan
            n_invalid += 1
            continue
            
        # Extract redshift samples for the current source
        if ae_redshift_samples is not None:
            zsamp_rav = ae_redshift_samples[i].ravel()
        else:
            if all_samples_data.ndim == 4:
                zsamp_rav = all_samples_data[i, :, :, z_idx].ravel()
            else:
                zsamp_rav = all_samples_data[i, :, z_idx]
        
        # Filter out non-finite samples
        zsamp_rav = zsamp_rav[np.isfinite(zsamp_rav)]
        
        if len(zsamp_rav) == 0:
            pit_values[i] = np.nan
            n_invalid += 1
            continue
        
        # Shift samples by negative of fractional bias scaled by (1+z_true)
        # This converts fractional bias back to absolute bias at this redshift
        absolute_bias_correction = mean_frac_bias * (1.0 + z_true[i])
        zsamp_corrected = zsamp_rav - absolute_bias_correction
        
        # Compute ECDF with corrected samples
        ecdf = ECDF(zsamp_corrected)
        pit_val = ecdf(z_true[i])
        
        # Clip to [0, 1] in case of numerical issues
        pit_values[i] = np.clip(pit_val, 0.0, 1.0)
    
    if n_invalid > 0:
        print(f"  WARNING: {n_invalid} sources had invalid PIT values (NaN/inf in inputs)")
    
    return pit_values, mean_frac_bias


def compute_pit_values_tf_bias_corrected(z_true, finez_grid, zpdf_tf_data, z_estimated):
    """
    Computes bias-corrected PIT values for TF by shifting PDFs by mean fractional bias.
    
    The bias correction works by:
    1. Computing mean fractional bias: <(z_est - z_true)/(1+z_true)> across all sources
    2. For each source, shifting PDF by -mean_frac_bias * (1+z_true) via grid interpolation
    3. Computing PIT from the shifted PDF
    
    This fractional approach accounts for redshift-dependent uncertainties, where
    the same absolute bias has different significance at different redshifts.
    
    Args:
        z_true (np.array): Array of true redshifts.
        finez_grid (np.array): Redshift grid for the PDFs.
        zpdf_tf_data (np.array): Array of TF PDFs (probability densities).
        z_estimated (np.array): Array of estimated redshifts (e.g., peak of PDF).
    
    Returns:
        tuple: (pit_values, mean_frac_bias) - PIT values after bias correction and the mean fractional bias applied
    """
    # Compute mean fractional bias only from finite values
    finite_mask = np.isfinite(z_estimated) & np.isfinite(z_true)
    if np.sum(finite_mask) == 0:
        print("WARNING: No finite values for bias computation, returning uncorrected PIT")
        return compute_pit_values_tf(z_true, finez_grid, zpdf_tf_data), 0.0
    
    fractional_bias = (z_estimated - z_true) / (1.0 + z_true)
    mean_frac_bias = np.median(fractional_bias[finite_mask])  # Use median for robustness
    print(f"  Bias correction: mean_frac_bias = {mean_frac_bias:.4f}")
    
    pit_values = np.zeros(len(z_true))
    dz_grid = finez_grid[1] - finez_grid[0]
    n_invalid = 0
    
    for i in range(len(z_true)):
        # Skip sources with invalid true redshift
        if not np.isfinite(z_true[i]):
            pit_values[i] = np.nan
            n_invalid += 1
            continue
        
        # Check for valid PDF
        if not np.all(np.isfinite(zpdf_tf_data[i])) or np.sum(zpdf_tf_data[i]) == 0:
            pit_values[i] = np.nan
            n_invalid += 1
            continue
        
        pdf_normalized = zpdf_tf_data[i] / np.sum(zpdf_tf_data[i])
        
        # Shift the PDF by interpolating onto a shifted grid
        # Convert fractional bias to absolute bias at this redshift
        absolute_bias_correction = mean_frac_bias * (1.0 + z_true[i])
        shifted_grid = finez_grid - absolute_bias_correction
        
        # Interpolate PDF onto original grid (effectively shifts PDF)
        pdf_interp = interp1d(shifted_grid, pdf_normalized, kind='linear', 
                              bounds_error=False, fill_value=0.0)
        pdf_shifted = pdf_interp(finez_grid)
        pdf_shifted = np.maximum(pdf_shifted, 0)  # Ensure non-negative
        
        # Renormalize
        if np.sum(pdf_shifted) > 0:
            pdf_shifted /= np.sum(pdf_shifted)
        else:
            # If shift moves PDF entirely out of bounds, use original
            print(f"    WARNING: Source {i} PDF shifted out of bounds, using original")
            pdf_shifted = pdf_normalized
        
        # Compute CDF
        cdf = np.cumsum(pdf_shifted)
        
        # Handle edge cases
        if not np.isfinite(cdf[-1]) or cdf[-1] == 0:
            pit_values[i] = np.nan
            n_invalid += 1
            continue
        
        cdf_interp = interp1d(finez_grid, cdf, kind='linear', 
                             bounds_error=False, fill_value=(0.0, 1.0))
        pit_val = cdf_interp(z_true[i])
        
        # Clip to [0, 1] and check validity
        pit_values[i] = np.clip(pit_val, 0.0, 1.0)
        
        if not np.isfinite(pit_values[i]):
            n_invalid += 1
    
    if n_invalid > 0:
        print(f"  WARNING: {n_invalid} sources had invalid PIT values (NaN/inf in inputs or PDF issues)")
    
    return pit_values, mean_frac_bias
    
    pit_values = np.clip(pit_values, 0.0, 1.0)
    return pit_values, mean_bias

    
def redshift_correlations(samples, nlatent=5):
    """
    Compute Pearson correlations between redshift (column 0) and latent parameters (columns 1–5).

    Parameters
    ----------
    samples : ndarray
        Array of shape (nsamples, nlatent+1), with redshift in column 0.

    Returns
    -------
    correlations : ndarray
        Array of shape (nlatent,) with Pearson r values for redshift vs. each latent param.
    """
    z = samples[:, -1]
    return np.array([
        np.corrcoef(z, samples[:, i])[0, 1]
        for i in range(nlatent)
    ])
    
def update_metric_dict(metric_dict, **kwargs):
    for key, val in kwargs.items():
        metric_dict[key].append(val)
    return metric_dict

def linear_interp_jax(x, x_points, y_points):
    return jnp.interp(x, x_points, y_points)

def convert_to_bfloat16(x):
    if isinstance(x, jnp.ndarray) and x.dtype == jnp.float32:
        return x.astype(jnp.bfloat16)
    return x

def convert_to_float16(x):
    if isinstance(x, jnp.ndarray) and x.dtype == jnp.float32:
        return x.astype(jnp.float16)
    return x

def check_for_zero_rows(arr, axis=1):

    all_zeros_rows = jnp.all(arr == 0, axis=axis)  # Boolean array indicating rows of all zeros

    # Check if there is at least one such row
    has_zero_row = jnp.any(all_zeros_rows)
    
    print('Zero mask:', all_zeros_rows)  # [ True False  True ] -> Rows 0 and 2 are all zeros
    print('At least one row is all zeros:', has_zero_row)    # True -> At least one row is all zeros
    
    return all_zeros_rows   

def check_for_any_zero_rows(arr, axis=1):

    rows_with_zeros = jnp.any(arr == 0, axis=1)

    # Check if there is at least one such row
    num_rows_with_zeros = jnp.sum(rows_with_zeros)
    
    print(rows_with_zeros)       # [ True False False  True ]
    print(num_rows_with_zeros)   # 2  (since rows 0 and 3 contain at least one zero)

    return rows_with_zeros  


def check_param_dtypes(params):
    """Recursively check the dtype of all parameters in a TrainState."""
    leaves = jax.tree_util.tree_leaves(params)
    for i, leaf in enumerate(leaves):
        print(f"Parameter {i} dtype: {leaf.dtype}")


def batch_generator(data, vals, batch_size):
    """Yields batches of data and values."""
    num_samples = data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for i in range(0, num_samples, batch_size):
        batch_idx = indices[i:i + batch_size]
        batch_data = data[batch_idx]
        batch_vals = [v[batch_idx] for v in vals]
        yield (batch_data, *batch_vals)

def reparameterize(mu, log_sigma):
    """
    Apply the reparameterization trick to sample from the Gaussian distribution
    defined by the mean `mu` and log-variance `log_sigma`.
    """

    # Compute the standard deviation
    sigma = jnp.exp(0.5 * log_sigma)  # Since log_sigma is log(variance), we take exp(0.5 * log(sigma^2))
    
    # Sample epsilon from a standard normal distribution
    epsilon = jax.random.normal(jax.random.PRNGKey(0), mu.shape)  # Same shape as `mu`
    
    # Reparameterization: z = mu + sigma * epsilon
    z = mu + sigma * epsilon
    return z


def sample_param_dict_gen(**kwargs):

    spd = dict({})

    spd['find_MAP_first'] = False
    spd['num_steps'] = 1000
    spd['u_step_size'] = 0.1

    spd['float_redshift'] = True
    spd['redshift_fix'] = None
    spd['redshift_step_size'] = None
    spd['zmin'] = 0.0
    spd['zmax'] = 4.0

    spd['mode'] = 'MCLMC'
    spd['hmc_mode'] = 'NUTS'
    spd['nchain'] = 1
    spd['test_mode'] = False
    spd['desired_energy_variance'] = 5e-4

    # for SMC
    spd['num_particles'] = 1000
    spd['num_steps_smc'] = 20 # number of annealing steps
    
    for key, arg in kwargs.items():

        spd[key] = arg
    
    return spd

def param_dict_gen(modl_type, **kwargs):
    
    params = dict({})

    # autoencoder model
    params['modl_type'] = modl_type
    params['plot_interval'] = 5
    params['nlatent'] = 5
    params['use_accelerator'] = True
    params['conv_decoder'] = False
    
    # data type and weights
    params['srcid_dict'] = dict({'COSMOS':'Tractor_ID', 'GAMA':'uberID'})
    params['restframe'] = False
    params['train_frac'] = 0.8
    if params['restframe']:
        params['weight_soft'] = 0. # noiseless spectra don't have weights, just MSE
        # params['use_noise_weights'] = False
    else:
        params['weight_soft'] = 3e-5 # softening high-SNR observations
        # params['use_noise_weights'] = True
    
    # application of filter integration
    params['nbands'] = 102 # fiducial SPHEREx, higher for upsampled rest frame 
    params['filter_integrate'] = False
    params['nlam_interp'] = 1000
    params['min_lam'] = 0.7

    params['lam_min_rest'] = 0.1
    params['lam_max_rest'] = 5.0
    params['nlam_rest'] = 500
    
    # training autoencoder
    params['epochs'] = 200
    params['dropout'] = 0.0
    params['weight_decay'] = 1e-4 # L2 regularization on network weights
    params['lr'] = 5e-4

    # optional similarity/consistency loss controls (paper-inspired)
    params['lambda_sim'] = 0.0
    params['lambda_consistency'] = 0.0
    params['sim_k0'] = 1.0
    params['sim_k1'] = 1.0
    params['sigma_s'] = 1.0
    params['similarity_subsample_size'] = 0
    params['similarity_eps'] = 1e-8
    params['consistency_aug_scale'] = 0.1

    # Reconstruction loss scaling mode (loss-only; does not change input preprocessing)
    params['recon_scale_mode'] = 'fixed'
    params['amp_eps'] = 1e-8
    params['amp_clip_min'] = 0.0
    params['amp_clip_max'] = None

    # testing adding small levels of noise to model SEDs in order to stabilize autoencoder representation
    params['add_noise'] = False
    params['inject_noise_norm_level'] = 0.01
    
    # training flow
    params['lr_flow'] = 1e-3
    params['nepoch_flow'] = 50
    params['mean_sub_latents'] = False

    params['knots'] = 8
    params['interval'] = 4
    
    # HMC parameters
    params['step_size'] = 0.1
    params['num_steps'] = 10
    params['burn'] = 100
    params['num_workers'] = None
    params['parallel'] = False
    
    # reconstructing observed spectra from rest-frame model
    if params['restframe']:
        params['recon_apply_redshift'] = False
        params['recon_float_redshift'] = False
        params['redshift_max'] = 4.0
        params['redshift_min'] = 1e-3
        params['use_noise_weights'] = False
        
    # the convolutional network is mixed with "spender" now, so mainly using that to specify model parameters
    if modl_type in ['spender', 'conv', 'jax']:
        params['filter_sizes'] = [5, 5, 5]
        params['n_hidden_encoder'] = (128, 64, 32)
        params['n_hidden_decoder'] = (16, 64, 256)
        params['filters'] = [32, 64, 128]
           
    # largely deprecated now. beta-VAE
    if modl_type=='mlp_vae':
        params['sizes'] = [102, 70, 20]
        params['beta'] = 0.1
        params['alpha'] = 1.0
        params['lambda'] = 5.0
        
    for key, arg in kwargs.items():
        params[key] = arg
        
            
    return params


def grab_default_vmin_vmax_plot(sed_set):
    ''' For latent variable color-coded plots against galaxy properties. '''
    if sed_set=='COSMOS':
        feature_vmin_vmax = dict({'imag':[18, 25], 'redshift':[0, 2.0], 'mass_best':[8, 11.5], 'ebv':[0.0, 0.25], 'bfit_tid':[0, 160], 'F_H-alpha':[1e-17, 1e-15], \
              'F_OIIIa':[1e-17, 3e-16], 'F_Pa-alpha':[1e-17, 3e-16], 'F_OII':[1e-17, 3e-16], 'ew_ha':[10, 300], \
             'snr_phot':[10, 200], 'phot_snr':[10, 200], 'chi2':[50, 100]})
    elif sed_set=='GAMA':
        feature_vmin_vmax = dict({'imag':[13, 18], 'redshift':[0, 0.5], 'mass_best':[8, 11.5], 'ebv':[0.0, 0.25], 'bfit_tid':[0, 160], 'F_H-alpha':[1e-16, 1e-14], \
                              'F_OIIIa':[1e-17, 3e-16], 'F_Pa-alpha':[1e-17, 1e-14], 'F_OII':[1e-16, 3e-16], 'ew_ha':[10, 300], \
                             'snr_phot':[10, 1000], 'phot_snr':[10, 1000], 'chi2':[50, 100]})
    return feature_vmin_vmax


# Sample a n-dimensional Gaussian Mixture Model (GMM)
def sample_gmm_nd(key, num_samples=5000, ncode=5):

    key1, key2, key3 = jr.split(key, 3)

    means = jnp.array([
        jnp.ones(ncode) * 2,  # First component mean
        jnp.ones(ncode) * -2   # Second component mean
    ])
    
    covs = jnp.array([
        jnp.eye(ncode) * 0.5,  # First component covariance
        jnp.eye(ncode) * 1.5   # Second component covariance
    ])

    assignments = jr.randint(key1, (num_samples,), 0, 2)  # Generates {0, 1} for each sample

    # Select means and covariances based on assignments
    chosen_means = means[assignments]
    chosen_covs = covs[assignments]

    # Sample from multivariate normal with chosen means and covariances
    noise = jax.vmap(lambda m, c, k: jr.multivariate_normal(k, m, c))(
        chosen_means, chosen_covs, jr.split(key2, num_samples)
    )

    return noise