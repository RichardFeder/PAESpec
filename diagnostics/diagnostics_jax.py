import numpy as np
import jax
import jax.numpy as jnp

import jax.random as jr

# import flax.linen as nn
# import optax
# import pickle
# from flax.training import train_state

def monte_carlo_correlation_null_distribution(num_samples, num_simulations):
    """
    Performs a Monte Carlo simulation to generate the null distribution of the 
    Pearson correlation coefficient for 6 independent variables.

    Args:
        num_samples (int): The number of data points in each simulation.
        num_simulations (int): The number of times to repeat the simulation.

    Returns:
        np.ndarray: An array of shape (5, num_simulations) containing the 
                    correlation coefficients between the first variable and 
                    each of the other five.
    """
    # Initialize an array to store the correlation coefficients.
    # We will have 5 distributions, one for each pair (P1 vs P2, P1 vs P3, etc.).
    correlations = np.zeros((5, num_simulations))

    # Perform the Monte Carlo simulation
    for i in range(num_simulations):
        # Generate 6 independent parameters (P1 to P6).
        # np.random.randn generates random samples from a standard normal distribution.
        # Since they are generated independently, they have no intrinsic correlation.
        p1 = np.random.randn(num_samples)
        p2 = np.random.randn(num_samples)
        p3 = np.random.randn(num_samples)
        p4 = np.random.randn(num_samples)
        p5 = np.random.randn(num_samples)
        p6 = np.random.randn(num_samples)
        
        # An alternative, more concise way to generate all parameters at once
        # params = np.random.randn(num_samples, 6)
        # p1 = params[:, 0]
        # p2 = params[:, 1] etc.

        # Calculate the correlation coefficient of the first parameter (p1)
        # with each of the other five.
        correlations[0, i] = np.corrcoef(p1, p2)[0, 1]
        correlations[1, i] = np.corrcoef(p1, p3)[0, 1]
        correlations[2, i] = np.corrcoef(p1, p4)[0, 1]
        correlations[3, i] = np.corrcoef(p1, p5)[0, 1]
        correlations[4, i] = np.corrcoef(p1, p6)[0, 1]

    return correlations


def proc_spec_post(PAE_obj, post_samples, x_obs, weight, burn_in=0, zmin=0.0, zmax=3.0, combine_chains=True, thin_fac=1, redshift_fix=None, \
                       verbose=False):

    _, _, nparam = post_samples.shape[0], post_samples.shape[1], post_samples.shape[2]

    samples_use = post_samples[:,burn_in::thin_fac]

    if verbose:
        print('samples use has shape', samples_use.shape)
    if combine_chains:
        samples_use = samples_use.reshape(-1, nparam)
        if verbose:
            print(samples_use.shape, 'combined')

        if redshift_fix:
            latents = samples_use
            redshifts = jnp.ones((samples_use.shape[0]))*redshift_fix
            if verbose:
                print('redshifts:', redshifts)
        else:
            latents = samples_use[:,:-1]
            redshifts = samples_use[:,-1]
        
    def recon_and_logL(latent, redshift):

        if redshift is None:
            if len(latent.shape)==1:
                u, redshift = latent[None,:-1], latent[None,-1]
            else:
                u, redshift = latent[:,:-1], latent[:,-1]
        else:
            if len(latent.shape)==1:
                u = latent[None,:]
            else:
                u = latent

        if verbose:
            print('u has shape', u.shape)
        
        recon_x, log_px_given_z, _ = PAE_obj.push_spec_marg(u, redshift, observed_flux=x_obs, weight=weight,\
                                                            marginalize_amplitude=True,\
                                                            return_rescaled_flux_and_loglike=True)  
        return recon_x, log_px_given_z
        

    push_fn = lambda u, z: recon_and_logL(u, z)
    all_recon_x, all_logL = jax.vmap(push_fn)(latents, redshifts)

    if verbose:
        print('redshifts:', redshifts)
        print(all_recon_x.shape, all_logL.shape)

    return all_recon_x, all_logL, redshifts

def monte_carlo_profile_likelihood_jax(post_burnin_samples, log_p_all, z_index=-1, 
                                       z_min=0.0, z_max=3.0, n_bins=200, return_map_latents=False):
    """
    JAX-based Monte Carlo profile likelihood computation.
    For each redshift bin, find the maximum log-likelihood.

    Parameters
    ----------
    post_burnin_samples : array, shape (n_chains, n_samples, dim)
        Post burn-in samples from MCMC.
    log_p_all : array, shape (n_chains, n_samples)
        Log-likelihoods for each sample.
    z_index : int
        Index of the redshift parameter in the samples array.
    z_min, z_max : float
        Range of z values for binning.
    n_bins : int
        Number of bins for the profile likelihood.
    return_map_latents : bool
        If True, also return the MAP (maximum a posteriori) latent parameters
        at each redshift bin. Default False.

    Returns
    -------
    z_bin_centers : array
        Midpoints of the redshift bins.
    profile_logL : array
        Maximum log-likelihood in each bin.
    map_latents : array, optional (shape: n_bins x n_latent)
        If return_map_latents=True, returns the latent parameters (excluding redshift)
        corresponding to the MAP point in each bin.
    """

    # Flatten chains into a single sample set
    samples_flat = post_burnin_samples.reshape(-1, post_burnin_samples.shape[-1])
    logp_flat = log_p_all.reshape(-1)

    # Extract z values
    z_vals = samples_flat[:, z_index]

    # Determine bin edges
    if z_min is None:
        z_min = jnp.min(z_vals)
    if z_max is None:
        z_max = jnp.max(z_vals)
    bin_edges = jnp.linspace(z_min, z_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Assign each sample to a bin (digitize equivalent in JAX)
    # jnp.digitize is not in JAX, so we simulate it:
    bin_indices = jnp.sum(z_vals[:, None] >= bin_edges[None, :], axis=1) - 1

    if not return_map_latents:
        # Original behavior: just return max logL in each bin
        def max_in_bin(i):
            mask = bin_indices == i
            # If no samples in this bin, return -inf
            return jnp.where(jnp.any(mask), jnp.max(jnp.where(mask, logp_flat, -jnp.inf)), -jnp.inf)

        # Vectorize over bins
        profile_logL = jax.vmap(max_in_bin)(jnp.arange(n_bins))
        return bin_centers, profile_logL
    
    else:
        # Also extract MAP latents (all dimensions except redshift)
        # Get number of latent dimensions (exclude redshift at z_index)
        n_dim = samples_flat.shape[-1]
        n_latent = n_dim - 1  # All dimensions except redshift
        
        # Create mask for latent indices (all except z_index)
        latent_indices = jnp.array([i for i in range(n_dim) if i != z_index])
        
        def max_in_bin_with_latents(i):
            mask = bin_indices == i
            
            # If no samples in this bin, return -inf and zeros for latents
            has_samples = jnp.any(mask)
            
            # Find the index of the maximum log-likelihood in this bin
            masked_logp = jnp.where(mask, logp_flat, -jnp.inf)
            max_logL = jnp.max(masked_logp)
            max_idx = jnp.argmax(masked_logp)
            
            # Extract the latent parameters at this MAP point
            map_sample = samples_flat[max_idx]
            map_latents = map_sample[latent_indices]
            
            return max_logL, map_latents
        
        # Vectorize over bins
        profile_logL, map_latents_all = jax.vmap(max_in_bin_with_latents)(jnp.arange(n_bins))
        
        return bin_centers, profile_logL, map_latents_all

# def monte_carlo_profile_likelihood(post_burnin_samples, log_p_all, z_index=0, 
#                                    z_min=None, z_max=None, n_bins=200):
#     """
#     Compute Monte Carlo profile likelihood from MCMC samples.

#     Parameters
#     ----------
#     post_burnin_samples : array, shape (n_chains, n_samples, dim)
#         Post burn-in samples from MCMC.
#     log_p_all : array, shape (n_chains, n_samples)
#         Log-likelihoods for each sample.
#     z_index : int
#         Index of the redshift parameter in the samples array.
#     z_min, z_max : float
#         Range of z values for binning.
#     n_bins : int
#         Number of bins for the profile likelihood.

#     Returns
#     -------
#     z_bin_centers : array
#         Midpoints of the redshift bins.
#     profile_logL : array
#         Maximum log-likelihood in each bin.
#     """

#     # Flatten chains into a single sample set
#     samples_flat = np.reshape(post_burnin_samples, (-1, post_burnin_samples.shape[-1]))
#     logp_flat = np.reshape(log_p_all, (-1,))

#     # Extract z values
#     z_vals = samples_flat[:, z_index]

#     # Determine bin edges
#     if z_min is None:
#         z_min = np.min(z_vals)
#     if z_max is None:
#         z_max = np.max(z_vals)
#     bin_edges = np.linspace(z_min, z_max, n_bins+1)
#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

#     # Compute max logL in each bin
#     profile_logL = np.full(n_bins, -np.inf)
#     bin_indices = np.digitize(z_vals, bin_edges) - 1

#     for i in range(n_bins):
#         mask = bin_indices == i
#         if np.any(mask):
#             profile_logL[i] = np.max(logp_flat[mask])

#     return bin_centers, profile_logL

def compute_rho_from_redshift_latents(sample_fpath, ngal=None, burn_in=1000, zidx=-1):

    floatzsamp = np.load(sample_fpath)
    all_samples = floatzsamp['all_samples']

    if ngal is None:
        ngal = all_samples.shape[0]

    all_redshift_corr = np.zeros((ngal, all_samples.shape[-1]-1))

    for x in range(ngal):
    
        samp_reshape = all_samples[x,:,burn_in:,:].reshape(-1, all_samples.shape[-1])
        corr_matrix = np.corrcoef(samp_reshape, rowvar=False)
        
        # The first row/column gives the correlations with redshift
        all_redshift_corr[x] = corr_matrix[zidx, :-1]
      
    return all_redshift_corr


def compute_redshift_stats(zml_select, zspec_select, sigma_z_select=None, nsig_outlier=3, outlier_pct=15):
	#statistics
	Ngal = len(zml_select)
	arg_bias = zml_select-zspec_select
	arg_std = arg_bias / (1. + zml_select)
	bias = np.median(arg_std)
	NMAD = 1.4821 * np.median( abs(arg_std-bias))
	
	if sigma_z_select is not None:
		cond_outl = (abs(arg_std) > nsig_outlier*sigma_z_select)
	else:
		cond_outl = ( abs(arg_std) > 0.01*outlier_pct)
	outl_rate = len(arg_std[cond_outl]) / float(Ngal)
	
	# Compute 15% fractional error outlier rate
	cond_outl_15pct = (abs(arg_std) > 0.15)
	outl_rate_15pct = len(arg_std[cond_outl_15pct]) / float(Ngal)
	
	return arg_bias, arg_std, bias, NMAD, cond_outl, outl_rate, outl_rate_15pct

def compute_redshift_percentiles(all_samples, percentiles):
    """
    Compute percentiles of redshift samples for each source.

    Parameters:
    -----------
    all_samples : np.ndarray
        Array of shape (num_sources, num_chains, num_steps, D), where D is the feature dimension.
        Assumes redshift is stored in the last column (-1).
    percentiles : list or array-like
        List of percentiles to compute, e.g., [16, 50, 84].

    Returns:
    --------
    percentiles_array : np.ndarray
        Array of shape (len(percentiles), num_sources), where each row corresponds to a percentile.
    """
    # Flatten chain and step dimensions
    zsamps_all = all_samples[:, :, :, -1].reshape(all_samples.shape[0], -1)

    # Compute percentiles over samples for each source
    return np.percentile(zsamps_all, percentiles, axis=1)


def hpd_interval(samples, alpha=0.68):
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
    """
    samples = jnp.sort(samples)
    n = samples.shape[0]
    interval_idx = int(jnp.floor(alpha * n))
    
    # Find the narrowest interval
    widths = samples[interval_idx:] - samples[:n - interval_idx]
    min_idx = jnp.argmin(widths)
    
    return samples[min_idx], samples[min_idx + interval_idx]

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

def compute_hdpi(zs, z_likelihood, frac=0.68):

	''' 
	This script computes the 1d highest posterior density interval from p(z). 
	The HDPI importantly assumes that the true posterior is unimodal,
	so care should be taken to identify multi-modal PDFs before using it on arbitrary PDFs.
	'''
	
	idxs_hdpi = []
	idxmax = np.argmax(z_likelihood)
	idxs_hdpi.append(idxmax)
	
	psum = z_likelihood[idxmax]
	
	idx0 = np.argmax(z_likelihood)+1
	idx1 = np.argmax(z_likelihood)-1
	
	if idx0>=len(z_likelihood):
		return None, None

	while True:
		
		if idx0==len(z_likelihood) or np.abs(idx1)==len(z_likelihood):
			# print('hit a limit, need to renormalize but passing None for now')
			return None, None

		if z_likelihood[idx0] > z_likelihood[idx1]:
			psum += z_likelihood[idx0]
			idxs_hdpi.append(idx0)
			idx0 += 1
		else:
			psum += z_likelihood[idx1]
			idxs_hdpi.append(idx1)
			idx1 -= 1
	
		if psum >= frac:
			break
		
	zs_credible = np.sort(zs[np.array(idxs_hdpi)])
	
	return zs_credible, np.array(idxs_hdpi)
    

def compute_chi2_perobj(ae_modl_state, dat_obj_result, accelerator=None):
    
    data_train = dat_obj_result.tdata_train
    
    norms = dat_obj_result.val_list[1]
    phot_snr = dat_obj_result.val_list[3]
    print('phot snr has length', len(phot_snr))
    flux_unc = dat_obj_result.flux_unc
    mean_dat = dat_obj_result.mean_dat
    
    # norm_weights = dat_obj_result.val_list[0]
        
    recon = ae_modl_state.apply(data_train) # tb rewritten
    
    data_flux_uJy = norms*(np.array(data_train))
    modl_recon_flux_uJy = norms*np.array(recon)
            
    chi2 = (data_flux_uJy-modl_recon_flux_uJy)**2/flux_unc**2    
        
    return chi2, phot_snr


def compute_mse_perobj_jax(ae_modl, dat_obj, batch_size=5000):
    
    data_train = dat_obj.data_valid  
            
    ntot = data_train.shape[0]
    nbatch = ntot//batch_size
    
    def encode(state, x):
        recon, _ = state.apply_fn({'params': state.params}, x)
        return recon

    mse_perobj, mse_vs_lam = np.zeros((ntot)), np.zeros((data_train.shape[1]))
    
    for n in range(nbatch):
        
        i0, i1 = n*batch_size, (n+1)*batch_size
        
        if i1 > ntot:
            i1 == ntot
            
        if n%10==0 or i1==ntot:
            print('new iteration!', i0, i1)

        recon = encode(ae_modl, data_train[i0:i1])
        
        delsq = (recon - data_train[i0:i1])**2

        mse_perobj[i0:i1] = np.sum(delsq, axis=1)
        mse_vs_lam += np.sum(delsq, axis=0)
        
        if i1==ntot:
            print('done iterating, move on')
            continue
            
    mse_perobj /= ntot
    mse_vs_lam /= ntot
    
    return mse_perobj, mse_vs_lam

# def compute_mse_perobj(ae_modl, dat_obj, device=None, batch_size=5000):

def cleanup_mask(all_samples, all_log_L, nlatent, chi2_red_threshold, nbands=102, gr_threshold=1.5):
    avg_chi2_red = -2*jnp.mean(all_log_L, axis=1)/(nbands-nlatent-1)
    bad_mask_chi2 = avg_chi2_red > chi2_red_threshold

    nsrc = all_samples.shape[0]

    redshift_gr_stat = np.zeros((nsrc))

    for x in range(nsrc):

        redshift_gr_stat[x] = gelman_rubin(all_samples[x,:,:,-1])

    bad_mask_Rhat = redshift_gr_stat > gr_threshold

    return bad_mask_chi2, bad_mask_Rhat
        

def calc_all_gr(all_samples, burn_in=1000):

    nparam = all_samples.shape[-1]
    nsrc = all_samples.shape[0]

    all_rhat = np.zeros((nparam, nsrc))

    for y in range(nparam):
        
        # Compute Gelman–Rubin for each source
        for x in range(nsrc):
            all_rhat[y, x] = gelman_rubin(all_samples[x, :, burn_in:, y])
    
    return all_rhat

def gelman_rubin(chains):
    """
    Compute Gelman-Rubin R-hat for given chains.
    chains: array of shape (n_chains, n_samples)
    """
    m = chains.shape[0]  # number of chains
    n = chains.shape[1]  # samples per chain

    # Mean per chain
    chain_means = np.mean(chains, axis=1)
    # Variance per chain
    chain_vars = np.var(chains, axis=1, ddof=1)

    # Between-chain variance
    B = n * np.var(chain_means, ddof=1)
    # Within-chain variance
    W = np.mean(chain_vars)

    # Estimate of marginal posterior variance
    var_hat = (n - 1) / n * W + B / n

    R_hat = np.sqrt(var_hat / W)
    return R_hat


def compute_convergence_flags(all_samples, all_log_L, nlatent, nbands=102, 
                               chi2_threshold=1.5, rhat_threshold=1.1,
                               chain_agreement_threshold=0.1,
                               n_measurements_nonzero=None):
    """
    Compute per-source convergence quality flags.
    
    Parameters:
    -----------
    all_samples : array (n_sources, n_chains, n_samples, n_params)
        MCMC samples for all sources
    all_log_L : array (n_sources, n_chains, n_samples)
        Log-likelihoods for all samples
    nlatent : int
        Number of latent dimensions
    nbands : int
        Number of photometric bands
    chi2_threshold : float
        Maximum acceptable reduced chi-squared
    rhat_threshold : float
        Maximum acceptable R-hat value
    chain_agreement_threshold : float
        Maximum acceptable normalized chain std dev
    
    Returns:
    --------
    flags : dict with arrays of shape (n_sources,)
        - 'converged': bool, True if R-hat < threshold
        - 'good_fit': bool, True if chi² < threshold  
        - 'chains_agree': bool, True if chain z estimates agree
        - 'quality_tier': int, 0=best, 1=good, 2=marginal, 3=poor
        - 'rhat_z': float, R-hat for redshift parameter
        - 'chi2_red': float, reduced chi-squared
        - 'chain_z_std': float, std dev of chain median redshifts
        - 'chain_z_std_norm': float, chain_z_std normalized by (1+z)
    """
    n_sources = all_samples.shape[0]
    
    # R-hat for redshift parameter (last dimension)
    rhat_z = np.array([gelman_rubin(all_samples[i, :, :, -1]) for i in range(n_sources)])
    
    # Reduced chi² (using mean log-likelihood)
    # Use per-source DOF when available to account for spectral incompleteness
    if n_measurements_nonzero is not None:
        dof_per_source = np.maximum(n_measurements_nonzero - nlatent - 1, 1).astype(float)
    else:
        dof_per_source = float(nbands - nlatent - 1)
    chi2_red = -2 * np.mean(all_log_L, axis=(1, 2)) / dof_per_source
    
    # Chain agreement: std of chain medians
    # Extract redshifts: exp(log_redshift) from last parameter
    chain_log_z = all_samples[:, :, :, -1]  # (n_sources, n_chains, n_samples)
    chain_z = np.exp(chain_log_z)
    
    chain_z_medians = np.median(chain_z, axis=2)  # (n_sources, n_chains)
    chain_z_std = np.std(chain_z_medians, axis=1)  # (n_sources,)
    z_median = np.median(chain_z_medians, axis=1)  # (n_sources,)
    chain_z_std_norm = chain_z_std / (1 + z_median)  # Normalized by (1+z)
    
    # Boolean flags
    converged = rhat_z < rhat_threshold
    good_fit = chi2_red < chi2_threshold
    chains_agree = chain_z_std_norm < chain_agreement_threshold
    
    # Quality tiers (0=best, 3=worst)
    quality_tier = np.zeros(n_sources, dtype=int)
    quality_tier[~converged] += 1
    quality_tier[~good_fit] += 1
    quality_tier[~chains_agree] += 1
    
    return {
        'converged': converged,
        'good_fit': good_fit,
        'chains_agree': chains_agree,
        'quality_tier': quality_tier,
        'rhat_z': rhat_z,
        'chi2_red': chi2_red,
        'chain_z_std': chain_z_std,
        'chain_z_std_norm': chain_z_std_norm
    }


def compute_photometry_flags(flux, flux_err, sigma_neg=-3.0, sigma_pos=10.0, 
                              min_err_threshold=1e-3):
    """
    Flag problematic photometry at the band level before fitting.
    
    Identifies negative flux outliers, positive flux outliers, bad errors,
    and effectively masked bands. Useful for pre-filtering data quality issues.
    
    Parameters:
    -----------
    flux : array (n_sources, n_bands)
        Observed flux values
    flux_err : array (n_sources, n_bands)
        Flux uncertainties (1-sigma)
    sigma_neg : float
        SNR threshold for negative outliers (default: -3.0)
        Flags bands with flux/err < sigma_neg
    sigma_pos : float
        SNR threshold for positive outliers (default: 10.0)
        Flags bands with flux/err > sigma_pos (unusual for typical sources)
    min_err_threshold : float
        Minimum valid error value (default: 1e-10)
    
    Returns:
    --------
    flags : dict with boolean arrays of shape (n_sources, n_bands)
        - 'negative_outlier': flux/err < sigma_neg (strong negative flux)
        - 'positive_outlier': flux/err > sigma_pos (unusually high SNR)
        - 'bad_error': err <= threshold or NaN
        - 'zero_weight': effectively masked (very large error)
        - 'any_bad': union of above flags
    n_bad_per_source : array (n_sources,)
        Number of flagged bands per source
    frac_bad_per_source : array (n_sources,)
        Fraction of bands flagged per source
    """
    n_sources, n_bands = flux.shape
    
    # Compute SNR
    snr = flux / (flux_err + 1e-30)  # Avoid division by zero
    
    # Flag different types of problems
    flags = {
        'negative_outlier': snr < sigma_neg,
        'bad_error': (flux_err <= min_err_threshold) | np.isnan(flux_err) | np.isnan(flux),
        'zero_weight': flux_err > 1e10,  # Effectively masked bands
    }
    
    # Combined "any bad" flag
    flags['any_bad'] = (
        flags['negative_outlier'] | 
        flags['bad_error'] | 
        flags['zero_weight']
    )
    
    # Per-source statistics
    n_bad_per_source = np.sum(flags['any_bad'], axis=1)
    frac_bad_per_source = n_bad_per_source / n_bands
    
    return flags, n_bad_per_source, frac_bad_per_source
    
    
def save_redshift_results(fpath, all_samples, all_log_L, all_log_prior, all_log_redshift, redshifts_true, burn_in=1000, \
                         compute_hdpi=True, zerr_pcts=[16, 84], nlatent=8, nbands=102, alpha=0.68, sample_fpath=None,\
                          sampling_mode='mclmc', redshift_in_flow=False, ae_redshifts=None, run_name=None, \
                         all_mean_logL_per_chain=None, all_max_logL=None, 
                         z_bins_mcpl=None, all_mcpl=None, z_TF=None, z_TF_err=None, 
                         compute_quality_flags=True, quality_chi2_threshold=1.5, 
                         quality_rhat_threshold=1.1, quality_chain_agreement_threshold=0.1,
                         weights=None,
                         all_tuned_L=None, all_tuned_step_size=None,
                         all_preinit_final_logL=None,
                         phot_norms=None,
                         sample_log_amplitude=False,
                         all_mcpl_map_latents=None,
                         src_ids=None,
                         data_indices=None,
                         frac_sampled_102=None,
                         minchi2_gals=None,
                         snr_quad=None,
                         init_z_per_chain=None,
                         reinit_best_chain_idx=None,
                         all_mean_logpx_per_chain=None):


    n_galaxies = all_samples.shape[0]

    z_mean, z_med, err_low, err_high, R_hat_vals = [np.zeros((n_galaxies,)) for _ in range(5)]
    
    # Initialize autocorrelation length array
    autocorr_length = np.zeros((n_galaxies,))
    
    # Initialize amplitude arrays (will remain None if not sampling amplitude)
    log_amp_median = None
    log_amp_std = None
    
    # Compute log-amplitude statistics if sampling amplitude
    if sample_log_amplitude:
        log_amp_median = np.zeros((n_galaxies,))
        log_amp_std = np.zeros((n_galaxies,))
        
        print("Computing log-amplitude statistics...")
        for x in range(n_galaxies):
            if len(all_samples.shape) == 4:
                # Shape: (n_sources, n_chains, n_steps, n_dim)
                # NOTE: log_amplitude is at index -2 (not -1!)
                # Actual storage: [u1...uN, log_A, z]
                log_amp_samples = all_samples[x, :, :, -2].ravel()
            else:
                # Shape: (n_sources, n_samples, n_dim)
                log_amp_samples = all_samples[x, :, -2]
            
            log_amp_median[x] = np.median(log_amp_samples)
            log_amp_std[x] = np.std(log_amp_samples)

    # Print diagnostic info about sample structure
    if len(all_samples.shape) == 4:
        n_dim = all_samples.shape[3]
        print(f"  Sample array shape: {all_samples.shape}")
        print(f"  Number of dimensions: {n_dim}")
        print(f"  sample_log_amplitude: {sample_log_amplitude}")
        if sample_log_amplitude:
            # NOTE: Despite initialization order being [u, z, log_A],
            # the actual storage appears to be [u, log_A, z]
            # This may be due to how MCLMC or the likelihood function reorders things
            print(f"  → log_amplitude at index -2, Redshift at index -1")
        else:
            print(f"  → Redshift at index -1")
    
    for x in range(n_galaxies):

        if ae_redshifts is not None:
            if len(all_samples.shape)==4:
                print('ae redshifts has shape', ae_redshifts.shape)
                zsamps = ae_redshifts[x].ravel()
            else:
                zsamps = ae_redshifts[x]
            print('zsamps has shape', zsamps.shape)
        else:
            if len(all_samples.shape)==4:
                # Determine redshift index based on whether we're sampling amplitude
                # NOTE: Despite expected order [u, z, log_A], actual storage is [u, log_A, z]
                z_idx = -1  # Redshift is always at the end
                zsamps = all_samples[x,:,:,z_idx].ravel() # ravel into a single array, can recover per chain from all_samples
                R_hat_vals[x] = gelman_rubin(all_samples[x, :, burn_in:, z_idx])

            else:
                z_idx = -1  # Redshift is always at the end
                zsamps = all_samples[x,:,z_idx]
                R_hat_vals[x] = np.nan  # can't compute R-hat without chains

        # Compute autocorrelation length for this source
        # Use post-burn-in samples for consistency with R-hat
        # Compute separately for each chain to avoid artificial correlation from raveling
        if len(all_samples.shape) == 4 and ae_redshifts is None:
            n_chains = all_samples.shape[1]
            chain_autocorr_lengths = []
            
            for chain_idx in range(n_chains):
                # Get post-burn-in redshift samples for this chain only
                zsamps_chain = all_samples[x, chain_idx, burn_in:, z_idx]
                
                # Compute autocorrelation function
                z_centered = zsamps_chain - zsamps_chain.mean()
                acf = np.correlate(z_centered, z_centered, mode='full')
                acf = acf[len(acf)//2:]  # Keep only positive lags
                if acf[0] == 0 or len(zsamps_chain) == 0:
                    # Chain is constant or empty (e.g., burn_in >= chain length) - skip
                    chain_autocorr_lengths.append(1)
                    continue
                acf = acf / acf[0]  # Normalize by zero-lag autocorrelation
                
                # Find first crossing below 0.5
                below_half = np.where(acf < 0.5)[0]
                if len(below_half) > 0:
                    chain_autocorr_lengths.append(below_half[0])
                else:
                    # Never crossed 0.5 - set to length of ACF
                    chain_autocorr_lengths.append(len(acf))
            
            # Take mean across chains for this source
            autocorr_length[x] = np.mean(chain_autocorr_lengths)
        else:
            # Can't compute autocorrelation for non-chain samples or ae_redshifts
            autocorr_length[x] = np.nan
        
        z_mean[x] = np.mean(zsamps)
        z_med[x] = np.median(zsamps)

        if compute_hdpi:
            if x==0:
                print('Using 68 pct HDPI')
            lower, upper = hpd_interval2(zsamps, alpha=alpha)
        else:
            if x==0:
                print('Using percentiles', zerr_pcts)
            lower, upper = [np.percentile(zsamps, pct) for pct in zerr_pcts]
            
        err_low[x] = z_mean[x] - lower
        err_high[x] = upper - z_mean[x]

    # meanerr = 0.5*(err_low+err_high)/(1+z_med)

    # Compute number of measurements with non-zero weight per source FIRST (needed for correct DOF)
    if weights is not None:
        # weights shape: (n_sources, nbands)
        # Non-zero weight means the measurement is actually used in the fit
        n_measurements_nonzero = np.sum(weights > 0, axis=1)
    else:
        n_measurements_nonzero = np.full(n_galaxies, nbands)  # Assume all bands used if weights not provided

    # Compute chi-squared metrics
    if all_mean_logL_per_chain is not None:
        print('all_log_L has shape', all_log_L.shape)
        # Determine mean log-likelihood based on all_log_L shape
        if len(all_log_L.shape) == 3:
            # Shape: (n_sources, n_chains, n_samples)
            mean_logL = np.mean(all_log_L, axis=(1, 2))
        elif len(all_log_L.shape) == 2:
            # Shape: (n_sources, n_chains) - already averaged over samples
            mean_logL = np.mean(all_log_L, axis=1)
        else:
            # Shape: (n_sources,) - single value per source
            mean_logL = all_log_L
        
        # Full chi-squared: -2 * log(L)
        chi2_full = -2.0 * mean_logL
        
        # Reduced chi-squared: chi2 / (n_effective_bands - nlatent - 1)
        # Use per-source n_measurements_nonzero so sources with spectral incompleteness
        # are not artificially assigned low chi2 from a denominator that includes
        # zero-weight bands that contributed nothing to the numerator.
        dof_per_source = np.maximum(n_measurements_nonzero - nlatent - 1, 1).astype(float)
        chi2_reduced = chi2_full / dof_per_source
        
        # For backward compatibility, keep 'chi2' as reduced chi2
        chi2 = chi2_reduced
    else:
        chi2 = None
        chi2_full = None
        chi2_reduced = None

    # Photometric chi-squared: -2 * mean(log p(x|z)), prior terms excluded.
    # Only available when batched_log_likelihood was provided during sampling.
    if all_mean_logpx_per_chain is not None:
        mean_logpx = np.mean(np.asarray(all_mean_logpx_per_chain), axis=1)  # mean over chains
        chi2_phot = -2.0 * mean_logpx
    else:
        chi2_phot = None
    
    zscore = (z_mean-redshifts_true)/np.maximum(0.5*(err_low+err_high), 1e-6)  # Guard against division by zero for degenerate chains
    
    # Compute quality flags if requested and we have multi-chain samples
    quality_flags = {}
    if compute_quality_flags and len(all_samples.shape) == 4:
        try:
            quality_flags = compute_convergence_flags(
                all_samples, 
                all_log_L, 
                nlatent,
                nbands=nbands,
                chi2_threshold=quality_chi2_threshold,
                rhat_threshold=quality_rhat_threshold,
                chain_agreement_threshold=quality_chain_agreement_threshold,
                n_measurements_nonzero=n_measurements_nonzero
            )
            
            # Print quality summary
            n_total = len(redshifts_true)
            n_tier0 = np.sum(quality_flags['quality_tier'] == 0)
            n_tier1 = np.sum(quality_flags['quality_tier'] == 1)
            n_tier2 = np.sum(quality_flags['quality_tier'] == 2)
            n_tier3 = np.sum(quality_flags['quality_tier'] == 3)
            
            print(f"\n{'='*70}")
            print("QUALITY TIER SUMMARY:")
            print(f"  Tier 0 (best):     {n_tier0:6d} ({100*n_tier0/n_total:5.1f}%)")
            print(f"  Tier 1 (good):     {n_tier1:6d} ({100*n_tier1/n_total:5.1f}%)")
            print(f"  Tier 2 (marginal): {n_tier2:6d} ({100*n_tier2/n_total:5.1f}%)")
            print(f"  Tier 3 (poor):     {n_tier3:6d} ({100*n_tier3/n_total:5.1f}%)")
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"Warning: Could not compute quality flags: {e}")
            quality_flags = {}

    print('Saving redshift results to ', fpath)
    save_dict = dict(
        ztrue=redshifts_true,
        z_mean=z_mean,
        z_med=z_med,
        err_low=err_low,
        err_high=err_high,
        autocorr_length=autocorr_length,  # Autocorrelation lengths computed during sampling
        chi2=chi2,  # Reduced chi2 (backward compatibility)
        chi2_reduced=chi2_reduced,  # Explicit reduced chi2
        chi2_full=chi2_full,  # Full chi2 = -2*log(L), includes prior terms
        chi2_phot=chi2_phot,  # Photometric chi2 = -2*mean(log p(x|z)), prior excluded (None if not computed)
        n_measurements_nonzero=n_measurements_nonzero,  # Number of bands with weight > 0
        phot_norms=phot_norms,  # Normalization factors applied to each source
        all_log_L=all_log_L,
        all_log_prior=all_log_prior,
        all_log_redshift=all_log_redshift,
        zscore=zscore,
        R_hat=R_hat_vals,
        sampling_mode=sampling_mode, 
        run_name=run_name, 
        all_mean_logL_per_chain=all_mean_logL_per_chain, 
        all_max_logL=all_max_logL,
        src_id=src_ids,  # Absolute source IDs from original catalog
        data_idx=data_indices,  # Indices into the data arrays (for re-loading spectra)
        z_TF=z_TF,
        z_TF_err=z_TF_err,
        frac_sampled_102=frac_sampled_102,  # Fraction of 102 bands sampled (spectral completeness)
        minchi2_gals=minchi2_gals,  # Template fitting chi2 from property catalog
        snr_quad=snr_quad,           # Broadband quadrature SNR: sqrt(sum((F/sigma)^2))
        # MCLMC tuning parameters (for diagnostics)
        tuned_L=all_tuned_L,
        tuned_step_size=all_tuned_step_size,
        # Pre-reinitialization diagnostics (if init_reinit=True)
        # Array shape: (n_sources, nchain_per_gal) - final log-prob of each chain before reinit
        preinit_final_logL=all_preinit_final_logL,
        # Initial redshift draw per chain before any sampling, shape (n_sources, nchain_per_gal)
        # Already converted from ln(z) to z when sample_log_redshift=True
        init_z_per_chain=init_z_per_chain,
        # Index (0-based) of the chain selected as best during reinitialization, shape (n_sources,)
        # Value is -1 if init_reinit=False (no reinitialization performed)
        reinit_best_chain_idx=reinit_best_chain_idx,
        # Quality flags
        **quality_flags,
        # Metadata
        quality_flags_computed=compute_quality_flags and len(quality_flags) > 0,
        quality_thresholds={
            'chi2': quality_chi2_threshold,
            'rhat': quality_rhat_threshold,
            'chain_agreement': quality_chain_agreement_threshold
        } if compute_quality_flags else None
    )
    
    # Add log-amplitude statistics if they were computed
    if log_amp_median is not None:
        save_dict['log_amp_median'] = log_amp_median
        save_dict['log_amp_std'] = log_amp_std
        print(f"  Saving log-amplitude statistics: median range [{np.min(log_amp_median):.3f}, {np.max(log_amp_median):.3f}]")
    
    np.savez(fpath, **save_dict)

    if sample_fpath is not None:
        print('Saving samples to ', sample_fpath)
        print('all mcpl has shape', all_mcpl.shape)
        save_dict_samples = dict(
            ztrue=redshifts_true, 
            all_samples=all_samples, 
            sampling_mode=sampling_mode,
            ae_redshifts=ae_redshifts, 
            z_bins_mcpl=z_bins_mcpl, 
            all_mcpl=all_mcpl, 
            z_TF=z_TF, 
            z_TF_err=z_TF_err,
            sample_log_amplitude=sample_log_amplitude
        )
        
        # Add MCPL MAP latents only if they should be saved
        # (They're always computed to avoid JIT issues, but only saved if requested)
        if all_mcpl_map_latents is not None:
            save_dict_samples['all_mcpl_map_latents'] = all_mcpl_map_latents
            print(f'  Saving MCPL MAP latents with shape {all_mcpl_map_latents.shape}')
        
        np.savez(sample_fpath, **save_dict_samples)

    return fpath


def effective_sample_size(samples):
    """
    Estimate the effective sample size (ESS) using autocorrelation.
    """
    n = samples.shape[0]
    mean = jnp.mean(samples)
    var = jnp.var(samples)

    def autocorr(k):
        return jnp.mean((samples[:n - k] - mean) * (samples[k:] - mean)) / var

    max_lag = min(1000, n // 2)
    ac = jnp.array([autocorr(k) for k in range(1, max_lag)])
    ac_sum = jnp.cumsum(ac)
    positive_ac = jnp.where(ac > 0, ac_sum, 0)
    tau = 1 + 2 * jnp.sum(positive_ac > 0)
    return n / tau

def quick_ess(samples):
    """
    Fast effective sample size (ESS) estimate using autocorrelation at lag-1 only.
    Assumes `samples` is a 1D array.
    """
    n = samples.shape[0]
    x = samples - jnp.mean(samples)
    acf1 = jnp.corrcoef(x[:-1], x[1:])[0, 1]
    ess = n / (1 + 2 * acf1)
    return jnp.clip(ess, 1, n)

def posterior_skewness(samples):
    median = jnp.median(samples)
    lo, hi = hpd_interval(samples, 0.68)
    return (hi - median) - (median - lo)

def count_peaks(samples, bins=50):
    hist, edges = jnp.histogram(samples, bins=bins)
    peaks = jnp.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))
    return peaks

def mean_outside_hpd(samples, alpha=0.68):
    lo, hi = hpd_interval(samples, alpha)
    mean = jnp.mean(samples)
    return (mean < lo) | (mean > hi)

def check_posterior_quality(samples):
    ess = quick_ess(samples)
    skew = posterior_skewness(samples)
    peaks = count_peaks(samples)
    mean_outside = mean_outside_hpd(samples)

    return ess, skew, peaks, mean_outside

def grab_encoded_vars_dataset(state, dat_obj, property_cat_df, save=False, rundir=None, \
                             sed_set='COSMOS', device=None, batch_size=2048, check_device=False):
    
    ''' 
    Pass the dataset through the encoder to get latents. 
    
    state : Autoencoder model train state
    dat_obj_results : spec_data class
    property_cat_df : Pandas DataFrame with redshifts/stellar masses/etc. for sample
    rundir : Name of the run used to save the model results. Used if saving the latents
    sed_set : either COSMOS (18 < i_AB < 25 sample, 166k total galaxies) or GAMA (i < 18, 44k galaxies over 217 deg2)
    
    '''
    redshift = np.array(property_cat_df['redshift'])
    
    if sed_set=='COSMOS':
        idkey = 'Tractor_ID'
    elif sed_set=='GAMA':
        idkey = 'uberID'


    ncode = state.apply_fn.__self__.n_latent
    all_z_train, all_z_valid = [np.zeros((len(redshift), ncode)) for x in range(2)]                          
    counter = 0
    
    data_train = dat_obj.data_train
    data_valid = dat_obj.data_valid

    all_zeros_rows = jnp.all(data_train == 0, axis=1)  # Boolean array indicating rows of all zeros
    # Check if there is at least one such row
    has_zero_row = jnp.any(all_zeros_rows)
    
    print(all_zeros_rows)  # [ True False  True ] -> Rows 0 and 2 are all zeros
    print(has_zero_row)    # True -> At least one row is all zeros

    # older from torch implementation
    # if check_device:
    #     device_dat = data.device()
    #     is_cpu = 'cpu' in device_dat.device_kind
    #     is_gpu = 'gpu' in device_dat.device_kind

    #     if is_cpu:
    #         print('Moving training/validation data to GPU..')
    #         data_train = jax.put_device(data_train, jax.devices('gpu')[0])
    #         data_valid = jax.put_device(data_valid, jax.devices('gpu')[0])

    print('data train has shape', data_train.shape)
    print('data valid has shape', data_valid.shape)
    
    ntot_train, ntot_valid = data_train.shape[0], data_valid.shape[0]
    nbatch_train, nbatch_valid = ntot_train//batch_size, ntot_valid//batch_size

    all_z_train = np.zeros((ntot_train, ncode))
    all_z_valid = np.zeros((ntot_valid, ncode))

    all_redshift_train, all_redshift_valid = redshift[dat_obj.trainidx], redshift[dat_obj.valididx]
    # , all_redshift_valid = np.zeros(ntot_train), np.zeros(ntot_valid)
    
    # all_latent_z = np.zeros((ntot, ncode))

    def encode(state, x):
        _, z = state.apply_fn({'params': state.params}, x)
        return z
    
    for n in range(nbatch_train):
        i0, i1 = n*batch_size, (n+1)*batch_size
        if i1 > ntot_train:
            i1 == ntot_train
        if n%10==0 or i1==ntot_train:
            print('new iteration!', i0, i1)
        all_z_train[i0:i1,:] = encode(state, data_train[i0:i1,:])
        if i1==ntot_train:
            print(i0, i1)
            print('done iterating, move on')
            continue

    for n in range(nbatch_valid):
        i0, i1 = n*batch_size, (n+1)*batch_size
        if i1 > ntot_valid:
            i1 == ntot_valid
        if n%10==0 or i1==ntot_valid:
            print('new iteration!', i0, i1)
        all_z_valid[i0:i1,:] = encode(state, data_valid[i0:i1,:])
        if i1==ntot_valid:
            print(i0, i1)
            print('done iterating, move on')
            continue
                
    if save:
        save_fpath = rundir+'/latents/latents.npz'
        print("Saving latents to ", save_fpath)
        save_dict = dict(
            src_ID=np.array(property_cat_df[idkey]),
            all_z_train=all_z_train,
            all_z_valid=all_z_valid,
            all_redshift_train=all_redshift_train,
            all_redshift_valid=all_redshift_valid,
            trainidx=dat_obj.trainidx,
            valididx=dat_obj.valididx,
        )
        # Save additional source properties for downstream cross-matching
        for col in ['redshift', 'ebv', 'bfit_tid']:
            if col in property_cat_df.columns:
                save_dict[f'{col}_train'] = np.array(property_cat_df[col])[dat_obj.trainidx]
                save_dict[f'{col}_valid'] = np.array(property_cat_df[col])[dat_obj.valididx]
        np.savez(save_fpath, **save_dict)
    
    return all_z_train, all_z_valid, ncode



def plot_latent_z_params(ae_modl, dat_obj_result, property_cat_df, all_mu=None, features_plot=None, figsize=(8, 8), \
                        color='k', alph=0.1, s=5, xlim=[-4, 4], ylim=[-4, 4], hist_bins=np.linspace(-5, 5, 30), \
                        yticks = [-2.5, 0.0, 2.5], save_figures=False, feature_cut=None, feature_min=None, feature_max=None, \
                        feature_vmin_vmax=None, sed_set='COSMOS', device=None):

    if feature_vmin_vmax is None:
        feature_vmin_vmax = dict({'imag':[18, 25], 'redshift':[0, 2.0], 'mass_best':[8, 11.5], 'ebv':[0.0, 0.25], 'bfit_tid':[0, 160], 'F_H-alpha':[1e-17, 1e-15], \
                                  'F_OIIIa':[1e-17, 3e-16], 'F_Pa-alpha':[1e-17, 3e-16], 'F_OII':[1e-17, 3e-16], 'ew_ha':[10, 300], \
                                 'snr_phot':[10, 200], 'phot_snr':[10, 200], 'chi2':[50, 100]})

    feature_logscale_bools = dict({'imag':False, 'redshift':False, 'mass_best':False, 'ebv':False, 'bfit_tid':False, 'F_H-alpha':True, \
                              'F_OIIIa':True, 'F_Pa-alpha':True, 'F_OII':True, 'ew_ha':True, 'phot_snr':True, 'chi2':True})
    
    feature_sel = np.ones_like(np.array(property_cat_df['redshift']))
    
    hist_bins = np.linspace(xlim[0], xlim[1], 30)
    
    yticks = [xlim[0]//2, 0., xlim[1]//2]

    redshift = np.array(property_cat_df['redshift'])
    
    if all_mu is None:
        all_mu, ncode = grab_encoded_vars_dataset(ae_modl, dat_obj_result, property_cat_df, sed_set=sed_set, device=device)
    else:
        ncode = all_mu.shape[1]
        
    if feature_cut is not None:
        feature_vals_cut = np.array(property_cat_df[feature_cut])
    
        if feature_min is not None:
            feature_sel *= (feature_vals_cut > feature_min)
        if feature_max is not None:
            feature_sel *= (feature_vals_cut < feature_max)
        print('After cut, '+str(np.sum(feature_sel))+' sources remain..')
        all_mu = all_mu[np.where(feature_sel)[0],:]
                                                
    figs = []
    
    for f, feature in enumerate(features_plot):
        
        feature_vals = np.array(property_cat_df[feature])[np.where(feature_sel)[0]]
        
        if len(feature_vals)>all_mu.shape[0]:
            feature_vals = feature_vals[:all_mu.shape[0]]
        
        feature_logscale = feature_logscale_bools[feature]
        vmin, vmax = feature_vmin_vmax[feature][0], feature_vmin_vmax[feature][1]
        
        fig = make_color_corner_plot(ncode, all_mu, feature_vals, feature, feature_logscale=feature_logscale_bools[feature], \
                                    vmin=vmin, vmax=vmax, figsize=figsize, hist_bins=hist_bins, color=color, \
                                    xlim=xlim, ylim=ylim, yticks=yticks, alph=alph, s=s)
    
        figs.append(fig)
        
    return figs


def ae_result_fig_wrapper(ae_modl, dat_obj, params, property_cat_df, rundir, features_plot=None, save=True, return_figs=False, \
                         dpi=300, pdf_or_png='png', plot_params=None, feature_vmin_vmax=None, sed_set='COSMOS', device=None, \
                          all_mu=None, ngal_plot=20000):
    
    if plot_params is None:
        plot_params = dict({'xlim':[-4, 4], 'figsize':(10, 8), 'alph':0.5, 's':2, 'feature_min':15, 'feature_cut':'phot_snr', 'nsamp_flow_plots':20000})
    
    if return_figs:
        all_figs = []
    
    # Plot training/validation loss curves if metrics file exists
    metrics_file = rundir + '/metrics.npz'
    if os.path.exists(metrics_file):
        from visualization.result_plotting_fns import plot_train_validation_loss_jax
        try:
            loss_fig = plot_train_validation_loss_jax(metrics_file=metrics_file, show=False, return_fig=True)
            if return_figs:
                all_figs.append(loss_fig)
            if save:
                loss_fig.savefig(rundir + '/figures/train_valid_loss_vs_epoch.' + pdf_or_png, bbox_inches='tight', dpi=dpi)
                print(f'Saved training/validation loss plot to {rundir}/figures/train_valid_loss_vs_epoch.{pdf_or_png}')
        except Exception as e:
            print(f'Warning: Could not generate loss plot: {e}')
        
    central_wavelengths = dat_obj.sed_um_wave

    if not params['restframe']:
        chi2_all, phot_snr = compute_chi2_perobj(ae_modl, dat_obj)

        figs = plot_chi2_stats(chi2_all, phot_snr, ae_modl.encoder.n_latent, central_wavelengths=central_wavelengths)

        if return_figs:
            all_figs.extend(figs)

        if save:
            figs[0].savefig(rundir+'/figures/chi2_hist.'+pdf_or_png, bbox_inches='tight')
            figs[1].savefig(rundir+'/figures/chi2_vs_source_snr_norm_renorm.'+pdf_or_png, bbox_inches='tight')
            figs[2].savefig(rundir+'/figures/chi2_vs_snr_vs_lam.'+pdf_or_png, bbox_inches='tight')

    else:
        
        mse_perobj, mse_vs_lam = compute_mse_perobj(ae_modl, dat_obj, device=device)
        fig1, fig2 = plot_mse_stats(mse_perobj, mse_vs_lam, wav=dat_obj.sed_um_wave)
        if return_figs:
            all_figs.extend([fig1, fig2])
        
        if save:
            print('saving..')
            fig1.savefig(rundir+'/figures/mse_hist_perobject.'+pdf_or_png, bbox_inches='tight')
            fig2.savefig(rundir+'/figures/mse_vs_lam_averaged.'+pdf_or_png, bbox_inches='tight')

    if features_plot is None:
        features_plot = ['redshift', 'ebv', 'mass_best', 'bfit_tid']
        if not params['restframe']:
            features_plot.append('phot_snr')

    if feature_vmin_vmax is None:

        feature_vmin_vmax = grab_default_vmin_vmax_plot(sed_set)

        if all_mu is not None:
            all_mu_use = all_mu[:ngal_plot,:]
        else:
            all_mu_use = None
        
        
        latent_variable_figs = plot_latent_z_params2(ae_modl, dat_obj, property_cat_df, \
                          all_mu=all_mu_use, features_plot=features_plot, feature_vmin_vmax=feature_vmin_vmax, sed_set=sed_set, device=device)

        if return_figs:
            all_figs.extend(latent_variable_figs)
        if save:
            for fidx, f in enumerate(latent_variable_figs):
                f.savefig(rundir+'/figures/corner_latent_z/corner_plot_latent_z_colored_by_'+str(features_plot[fidx])+'.'+pdf_or_png, bbox_inches='tight', dpi=dpi)

    if return_figs:
        return all_figs
    
    
def nf_result_fig_wrapper(NDE_theta, ae_latents, rundir=None, save=False, return_figs=False, dpi=200, pdf_or_png='png', \
                         nsamp_flow_plots=10000):

    if return_figs:
        all_figs = []
    
    # Plot training/validation loss curves if metrics file exists
    if rundir is not None:
        metrics_file = rundir + '/flow_metrics.npz'
        if os.path.exists(metrics_file):
            from visualization.result_plotting_fns import plot_train_validation_loss_jax
            try:
                loss_fig = plot_train_validation_loss_jax(metrics_file=metrics_file, show=False, return_fig=True)
                if return_figs:
                    all_figs.append(loss_fig)
                if save:
                    loss_fig.savefig(rundir + '/figures/flow_train_valid_loss_vs_epoch.' + pdf_or_png, bbox_inches='tight', dpi=dpi)
                    print(f'Saved flow training/validation loss plot to {rundir}/figures/flow_train_valid_loss_vs_epoch.{pdf_or_png}')
            except Exception as e:
                print(f'Warning: Could not generate flow loss plot: {e}')
    
    fig = compare_flow_to_latentz(nsamp_flow_plots, NDE_theta, ae_latents, alph=0.02)
    # fig2 = plot_udist_train_validation(nf_latents_train, nf_latents_valid=nf_latents_valid)

    if save and rundir is not None:
        fig.savefig(rundir+'/figures/flow_vs_latents.'+pdf_or_png, bbox_inches='tight', dpi=dpi)
        # fig2.savefig(rundir+'/figures/nf_latents.'+pdf_or_png, bbox_inches='tight', dpi=dpi)

    if return_figs:
        all_figs.append(fig)
        return all_figs if len(all_figs) > 1 else fig
    
    return None


def compare_flow_to_latentz(ncode, nsamp, flow, latents_train, xlim=[-6, 6], yticks=[0], colors=['k', 'C3'], alph=0.05, figsize=(8,8), 
                            labels=['Latents', 'Flow samples'], legend_fs=14, bbox_to_anchor=[0.95, 0.8]):


    from visualization.result_plotting_fns import make_color_corner_plot
    key = jr.key(42)

    key, subkey = jr.split(key)
    latent_z_samp = flow.sample(subkey, (nsamp,))

    latents = np.array(latents_train)
    latent_z_samp = np.array(latent_z_samp)
    
    hist_bins = np.linspace(xlim[0], xlim[1], 30)
    
    fig = make_color_corner_plot(ncode, [latents, latent_z_samp], None, None, xlim=xlim, ylim=xlim, yticks=yticks, color=colors, alph=alph, hist_bins=hist_bins, \
                                figsize=figsize, use_contour=False, labels=labels, bbox_to_anchor=bbox_to_anchor, legend_fs=legend_fs)

    
    return fig
    