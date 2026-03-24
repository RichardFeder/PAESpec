import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import json
from dataclasses import dataclass, replace

# import blackjax
# import jaxlib
# from flax.training import train_state
# from jax import jit
# from jax import grad
# import config
# import time


# Specific imports to avoid loading unnecessary modules
from .mclmc import run_mclmc_simp, run_mclmc_simp_with_pretune
from inference.like_prior import logdensity_fn_fixz, make_logdensity_fn_optzprior, make_batched_logdensity_fn, make_batched_loglikelihood_fn
from diagnostics.diagnostics_jax import cleanup_mask, monte_carlo_profile_likelihood_jax, save_redshift_results
from data_proc.dataloader_jax import SPHERExData
# Note: models.pae_jax is not imported here - PAE_obj is passed as parameter


@dataclass
class MCLMCSamplingConfig:
    num_steps: int = 2000
    burn_in: int = 1000
    nsamp_init: int = 200
    nchain_per_gal: int = 4
    pretuned_L: float = 0.5
    pretuned_step_size: float = 0.1
    desired_energy_variance: float = 5e-3
    reinit_scatter: float = 1e-3
    fix_z: bool = False
    nf_alpha: float = 1.0
    # If >= 0, use this nf_alpha during burn-in/init phase only; final sampling uses nf_alpha.
    # Set nf_alpha_burnin=1.0 and nf_alpha=0.0 to guide chains with the flow prior
    # during burn-in, then drop it for the final posterior run.
    # Default -1.0 means disabled (burn-in and final use the same nf_alpha).
    nf_alpha_burnin: float = -1.0
    zmin: float = 0.0
    zmax: float = 3.0
    init_reinit: bool = True
    redshift_in_flow: bool = False
    chi2_red_threshold: float = 50.0
    rescale_nchain: int = 2
    gr_threshold: float = 1.5
    # BPZ magnitude dependent redshift prior
    
    include_gaussian_prior: bool = False
    include_bpz_prior: bool = False

    # Redshift prior types:
    # 0=no prior, 1=Gaussian p(z), 2=BPZ type p(z), 3=amplitude-dependent BPZ p(z|log_A)
    redshift_prior_type: int = 0
    z0_prior: float = 0.65
    sigma_prior: float = 0.4
    alpha_prior: float = 2.0
    beta_prior: float = 1.5
    m0_prior: float = 20.0
    kz_prior: float = 0.05
    
    # Amplitude-dependent BPZ prior (type=3)
    # z0(log10_A) = z0_amp_slope * log10_A + z0_amp_intercept
    z0_amp_slope: float = -0.114508
    z0_amp_intercept: float = 0.396857
    alpha_amp: float = 2.0  # Can make this amplitude-dependent too if needed
    
    # Performance optimization: use single batched logdensity function
    use_batched_logdensity: bool = True
    
    # Multi-core parallelization (distributes work across GPU cores using pmap)
    use_multicore: bool = False
    n_devices_per_node: int = 4  # Number of GPU cores to use per node
    
    # Robust reinitialization parameters
    use_robust_reinit: bool = False  # Use robust reinitialization logic
    reinit_min_chains_agree: int = 2  # Minimum chains that must agree
    reinit_logL_tolerance: float = 5.0  # Log-likelihood tolerance for agreement
    
    # Amplitude handling
    sample_log_amplitude: bool = False  # If True, sample log(amplitude) instead of marginalizing
    log_amplitude_prior_std: float = 2.0  # Std of Gaussian prior on log(amplitude), centered at 0
    
    # MCPL MAP extraction (for warm-starting profile likelihood optimization)
    save_mcpl_map_latents: bool = False  # If True, extract and save MAP latents at each MCPL redshift bin

    # Sample ln(z) instead of z to avoid gradient singularities near z=0
    # When True, the sampler operates in u=ln(z) space; results are converted back to z before saving
    sample_log_redshift: bool = False

    # Minimum tuned parameters: guards against rare tuner-collapse producing near-zero L/step_size
    # which causes NaN propagation throughout the sampling chain.
    min_L: float = 1e-2
    min_step_size: float = 1e-3

    # Optional SNR-based prefit initialization (for burn-in and optional direct inference).
    use_snr_prefit_init: bool = False
    snr_prefit_json: str | None = None
    snr_prefit_column: str = "phot_snr"
    skip_autotune_with_prefit: bool = False
    

def initialize_latents_scale(N, num_samples, key, include_z=True, z_min=0.0, z_max=1.0, prior_scale=1.0, redshift_in_flow=False, include_log_amplitude=False, sample_log_redshift=False,
                             redshift_prior_type=0, z0_prior=0.65, sigma_prior=0.4, alpha_prior=2.0, beta_prior=1.5):
    """
    Initialize a latent variable vector of shape (num_samples, N+k),
    where the first N dimensions follow a standard multivariate Gaussian,
    optionally followed by redshift, and optionally followed by log-amplitude.

    Parameters:
    - N (int): Latent dimension of the normalizing flow
    - num_samples (int): Number of samples
    - key (jax.random.PRNGKey): Random number generator key
    - include_z (bool): Whether to include redshift in the latent vector
    - z_min (float): Minimum redshift value
    - z_max (float): Maximum redshift value
    - prior_scale (float): Scale for the latent variable prior
    - redshift_in_flow (bool): If True, treat redshift as flow variable (Gaussian init)
    - include_log_amplitude (bool): If True, add log-amplitude dimension (initialized near 0)
    - redshift_prior_type (int): 0=uniform, 1=Gaussian, 2=BPZ, 3=amplitude-dep BPZ (falls back to 2)
    - z0_prior, sigma_prior, alpha_prior, beta_prior: prior parameters

    Returns:
    - initial_position (jnp.ndarray): Shape (num_samples, N + include_z + include_log_amplitude)
    """
    
    key_gauss, key_uniform, key_amp = jax.random.split(key, 3)

    # Sample the first N latent variables from a standard normal distribution
    # Note: N should NOT include z or log_amplitude dimensions here
    if include_z:
        N = N - 1  # Subtract z dimension
    if include_log_amplitude:
        N = N - 1  # Subtract log_amplitude dimension

    latents = prior_scale*jax.random.normal(key_gauss, shape=(num_samples, N))
    
    components = [latents]

    if include_z:
        if redshift_in_flow:
            print('Initializing Gaussian for additional flow variable')
            redshifts = jax.random.normal(key_uniform, shape=(num_samples, 1))
        elif sample_log_redshift:
            # Initialize in ln(z) space: draw z from prior, convert to ln(z)
            z_min_safe = max(z_min, 1e-4)  # Avoid ln(0)
            z_draw = _draw_z_from_prior(key_uniform, num_samples, z_min_safe, z_max,
                                        redshift_prior_type, z0_prior, sigma_prior, alpha_prior, beta_prior)
            redshifts = jnp.log(z_draw)
            print(f'Initializing ln(z) from prior type={redshift_prior_type} over [{z_min_safe:.4f}, {z_max}]')
        else:
            z_min_safe = max(z_min, 1e-6)
            redshifts = _draw_z_from_prior(key_uniform, num_samples, z_min_safe, z_max,
                                           redshift_prior_type, z0_prior, sigma_prior, alpha_prior, beta_prior)
            print(f'Initializing z from prior type={redshift_prior_type} over [{z_min_safe:.4f}, {z_max}]')
    
    if include_log_amplitude:
        # Initialize log-amplitude near 0 (corresponding to amplitude ~ 1)
        # Use small scatter around 0
        log_amplitudes = 0.1 * jax.random.normal(key_amp, shape=(num_samples, 1))
        # Append log_amplitude BEFORE redshift to get order: [u, log_A, z]
        components.append(log_amplitudes)
        print('Initializing log-amplitude near 0 (amplitude ~ 1)')
    
    if include_z and not redshift_in_flow:
        # Append redshift last
        components.append(redshifts)
    
    # Concatenate all components: [u1...uN, log_A, z] when both included
    # or [u1...uN, z] when only redshift
    initial_position = jnp.concatenate(components, axis=-1)
    
    return initial_position


def _draw_z_from_prior(key, num_samples, z_min, z_max, redshift_prior_type,
                       z0_prior, sigma_prior, alpha_prior, beta_prior, n_cdf=2000):
    """
    Draw redshift samples from the specified prior via inverse-CDF.

    Prior types:
      0 – Uniform(z_min, z_max)  (default)
      1 – Truncated Gaussian N(z0_prior, sigma_prior) on [z_min, z_max]
      2 – BPZ-type: p(z) ∝ z^alpha * exp(-(z/z0)^beta)
      3 – Amplitude-dependent BPZ (falls back to type-2 at init time)

    Returns array of shape (num_samples, 1).
    """
    if redshift_prior_type == 0:
        # Uniform
        return jax.random.uniform(key, shape=(num_samples, 1), minval=z_min, maxval=z_max)

    elif redshift_prior_type == 1:
        # Truncated Gaussian via inverse normal CDF
        a = (z_min - z0_prior) / sigma_prior
        b = (z_max - z0_prior) / sigma_prior
        pa = jax.scipy.special.ndtr(a)
        pb = jax.scipy.special.ndtr(b)
        u = jax.random.uniform(key, shape=(num_samples, 1), minval=float(pa), maxval=float(pb))
        return z0_prior + sigma_prior * jax.scipy.special.ndtri(u)

    elif redshift_prior_type in (2, 3):
        # BPZ-type: inverse CDF via pre-computed grid + linear interpolation
        z_grid = jnp.linspace(z_min, z_max, n_cdf)
        pdf = z_grid ** alpha_prior * jnp.exp(-(z_grid / z0_prior) ** beta_prior)
        dz = z_grid[1] - z_grid[0]
        cdf = jnp.cumsum(pdf * dz)
        cdf = cdf / cdf[-1]  # Normalize to [0, 1]
        u = jax.random.uniform(key, shape=(num_samples,), minval=0.0, maxval=1.0)
        z_samples = jnp.interp(u, cdf, z_grid)
        return z_samples[:, None]

    else:
        # Unknown type: fall back to uniform
        print(f'Warning: unknown redshift_prior_type={redshift_prior_type}, falling back to uniform.')
        return jax.random.uniform(key, shape=(num_samples, 1), minval=z_min, maxval=z_max)

def reinit_chains(log_p, initial_samples, key, num_chains, reinit_scatter=1e-2, precomputed_log_densities=None):

    # Use pre-computed log-densities if available, otherwise compute them
    if precomputed_log_densities is not None:
        log_densities = precomputed_log_densities
    else:
        log_densities = jax.vmap(log_p)(initial_samples[:, -1])  # Use last sample per chain
    
    best_idx = jnp.argmax(log_densities)
    best_sample = initial_samples[best_idx, -1]
    reinitialized_positions = jnp.repeat(best_sample[None, :], num_chains, axis=0)
    
    # print('replacing bad chains')
    # # Replace bad chains by resampling from good ones with small noise
    # # new_initial_positions = jax.random.choice(key, good_chains, shape=(num_chains,))
    noise = jax.random.normal(key, shape=reinitialized_positions.shape) * reinit_scatter
    reinitialized_positions += noise  
    # reinitialized_positions = new_initial_positions  

    print('reinitialized positions has shape:', reinitialized_positions.shape)

    return reinitialized_positions, best_idx


def reinit_chains_robust(log_p, initial_samples, key, num_chains, 
                         reinit_scatter=1e-2, min_chains_agree=2, 
                         logL_tolerance=5.0, precomputed_log_densities=None):
    """
    Robust chain reinitialization that checks for agreement before reinitializing.
    
    Only reinitializes from the best chain if enough chains agree (within tolerance).
    Otherwise, keeps chains at their current positions to continue independent exploration.
    
    Parameters:
    -----------
    log_p : callable
        Log-probability function
    initial_samples : array (num_chains, n_samples, n_params)
        Chain samples from burn-in phase
    key : jax.random.PRNGKey
        Random key for noise generation
    num_chains : int
        Number of chains
    reinit_scatter : float
        Small noise added to break symmetry (default: 1e-2)
    min_chains_agree : int
        Minimum number of chains that must agree to reinitialize (default: 2)
    logL_tolerance : float
        Log-likelihood difference within which chains are considered "agreeing" (default: 2.0)
    precomputed_log_densities : array, optional
        Pre-computed log densities for final positions to avoid recomputation
    
    Returns:
    --------
    reinitialized_positions : array (num_chains, n_params)
        New starting positions for chains
    reinit_performed : bool
        True if reinitialization was performed, False if chains disagreed
    agreement_info : dict
        Dictionary with agreement diagnostics
    """
    # Extract final positions from samples (always needed for do_reinit)
    final_positions = initial_samples[:, -1]  # (num_chains, n_params)
    
    # Use pre-computed log-densities if available, otherwise compute them
    if precomputed_log_densities is not None:
        log_densities = precomputed_log_densities
    else:
        log_densities = jax.vmap(log_p)(final_positions)
    
    best_idx = jnp.argmax(log_densities)
    best_logL = log_densities[best_idx]
    
    # Count how many chains are within tolerance of best
    n_agreeing = jnp.sum(log_densities >= best_logL - logL_tolerance)
    
    # Only reinitialize if enough chains agree
    should_reinit = n_agreeing >= min_chains_agree
    
    # Use jax.lax.cond for conditional logic (avoids boolean conversion error)
    def do_reinit(args):
        final_pos, key_use, best_idx_use = args
        best_sample = final_pos[best_idx_use]
        positions = jnp.repeat(best_sample[None, :], num_chains, axis=0)
        noise = jax.random.normal(key_use, shape=positions.shape) * reinit_scatter
        return positions + noise
    
    def no_reinit(args):
        final_pos, _, _ = args
        return final_pos
    
    reinitialized_positions = jax.lax.cond(
        should_reinit,
        do_reinit,
        no_reinit,
        (final_positions, key, best_idx)
    )
    
    # Return JAX arrays - conversions will happen outside traced functions
    agreement_info = {
        'n_agreeing': n_agreeing,
        'best_logL': best_logL,
        'logL_range': jnp.max(log_densities) - jnp.min(log_densities),
        'all_logL': log_densities
    }
    
    return reinitialized_positions, should_reinit, agreement_info, best_idx

def batched_log_likelihood(log_p_indiv, samples):
    n_total = samples.shape[0]
    all_logL = []

    for i in range(n_total):
        # batch = samples[i:i+batch_size]
        _, _, logL = jax.vmap(log_p_indiv)(samples[i])
        all_logL.append(logL)

    return jnp.concatenate(all_logL, axis=0)

    
def pae_spec_sample_fixz_vmap(
    PAE_obj,
    x_obs,
    weight,
    redshift_fix,
    rkey,
    sampler_cfg: MCLMCSamplingConfig
):
    """
    Fixed-redshift version of MCLMC sampling for a single galaxy.

    Parameters
    ----------
    PAE_obj : object
        Trained probabilistic autoencoder object.
    x_obs : array
        Observed photometry or spectrum for one galaxy.
    weight : array
        Inverse variance weights or uncertainties.
    redshift_fix : float
        Fixed redshift for this galaxy.
    rkey : jax.random.PRNGKey
        Random number generator key.
    sampler_cfg : MCLMCSamplingConfig
        Configuration object with sampling parameters.
    """

    # Build the log-density function with fixed redshift
    def make_logdensity_fn(push_spec_fn, x_obs, weight, redshift):
        def logdensity_fn_use(x):
            return logdensity_fn_fixz(x.reshape(1, -1), push_spec_fn, x_obs, weight, redshift)
        return logdensity_fn_use

    # log_p = make_logdensity_fn(PAE_obj.push_spec, x_obs, weight, redshift_fix)
    log_p = make_logdensity_fn(PAE_obj.push_spec_marg, x_obs, weight, redshift_fix)

    # Initialize latent positions (no z-dimension here)
    n_latent = PAE_obj.params['nlatent']
    initial_position = jnp.array(initialize_latents_scale(
        n_latent,
        sampler_cfg.nchain_per_gal,
        rkey,
        z_min=sampler_cfg.zmin,
        z_max=sampler_cfg.zmax,
        include_z=False,
        prior_scale=1.0
    ))

    transform = lambda state, info: state.position
    sample_key, _ = jax.random.split(rkey)
    keys = jax.random.split(sample_key, initial_position.shape[0])

    # ---------------------------- Optional re-initialization step --------------------
    if sampler_cfg.init_reinit:
        def run_initial_chain(position, key):
            return run_mclmc_simp(
                log_p,
                sampler_cfg.nsamp_init,
                position,
                key,
                transform,
                sampler_cfg.pretuned_L,
                sampler_cfg.pretuned_step_size,
                sampler_cfg.desired_energy_variance
            )

        all_init_samples, new_keys = jax.vmap(run_initial_chain)(initial_position, keys)
        
        if sampler_cfg.use_robust_reinit:
            print('Using robust reinitialization...')
            reinitialized_positions, reinit_performed, agreement_info = reinit_chains_robust(
                log_p,
                all_init_samples,
                rkey,
                sampler_cfg.nchain_per_gal,
                reinit_scatter=sampler_cfg.reinit_scatter,
                min_chains_agree=sampler_cfg.reinit_min_chains_agree,
                logL_tolerance=sampler_cfg.reinit_logL_tolerance
            )
            # Convert JAX arrays to Python types for printing
            print(f'Robust reinit: performed={bool(reinit_performed)}, n_agreeing={int(agreement_info["n_agreeing"])}')
        else:
            reinitialized_positions = reinit_chains(
                log_p,
                all_init_samples,
                rkey,
                sampler_cfg.nchain_per_gal,
                reinit_scatter=sampler_cfg.reinit_scatter
            )
        keys_for_final_run = new_keys
    else:
        reinitialized_positions = initial_position
        keys_for_final_run = keys

    # Final MCLMC sampling with pre-tuning
    def run_inference_chain(position, key):
        return run_mclmc_simp_with_pretune(
            log_p,
            sampler_cfg.num_steps,
            position,
            key,
            transform,
            sampler_cfg.pretuned_L,
            sampler_cfg.pretuned_step_size,
            sampler_cfg.desired_energy_variance,
            min_L=sampler_cfg.min_L,
            min_step_size=sampler_cfg.min_step_size
        )

    final_samples, _, tuned_L, tuned_step_size = jax.vmap(run_inference_chain)(
        reinitialized_positions, keys_for_final_run
    )

    # Compute posterior mean log-probability
    post_mean = jnp.mean(final_samples, axis=(0, 1)).reshape(1, -1)

    def make_logdensity_fn_indiv(push_spec_fn, x_obs, weight, redshift):
        def logdensity_fn_use(x):
            return logdensity_fn_fixz(x.reshape(1, -1), push_spec_fn, x_obs, weight, redshift,return_indiv=True)
        return logdensity_fn_use

    # log_p_indiv = make_logdensity_fn_indiv(PAE_obj.push_spec, x_obs, weight, redshift_fix)
    log_p_indiv = make_logdensity_fn_indiv(PAE_obj.push_spec_marg, x_obs, weight, redshift_fix)

    postmean_log_prior, postmean_logL = jax.vmap(log_p_indiv)(post_mean)

    return final_samples, postmean_log_prior, postmean_logL, initial_position, tuned_L, tuned_step_size


def pae_spec_sample_floatz(PAE_obj, x_obs, weight, rkey, sampler_cfg: MCLMCSamplingConfig,
                           batched_log_density=None, log_amplitude_data=None,
                           batched_log_density_burnin=None, batched_log_likelihood=None,
                           init_L=None, init_step_size=None):
    """
    Sample from PAE posterior for a single galaxy with floating redshift.
    
    Args:
        PAE_obj: PAE model object
        x_obs: (n_bands,) observed spectrum
        weight: (n_bands,) inverse variance weights
        rkey: JAX random key
        sampler_cfg: MCLMCSamplingConfig
        batched_log_density: Optional pre-compiled batched logdensity function (final run).
                            If provided, uses it (avoids per-galaxy JIT compilation).
                            If None, creates galaxy-specific closure (old behavior).
        log_amplitude_data: log10 of weighted mean flux (for amplitude-dependent prior type=3)
        batched_log_density_burnin: Optional pre-compiled batched logdensity for burn-in phase.
                            Only used when sampler_cfg.nf_alpha_burnin >= 0.
                            If None and nf_alpha_burnin >= 0, builds a per-galaxy closure.
        batched_log_likelihood: Optional pre-compiled pure photometric log-likelihood function.
                            If provided, its per-chain mean is saved as mean_logpx_per_chain
                            (enabling chi2_phot = -2 * mean_logpx, prior-free).
                            If None, mean_logpx_per_chain is set to zeros.
    """
    
    # Create log_p function for this galaxy (used for FINAL sampling phase)
    if batched_log_density is not None:
        # NEW: Use pre-compiled batched function (data as parameters)
        log_p = lambda latent: batched_log_density(latent, x_obs, weight, log_amplitude_data)
    else:
        # OLD: Create new closure for this galaxy (closes over x_obs, weight)
        log_p = make_logdensity_fn_optzprior(
            PAE_obj, x_obs, weight, z_min=sampler_cfg.zmin, z_max=sampler_cfg.zmax,
            nf_alpha=sampler_cfg.nf_alpha,
            redshift_in_flow=sampler_cfg.redshift_in_flow,
            z0_prior=sampler_cfg.z0_prior,
            sigma_prior=sampler_cfg.sigma_prior,
            include_gaussian_prior=sampler_cfg.include_gaussian_prior,
            redshift_prior_type=sampler_cfg.redshift_prior_type
            # Note: BPZ prior (type=2) not yet implemented in make_logdensity_fn_optzprior
            # When implemented, add: alpha_prior, beta_prior, m0_prior, kz_prior
        )

    # Create pure photometric log-likelihood function for afterburner diagnostics
    if batched_log_likelihood is not None:
        log_px = lambda latent: batched_log_likelihood(latent, x_obs, weight, log_amplitude_data)
    else:
        log_px = None

    # Create log_p_burnin for the BURN-IN / init phase.
    # When nf_alpha_burnin >= 0, the burn-in uses a different nf_alpha from the final run,
    # allowing the NF prior to guide initialization without biasing the final posterior.
    if sampler_cfg.nf_alpha_burnin < 0:
        # Disabled: burn-in uses the same log_p as the final run
        log_p_burnin = log_p
    elif batched_log_density_burnin is not None:
        log_p_burnin = lambda latent: batched_log_density_burnin(latent, x_obs, weight, log_amplitude_data)
    else:
        log_p_burnin = make_logdensity_fn_optzprior(
            PAE_obj, x_obs, weight, z_min=sampler_cfg.zmin, z_max=sampler_cfg.zmax,
            nf_alpha=sampler_cfg.nf_alpha_burnin,
            redshift_in_flow=sampler_cfg.redshift_in_flow,
            z0_prior=sampler_cfg.z0_prior,
            sigma_prior=sampler_cfg.sigma_prior,
            include_gaussian_prior=sampler_cfg.include_gaussian_prior,
            redshift_prior_type=sampler_cfg.redshift_prior_type
        )

    # Compute total latent dimension: u1..uN + z (always included) + log_A (if sampling amplitude)
    # Note: redshift_in_flow doesn't change dimension count, just changes where z goes
    n_latent = PAE_obj.params['nlatent'] + 1  # Always +1 for redshift
    if sampler_cfg.sample_log_amplitude:
        n_latent += 1  # Add dimension for log(amplitude)
    
    initial_position = jnp.array(initialize_latents_scale(
        n_latent, sampler_cfg.nchain_per_gal, rkey, z_min=sampler_cfg.zmin, z_max=sampler_cfg.zmax,
        include_z=True, prior_scale=1.0,
        redshift_in_flow=sampler_cfg.redshift_in_flow,
        include_log_amplitude=sampler_cfg.sample_log_amplitude,
        sample_log_redshift=sampler_cfg.sample_log_redshift,
        redshift_prior_type=sampler_cfg.redshift_prior_type,
        z0_prior=sampler_cfg.z0_prior,
        sigma_prior=sampler_cfg.sigma_prior,
        alpha_prior=sampler_cfg.alpha_prior,
        beta_prior=sampler_cfg.beta_prior,
    ))

    transform = lambda state, info: state.position
    sample_key, _ = jax.random.split(rkey)
    keys = jax.random.split(sample_key, initial_position.shape[0])

    init_L_use = sampler_cfg.pretuned_L if init_L is None else init_L
    init_step_use = sampler_cfg.pretuned_step_size if init_step_size is None else init_step_size

    # Initialization phase
    if sampler_cfg.init_reinit:
        all_init_samples, new_keys = jax.vmap(
            lambda pos, key: run_mclmc_simp(
                log_p_burnin, sampler_cfg.nsamp_init, pos, key, transform,
                init_L_use, init_step_use, sampler_cfg.desired_energy_variance
            )
        )(initial_position, keys)
        
        # Compute log-probabilities for the final pre-reinitialization positions.
        # Evaluated under log_p_burnin (consistent with the landscape the chains explored).
        preinit_final_positions = all_init_samples[:, -1, :]  # Last sample per chain
        preinit_final_log_p = jax.vmap(log_p_burnin)(preinit_final_positions)  # (nchain_per_gal,)
        
        if sampler_cfg.use_robust_reinit:
            # Pass precomputed log densities to avoid recomputation
            reinitialized_positions, reinit_performed, agreement_info, reinit_best_chain_idx = reinit_chains_robust(
                log_p_burnin, all_init_samples, rkey, sampler_cfg.nchain_per_gal,
                reinit_scatter=sampler_cfg.reinit_scatter,
                min_chains_agree=sampler_cfg.reinit_min_chains_agree,
                logL_tolerance=sampler_cfg.reinit_logL_tolerance,
                precomputed_log_densities=preinit_final_log_p
            )
        else:
            # Pass precomputed log densities to avoid recomputation
            reinitialized_positions, reinit_best_chain_idx = reinit_chains(log_p_burnin, all_init_samples, rkey, sampler_cfg.nchain_per_gal, 
                                                   reinit_scatter=sampler_cfg.reinit_scatter,
                                                   precomputed_log_densities=preinit_final_log_p)
        keys_for_final_run = new_keys
    else:
        reinitialized_positions = initial_position
        keys_for_final_run = keys
        # No pre-init samples to save - return dummy array for vmap compatibility
        # Shape: (nchain_per_gal,) to match the case when init_reinit=True
        preinit_final_log_p = jnp.zeros(sampler_cfg.nchain_per_gal)
        reinit_best_chain_idx = jnp.array(-1, dtype=jnp.int32)

    # Final sampling phase with optional autotune bypass.
    if sampler_cfg.skip_autotune_with_prefit:
        final_samples, _ = jax.vmap(
            lambda pos, key: run_mclmc_simp(
                log_p, sampler_cfg.num_steps, pos, key, transform,
                init_L_use, init_step_use, sampler_cfg.desired_energy_variance
            )
        )(reinitialized_positions, keys_for_final_run)
        tuned_L = jnp.full((sampler_cfg.nchain_per_gal,), jnp.maximum(init_L_use, sampler_cfg.min_L))
        tuned_step_size = jnp.full((sampler_cfg.nchain_per_gal,), jnp.maximum(init_step_use, sampler_cfg.min_step_size))
    else:
        final_samples, _, tuned_L, tuned_step_size = jax.vmap(
            lambda pos, key: run_mclmc_simp_with_pretune(
                log_p, sampler_cfg.num_steps, pos, key, transform,
                init_L_use, init_step_use, sampler_cfg.desired_energy_variance,
                min_L=sampler_cfg.min_L, min_step_size=sampler_cfg.min_step_size
            )
        )(reinitialized_positions, keys_for_final_run)

    # Placeholder: only returning log_p, not split into prior / redshift terms
    post_mean_log_p = jax.vmap(log_p)(jnp.mean(final_samples, axis=1))

    post_burnin_samples = final_samples[:, sampler_cfg.burn_in:, :]
    # # Evaluate log_p for all post-burn-in samples
    # log_p_all = jax.vmap(lambda chain: jax.vmap(log_p)(chain))(post_burnin_samples)

    # log_p_all = []
    # for chain_samples in post_burnin_samples:
    #     log_p_chain = jax.vmap(log_p)(chain_samples)  # vectorize over samples in this chain
    #     log_p_all.append(log_p_chain)
    # log_p_all = jnp.stack(log_p_all)

    log_p_all = jax.lax.map(
                    lambda chain: jax.lax.map(log_p, chain),
                    post_burnin_samples
                )
    # # Mean logL per chain
    mean_logL_per_chain = jnp.mean(log_p_all, axis=1)

    # # Max logL across all chains and samples
    max_logL_all = jnp.max(log_p_all)

    # Pure photometric log-likelihood per chain (prior terms excluded)
    if log_px is not None:
        log_px_all = jax.lax.map(
                        lambda chain: jax.lax.map(log_px, chain),
                        post_burnin_samples
                    )
        mean_logpx_per_chain = jnp.mean(log_px_all, axis=1)
    else:
        mean_logpx_per_chain = jnp.zeros_like(mean_logL_per_chain)

    # Convert from ln(z) space back to z space if sampling in log-redshift mode.
    # This must happen AFTER log_p_all is computed (which needs ln(z) parameterization)
    # but BEFORE MCPL / saving (which expect actual z values).
    if sampler_cfg.sample_log_redshift:
        final_samples = final_samples.at[:, :, -1].set(jnp.exp(final_samples[:, :, -1]))
        post_burnin_samples = final_samples[:, sampler_cfg.burn_in:, :]

    # Always compute MCPL and MAP latents (small overhead, avoids JIT issues)
    z_bins, profile_logL, mcpl_map_latents = monte_carlo_profile_likelihood_jax(
                            post_burnin_samples, log_p_all, z_index=-1, n_bins=200,
                            z_min=sampler_cfg.zmin, z_max=sampler_cfg.zmax,
                            return_map_latents=True
                        )

    
    return final_samples, None, jnp.zeros_like(post_mean_log_p), jnp.zeros_like(post_mean_log_p), post_mean_log_p, initial_position, tuned_L, tuned_step_size, mean_logL_per_chain, max_logL_all, z_bins, profile_logL, preinit_final_log_p, mcpl_map_latents, reinit_best_chain_idx, mean_logpx_per_chain

def sample_mclmc_wrapper(PAE_obj,
                          data: SPHERExData, 
                          sampler_cfg: MCLMCSamplingConfig, 
                          src_idxs_run = None,
                          src_idxs_sub= None,
                          property_cat_df=None,
                          ngal = None,
                          batch_size: int=100,
                          keyidx: int=102,
                          save_results: bool=False,
                          save_fpath: str | None=None,
                          sample_fpath: str | None=None,
                          return_results: bool=True,
                          do_cleanup: bool=False,
                          use_native_filters: bool=False,
                          filter_curves=None,
                          lam_interp=None,
                          phot_norms=None
                         ):
                          

    if ngal is None:
        ngal = len(src_idxs_run) if src_idxs_run is not None else len(data.src_idxs)

    if src_idxs_run is None:
        idxs = np.arange(ngal)
    else:
        idxs = src_idxs_run

    rng_key = jr.key(keyidx)

    # Pull data slices

    if src_idxs_sub is None:
        query_idx = data.src_idxs[idxs]
    else:
        query_idx = src_idxs_sub
    
    specs = data.all_spec_obs[query_idx]
    weights = data.weights[query_idx]
    log_amplitudes = data.log_amplitude[query_idx]  # Extract log_amplitude for amplitude-dependent prior
    
    # Diagnostic output for amplitude-dependent prior
    if sampler_cfg.redshift_prior_type == 3:
        jax.debug.print("  Using amplitude-dependent BPZ prior (type=3)")
        jax.debug.print("  Log-amplitude range: [{min:.3f}, {max:.3f}]", 
                        min=log_amplitudes.min(), max=log_amplitudes.max())
        jax.debug.print("  Log-amplitude mean: {mean:.3f} ± {std:.3f}",
                        mean=log_amplitudes.mean(), std=log_amplitudes.std())
        
        # Check for NaN or inf values
        n_invalid = np.sum(~np.isfinite(log_amplitudes))
        if n_invalid > 0:
            jax.debug.print("  ⚠ WARNING: {n} sources have invalid log-amplitudes (NaN or inf)", n=n_invalid)
            jax.debug.print("  These will be replaced with 1.0 (corresponding to amplitude=10) in the prior")
    
    # For real data: separate spec-z (ztrue) from template-fitting photo-z (z_TF)
    # For simulated data: data.redshift contains the true redshift
    if property_cat_df is not None:
        # Real data: extract spec-z as ztrue (may be NaN/None)
        try:
            redshifts_true = np.array(property_cat_df['z_specz'])[query_idx]
            # Convert NaN to a placeholder value for sampler (won't be used for validation anyway)
            redshifts_sampler = np.where(np.isnan(redshifts_true), 1.0, redshifts_true)
        except (KeyError, TypeError):
            # Fallback if z_specz not available
            redshifts_true = data.redshift[query_idx]
            redshifts_sampler = redshifts_true
    else:
        # Simulated data: use catalog redshift
        redshifts_true = data.redshift[query_idx]
        redshifts_sampler = redshifts_true

    prefit_L_source = None
    prefit_step_source = None
    if sampler_cfg.use_snr_prefit_init:
        prefit_L_source, prefit_step_source = _build_prefit_init_from_snr(
            sampler_cfg,
            property_cat_df,
            query_idx,
        )
        print(
            "Using SNR-prefit init: "
            f"L range=[{np.nanmin(prefit_L_source):.4f}, {np.nanmax(prefit_L_source):.4f}], "
            f"step range=[{np.nanmin(prefit_step_source):.4f}, {np.nanmax(prefit_step_source):.4f}]"
        )
        if sampler_cfg.skip_autotune_with_prefit:
            print("SNR-prefit mode: skipping final MCLMC autotuning and running direct inference")

    
    # === Main sampling phase ===
    (all_samples, all_ae_redshifts, all_pm_log_prior, all_pm_log_redshift,
             all_pm_log_L, all_pm_initial_position, all_tuned_L, all_tuned_step_size, 
            all_mean_logL_per_chain, all_max_logL, 
            z_bins_mcpl, all_mcpl, all_preinit_final_logL, all_mcpl_map_latents,
            all_reinit_best_chain_idx, all_mean_logpx_per_chain) = run_batched_sampler(
                                                        PAE_obj, 
                                                        specs, 
                                                        weights, 
                                                        redshifts_sampler,
                                                        log_amplitudes=log_amplitudes,
                                                        prefit_L_by_source=prefit_L_source,
                                                        prefit_step_by_source=prefit_step_source,
                                                        batch_size=batch_size,
                                                        rng_key=rng_key,
                                                        sampler_cfg=sampler_cfg)
    
    dchi2 = None

    if do_cleanup:
        bad_mask_chi2, bad_mask_Rhat = cleanup_mask(all_samples, all_pm_log_L, PAE_obj.params['nlatent'], sampler_cfg.chi2_red_threshold, nbands=102, gr_threshold=sampler_cfg.gr_threshold)
        # bad_mask_combined = bad_mask_chi2 * bad_mask_Rhat
        bad_mask_combined = jnp.logical_or(bad_mask_chi2, bad_mask_Rhat)

        nbad_chi2, nbad_Rhat, n_bad = int(jnp.sum(bad_mask_chi2)), int(jnp.sum(bad_mask_Rhat)), int(jnp.sum(bad_mask_combined))
                
        print(f"Identified {nbad_chi2} bad fits out of {ngal} galaxies based on reduced chi2..")
        print(f"Identified {nbad_Rhat} bad fits out of {ngal} galaxies based on G-R statistic for redshift parameter..")
        print(f"Identified {n_bad} bad fits out of {ngal} galaxies based on both reduced chi2 and G-R statistic..")

        if n_bad > 0:
            bad_idxs = idxs[bad_mask_combined]
            rng_key, cleanup_key = jr.split(rng_key)
    
            # not sure this line works, setting do_cleanup = False for now
            cleanup_config = replace(sampler_cfg, nchain_per_gal=sampler_cfg.nchain_per_gal * sampler_cfg.rescale_nchain)
    
            bad_data_slice = data.src_idxs[bad_idxs]
            
            # === Clean-up re-sampling phase ===
            (rerun_samples, _, rerun_log_prior, rerun_log_redshift,
                 rerun_log_L, rerun_init_pos, rerun_tuned_L, rerun_tuned_stepsize, 
                    rerun_mean_logL_per_chain, rerun_max_logL, 
                    _, _, _, _, rerun_mean_logpx_per_chain) = run_batched_sampler(
                    PAE_obj,
                    data.all_spec_obs[bad_data_slice],
                    data.weights[bad_data_slice],
                    data.redshift[bad_data_slice],
                    batch_size=batch_size // sampler_cfg.rescale_nchain,
                    rng_key=cleanup_key,
                    sampler_cfg=cleanup_config
                )
    
            # Keep only the same number of chains as in the original run
            rerun_samples = rerun_samples[:, -sampler_cfg.nchain_per_gal:, :, :]
            rerun_log_L = rerun_log_L[:, -sampler_cfg.nchain_per_gal:]
            rerun_log_prior = rerun_log_prior[:, -sampler_cfg.nchain_per_gal:]
            rerun_init_pos = rerun_init_pos[:, -sampler_cfg.nchain_per_gal:, :]
            rerun_tuned_L = rerun_tuned_L[:, -sampler_cfg.nchain_per_gal:]
            rerun_tuned_stepsize = rerun_tuned_stepsize[:, -sampler_cfg.nchain_per_gal:]
            rerun_mean_logL_per_chain = rerun_mean_logL_per_chain[:, -sampler_cfg.nchain_per_gal:]

            # rerun_max_logL = rerun_max_logL[:, -sampler_cfg.nchain_per_gal:]
            # rerun_max_logL = rerun_max_logL[:, -sampler_cfg.nchain_per_gal:]
            
            if not sampler_cfg.fix_z:
                rerun_log_redshift = rerun_log_redshift[:, -sampler_cfg.nchain_per_gal:]
    
    
            dchi2 = 2*(jnp.mean(rerun_log_L, axis=1)-jnp.mean(all_pm_log_L[bad_mask_combined], axis=1))
            print('improvement:', jnp.mean(rerun_log_L, axis=1)-jnp.mean(all_pm_log_L[bad_mask_combined], axis=1))
            
            # === Replace in full arrays ===
            all_samples = jax.tree_util.tree_map(
                lambda a, b: a.at[bad_mask_combined].set(b),
                all_samples, rerun_samples
            )
            all_pm_log_L = all_pm_log_L.at[bad_mask_combined].set(rerun_log_L)
            all_pm_log_prior = all_pm_log_prior.at[bad_mask_combined].set(rerun_log_prior)
            all_pm_initial_position = all_pm_initial_position.at[bad_mask_combined].set(rerun_init_pos)
            all_tuned_L = all_tuned_L.at[bad_mask_combined].set(rerun_tuned_L)
            all_tuned_step_size = all_tuned_step_size.at[bad_mask_combined].set(rerun_tuned_stepsize)
    
            if not sampler_cfg.fix_z:
                all_pm_log_redshift = all_pm_log_redshift.at[bad_mask_combined].set(rerun_log_redshift)

            all_mean_logL_per_chain = all_mean_logL_per_chain.at[bad_mask_combined].set(rerun_mean_logL_per_chain)
            all_max_logL = all_max_logL.at[bad_mask_combined].set(rerun_max_logL)
            if all_mean_logpx_per_chain is not None and rerun_mean_logpx_per_chain is not None:
                all_mean_logpx_per_chain = all_mean_logpx_per_chain.at[bad_mask_combined].set(rerun_mean_logpx_per_chain)
    
            # if sampler_cfg.redshift_in_flow and all_ae_redshifts is not None and rerun_ae_redshifts is not None:
            #     all_ae_redshifts = all_ae_redshifts.at[bad_mask_combined].set(rerun_ae_redshifts)
    

    if sampler_cfg.fix_z:
        all_pm_log_redshift = None
    
    # === Save final results ===
    if save_results:
        # Extract template-fitting photo-z (z_TF) if available
        z_TF = None
        try:
            if property_cat_df is not None:
                # Try to get photo-z from template fitting
                # Priority: z_phot (stored explicitly) > z_best_gals > redshift column
                if 'z_phot' in property_cat_df.columns:
                    z_TF = np.array(property_cat_df['z_phot'])[query_idx]
                elif 'z_best_gals' in property_cat_df.columns:
                    z_TF = np.array(property_cat_df['z_best_gals'])[query_idx]
                else:
                    # Fallback to 'redshift' column (which should be photo-z for real data)
                    z_TF = np.array(property_cat_df['redshift'])[query_idx]
                
                # Convert NaN to None for cleaner output
                if np.any(np.isnan(z_TF)):
                    z_TF = np.where(np.isnan(z_TF), None, z_TF)
        except Exception as e:
            print(f"Warning: Could not extract z_TF: {e}")
            z_TF = None

        # try to extract template-fitting photo-z error if available
        z_TF_err = None
        try:
            if property_cat_df is not None:
                if 'z_err' in property_cat_df.columns:
                    z_TF_err = np.array(property_cat_df['z_err'])[query_idx]
                elif 'z_err_std_gals' in property_cat_df.columns:
                    z_TF_err = np.array(property_cat_df['z_err_std_gals'])[query_idx]
                else:
                    # fallback: try common alternate names
                    for col in ['z_err_gals', 'z_err_std', 'z_err_tf']:
                        if col in property_cat_df.columns:
                            z_TF_err = np.array(property_cat_df[col])[query_idx]
                            break
                if z_TF_err is not None and np.any(np.isnan(z_TF_err)):
                    z_TF_err = np.where(np.isnan(z_TF_err), None, z_TF_err)
        except Exception:
            z_TF_err = None

        # Extract spectral completeness (frac_sampled_102) if available
        frac_sampled_102 = None
        try:
            if property_cat_df is not None and 'frac_sampled_102' in property_cat_df.columns:
                frac_sampled_102 = np.array(property_cat_df['frac_sampled_102'])[query_idx]
        except Exception as e:
            print(f"Warning: Could not extract frac_sampled_102: {e}")
            frac_sampled_102 = None

        # Extract template fitting chi2 (minchi2_gals) if available
        minchi2_gals = None
        try:
            if property_cat_df is not None and 'minchi2_gals' in property_cat_df.columns:
                minchi2_gals = np.array(property_cat_df['minchi2_gals'])[query_idx]
        except Exception as e:
            print(f"Warning: Could not extract minchi2_gals: {e}")
            minchi2_gals = None

        # Extract broadband quadrature SNR (snr_quad = sqrt(sum((F/sigma)^2))) if available
        snr_quad = None
        try:
            if property_cat_df is not None and 'snr_quad' in property_cat_df.columns:
                snr_quad = np.array(property_cat_df['snr_quad'])[query_idx]
        except Exception as e:
            print(f"Warning: Could not extract snr_quad: {e}")
            snr_quad = None

        # Extract initial z per chain (last latent dim); convert from ln(z) if needed
        init_z_per_chain = np.asarray(all_pm_initial_position[:, :, -1])
        if sampler_cfg.sample_log_redshift:
            init_z_per_chain = np.exp(init_z_per_chain)

        if sampler_cfg.redshift_in_flow:
            print('all ae redshifts has shape', all_ae_redshifts.shape)
            save_redshift_results(save_fpath, all_samples, all_pm_log_L, all_pm_log_prior, all_pm_log_redshift, redshifts_true,
                                  sample_fpath=sample_fpath, ae_redshifts=all_ae_redshifts, z_TF=z_TF, z_TF_err=z_TF_err,
                                  weights=weights,
                                  all_tuned_L=np.asarray(all_tuned_L), all_tuned_step_size=np.asarray(all_tuned_step_size),
                                  all_preinit_final_logL=all_preinit_final_logL,
                                  phot_norms=phot_norms,
                                  sample_log_amplitude=sampler_cfg.sample_log_amplitude,
                                  src_ids=data.srcid_obs[query_idx],
                                  data_indices=query_idx,
                                  frac_sampled_102=frac_sampled_102,
                                  minchi2_gals=minchi2_gals,
                                  snr_quad=snr_quad,
                                  init_z_per_chain=init_z_per_chain,
                                  reinit_best_chain_idx=all_reinit_best_chain_idx,
                                  all_mean_logpx_per_chain=all_mean_logpx_per_chain)
        else:
            # Only save MAP latents if flag is set (they're always computed to avoid JIT issues)
            mcpl_map_latents_to_save = all_mcpl_map_latents if sampler_cfg.save_mcpl_map_latents else None
            
            save_redshift_results(save_fpath, all_samples, all_pm_log_L, all_pm_log_prior, all_pm_log_redshift, redshifts_true,
                                  sample_fpath=sample_fpath, run_name=PAE_obj.run_name, 
                                 all_mean_logL_per_chain=all_mean_logL_per_chain, all_max_logL=all_max_logL, 
                                 z_bins_mcpl=z_bins_mcpl, all_mcpl=all_mcpl, z_TF=z_TF, z_TF_err=z_TF_err,
                                 weights=weights,
                                 burn_in=sampler_cfg.burn_in,
                                 all_tuned_L=np.asarray(all_tuned_L), all_tuned_step_size=np.asarray(all_tuned_step_size),
                                 all_preinit_final_logL=all_preinit_final_logL,
                                 phot_norms=phot_norms,
                                 sample_log_amplitude=sampler_cfg.sample_log_amplitude,
                                 src_ids=data.srcid_obs[query_idx],
                                 data_indices=query_idx,
                                 all_mcpl_map_latents=mcpl_map_latents_to_save,
                                 frac_sampled_102=frac_sampled_102,
                                 minchi2_gals=minchi2_gals,
                                 snr_quad=snr_quad,
                                 init_z_per_chain=init_z_per_chain,
                                 reinit_best_chain_idx=all_reinit_best_chain_idx,
                                 all_mean_logpx_per_chain=all_mean_logpx_per_chain)


    # === Return final results ===
    if return_results:
        # Convert JAX device arrays to host numpy arrays to free GPU memory
        # This is critical for preventing memory accumulation across batches
        
        def to_host_numpy(x):
            """Convert JAX array to numpy, moving to host if needed."""
            if x is None:
                return None
            if hasattr(x, 'device'):  # JAX array
                return np.array(jax.device_get(x))
            return x
        
        # Convert all results to host numpy
        all_samples_host = jax.tree_util.tree_map(to_host_numpy, all_samples)
        all_pm_log_L_host = to_host_numpy(all_pm_log_L)
        all_pm_log_prior_host = to_host_numpy(all_pm_log_prior)
        all_pm_log_redshift_host = to_host_numpy(all_pm_log_redshift)
        all_ae_redshifts_host = to_host_numpy(all_ae_redshifts)
        dchi2_host = to_host_numpy(dchi2)
        
        # Delete JAX device arrays explicitly
        del all_samples, all_pm_log_L, all_pm_log_prior, all_pm_log_redshift
        del all_ae_redshifts, dchi2
        
        res = {'all_samples': all_samples_host, 
               'all_ae_redshifts': all_ae_redshifts_host, 
               'all_pm_log_L': all_pm_log_L_host,
               'all_pm_log_prior': all_pm_log_prior_host,
               'all_pm_log_redshift': all_pm_log_redshift_host,
               'dchi2': dchi2_host}

        return res
        
        
def split_into_batches(x, batch_size):
    n = x.shape[0]
    return [x[i:i+batch_size] for i in range(0, n, batch_size)]


def _predict_log_from_fit_model(fit_model, x):
    kind = fit_model.get("kind", "linear")
    x = np.asarray(x, dtype=float)

    if kind in ("linear", "poly2", "poly3"):
        coeffs = np.asarray(fit_model["coeffs"], dtype=float)
        return np.polyval(coeffs, x)

    if kind == "bininterp":
        x_nodes = np.asarray(fit_model["x_nodes"], dtype=float)
        y_nodes = np.asarray(fit_model["y_nodes"], dtype=float)
        x_clip = np.clip(x, x_nodes[0], x_nodes[-1])
        return np.interp(x_clip, x_nodes, y_nodes)

    raise ValueError(f"Unsupported fit model kind: {kind}")


def _build_prefit_init_from_snr(sampler_cfg: MCLMCSamplingConfig, property_cat_df, query_idx):
    """Predict per-source prefit L and step_size from SNR using saved fit JSON."""
    if not sampler_cfg.use_snr_prefit_init:
        return None, None

    if sampler_cfg.snr_prefit_json is None:
        raise ValueError("use_snr_prefit_init=True but snr_prefit_json is not set")

    if property_cat_df is None:
        raise ValueError("SNR-prefit init requires property_cat_df with SNR column")

    if sampler_cfg.snr_prefit_column not in property_cat_df.columns:
        raise KeyError(
            f"SNR-prefit column '{sampler_cfg.snr_prefit_column}' not in property_cat_df"
        )

    with open(sampler_cfg.snr_prefit_json, "r", encoding="utf-8") as f:
        fit_payload = json.load(f)

    fit_step = fit_payload.get("fit_step_size", None)
    fit_L = fit_payload.get("fit_L", None)
    if fit_step is None or fit_L is None:
        raise KeyError("Fit JSON missing fit_step_size and/or fit_L model")

    snr_vals = np.asarray(property_cat_df[sampler_cfg.snr_prefit_column], dtype=float)[query_idx]
    snr_safe = np.clip(snr_vals, 1e-8, None)
    log_snr = np.log10(snr_safe)

    log_step = _predict_log_from_fit_model(fit_step, log_snr)
    log_L = _predict_log_from_fit_model(fit_L, log_snr)

    step = np.maximum(10 ** log_step, sampler_cfg.min_step_size)
    L = np.maximum(10 ** log_L, sampler_cfg.min_L)

    return L.astype(np.float32), step.astype(np.float32)


def run_batched_sampler_multicore(
    PAE_obj,
    specs,
    weights,
    redshifts,
    batch_size: int,
    rng_key,
    sampler_cfg: MCLMCSamplingConfig,
    log_amplitudes=None,
    prefit_L_by_source=None,
    prefit_step_by_source=None,
):
    """
    Multi-core version using pmap to distribute work across GPU cores.
    
    Strategy:
    - Split galaxies into n_devices groups
    - Each device processes batch_size//n_devices galaxies in parallel
    - Results are gathered and concatenated
    
    Performance: Each core saturates around 600 chains (150 sources × 4 chains)
    """
    
    devices = jax.devices()
    n_devices = min(sampler_cfg.n_devices_per_node, len(devices))
    
    print("=" * 70)
    print(f"MULTI-CORE MODE: Distributing work across {n_devices} GPU cores")
    print(f"Available devices: {[str(d) for d in devices[:n_devices]]}")
    print(f"Total galaxies: {specs.shape[0]}")
    print(f"Chains per galaxy: {sampler_cfg.nchain_per_gal}")
    print(f"Total chains: {specs.shape[0] * sampler_cfg.nchain_per_gal}")
    print(f"Batch size per core: ~{batch_size // n_devices}")
    print(f"Init reinit: {sampler_cfg.init_reinit}")
    print(f"Robust reinit: {sampler_cfg.use_robust_reinit}")
    if sampler_cfg.use_robust_reinit:
        print(f"  - Min chains agree: {sampler_cfg.reinit_min_chains_agree}")
        print(f"  - LogL tolerance: {sampler_cfg.reinit_logL_tolerance}")
        print(f"  - Reinit scatter: {sampler_cfg.reinit_scatter}")
    
    # Diagnostic for amplitude-dependent prior
    if log_amplitudes is not None:
        jax.debug.print("Log-amplitudes provided: shape={shape}", shape=log_amplitudes.shape)
        if sampler_cfg.redshift_prior_type == 3:
            jax.debug.print("  → Will be used for amplitude-dependent BPZ prior (type=3)")
    else:
        if sampler_cfg.redshift_prior_type == 3:
            jax.debug.print("⚠ WARNING: Prior type=3 requires log_amplitudes, but none provided!")
    print("=" * 70)
    
    # Create batched logdensity once (shared across all devices)
    batched_log_density = None
    batched_log_likelihood = None
    if sampler_cfg.use_batched_logdensity and not sampler_cfg.fix_z:
        print("Creating single batched logdensity function for all cores...")
        batched_log_density = make_batched_logdensity_fn(PAE_obj, sampler_cfg)
        batched_log_likelihood = make_batched_loglikelihood_fn(PAE_obj, sampler_cfg)
        
        # Pre-compile
        n_latent = PAE_obj.params['nlatent']
        n_bands = specs.shape[1]
        dummy_latent = jnp.zeros(n_latent + 1)
        dummy_spec = jnp.ones(n_bands)
        dummy_weight = jnp.ones(n_bands)
        _ = batched_log_density(dummy_latent, dummy_spec, dummy_weight, None, None)
        _ = batched_log_likelihood(dummy_latent, dummy_spec, dummy_weight, None, None)
        print("✓ Logdensity compiled and ready")
    
    # Split data across devices
    n_gal = specs.shape[0]
    gal_per_device = (n_gal + n_devices - 1) // n_devices  # Ceiling division
    
    # Pad data to make it evenly divisible
    pad_size = gal_per_device * n_devices - n_gal
    if pad_size > 0:
        specs = jnp.concatenate([specs, jnp.zeros((pad_size, specs.shape[1]))], axis=0)
        weights = jnp.concatenate([weights, jnp.ones((pad_size, weights.shape[1]))], axis=0)
        redshifts = jnp.concatenate([redshifts, jnp.zeros(pad_size)], axis=0)
        if log_amplitudes is not None:
            log_amplitudes = jnp.concatenate([log_amplitudes, jnp.ones(pad_size)], axis=0)  # Default to 1.0 for padding
        if prefit_L_by_source is not None:
            prefit_L_by_source = jnp.concatenate([prefit_L_by_source, jnp.full((pad_size,), sampler_cfg.pretuned_L)], axis=0)
        if prefit_step_by_source is not None:
            prefit_step_by_source = jnp.concatenate([prefit_step_by_source, jnp.full((pad_size,), sampler_cfg.pretuned_step_size)], axis=0)
    
    # Reshape to (n_devices, gal_per_device, ...)
    specs_split = specs.reshape(n_devices, gal_per_device, -1)
    weights_split = weights.reshape(n_devices, gal_per_device, -1)
    redshifts_split = redshifts.reshape(n_devices, gal_per_device)
    log_amplitudes_split = log_amplitudes.reshape(n_devices, gal_per_device) if log_amplitudes is not None else None
    prefit_L_split = prefit_L_by_source.reshape(n_devices, gal_per_device) if prefit_L_by_source is not None else None
    prefit_step_split = prefit_step_by_source.reshape(n_devices, gal_per_device) if prefit_step_by_source is not None else None
    
    # Split random keys across devices
    device_keys = jax.random.split(rng_key, n_devices)
    
    # Define the per-device processing function
    def process_on_device(device_specs, device_weights, device_redshifts, device_key, device_log_amplitudes=None,
                          device_prefit_L=None, device_prefit_step=None):
        """
        Process a subset of galaxies on one device.
        This function will be pmapped across devices.
        """
        # Process in smaller batches within this device
        device_batch_size = max(1, batch_size // n_devices)
        
        spec_batches = split_into_batches(device_specs, device_batch_size)
        weight_batches = split_into_batches(device_weights, device_batch_size)
        z_batches = split_into_batches(device_redshifts, device_batch_size)
        # If log_amplitudes not provided, create dummy arrays (will be ignored by prior if not type=3)
        if device_log_amplitudes is not None:
            log_amp_batches = split_into_batches(device_log_amplitudes, device_batch_size)
        else:
            # Create zero-filled arrays matching the batch shapes
            log_amp_batches = [jnp.zeros(spec_batch.shape[0]) for spec_batch in spec_batches]

        if device_prefit_L is not None:
            prefit_L_batches = split_into_batches(device_prefit_L, device_batch_size)
        else:
            prefit_L_batches = [jnp.full((spec_batch.shape[0],), sampler_cfg.pretuned_L) for spec_batch in spec_batches]

        if device_prefit_step is not None:
            prefit_step_batches = split_into_batches(device_prefit_step, device_batch_size)
        else:
            prefit_step_batches = [jnp.full((spec_batch.shape[0],), sampler_cfg.pretuned_step_size) for spec_batch in spec_batches]
        
        all_samples, all_log_prior, all_log_redshift, all_log_L = [], [], [], []
        all_initial_position, all_tuned_L, all_tuned_step_size = [], [], []
        all_mean_logL_per_chain, all_max_logL, all_mcpl = [], [], []
        all_mean_logpx_per_chain = []
        all_preinit_final_logL = []
        all_reinit_best_chain_idx = []
        
        for spec_batch, w_batch, z_batch, log_amp_batch, L_batch, step_batch in zip(
            spec_batches, weight_batches, z_batches, log_amp_batches, prefit_L_batches, prefit_step_batches
        ):
            device_key, subkey = jax.random.split(device_key)
            subkeys = jax.random.split(subkey, num=spec_batch.shape[0])
            
            if sampler_cfg.fix_z:
                samples, log_prior, log_L, initial_pos, tuned_L, tuned_step_size = jax.vmap(
                    lambda key, spec, w, ztrue: pae_spec_sample_fixz_vmap(
                        PAE_obj, spec, w, ztrue, key, sampler_cfg
                    )
                )(subkeys, spec_batch, w_batch, z_batch)
                
                all_log_redshift.append(jnp.zeros((spec_batch.shape[0], sampler_cfg.nchain_per_gal)))
                all_mean_logL_per_chain.append(jnp.zeros((spec_batch.shape[0], sampler_cfg.nchain_per_gal)))
                all_mean_logpx_per_chain.append(jnp.zeros((spec_batch.shape[0], sampler_cfg.nchain_per_gal)))
                all_max_logL.append(jnp.zeros(spec_batch.shape[0]))
                all_mcpl.append(jnp.zeros((spec_batch.shape[0], 200)))
            else:
                # Note: robust_reinit is configured via sampler_cfg and used inside pae_spec_sample_floatz
                # Print statements inside vmapped functions won't show due to GPU execution
                samples, _, log_prior, log_redshift, log_L, initial_pos, tuned_L, tuned_step_size, \
                    mean_logL_per_chain, max_logL_all, z_bins_mcpl, mcpl, preinit_final_logL, mcpl_map_latents, \
                    reinit_best_chain_idx_batch, mean_logpx_per_chain = jax.vmap(
                        lambda key, spec, w, ztrue, log_amp, init_L, init_step: pae_spec_sample_floatz(
                            PAE_obj, spec, w, key, sampler_cfg,
                            batched_log_density=batched_log_density,
                            log_amplitude_data=log_amp,
                            batched_log_likelihood=batched_log_likelihood,
                            init_L=init_L,
                            init_step_size=init_step,
                        )
                    )(subkeys, spec_batch, w_batch, z_batch, log_amp_batch, L_batch, step_batch)
                
                all_log_redshift.append(log_redshift)
                all_mean_logL_per_chain.append(mean_logL_per_chain)
                all_mean_logpx_per_chain.append(mean_logpx_per_chain)
                all_max_logL.append(max_logL_all)
                all_mcpl.append(mcpl)
                all_reinit_best_chain_idx.append(reinit_best_chain_idx_batch)
                # Only append preinit_final_logL if it's not None (init_reinit=True)
                if preinit_final_logL is not None:
                    all_preinit_final_logL.append(preinit_final_logL)
            
            all_samples.append(samples)
            all_log_prior.append(log_prior)
            all_log_L.append(log_L)
            all_initial_position.append(initial_pos)
            all_tuned_L.append(tuned_L)
            all_tuned_step_size.append(tuned_step_size)
            
            # Free batch arrays to reduce device memory accumulation
            del spec_batch, w_batch, z_batch, subkeys
            if not sampler_cfg.fix_z:
                del samples, log_prior, log_redshift, log_L, initial_pos, tuned_L, tuned_step_size
                del mean_logL_per_chain, max_logL_all, mcpl, reinit_best_chain_idx_batch, mean_logpx_per_chain
            else:
                del samples, log_prior, log_L, initial_pos, tuned_L, tuned_step_size
        
        # Concatenate results from all batches on this device
        preinit_final_logL_concat = jnp.concatenate(all_preinit_final_logL, axis=0) if len(all_preinit_final_logL) > 0 else None
        reinit_best_chain_idx_concat = jnp.concatenate(all_reinit_best_chain_idx, axis=0) if len(all_reinit_best_chain_idx) > 0 else None
        
        return (
            jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *all_samples),
            jnp.concatenate(all_log_prior, axis=0),
            jnp.concatenate(all_log_redshift, axis=0),
            jnp.concatenate(all_log_L, axis=0),
            jnp.concatenate(all_initial_position, axis=0),
            jnp.concatenate(all_tuned_L, axis=0),
            jnp.concatenate(all_tuned_step_size, axis=0),
            jnp.concatenate(all_mean_logL_per_chain, axis=0),
            jnp.concatenate(all_max_logL, axis=0),
            jnp.concatenate(all_mcpl, axis=0),
            preinit_final_logL_concat,
            reinit_best_chain_idx_concat,
            jnp.concatenate(all_mean_logpx_per_chain, axis=0)
        )
    
    # Use manual device placement instead of pmap to avoid shape issues
    print(f"Launching parallel processing across {n_devices} devices...")
    
    # Process each device's data separately and gather results
    import concurrent.futures
    
    def process_device_data(device_idx):
        """Process data for one device."""
        device = devices[device_idx]
        device_spec = jax.device_put(specs_split[device_idx], device)
        device_weight = jax.device_put(weights_split[device_idx], device)
        device_redshift = jax.device_put(redshifts_split[device_idx], device)
        device_log_amp = jax.device_put(log_amplitudes_split[device_idx], device) if log_amplitudes_split is not None else None
        device_prefit_L = jax.device_put(prefit_L_split[device_idx], device) if prefit_L_split is not None else None
        device_prefit_step = jax.device_put(prefit_step_split[device_idx], device) if prefit_step_split is not None else None
        device_key = device_keys[device_idx]
        
        return process_on_device(
            device_spec, device_weight, device_redshift, device_key,
            device_log_amp, device_prefit_L, device_prefit_step
        )
    
    # Run on all devices in parallel using threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as executor:
        futures = [executor.submit(process_device_data, i) for i in range(n_devices)]
        device_results = [f.result() for f in futures]
    
    # Move all results to CPU before stacking to avoid device mismatch errors
    print("✓ Parallel processing complete, gathering results...")
    cpu = jax.devices('cpu')[0]
    results = []
    for i in range(len(device_results[0])):
        # Check if all devices returned None for this result (e.g., preinit_final_logL when init_reinit=False)
        if all(r[i] is None for r in device_results):
            results.append(None)
        else:
            # Move each array from all devices to CPU, filtering out None values, then stack
            arrays_on_cpu = [jax.device_put(r[i], cpu) for r in device_results if r[i] is not None]
            if len(arrays_on_cpu) == 0:
                results.append(None)
            else:
                stacked = jnp.stack(arrays_on_cpu, axis=0)
                results.append(stacked)
    results = tuple(results)
    
    # Clear device memory after gathering results
    del device_results
    import gc
    gc.collect()
    
    # Also clear device arrays that were created
    del specs_split, weights_split, redshifts_split
    if log_amplitudes_split is not None:
        del log_amplitudes_split
    if prefit_L_split is not None:
        del prefit_L_split
    if prefit_step_split is not None:
        del prefit_step_split
    gc.collect()
    
    # Unpack and concatenate results from all devices
    all_samples, all_log_prior, all_log_redshift, all_log_L, \
        all_initial_position, all_tuned_L, all_tuned_step_size, \
        all_mean_logL_per_chain, all_max_logL, all_mcpl, all_preinit_final_logL, \
        all_reinit_best_chain_idx, all_mean_logpx_per_chain = results
    
    # Reshape from (n_devices, gal_per_device, ...) to (n_gal_total, ...)
    all_samples = jax.tree_util.tree_map(
        lambda x: x.reshape(-1, *x.shape[2:])[:n_gal],  # Remove padding
        all_samples
    )
    all_log_prior = all_log_prior.reshape(-1, *all_log_prior.shape[2:])[:n_gal]
    all_log_redshift = all_log_redshift.reshape(-1, *all_log_redshift.shape[2:])[:n_gal]
    all_log_L = all_log_L.reshape(-1, *all_log_L.shape[2:])[:n_gal]
    all_initial_position = all_initial_position.reshape(-1, *all_initial_position.shape[2:])[:n_gal]
    all_tuned_L = all_tuned_L.reshape(-1, *all_tuned_L.shape[2:])[:n_gal]
    all_tuned_step_size = all_tuned_step_size.reshape(-1, *all_tuned_step_size.shape[2:])[:n_gal]
    all_mean_logL_per_chain = all_mean_logL_per_chain.reshape(-1, *all_mean_logL_per_chain.shape[2:])[:n_gal]
    all_max_logL = all_max_logL.reshape(-1)[:n_gal]
    all_mcpl = all_mcpl.reshape(-1, *all_mcpl.shape[2:])[:n_gal]
    
    # Handle preinit_final_logL which might be None
    if all_preinit_final_logL is not None:
        all_preinit_final_logL = all_preinit_final_logL.reshape(-1, *all_preinit_final_logL.shape[2:])[:n_gal]
    if all_reinit_best_chain_idx is not None:
        all_reinit_best_chain_idx = all_reinit_best_chain_idx.reshape(-1)[:n_gal]
    if all_mean_logpx_per_chain is not None:
        all_mean_logpx_per_chain = all_mean_logpx_per_chain.reshape(-1, *all_mean_logpx_per_chain.shape[2:])[:n_gal]
    
    print(f"✓ Results gathered from all {n_devices} devices")
    
    # Convert all to host numpy to free device memory before returning
    def to_host_numpy(x):
        if x is None:
            return None
        if hasattr(x, 'device'):  # JAX array
            return np.array(jax.device_get(x))
        return x
    
    all_samples = jax.tree_util.tree_map(to_host_numpy, all_samples)
    all_log_prior = to_host_numpy(all_log_prior)
    all_log_redshift = to_host_numpy(all_log_redshift)
    all_log_L = to_host_numpy(all_log_L)
    all_initial_position = to_host_numpy(all_initial_position)
    all_tuned_L = to_host_numpy(all_tuned_L)
    all_tuned_step_size = to_host_numpy(all_tuned_step_size)
    all_mean_logL_per_chain = to_host_numpy(all_mean_logL_per_chain)
    all_max_logL = to_host_numpy(all_max_logL)
    all_mcpl = to_host_numpy(all_mcpl)
    all_preinit_final_logL = to_host_numpy(all_preinit_final_logL)
    all_reinit_best_chain_idx = to_host_numpy(all_reinit_best_chain_idx)
    all_mean_logpx_per_chain = to_host_numpy(all_mean_logpx_per_chain)
    
    # Explicitly delete device-backed results tuple
    del results
    gc.collect()
    
    if sampler_cfg.fix_z:
        return (
            all_samples, None, all_log_prior, None,
            all_log_L, all_initial_position,
            all_tuned_L, all_tuned_step_size,
            None,  # all_mean_logL_per_chain
            None,  # all_max_logL
            None,  # z_bins_mcpl
            None,  # all_mcpl
            None,  # preinit_final_logL
            None,  # all_mcpl_map_latents (not supported in multicore mode)
            None,  # reinit_best_chain_idx
            None,  # all_mean_logpx_per_chain
        )
    else:
        # Create dummy z_bins_mcpl (same for all galaxies)
        z_bins_mcpl = jnp.linspace(sampler_cfg.zmin, sampler_cfg.zmax, 200)
        
        return (
            all_samples, None, all_log_prior, all_log_redshift,
            all_log_L, all_initial_position,
            all_tuned_L, all_tuned_step_size,
            all_mean_logL_per_chain, all_max_logL,
            z_bins_mcpl, all_mcpl, all_preinit_final_logL,
            None,  # all_mcpl_map_latents (not supported in multicore mode)
            all_reinit_best_chain_idx,
            all_mean_logpx_per_chain,
        )


def run_batched_sampler(
    PAE_obj,
    specs,
    weights,
    redshifts,
    batch_size: int,
    rng_key,
    sampler_cfg: MCLMCSamplingConfig,
    log_amplitudes=None,
    prefit_L_by_source=None,
    prefit_step_by_source=None,
):
    """
    Run MCMC sampling on a batch of galaxies.
    
    If use_multicore=True, distributes work across multiple GPU cores using pmap.
    Otherwise runs sequentially on a single core.
    """
    assert rng_key is not None, "You must provide a JAX random key."

    # ============ MULTI-CORE PARALLELIZATION ============
    if sampler_cfg.use_multicore:
        return run_batched_sampler_multicore(
            PAE_obj, specs, weights, redshifts, batch_size, rng_key, sampler_cfg,
            log_amplitudes,
            prefit_L_by_source=prefit_L_by_source,
            prefit_step_by_source=prefit_step_by_source,
        )
    # ====================================================

    # ============ OPTIONAL: CREATE BATCHED LOGDENSITY ONCE ============
    batched_log_density = None
    batched_log_density_burnin = None
    batched_log_likelihood = None
    if sampler_cfg.use_batched_logdensity and not sampler_cfg.fix_z:
        print("=" * 60)
        print("Creating single batched logdensity function for all galaxies...")
        batched_log_density = make_batched_logdensity_fn(PAE_obj, sampler_cfg)
        batched_log_likelihood = make_batched_loglikelihood_fn(PAE_obj, sampler_cfg)
        
        # Pre-compile by evaluating once with dummy data
        print("Pre-compiling logdensity function...")
        n_latent = PAE_obj.params['nlatent']
        n_bands = specs.shape[1]
        dummy_latent = jnp.zeros(n_latent + 1)
        dummy_spec = jnp.ones(n_bands)
        dummy_weight = jnp.ones(n_bands)
        _ = batched_log_density(dummy_latent, dummy_spec, dummy_weight, None, None)
        _ = batched_log_likelihood(dummy_latent, dummy_spec, dummy_weight, None, None)
        print("✓ Logdensity compiled and ready (will be reused for all galaxies)")

        # If burn-in uses a different nf_alpha, pre-compile a separate burn-in function
        if sampler_cfg.nf_alpha_burnin >= 0:
            print(f"Creating separate burn-in logdensity (nf_alpha_burnin={sampler_cfg.nf_alpha_burnin})...")
            burnin_cfg = replace(sampler_cfg, nf_alpha=sampler_cfg.nf_alpha_burnin)
            batched_log_density_burnin = make_batched_logdensity_fn(PAE_obj, burnin_cfg)
            _ = batched_log_density_burnin(dummy_latent, dummy_spec, dummy_weight, None, None)
            print("✓ Burn-in logdensity compiled")
        print("=" * 60)
    elif not sampler_cfg.use_batched_logdensity:
        print("Using per-galaxy logdensity functions (old behavior)")
    # ===================================================================

    spec_batches = split_into_batches(specs, batch_size)
    weight_batches = split_into_batches(weights, batch_size)
    z_batches = split_into_batches(redshifts, batch_size)
    log_amp_batches = split_into_batches(log_amplitudes, batch_size) if log_amplitudes is not None else [None] * len(spec_batches)
    prefit_L_batches = split_into_batches(prefit_L_by_source, batch_size) if prefit_L_by_source is not None else [None] * len(spec_batches)
    prefit_step_batches = split_into_batches(prefit_step_by_source, batch_size) if prefit_step_by_source is not None else [None] * len(spec_batches)

    nbatch = len(spec_batches)
    
    all_samples, all_ae_redshifts, all_log_prior, all_log_redshift, all_log_L, all_initial_position = [[] for x in range(6)]
    all_tuned_L, all_tuned_step_size = [], []
    all_mean_logL_per_chain, all_max_logL, all_mcpl = [], [], []
    all_mean_logpx_per_chain = []
    all_preinit_final_logL = []
    all_mcpl_map_latents = []
    all_reinit_best_chain = []
    
    for batch_idx, (spec_batch, w_batch, z_batch, log_amp_batch, L_batch, step_batch) in enumerate(
        zip(spec_batches, weight_batches, z_batches, log_amp_batches, prefit_L_batches, prefit_step_batches),
        start=1,
    ):
        
        print(f"On batch {batch_idx} of {nbatch}..")
        rng_key, subkey = jax.random.split(rng_key)
        subkeys = jax.random.split(subkey, num=spec_batch.shape[0])

        if sampler_cfg.fix_z:
            print('Fixed redshifts..')
            samples, log_prior, log_L,\
                initial_pos, tuned_L, tuned_step_size = jax.vmap(lambda key, spec, w, ztrue: 
                                                                pae_spec_sample_fixz_vmap(PAE_obj, spec, w, ztrue, key, sampler_cfg))(subkeys, spec_batch, w_batch, z_batch)
            

        else:
            
            samples, _, log_prior, log_redshift,\
                log_L, initial_pos, tuned_L, tuned_step_size, \
                    mean_logL_per_chain, max_logL_all, z_bins_mcpl, mcpl, \
                    preinit_final_logL, mcpl_map_latents, reinit_best_chain_idx, \
                    mean_logpx_per_chain = jax.vmap(
                        lambda key, spec, w, ztrue, log_amp, init_L, init_step: pae_spec_sample_floatz(
                            PAE_obj, spec, w, key, sampler_cfg,
                            batched_log_density=batched_log_density,
                            log_amplitude_data=log_amp,
                            batched_log_density_burnin=batched_log_density_burnin,
                            batched_log_likelihood=batched_log_likelihood,
                            init_L=init_L,
                            init_step_size=init_step,
                        )
                    )(
                        subkeys,
                        spec_batch,
                        w_batch,
                        z_batch,
                        log_amp_batch,
                        jnp.full((spec_batch.shape[0],), sampler_cfg.pretuned_L) if L_batch is None else L_batch,
                        jnp.full((spec_batch.shape[0],), sampler_cfg.pretuned_step_size) if step_batch is None else step_batch,
                    )

            all_log_redshift.append(log_redshift)

            all_mean_logL_per_chain.append(mean_logL_per_chain)
            all_mean_logpx_per_chain.append(mean_logpx_per_chain)
            all_max_logL.append(max_logL_all)

            all_mcpl.append(mcpl)
            all_mcpl_map_latents.append(mcpl_map_latents)
            
            all_preinit_final_logL.append(preinit_final_logL)
            all_reinit_best_chain.append(reinit_best_chain_idx)
            
            # if sampler_cfg.redshift_in_flow:
            #     all_ae_redshifts.append(ae_redshifts)

        all_log_prior.append(log_prior)
        all_log_L.append(log_L)
        all_initial_position.append(initial_pos)
        all_samples.append(samples)
        all_tuned_L.append(tuned_L)
        all_tuned_step_size.append(tuned_step_size)
        
        # Free batch arrays to reduce memory accumulation
        del spec_batch, w_batch, z_batch, subkeys
        if not sampler_cfg.fix_z:
            del samples, log_prior, log_redshift, log_L, initial_pos, tuned_L, tuned_step_size
            del mean_logL_per_chain, max_logL_all, mcpl, mcpl_map_latents, reinit_best_chain_idx, mean_logpx_per_chain
        else:
            del samples, log_prior, log_L, initial_pos, tuned_L, tuned_step_size


    if sampler_cfg.fix_z:
        return (
            jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *all_samples),
            None,
            jnp.concatenate(all_log_prior, axis=0),
            None,
            jnp.concatenate(all_log_L, axis=0), 
            jnp.concatenate(all_initial_position, axis=0), 
            jnp.concatenate(all_tuned_L, axis=0),
            jnp.concatenate(all_tuned_step_size, axis=0),
            None,  # all_mean_logL_per_chain
            None,  # all_max_logL
            None,  # z_bins_mcpl
            None,  # all_mcpl
            None,  # preinit_final_logL
            None,  # all_mcpl_map_latents
            None,  # reinit_best_chain_idx
            None,  # all_mean_logpx_per_chain
        )

    else:
        # print('
        # if sampler_cfg.redshift_in_flow:
        #     ae_concat = jnp.concatenate(all_ae_redshifts, axis=0)
        # else:
        #     ae_concat = None
        # ae_concat = None
        
        # Concatenate pre-init diagnostics if available
        if len(all_preinit_final_logL) > 0:
            preinit_final_logL_concat = jnp.concatenate(all_preinit_final_logL, axis=0)
        else:
            preinit_final_logL_concat = None
        
        return (
    
            jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=0), *all_samples),
            None,
            jnp.concatenate(all_log_prior, axis=0),
            jnp.concatenate(all_log_redshift, axis=0),
            jnp.concatenate(all_log_L, axis=0), 
            jnp.concatenate(all_initial_position, axis=0), 
            jnp.concatenate(all_tuned_L, axis=0),
            jnp.concatenate(all_tuned_step_size, axis=0),
            jnp.concatenate(all_mean_logL_per_chain, axis=0),
            jnp.concatenate(all_max_logL, axis=0),
            z_bins_mcpl,
            jnp.concatenate(all_mcpl, axis=0),
            preinit_final_logL_concat,
            jnp.concatenate(all_mcpl_map_latents, axis=0),
            jnp.concatenate(all_reinit_best_chain, axis=0),
            jnp.concatenate(all_mean_logpx_per_chain, axis=0),
        )
