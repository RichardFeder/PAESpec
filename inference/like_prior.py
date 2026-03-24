import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import vmap
import jaxopt

import numpy as np
from scipy.stats import truncnorm, rv_continuous
import time

# from scipy.stats import multivariate_normal, truncnorm, rv_continuous


from functools import partial

# -------------- log prior functions ------------------

def bpz_prior(z, m, alpha=2.0, beta=1.5, z0=0.7, k_z=0.05, m0=20.0):
    z_m = z0 + k_z * (m - m0)
    p_unnorm = z**alpha * np.exp(-(z / z_m)**beta)
    # Normalize to integrate to 1 over z range
    norm_factor = np.trapz(p_unnorm, z)
    return p_unnorm / norm_factor

def redshift_trunc_prior(z0=0.65, sigma=0.4, zmax=3.0):
    z_prior = truncnorm((0 - z0) / sigma, (zmax - z0) / sigma, loc=z0, scale=sigma)
    return z_prior
    
def soft_ball_log_prior(u, R=5.0, k=4):
    norm_sq = jnp.sum(u**2, axis=-1)
    return -0.5 * (norm_sq / R**2) ** (k / 2)

def make_uniform_ball_prior(center, radius):
    """
    Create a prior uniform inside an N-dimensional ball of given radius.

    Parameters
    ----------
    center : array_like
        Center of the ball (shape: [D,]).
    radius : float
        Radius of the ball.

    Returns
    -------
    prior : rv_continuous
        A scipy.stats-compatible object with .logpdf() and .pdf().
    """
    center = np.asarray(center)
    dim = center.size
    volume = (np.pi ** (dim / 2)) / np.math.gamma(dim / 2 + 1) * radius ** dim

    class UniformBall(rv_continuous):
        def __init__(self, center, radius):
            super().__init__()
            self.center = center
            self.radius = radius
            self.log_volume = np.log(volume)

        def _logpdf(self, x):
            x = np.atleast_2d(x)
            dist = np.linalg.norm(x - self.center, axis=1)
            in_ball = dist <= self.radius
            logp = np.where(in_ball, -self.log_volume, -np.inf)
            return logp if len(logp) > 1 else logp[0]

        def _pdf(self, x):
            return np.exp(self._logpdf(x))

    return UniformBall(center=center, radius=radius)

# ------------------------------------ log-density functions ---------------------------------

def logdensity_fn_fixz(u, push_fn, x_obs, w, redshift, nf_alpha=1.0, return_indiv=False):
    """
    Log-density function for BlackJAX HMC.
    
    latent: tuple (z, z_redshift)
    params: model parameters
    flow: normalizing flow model
    decoder: decoder model
    x_obs: observed data
    """    

    u = u.reshape(1, -1)

    # Normalizing flow prior (Multivariate Gaussian)
    log_pz = -0.5 * jnp.sum(u**2) - 0.5 * u.shape[-1] * jnp.log(2 * jnp.pi)

    # print('redshift:', redshift)
    # Compute likelihood p(x | u, z_redshift)
    recon_x = push_fn(u, redshift)

    # print('recon:', recon_x)
    log_px_given_z = -0.5 * jnp.sum(w*(x_obs - recon_x) ** 2)

    if return_indiv:
        return (nf_alpha * log_pz), log_px_given_z
    else:
        return (nf_alpha * log_pz) + log_px_given_z

def make_logdensity_fn_optzprior(PAE_obj, x_obs, w, z_min, z_max, nf_alpha=1.0, sharpness=100.0, redshift_in_flow=False, \
                           z0_prior=0.65, sigma_prior=0.4, include_gaussian_prior=False, 
                                redshift_prior_type=0, m_mag=None):
    """
    Factory function that creates and JIT-compiles the log-density function.
    This "freezes" the PAE object and its parameters, avoiding recompilation.
    """
    
    # These functions are now "closed over" by the log_density function below
    push_spec_marg = PAE_obj.push_spec_marg
    
    @jax.jit
    def log_density(latent):
        """
        Log-density function for BlackJAX HMC.
        """
        latent = latent[None, :] if latent.ndim == 1 else latent
        
        if redshift_in_flow:
            u = latent
            z_redshift = None
        else:
            u = latent[:, :-1]
            z_redshift = latent[:, -1]

        # Normalizing flow prior (Multivariate Gaussian), gets rescaled by nf_alpha
        log_pz = -0.5 * jnp.sum(u**2) - 0.5 * u.shape[-1] * jnp.log(2 * jnp.pi)
        
        # Compute decoder likelihood p(x | z, z_redshift)
        _, log_px_given_z, z_redshift_out = push_spec_marg(
            u, z_redshift,
            observed_flux=x_obs,
            weight=w,
            marginalize_amplitude=True,
            return_rescaled_flux_and_loglike=True,
            redshift_in_flow=redshift_in_flow)

        # roll-off prior at boundaries
        below_penalty = -sharpness * jnp.maximum(0., z_min - z_redshift)
        above_penalty = -sharpness * jnp.maximum(0., z_redshift - z_max)
        log_pz_rolloff = below_penalty + above_penalty

        # --- Optionally Active: Gaussian Core Prior ---
        def calculate_gaussian_core(z, m):
            return norm.logpdf(z, loc=z0_prior, scale=sigma_prior)

        def calculate_bpz_prior(z, m):
            # BPZ-type prior: p(z) ~ z^2 * exp(-z/z0)
            # z0 is stored in sigma_prior parameter
            # Log-prior (unnormalized): 2*log(z) - z/z0
            z0 = sigma_prior
            log_p = 2.0 * jnp.log(z + 1e-10) - z / z0  # Add small constant to avoid log(0)
            return log_p

        def no_gaussian_core(z, m):
            return jnp.zeros_like(z)

        log_pz_prior = jax.lax.switch(
                    redshift_prior_type,
                    (no_gaussian_core, calculate_gaussian_core, calculate_bpz_prior),
                    z_redshift, m_mag
                )

        log_pz_redshift = jnp.sum(log_pz_rolloff) + jnp.sum(log_pz_prior)

        # log_pz_gaussian_core = jax.lax.cond(
        #     include_gaussian_prior,
        #     calculate_gaussian_core,
        #     no_gaussian_core,
        #     z_redshift
        # )

        # log_pz_redshift = jnp.sum(log_pz_rolloff) + jnp.sum(log_pz_gaussian_core)
        
        return (nf_alpha * log_pz) + log_pz_redshift + jnp.sum(log_px_given_z)

    return log_density


def make_batched_logdensity_fn(PAE_obj, sampler_cfg):
    """
    Create a single JIT-compiled logdensity function that takes data as parameters.
    
    This function is created ONCE for all galaxies, avoiding repeated JIT compilation.
    The key difference from make_logdensity_fn_optzprior is that x_obs and weight
    are parameters rather than being closed over, allowing the same compiled function
    to be reused across all galaxies.
    
    Args:
        PAE_obj: The PAE model object (closed over)
        sampler_cfg: MCLMCSamplingConfig with prior settings (closed over)
    
    Returns:
        A JIT-compiled function: log_density(latent, x_obs, weight, log_amplitude_data) -> log_prob
    """
    # Extract settings from config (these get closed over)
    z_min = sampler_cfg.zmin
    z_max = sampler_cfg.zmax
    nf_alpha = sampler_cfg.nf_alpha
    sharpness = 100.0
    redshift_in_flow = sampler_cfg.redshift_in_flow
    z0_prior = sampler_cfg.z0_prior
    sigma_prior = sampler_cfg.sigma_prior
    redshift_prior_type = sampler_cfg.redshift_prior_type
    # For amplitude-dependent BPZ (type=3)
    z0_amp_slope = sampler_cfg.z0_amp_slope
    z0_amp_intercept = sampler_cfg.z0_amp_intercept
    alpha_amp = sampler_cfg.alpha_amp
    # For fixed BPZ prior (type=2): use alpha_prior, beta_prior, and sigma_prior (as z0)
    alpha_prior = sampler_cfg.alpha_prior
    beta_prior = sampler_cfg.beta_prior
    
    # Log-redshift sampling mode: sample u=ln(z) instead of z
    sample_log_redshift = sampler_cfg.sample_log_redshift
    
    # Close over PAE object methods, but NOT the data
    push_spec_marg = PAE_obj.push_spec_marg
    
    @jax.jit
    def log_density(latent, x_obs, weight, m_mag=None, log_amplitude_data=None):
        """
        Batched log-density function that takes data as parameters.
        
        Args:
            latent: (n_latent + 1,) array for single chain
                    Last element is z (default) or ln(z) (if sample_log_redshift=True)
            x_obs: (n_bands,) observed fluxes for ONE galaxy
            weight: (n_bands,) inverse variance weights for ONE galaxy
            m_mag: Optional magnitude (for BPZ prior, currently unused)
            log_amplitude_data: Optional log10(amplitude) for amplitude-dependent BPZ prior
        
        Returns:
            Scalar log probability
        """
        # Ensure latent is 2D for consistency
        latent = latent[None, :] if latent.ndim == 1 else latent
        
        if redshift_in_flow:
            u = latent
            z_redshift = None
        else:
            u = latent[:, :-1]
            if sample_log_redshift:
                # Sampler operates in ln(z) space: extract ln(z), compute z = exp(ln_z)
                ln_z = latent[:, -1]
                z_redshift = jnp.exp(ln_z)
            else:
                z_redshift = latent[:, -1]
        
        # Normalizing flow prior (standard Gaussian)
        log_pz = -0.5 * jnp.sum(u**2) - 0.5 * u.shape[-1] * jnp.log(2 * jnp.pi)
        
        # Likelihood: p(x_obs | latent, redshift)
        # Note: x_obs and weight are now PARAMETERS, not closed over
        # Always pass the actual z value (not ln(z)) to the PAE model
        _, log_px_given_z, z_redshift_out = push_spec_marg(
            u, z_redshift,
            observed_flux=x_obs[None, :],  # Add batch dim for consistency
            weight=weight[None, :],        # Add batch dim for consistency
            marginalize_amplitude=True,
            return_rescaled_flux_and_loglike=True,
            redshift_in_flow=redshift_in_flow
        )
        
        # Redshift boundary roll-off
        if sample_log_redshift:
            # Boundaries in ln(z) space
            ln_z_min = jnp.log(jnp.maximum(z_min, 1e-6))
            ln_z_max = jnp.log(z_max)
            below_penalty = -sharpness * jnp.maximum(0., ln_z_min - ln_z)
            above_penalty = -sharpness * jnp.maximum(0., ln_z - ln_z_max)
        else:
            below_penalty = -sharpness * jnp.maximum(0., z_min - z_redshift)
            above_penalty = -sharpness * jnp.maximum(0., z_redshift - z_max)
        log_pz_rolloff = below_penalty + above_penalty
        
        # Optional redshift priors — always evaluated at actual z
        # When sample_log_redshift=True, the Jacobian |dz/du|=z=exp(ln_z)
        # is added separately below.
        def no_prior(z, log_amp):
            return jnp.zeros_like(z)
        
        def calculate_gaussian_prior(z, log_amp):
            return norm.logpdf(z, loc=z0_prior, scale=sigma_prior)
        
        def calculate_bpz_prior(z, log_amp):
            # Fixed BPZ prior: p(z) ~ z^alpha * exp(-(z/z0)^beta)
            # alpha from alpha_prior, beta from beta_prior, z0 from sigma_prior
            z0 = sigma_prior
            log_p = alpha_prior * jnp.log(z + 1e-10) - (z / z0)**beta_prior
            return log_p
        
        def calculate_amplitude_dependent_bpz_prior(z, log_amp):
            # Amplitude-dependent BPZ prior: z0 = slope * log10_A + intercept
            # Safety: ensure log_amp is valid (not NaN, not inf)
            log_amp_safe = jnp.where(jnp.isfinite(log_amp), log_amp, 1.0)  # Use log10(10)=1 as fallback
            z0 = z0_amp_slope * log_amp_safe + z0_amp_intercept
            z0 = jnp.maximum(z0, 0.01)  # Ensure positive z0
            log_p = alpha_amp * jnp.log(z + 1e-10) - z / z0
            return log_p
        
        log_amplitude_val = log_amplitude_data if log_amplitude_data is not None else 1.0  # Default to log10(10)=1
        
        log_pz_prior = jax.lax.switch(
            redshift_prior_type,
            (no_prior, calculate_gaussian_prior, calculate_bpz_prior, calculate_amplitude_dependent_bpz_prior),
            z_redshift, log_amplitude_val
        )
        
        # Jacobian for change of variables: log |dz/d(ln_z)| = ln_z
        # This ensures the posterior in z-space is unchanged.
        if sample_log_redshift:
            log_jacobian = jnp.sum(ln_z)
        else:
            log_jacobian = 0.0
        
        log_pz_redshift = jnp.sum(log_pz_rolloff) + jnp.sum(log_pz_prior) + log_jacobian
        
        return (nf_alpha * log_pz) + log_pz_redshift + jnp.sum(log_px_given_z)
    
    return log_density


def make_batched_loglikelihood_fn(PAE_obj, sampler_cfg):
    """
    Create a JIT-compiled function returning only the photometric log-likelihood.

    Same call signature as make_batched_logdensity_fn, but returns only
    jnp.sum(log_px_given_z) — the data term with all prior contributions omitted.
    Used in the afterburner to track pure photometric chi-squared
    (chi2_phot = -2 * mean(log_px_given_z)) independently of the NF / redshift priors.

    Args:
        PAE_obj: The PAE model object (closed over)
        sampler_cfg: MCLMCSamplingConfig (only redshift/amplitude settings used)

    Returns:
        A JIT-compiled function:
            log_likelihood(latent, x_obs, weight, m_mag=None, log_amplitude_data=None)
            -> scalar photometric log-likelihood
    """
    redshift_in_flow = sampler_cfg.redshift_in_flow
    sample_log_redshift = sampler_cfg.sample_log_redshift
    push_spec_marg = PAE_obj.push_spec_marg

    @jax.jit
    def log_likelihood(latent, x_obs, weight, m_mag=None, log_amplitude_data=None):
        """Return sum(log p(x_obs | latent, z)) with no prior terms."""
        latent = latent[None, :] if latent.ndim == 1 else latent

        if redshift_in_flow:
            u = latent
            z_redshift = None
        else:
            u = latent[:, :-1]
            if sample_log_redshift:
                ln_z = latent[:, -1]
                z_redshift = jnp.exp(ln_z)
            else:
                z_redshift = latent[:, -1]

        _, log_px_given_z, _ = push_spec_marg(
            u, z_redshift,
            observed_flux=x_obs[None, :],
            weight=weight[None, :],
            marginalize_amplitude=True,
            return_rescaled_flux_and_loglike=True,
            redshift_in_flow=redshift_in_flow
        )
        return jnp.sum(log_px_given_z)

    return log_likelihood


def make_batched_logdensity_fn_native(PAE_obj, sampler_cfg, lam_interp):
    """
    Create a JIT-compiled logdensity function for native filters.
    
    Similar to make_batched_logdensity_fn, but accepts per-source filter_curves
    as a parameter. The filter_curves must have consistent shape (max_nbands, n_lam_interp)
    across all sources (padded to maximum), so JIT compilation works.
    
    If sampler_cfg.sample_log_amplitude is True, the latent vector is expected to have
    an additional dimension for log(amplitude), i.e., [u1, ..., uN, z, log_A].
    This enforces positive amplitudes and can reduce overfitting at high redshift.
    
    Args:
        PAE_obj: The PAE model object (closed over)
        sampler_cfg: MCLMCSamplingConfig with prior settings (closed over)
        lam_interp: Wavelength grid for filter interpolation (closed over)
    
    Returns:
        A JIT-compiled function: log_density(latent, x_obs, weight, filter_curves) -> log_prob
    """
    # Extract settings from config (these get closed over)
    z_min = sampler_cfg.zmin
    z_max = sampler_cfg.zmax
    nf_alpha = sampler_cfg.nf_alpha
    sharpness = 100.0
    redshift_in_flow = sampler_cfg.redshift_in_flow
    z0_prior = sampler_cfg.z0_prior
    sigma_prior = sampler_cfg.sigma_prior
    redshift_prior_type = sampler_cfg.redshift_prior_type
    
    # Amplitude-dependent BPZ prior parameters (type=3)
    z0_amp_slope = sampler_cfg.z0_amp_slope
    z0_amp_intercept = sampler_cfg.z0_amp_intercept
    alpha_amp = sampler_cfg.alpha_amp
    
    # For fixed BPZ prior (type=2): use alpha_prior and beta_prior
    alpha_prior = sampler_cfg.alpha_prior
    beta_prior = sampler_cfg.beta_prior
    
    # Amplitude handling
    sample_log_amplitude = sampler_cfg.sample_log_amplitude
    log_amplitude_prior_std = sampler_cfg.log_amplitude_prior_std
    
    # Log-redshift sampling mode
    sample_log_redshift = sampler_cfg.sample_log_redshift
    
    # Close over PAE object methods and lam_interp
    push_spec_marg = PAE_obj.push_spec_marg
    
    @jax.jit
    def log_density(latent, x_obs, weight, filter_curves, log_amplitude_data=None, m_mag=None):
        """
        Batched log-density function with native filters.
        
        Args:
            latent: (n_latent + 1,) or (n_latent + 2,) array for single chain
                    If sample_log_amplitude=True, last dim is log(amplitude)
                    If sample_log_redshift=True, the redshift dimension is ln(z)
            x_obs: (n_bands,) observed fluxes for ONE galaxy
            weight: (n_bands,) inverse variance weights for ONE galaxy
            filter_curves: (n_bands, n_lam_interp) per-source filter matrix
            log_amplitude_data: log10 of weighted mean flux (for amplitude-dependent prior type=3)
            m_mag: Optional magnitude (for BPZ prior, currently unused)
        
        Returns:
            Scalar log probability
        """
        # Ensure latent is 2D for consistency
        latent = latent[None, :] if latent.ndim == 1 else latent
        
        # Extract components based on structure:
        # - When sample_log_amplitude=False: [u1...uN, z_or_lnz] (original behavior)
        # - When sample_log_amplitude=True: [u1...uN, log_A, z_or_lnz]
        if sample_log_amplitude:
            # Structure: [u1...uN, log_A, z_or_lnz]
            log_amplitude = latent[:, -2]
            u = latent[:, :-2]
            raw_z = latent[:, -1]
        else:
            log_amplitude = None
            u = latent[:, :-1] if not redshift_in_flow else latent
            raw_z = latent[:, -1] if not redshift_in_flow else None
        
        # Convert from ln(z) to z if in log-redshift mode
        if sample_log_redshift and raw_z is not None:
            ln_z = raw_z
            z_redshift = jnp.exp(ln_z)
        else:
            z_redshift = raw_z
        
        # Normalizing flow prior (standard Gaussian)
        log_pz = -0.5 * jnp.sum(u**2) - 0.5 * u.shape[-1] * jnp.log(2 * jnp.pi)
        
        # Likelihood with native filters (always uses actual z)
        _, log_px_given_z, z_redshift_out = push_spec_marg(
            u, z_redshift,
            observed_flux=x_obs[None, :],
            weight=weight[None, :],
            marginalize_amplitude=(not sample_log_amplitude),
            return_rescaled_flux_and_loglike=True,
            redshift_in_flow=redshift_in_flow,
            filter_curves=filter_curves[None, :, :],  # Add batch dim
            lam_interp=lam_interp,
            log_amplitude=log_amplitude
        )
        
        # Redshift boundary roll-off
        if sample_log_redshift and raw_z is not None:
            ln_z_min = jnp.log(jnp.maximum(z_min, 1e-6))
            ln_z_max = jnp.log(z_max)
            below_penalty = -sharpness * jnp.maximum(0., ln_z_min - ln_z)
            above_penalty = -sharpness * jnp.maximum(0., ln_z - ln_z_max)
        else:
            below_penalty = -sharpness * jnp.maximum(0., z_min - z_redshift)
            above_penalty = -sharpness * jnp.maximum(0., z_redshift - z_max)
        log_pz_rolloff = below_penalty + above_penalty
        
        # Optional redshift priors — evaluated at actual z
        def calculate_gaussian_core(z, log_amp_data):
            return norm.logpdf(z, loc=z0_prior, scale=sigma_prior)
        
        def calculate_bpz_prior(z, log_amp_data):
            z0 = sigma_prior
            log_p = alpha_prior * jnp.log(z + 1e-10) - (z / z0)**beta_prior
            return log_p
        
        def calculate_amplitude_dependent_bpz_prior(z, log_amp_data):
            log_amp_safe = jnp.where(jnp.isfinite(log_amp_data), log_amp_data, 1.0)
            z0 = z0_amp_slope * log_amp_safe + z0_amp_intercept
            z0 = jnp.maximum(z0, 0.01)
            log_p = alpha_amp * jnp.log(z + 1e-10) - z / z0
            return log_p
        
        def no_gaussian_core(z, log_amp_data):
            return jnp.zeros_like(z)
        
        log_pz_prior = jax.lax.switch(
            redshift_prior_type,
            (no_gaussian_core, calculate_gaussian_core, calculate_bpz_prior, calculate_amplitude_dependent_bpz_prior),
            z_redshift, log_amplitude_data
        )
        
        # Jacobian for change of variables: log |dz/d(ln_z)| = ln_z
        if sample_log_redshift and raw_z is not None:
            log_jacobian = jnp.sum(ln_z)
        else:
            log_jacobian = 0.0
        
        log_pz_redshift = jnp.sum(log_pz_rolloff) + jnp.sum(log_pz_prior) + log_jacobian
        
        # Log-amplitude prior if sampling it (Gaussian centered at 0)
        log_p_amplitude = jnp.where(
            sample_log_amplitude,
            -0.5 * (log_amplitude**2) / (log_amplitude_prior_std**2) - 0.5 * jnp.log(2 * jnp.pi * log_amplitude_prior_std**2),
            0.0
        )
        
        return (nf_alpha * log_pz) + log_pz_redshift + jnp.sum(log_px_given_z) + jnp.sum(log_p_amplitude)
    
    return log_density

# -------------- log density functions ------------------
def logdensity_fn_marg_pmc(latent, PAE_obj, x_obs, w, z_min, z_max, nf_alpha=1.0, sharpness=100.0, \
                      return_recon=False, flow_prior=None, logp_threshold=None, redshift_in_flow=False):
    """
    Log-density function for pocoMC.
    
    latent: tuple (z, z_redshift)
    params: model parameters
    flow: normalizing flow model
    decoder: decoder model
    x_obs: observed data
    """    

    # Ensure latent is 2D
    latent = latent[None, :] if latent.ndim == 1 else latent
    
    if redshift_in_flow:
        u = latent
        z_redshift = None
    else:
        u = latent[:, :-1]
        z_redshift = latent[:, -1]

    recon_x, log_px_given_z, z_redshift = PAE_obj.push_spec_marg(
        u, z_redshift,
        observed_flux=x_obs,
        weight=w,
        marginalize_amplitude=True,
        return_rescaled_flux_and_loglike=True,
        redshift_in_flow=redshift_in_flow)

    if flow_prior is not None:
        logp_u = flow_prior.logpdf(u) 
        logpmask = jnp.where((logp_u < logp_threshold))[0]
        log_px_given_z = log_px_given_z.at[logpmask].set(-1e10)

    if return_recon:        
        return log_px_given_z, recon_x
    else:
        return log_px_given_z



def neg_log_posterior_theta(theta, z, PAE_obj, x_obs, weight, nf_alpha):
    theta_batch = theta[None, :] 
    z_batch = jnp.array([z])

    _, log_px_given_z, _ = PAE_obj.push_spec_marg(
        theta_batch, z_batch, observed_flux=x_obs, weight=weight,
        marginalize_amplitude=True, return_rescaled_flux_and_loglike=True, redshift_in_flow=False
    )
    neg_log_likelihood = -jnp.sum(log_px_given_z)

    log_prior_theta = -0.5 * jnp.sum(theta**2) - 0.5 * theta.shape[-1] * jnp.log(2 * jnp.pi)
    neg_log_prior = -log_prior_theta

    neg_log_posterior = neg_log_likelihood + nf_alpha*neg_log_prior
    
    return neg_log_posterior

def neg_log_posterior_theta_full_args(theta, z, PAE_obj, x_obs, weight):
    """
    Calculates the negative log-posterior for a given theta and a fixed z.
    theta: (n_latent,)
    z: scalar
    x_obs: (N_WAVELENGTH_BINS,)
    weight: (N_WAVELENGTH_BINS,)
    """
    # 1. Calculate negative log-likelihood (-log p(x | theta, z))
    theta_batch = theta[None, :] 
    z_batch = jnp.array([z])

    _, log_px_given_z, _ = PAE_obj.push_spec_marg(
        theta_batch, z_batch, observed_flux=x_obs, weight=weight,
        marginalize_amplitude=True, return_rescaled_flux_and_loglike=True, redshift_in_flow=False
    )
    neg_log_likelihood = -jnp.sum(log_px_given_z)

    # # 2. Calculate negative log-prior (-log p(theta | z))
    log_prior_theta = -0.5 * jnp.sum(theta**2) - 0.5 * theta.shape[-1] * jnp.log(2 * jnp.pi)
    neg_log_prior = -log_prior_theta
    
    neg_log_posterior = neg_log_likelihood + neg_log_prior
    
    return neg_log_posterior

def neg_log_likelihood_theta_full_args(theta, z, PAE_obj, x_obs, weight):
    """
    Calculates the negative log-posterior for a given theta and a fixed z.
    theta: (n_latent,)
    z: scalar
    x_obs: (N_WAVELENGTH_BINS,)
    weight: (N_WAVELENGTH_BINS,)
    """
    # 1. Calculate negative log-likelihood (-log p(x | theta, z))
    theta_batch = theta[None, :] 
    z_batch = jnp.array([z])

    _, log_px_given_z, _ = PAE_obj.push_spec_marg(
        theta_batch, z_batch, observed_flux=x_obs, weight=weight,
        marginalize_amplitude=True, return_rescaled_flux_and_loglike=True, redshift_in_flow=False
    )
    neg_log_likelihood = -jnp.sum(log_px_given_z)
    
    return neg_log_likelihood


def _optimize_theta_with_restarts(
    z,
    PAE_obj,
    x_obs,
    weight,
    nf_alpha,
    optimizer_maxiter,
    initial_guesses,
):
    """Run LBFGS from multiple initial guesses and return best solution plus diagnostics."""
    objective_value_and_grad = jax.value_and_grad(neg_log_posterior_theta, argnums=0)
    partial_objective_value_and_grad = partial(
        objective_value_and_grad,
        z=z,
        PAE_obj=PAE_obj,
        x_obs=x_obs,
        weight=weight,
        nf_alpha=nf_alpha,
    )

    best_theta = None
    best_neg_log_post = np.inf
    best_error = np.inf
    best_iter = optimizer_maxiter

    for theta0 in initial_guesses:
        optimizer = jaxopt.LBFGS(
            fun=partial_objective_value_and_grad,
            maxiter=optimizer_maxiter,
            value_and_grad=True,
        )
        opt_result, opt_state = optimizer.run(theta0)
        value = float(opt_state.value)
        if value < best_neg_log_post:
            best_theta = opt_result
            best_neg_log_post = value
            best_error = float(getattr(opt_state, "error", np.inf))
            best_iter = int(getattr(opt_state, "iter_num", optimizer_maxiter))

    return best_theta, best_neg_log_post, best_error, best_iter

@partial(jax.jit, static_argnames=['PAE_obj', 'Nz', 'z_min', 'z_max', 'n_latent', 'verbose', 'num_restarts', 'save_bestfit_models', 'save_restframe_seds'])
def evaluate_PAE_proflike_with_posterior_opt(PAE_obj, x_obs, weight, Nz=100, z_min=0.01, z_max=3.0, optimizer_maxiter=100, 
                                             latent_norm_max=3.0, penalty_strength=100.0, rkey=None, n_latent=None, verbose=True,
                                               num_restarts=1, nf_alpha=1.0, save_bestfit_models=False, save_restframe_seds=False,
                                               mcpl_map_latents=None, z_bins_mcpl=None):
    """
    Evaluates an approximate profile likelihood by optimizing against the posterior,
    using multiple random restarts for the theta optimization at each redshift.
    Now uses value_and_grad=True for LBFGS.
    
    Parameters:
    -----------
    save_bestfit_models : bool, optional
        If True, compute and return the reconstructed observed-frame SED at each redshift point.
        The returned array will have shape (Nz, nbands). Default is False.
    save_restframe_seds : bool, optional
        If True, compute and return the rest-frame SED (before redshifting and filter convolution)
        at each redshift point. The returned array will have shape (Nz, n_wavelengths). Default is False.
    mcpl_map_latents : array, optional
        MAP latents from MCLMC at redshift bins (shape: n_bins x n_latent).
        If provided, used as initialization for optimization. Default is None.
    z_bins_mcpl : array, optional
        Redshift bin centers corresponding to mcpl_map_latents (shape: n_bins).
        Required if mcpl_map_latents is provided. Default is None.
    
    Returns:
    --------
    z_grid : array
        Redshift grid points
    profile_logL : array
        Profile log-likelihood at each redshift
    all_map_thetas : array
        Optimal latent parameters at each redshift (shape: Nz x n_latent)
    all_bestfit_models : array or None
        If save_bestfit_models=True, reconstructed observed-frame SEDs at each redshift (shape: Nz x nbands)
        Otherwise None
    all_restframe_seds : array or None
        If save_restframe_seds=True, rest-frame SEDs at each redshift (shape: Nz x n_wavelengths)
        Otherwise None
    """
    if rkey is None:
        raise ValueError("A random key (rkey) must be provided for initial guess generation.")
    if n_latent is None:
        raise ValueError("n_latent must be provided for random initialization of theta.")
    if num_restarts < 1:
        raise ValueError("num_restarts must be at least 1.")

    if verbose:
        print(f"Starting profile likelihood estimation (optimizing posterior) for {Nz} redshift points.")
        print(f"Redshift range: [{z_min:.2f}, {z_max:.2f}]")
        print(f"Latent space dimension (n_latent): {n_latent}")
        print(f"Optimizer max iterations: {optimizer_maxiter}")
        print(f"Number of random restarts per redshift point: {num_restarts}")
        if save_bestfit_models:
            print(f"Will save best-fit reconstructed observed-frame models at each redshift point.")
        if save_restframe_seds:
            print(f"Will save rest-frame SEDs (before redshifting) at each redshift point.")
        # start_time = time.time()

    z_grid = jnp.linspace(z_min, z_max, num=Nz)
    
    # Prepare MCLMC-based initialization if provided
    # Find nearest MCPL latent for each z_grid point
    use_mcpl_init = mcpl_map_latents is not None and z_bins_mcpl is not None
    if use_mcpl_init:
        # For each z in z_grid, find nearest z in z_bins_mcpl
        # Using broadcasting to compute distances: |z_grid - z_bins_mcpl|
        # Shape: (Nz, n_bins_mcpl)
        dists = jnp.abs(z_grid[:, None] - z_bins_mcpl[None, :])
        nearest_bin_indices = jnp.argmin(dists, axis=1)  # Shape: (Nz,)
        # Map each z_grid point to its nearest MCPL latent
        mcpl_init_latents = mcpl_map_latents[nearest_bin_indices]  # Shape: (Nz, n_latent)

    def find_best_map_theta_for_z(z, rkey_for_z, z_idx):
        # 1. Create the `value_and_grad` version of the objective function.
        #    We specify `argnums=0` because `theta` is the first argument we want to differentiate.
        # objective_value_and_grad = jax.value_and_grad(neg_log_posterior_theta_full_args, argnums=0)
        objective_value_and_grad = jax.value_and_grad(neg_log_posterior_theta, argnums=0)

        # 2. Use `partial` to fix the non-differentiated arguments.
        #    The *first* argument of the partially applied function will be `theta`.
        
        # partial_objective_value_and_grad = partial(objective_value_and_grad, z=z, PAE_obj=PAE_obj, x_obs=x_obs, weight=weight, \
        #                                           latent_norm_max=latent_norm_max, penalty_strength=penalty_strength)
        partial_objective_value_and_grad = partial(objective_value_and_grad, z=z, PAE_obj=PAE_obj, x_obs=x_obs, weight=weight, nf_alpha=nf_alpha)

        # Define the single optimization run for a given initial guess
        def run_single_optimization(initial_theta_guess_restart):
            # Pass the partially applied function directly to the optimizer.
            # LBFGS will now expect (value, grad) from this function.
            optimizer = jaxopt.LBFGS(
                fun=partial_objective_value_and_grad, # This is the function that returns (value, grad)
                maxiter=optimizer_maxiter,
                value_and_grad=True # Crucial: Tell the optimizer to expect both
            )
            opt_result, opt_state = optimizer.run(initial_theta_guess_restart)
            return opt_result, opt_state.value # Return optimized theta and its objective value (neg_log_posterior)

        # Split the key for each restart within this redshift
        rkeys_for_restarts = jax.random.split(rkey_for_z, num_restarts)
        
        # Generate initial guesses
        if use_mcpl_init:
            # Use MCPL MAP latent for first restart, random for others
            mcpl_init = mcpl_init_latents[z_idx]
            # Generate random inits using vmap over keys
            random_inits = vmap(lambda key: jax.random.normal(key, shape=(n_latent,)))(rkeys_for_restarts[1:])
            # Stack MCPL init first, then random inits
            initial_guesses = jnp.concatenate([mcpl_init[None, :], random_inits], axis=0)
        else:
            # All random initializations - use vmap over keys
            initial_guesses = vmap(lambda key: jax.random.normal(key, shape=(n_latent,)))(rkeys_for_restarts)

        # Vmap over the restarts for this single redshift
        vmapped_restart_optimizer = vmap(
            lambda theta_init: run_single_optimization(theta_init),
            in_axes=0
        )

        optimized_thetas_all_restarts, neg_log_posteriors_all_restarts = vmapped_restart_optimizer(initial_guesses)

        best_restart_idx = jnp.argmin(neg_log_posteriors_all_restarts)

        map_theta = optimized_thetas_all_restarts[best_restart_idx]

        # Re-evaluate the *likelihood* at this best MAP theta and optionally get the reconstructed model
        if save_bestfit_models:
            recon_flux, log_px_given_z_at_map, _ = PAE_obj.push_spec_marg(
                map_theta[None, :], 
                jnp.array([z]),
                observed_flux=x_obs,
                weight=weight,
                marginalize_amplitude=True,
                return_rescaled_flux_and_loglike=True,
                redshift_in_flow=False
            )
            max_logL_at_map = jnp.sum(log_px_given_z_at_map)
            
            if save_restframe_seds:
                # Call separate function to get rest-frame SED
                restframe_sed = PAE_obj.get_restframe_sed(
                    map_theta[None, :],
                    redshift_in_flow=False
                )
                # recon_flux is (1, nbands), restframe_sed is (1, n_wav) - extract first row
                return map_theta, max_logL_at_map, recon_flux[0], restframe_sed[0]
            else:
                # recon_flux is (1, nbands) - extract first row
                return map_theta, max_logL_at_map, recon_flux[0]
        elif save_restframe_seds:
            # Get likelihood without returning flux
            _, log_px_given_z_at_map, _ = PAE_obj.push_spec_marg(
                map_theta[None, :], 
                jnp.array([z]),
                observed_flux=x_obs,
                weight=weight,
                marginalize_amplitude=True,
                return_rescaled_flux_and_loglike=True,
                redshift_in_flow=False
            )
            max_logL_at_map = jnp.sum(log_px_given_z_at_map)
            
            # Call separate function to get rest-frame SED
            restframe_sed = PAE_obj.get_restframe_sed(
                map_theta[None, :],
                redshift_in_flow=False
            )
            # restframe_sed is (1, n_wav) - extract first row
            return map_theta, max_logL_at_map, restframe_sed[0]
        else:
            _, log_px_given_z_at_map, _ = PAE_obj.push_spec_marg(
                map_theta[None, :], 
                jnp.array([z]),
                observed_flux=x_obs,
                weight=weight,
                marginalize_amplitude=True,
                return_rescaled_flux_and_loglike=True,
                redshift_in_flow=False
            )
            max_logL_at_map = jnp.sum(log_px_given_z_at_map)
            return map_theta, max_logL_at_map

    rkeys_for_z_grid = jax.random.split(rkey, Nz)
    z_indices = jnp.arange(Nz)

    if save_bestfit_models and save_restframe_seds:
        vmapped_z_optimizer = vmap(find_best_map_theta_for_z, in_axes=[0, 0, 0], out_axes=(0, 0, 0, 0))
        all_map_thetas, profile_logL, all_bestfit_models, all_restframe_seds = vmapped_z_optimizer(z_grid, rkeys_for_z_grid, z_indices)
    elif save_bestfit_models:
        vmapped_z_optimizer = vmap(find_best_map_theta_for_z, in_axes=[0, 0, 0], out_axes=(0, 0, 0))
        all_map_thetas, profile_logL, all_bestfit_models = vmapped_z_optimizer(z_grid, rkeys_for_z_grid, z_indices)
        all_restframe_seds = None
    elif save_restframe_seds:
        vmapped_z_optimizer = vmap(find_best_map_theta_for_z, in_axes=[0, 0, 0], out_axes=(0, 0, 0))
        all_map_thetas, profile_logL, all_restframe_seds = vmapped_z_optimizer(z_grid, rkeys_for_z_grid, z_indices)
        all_bestfit_models = None
    else:
        vmapped_z_optimizer = vmap(find_best_map_theta_for_z, in_axes=[0, 0, 0])
        all_map_thetas, profile_logL = vmapped_z_optimizer(z_grid, rkeys_for_z_grid, z_indices)
        all_bestfit_models = None
        all_restframe_seds = None

    return z_grid, profile_logL, all_map_thetas, all_bestfit_models, all_restframe_seds


def evaluate_PAE_proflike_with_adaptive_warmstart(
    PAE_obj,
    x_obs,
    weight,
    Nz=100,
    z_min=0.01,
    z_max=3.0,
    optimizer_maxiter=100,
    rkey=None,
    n_latent=None,
    nf_alpha=1.0,
    save_bestfit_models=False,
    save_restframe_seds=False,
    warmstart_coarse_factor=10,
    warmstart_num_restarts_coarse=4,
    warmstart_num_restarts_fine=0,
    warmstart_adaptive_fallback_restarts=3,
    warmstart_error_tol=1e-3,
    mcpl_map_latents=None,
    z_bins_mcpl=None,
):
    """Coarse-to-fine profile likelihood with adaptive fallback restarts for hard z points."""
    if rkey is None:
        raise ValueError("A random key (rkey) must be provided for initial guess generation.")
    if n_latent is None:
        raise ValueError("n_latent must be provided for initialization of theta.")

    z_grid = np.asarray(jnp.linspace(z_min, z_max, num=Nz))
    rkeys_for_z_grid = jax.random.split(rkey, Nz)

    use_mcpl_init = mcpl_map_latents is not None and z_bins_mcpl is not None
    if use_mcpl_init:
        dists = np.abs(z_grid[:, None] - np.asarray(z_bins_mcpl)[None, :])
        nearest_bin_indices = np.argmin(dists, axis=1)
        mcpl_init_latents = np.asarray(mcpl_map_latents)[nearest_bin_indices]
    else:
        mcpl_init_latents = None

    coarse_step = max(1, int(warmstart_coarse_factor))
    coarse_indices = list(range(0, Nz, coarse_step))
    if coarse_indices[-1] != Nz - 1:
        coarse_indices.append(Nz - 1)
    coarse_set = set(coarse_indices)

    all_map_thetas = np.zeros((Nz, n_latent), dtype=np.float32)
    profile_logL = np.full((Nz,), -np.inf, dtype=np.float32)
    converged = np.zeros((Nz,), dtype=bool)

    def _random_initials(z_idx, count, salt=0):
        if count <= 0:
            return []
        base_key = jax.random.fold_in(rkeys_for_z_grid[z_idx], int(salt))
        keys = jax.random.split(base_key, count)
        return [jax.random.normal(k, shape=(n_latent,)) for k in keys]

    def _solve_z(z_idx, warm_theta, n_random, fallback_restarts):
        initials = []
        if warm_theta is not None:
            initials.append(jnp.asarray(warm_theta))
        initials.extend(_random_initials(z_idx, n_random, salt=0))
        if len(initials) == 0:
            initials = _random_initials(z_idx, 1, salt=0)

        best_theta, _, best_error, best_iter = _optimize_theta_with_restarts(
            z=z_grid[z_idx],
            PAE_obj=PAE_obj,
            x_obs=x_obs,
            weight=weight,
            nf_alpha=nf_alpha,
            optimizer_maxiter=optimizer_maxiter,
            initial_guesses=initials,
        )

        is_converged = (best_error <= warmstart_error_tol) and (best_iter < optimizer_maxiter)
        if (not is_converged) and fallback_restarts > 0:
            fb_initials = [jnp.asarray(best_theta)]
            fb_initials.extend(_random_initials(z_idx, fallback_restarts, salt=1))
            fb_theta, _, fb_error, fb_iter = _optimize_theta_with_restarts(
                z=z_grid[z_idx],
                PAE_obj=PAE_obj,
                x_obs=x_obs,
                weight=weight,
                nf_alpha=nf_alpha,
                optimizer_maxiter=optimizer_maxiter,
                initial_guesses=fb_initials,
            )
            if fb_error <= best_error:
                best_theta = fb_theta
                best_error = fb_error
                best_iter = fb_iter
            is_converged = (best_error <= warmstart_error_tol) and (best_iter < optimizer_maxiter)

        _, log_px_given_z_at_map, _ = PAE_obj.push_spec_marg(
            jnp.asarray(best_theta)[None, :],
            jnp.array([z_grid[z_idx]]),
            observed_flux=x_obs,
            weight=weight,
            marginalize_amplitude=True,
            return_rescaled_flux_and_loglike=True,
            redshift_in_flow=False,
        )
        max_logL_at_map = float(jnp.sum(log_px_given_z_at_map))
        return np.asarray(best_theta), max_logL_at_map, is_converged

    # Coarse pass
    for z_idx in coarse_indices:
        warm_theta = mcpl_init_latents[z_idx] if use_mcpl_init else None
        n_random = max(0, int(warmstart_num_restarts_coarse) - (1 if warm_theta is not None else 0))
        theta, logl, ok = _solve_z(
            z_idx,
            warm_theta=warm_theta,
            n_random=n_random,
            fallback_restarts=max(0, int(warmstart_adaptive_fallback_restarts)),
        )
        all_map_thetas[z_idx] = theta
        profile_logL[z_idx] = logl
        converged[z_idx] = ok

    # Fine pass
    for z_idx in range(Nz):
        if z_idx in coarse_set:
            continue

        left_candidates = [c for c in coarse_indices if c < z_idx]
        right_candidates = [c for c in coarse_indices if c > z_idx]
        left = max(left_candidates) if left_candidates else None
        right = min(right_candidates) if right_candidates else None

        warm_theta = None
        if left is not None and right is not None and right != left:
            alpha = (z_idx - left) / float(right - left)
            warm_theta = (1.0 - alpha) * all_map_thetas[left] + alpha * all_map_thetas[right]
        elif left is not None:
            warm_theta = all_map_thetas[left]
        elif right is not None:
            warm_theta = all_map_thetas[right]

        if warm_theta is None and use_mcpl_init:
            warm_theta = mcpl_init_latents[z_idx]

        theta, logl, ok = _solve_z(
            z_idx,
            warm_theta=warm_theta,
            n_random=max(0, int(warmstart_num_restarts_fine)),
            fallback_restarts=max(0, int(warmstart_adaptive_fallback_restarts)),
        )
        all_map_thetas[z_idx] = theta
        profile_logL[z_idx] = logl
        converged[z_idx] = ok

    z_grid_jnp = jnp.asarray(z_grid)
    all_map_thetas_jnp = jnp.asarray(all_map_thetas)
    profile_logL_jnp = jnp.asarray(profile_logL)

    if save_bestfit_models:
        all_bestfit_models = []
    else:
        all_bestfit_models = None

    if save_restframe_seds:
        all_restframe_seds = []
    else:
        all_restframe_seds = None

    if save_bestfit_models or save_restframe_seds:
        for z_idx in range(Nz):
            theta = all_map_thetas_jnp[z_idx]
            z_val = z_grid_jnp[z_idx]
            recon_flux, _, _ = PAE_obj.push_spec_marg(
                theta[None, :],
                jnp.array([z_val]),
                observed_flux=x_obs,
                weight=weight,
                marginalize_amplitude=True,
                return_rescaled_flux_and_loglike=True,
                redshift_in_flow=False,
            )
            if save_bestfit_models:
                all_bestfit_models.append(recon_flux[0])
            if save_restframe_seds:
                restframe_sed = PAE_obj.get_restframe_sed(theta[None, :], redshift_in_flow=False)
                all_restframe_seds.append(restframe_sed[0])

        if save_bestfit_models:
            all_bestfit_models = jnp.asarray(all_bestfit_models)
        if save_restframe_seds:
            all_restframe_seds = jnp.asarray(all_restframe_seds)

    print(
        f"Adaptive warmstart: converged {int(np.sum(converged))}/{Nz} z-points "
        f"(tol={warmstart_error_tol}, coarse_step={coarse_step})"
    )

    return z_grid_jnp, profile_logL_jnp, all_map_thetas_jnp, all_bestfit_models, all_restframe_seds



def prof_like(x_obs, weight, PAE_obj, Z_MIN=0.01, Z_MAX=2.0, NZ_GRID=200, OPTIMIZER_MAXITER=50, NUM_RESTARTS_PER_Z=5, 
             nf_alpha=1.0, save_bestfit_models=False, save_restframe_seds=False,
             use_warmstart=False, warmstart_coarse_factor=10,
             warmstart_num_restarts_coarse=4, warmstart_num_restarts_fine=0,
             warmstart_adaptive_fallback_restarts=3, warmstart_error_tol=1e-3,
             mcpl_map_latents=None, z_bins_mcpl=None):
    """
    Compute profile likelihood for a single source.
    
    Parameters:
    -----------
    save_bestfit_models : bool, optional
        If True, compute and return the reconstructed observed-frame SED at each redshift point.
        Default is False.
    save_restframe_seds : bool, optional
        If True, compute and return the rest-frame SED (before redshifting and filter convolution)
        at each redshift point. Default is False.
    mcpl_map_latents : array, optional
        MAP latents from MCLMC at redshift bins (shape: n_bins x n_latent).
        If provided, used as initialization for optimization. Default is None.
    z_bins_mcpl : array, optional
        Redshift bin centers corresponding to mcpl_map_latents (shape: n_bins).
        Required if mcpl_map_latents is provided. Default is None.
    
    Returns:
    --------
    z_grid : array
        Redshift grid points
    profile_logL : array
        Profile log-likelihood at each redshift
    all_map_thetas : array
        Optimal latent parameters at each redshift
    all_bestfit_models : array or None
        If save_bestfit_models=True, reconstructed observed-frame SEDs at each redshift (shape: Nz x nbands)
        Otherwise None
    all_restframe_seds : array or None
        If save_restframe_seds=True, rest-frame SEDs at each redshift (shape: Nz x n_wavelengths)
        Otherwise None
    """
    key = jax.random.PRNGKey(44) # You can use any integer as seed

    t0 = time.time()
    # 6. Run the profile likelihood estimation
    print("Starting profile likelihood estimation...")
    if mcpl_map_latents is not None:
        print(f"  Using MCLMC-derived MAP latents for initialization")
    if use_warmstart:
        z_grid, profile_logL, all_map_thetas, all_bestfit_models, all_restframe_seds = evaluate_PAE_proflike_with_adaptive_warmstart(
            PAE_obj=PAE_obj,
            x_obs=x_obs,
            weight=weight,
            Nz=NZ_GRID,
            z_min=Z_MIN,
            z_max=Z_MAX,
            optimizer_maxiter=OPTIMIZER_MAXITER,
            rkey=key,
            n_latent=PAE_obj.params['nlatent'],
            nf_alpha=nf_alpha,
            save_bestfit_models=save_bestfit_models,
            save_restframe_seds=save_restframe_seds,
            warmstart_coarse_factor=warmstart_coarse_factor,
            warmstart_num_restarts_coarse=warmstart_num_restarts_coarse,
            warmstart_num_restarts_fine=warmstart_num_restarts_fine,
            warmstart_adaptive_fallback_restarts=warmstart_adaptive_fallback_restarts,
            warmstart_error_tol=warmstart_error_tol,
            mcpl_map_latents=mcpl_map_latents,
            z_bins_mcpl=z_bins_mcpl,
        )
    else:
        z_grid, profile_logL, all_map_thetas, all_bestfit_models, all_restframe_seds = evaluate_PAE_proflike_with_posterior_opt(
            PAE_obj=PAE_obj,
            x_obs=x_obs,
            weight=weight,
            Nz=NZ_GRID,
            z_min=Z_MIN,
            z_max=Z_MAX,
            optimizer_maxiter=OPTIMIZER_MAXITER,
            rkey=key,  # Pass the random key
            n_latent=PAE_obj.params['nlatent'],  # Pass the latent space dimension
            num_restarts=NUM_RESTARTS_PER_Z, \
            nf_alpha=nf_alpha,
            save_bestfit_models=save_bestfit_models,
            save_restframe_seds=save_restframe_seds,
            mcpl_map_latents=mcpl_map_latents,
            z_bins_mcpl=z_bins_mcpl
        )
    print("Profile likelihood estimation complete.")
    
    print('time elapsed:', time.time()-t0)
    print(f"\nMaximum profile log-likelihood found at z = {z_grid[jnp.argmax(profile_logL)]:.3f}")
    print(f"Corresponding max log-likelihood value: {jnp.max(profile_logL):.3f}")
    print(f"Shape of optimized latent variables (all_map_thetas): {all_map_thetas.shape}") # Should be (Nz, n_latent)
    if save_bestfit_models:
        print(f"Shape of best-fit observed-frame models: {all_bestfit_models.shape}")  # Should be (Nz, nbands)
    if save_restframe_seds:
        print(f"Shape of rest-frame SEDs: {all_restframe_seds.shape}")  # Should be (Nz, n_wavelengths)

    return z_grid, profile_logL, all_map_thetas, all_bestfit_models, all_restframe_seds
    


# def make_logdensity_fn_marg(PAE_obj, x_obs, w, z_min, z_max, nf_alpha=1.0, sharpness=100.0, redshift_in_flow=False, \
#                            z0_prior=0.65, sigma_prior=0.4, include_truncated_prior=False):
#     """
#     Factory function that creates and JIT-compiles the log-density function.
#     This "freezes" the PAE object and its parameters, avoiding recompilation.
#     """
    
#     # These functions are now "closed over" by the log_density function below
#     push_spec_marg = PAE_obj.push_spec_marg
    
#     @jax.jit
#     def log_density(latent):
#         """
#         Log-density function for BlackJAX HMC.
#         """
#         latent = latent[None, :] if latent.ndim == 1 else latent
        
#         if redshift_in_flow:
#             u = latent
#             z_redshift = None
#         else:
#             u = latent[:, :-1]
#             z_redshift = latent[:, -1]

#         # Normalizing flow prior (Multivariate Gaussian)
#         log_pz = -0.5 * jnp.sum(u**2) - 0.5 * u.shape[-1] * jnp.log(2 * jnp.pi)

#         def log_soft_prior(z):
#             below = -sharpness * (z_min - z) * (z < z_min)
#             above = -sharpness * (z - z_max) * (z > z_max)
#             return below + above
        
#         # Compute decoder likelihood p(x | z, z_redshift)
#         _, log_px_given_z, z_redshift_out = push_spec_marg(
#             u, z_redshift,
#             observed_flux=x_obs,
#             weight=w,
#             marginalize_amplitude=True,
#             return_rescaled_flux_and_loglike=True,
#             redshift_in_flow=redshift_in_flow)

#         # Use the redshift returned from the forward pass
#         log_pz_redshift = log_soft_prior(z_redshift_out)

#         # ------------------ Key Change Here ------------------
#         # Use jnp.sum() to robustly handle both scalar and 1-element array outputs.
#         # This ensures the function always returns a single scalar value as required by BlackJAX.
#         return (nf_alpha * log_pz) + jnp.sum(log_pz_redshift) + jnp.sum(log_px_given_z)

#     return log_density


# def logdensity_fn_marg(latent, PAE_obj, x_obs, w, z_min, z_max, nf_alpha=1.0, return_indiv=False, sharpness=100.0, \
#                       return_recon=False, redshift_in_flow=False):
#     """
#     Log-density function for BlackJAX HMC.
    
#     latent: tuple (z, z_redshift)
#     params: model parameters
#     flow: normalizing flow model
#     decoder: decoder model
#     x_obs: observed data
#     """    

#     # already configured initialize_latents_scale to add redshift to initial position when redshift_in_flow


#     # Ensure latent is 2D
#     latent = latent[None, :] if latent.ndim == 1 else latent
    
#     if redshift_in_flow:
#         u = latent
#         z_redshift = None
#     else:
#         u = latent[:, :-1]
#         z_redshift = latent[:, -1]

#     # Normalizing flow prior (Multivariate Gaussian)
#     log_pz = -0.5 * jnp.sum(u**2) - 0.5 * u.shape[-1] * jnp.log(2 * jnp.pi)

#     def log_soft_prior(z):
#         below = -sharpness * (z_min - z) * (z < z_min)
#         above = -sharpness * (z - z_max) * (z > z_max)
#         return below + above
    
#     # Compute decoder likelihood p(x | z, z_redshift)
#     recon_x, log_px_given_z, z_redshift = PAE_obj.push_spec_marg(
#         u, z_redshift,
#         observed_flux=x_obs,
#         weight=w,
#         marginalize_amplitude=True,
#         return_rescaled_flux_and_loglike=True,
#         redshift_in_flow=redshift_in_flow)

#     # placing after log-likelihood evaluation for case where redshift_in_flow produces z_redshift from NF
#     log_pz_redshift = log_soft_prior(z_redshift)

#     if return_recon:
#         tot_logp = nf_alpha*log_pz + log_pz_redshift[0] + log_px_given_z
        
#         return tot_logp, recon_x

#     if return_indiv:
#         return nf_alpha*log_pz, log_pz_redshift[0], log_px_given_z
#     else:
#         return nf_alpha*log_pz + log_pz_redshift[0] + log_px_given_z


# def logdensity_fn_nfalpha(latent, PAE_obj, x_obs, w, z_min, z_max, nf_alpha=1.0, return_indiv=False, sharpness=100.0):
#     """    
#     latent: tuple (z, z_redshift)
#     params: model parameters
#     flow: normalizing flow model
#     decoder: decoder model
#     x_obs: observed data
#     """    

#     if len(latent.shape)==1:
#         u, z_redshift = latent[None,:-1], latent[None,-1]
#     else:
#         u, z_redshift = latent[:,:-1], latent[:,-1]
    
#     # Normalizing flow prior (Multivariate Gaussian)
#     log_pz = -0.5 * jnp.sum(u**2) - 0.5 * u.shape[-1] * jnp.log(2 * jnp.pi)

#     def log_soft_prior(z):
#         below = -sharpness * (z_min - z) * (z < z_min)
#         above = -sharpness * (z - z_max) * (z > z_max)
#         return below + above

#     log_pz_redshift = log_soft_prior(z_redshift)
    
#     # Compute decoder likelihood p(x | z, z_redshift)
#     recon_x = PAE_obj.push_spec(u, z_redshift)
#     # print('recon_x.shape:', recon_x.shape)
#     log_px_given_z = -0.5 * jnp.sum(w*(x_obs - recon_x) ** 2)

#     if return_indiv:
#         # print('log_pz, log_pz_redshift[0], log_px_given_z shape:', log_pz.shape, log_pz_redshift[0].shape, log_px_given_z.shape)
#         return nf_alpha*log_pz, log_pz_redshift[0], log_px_given_z
#     else:
#         return nf_alpha*log_pz + log_pz_redshift[0] + log_px_given_z


# def logdensity_fn(latent, PAE_obj, x_obs, w, z_min, z_max, nf_alpha=1.0, return_indiv=False, sharpness=100.0):
#     """
#     Log-density function for BlackJAX.
    
#     latent: tuple (z, z_redshift)
#     params: model parameters
#     flow: normalizing flow model
#     decoder: decoder model
#     x_obs: observed data
#     """    

#     if len(latent.shape)==1:
#         u, z_redshift = latent[None,:-1], latent[None,-1]
#     else:
#         u, z_redshift = latent[:,:-1], latent[:,-1]
    
#     # Normalizing flow prior (Multivariate Gaussian)
#     log_pz = -0.5 * jnp.sum(u**2) - 0.5 * u.shape[-1] * jnp.log(2 * jnp.pi)

#     def log_soft_prior(z):
#         below = -sharpness * (z_min - z) * (z < z_min)
#         above = -sharpness * (z - z_max) * (z > z_max)
#         return below + above

#     log_pz_redshift = log_soft_prior(z_redshift)
    
#     # Compute decoder likelihood p(x | z, z_redshift)
#     recon_x = PAE_obj.push_spec(u, z_redshift)
#     log_px_given_z = -0.5 * jnp.sum(w*(x_obs - recon_x) ** 2)

#     if return_indiv:
#         return log_pz, nf_alpha*log_pz_redshift[0], log_px_given_z
#     else:
#         return log_pz + nf_alpha*log_pz_redshift[0] + log_px_given_z


# def logdensity_fn_softball(latent, PAE_obj, x_obs, w, z_min, z_max, return_indiv=False, sharpness=100.0):
#     """
#     Log-density function for BlackJAX.
    
#     latent: tuple (z, z_redshift)
#     params: model parameters
#     flow: normalizing flow model
#     decoder: decoder model
#     x_obs: observed data
#     """    
#     if len(latent.shape)==1:
#         u, z_redshift = latent[None,:-1], latent[None,-1]
#     else:
#         u, z_redshift = latent[:,:-1], latent[:,-1]
#     # Normalizing flow prior (Multivariate Gaussian)
#     log_pz = soft_ball_log_prior(u)

#     def log_soft_prior(z):
#         below = -sharpness * (z_min - z) * (z < z_min)
#         above = -sharpness * (z - z_max) * (z > z_max)
#         return below + above
        
#     log_pz_redshift = log_soft_prior(z_redshift)
#     # Compute decoder likelihood p(x | z, z_redshift)
#     recon_x = PAE_obj.push_spec(u, z_redshift)
#     log_px_given_z = -0.5 * jnp.sum(w*(x_obs - recon_x) ** 2)

#     if return_indiv:
#         return log_pz, log_pz_redshift[0], log_px_given_z
#     else:
#         return log_pz + log_pz_redshift[0] + log_px_given_z

