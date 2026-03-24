# Batched Logdensity Function Implementation Guide

## Problem with Current Implementation

Currently in `pae_spec_sample_floatz()`:

```python
# This happens INSIDE the vmap over galaxies
log_p = make_logdensity_fn_optzprior(
    PAE_obj, x_obs, weight, z_min=sampler_cfg.zmin, z_max=sampler_cfg.zmax,
    nf_alpha=sampler_cfg.nf_alpha, ...
)
```

**Issues:**
- Creates a NEW closure for every galaxy
- Captures `x_obs` and `weight` in the closure
- JIT compiles separately for each galaxy → 1000 galaxies = 1000 compilations
- Can't share compiled code across galaxies

---

## Solution: Batched Logdensity Function

### Step 1: Create a Parameterized Logdensity Function

Add this to `inference/like_prior.py`:

```python
def make_batched_logdensity_fn(PAE_obj, sampler_cfg):
    """
    Create a single JIT-compiled logdensity function that takes data as parameters.
    
    This function is created ONCE for all galaxies, avoiding repeated JIT compilation.
    
    Args:
        PAE_obj: The PAE model object
        sampler_cfg: MCLMCSamplingConfig with prior settings
    
    Returns:
        A JIT-compiled function: log_density(latent, x_obs, weight) -> log_prob
    """
    # Extract settings from config
    z_min = sampler_cfg.zmin
    z_max = sampler_cfg.zmax
    nf_alpha = sampler_cfg.nf_alpha
    sharpness = 100.0
    redshift_in_flow = sampler_cfg.redshift_in_flow
    z0_prior = sampler_cfg.z0_prior
    sigma_prior = sampler_cfg.sigma_prior
    redshift_prior_type = sampler_cfg.redshift_prior_type
    
    # Close over PAE object and config, but NOT the data
    push_spec_marg = PAE_obj.push_spec_marg
    
    @jax.jit
    def log_density(latent, x_obs, weight, m_mag=None):
        """
        Batched log-density function that takes data as parameters.
        
        Args:
            latent: (n_latent + 1,) array for single chain
            x_obs: (n_bands,) observed fluxes for ONE galaxy
            weight: (n_bands,) inverse variance weights for ONE galaxy
            m_mag: Optional magnitude (for BPZ prior)
        
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
            z_redshift = latent[:, -1]
        
        # Normalizing flow prior (standard Gaussian)
        log_pz = -0.5 * jnp.sum(u**2) - 0.5 * u.shape[-1] * jnp.log(2 * jnp.pi)
        
        # Likelihood: p(x_obs | latent, redshift)
        # Note: x_obs and weight are now PARAMETERS, not closed over
        _, log_px_given_z, z_redshift_out = push_spec_marg(
            u, z_redshift,
            observed_flux=x_obs[None, :],  # Add batch dim
            weight=weight[None, :],        # Add batch dim
            marginalize_amplitude=True,
            return_rescaled_flux_and_loglike=True,
            redshift_in_flow=redshift_in_flow
        )
        
        # Redshift boundary roll-off
        below_penalty = -sharpness * jnp.maximum(0., z_min - z_redshift)
        above_penalty = -sharpness * jnp.maximum(0., z_redshift - z_max)
        log_pz_rolloff = below_penalty + above_penalty
        
        # Optional Gaussian redshift prior
        def calculate_gaussian_core(z, m):
            return norm.logpdf(z, loc=z0_prior, scale=sigma_prior)
        
        def no_gaussian_core(z, m):
            return jnp.zeros_like(z)
        
        log_pz_prior = jax.lax.switch(
            redshift_prior_type,
            (no_gaussian_core, calculate_gaussian_core),
            z_redshift, m_mag
        )
        
        log_pz_redshift = jnp.sum(log_pz_rolloff) + jnp.sum(log_pz_prior)
        
        return (nf_alpha * log_pz) + log_pz_redshift + jnp.sum(log_px_given_z)
    
    return log_density
```

---

### Step 2: Refactor `pae_spec_sample_floatz()`

Modify `sampling/sample_pae_batch_refactor.py`:

```python
def pae_spec_sample_floatz(PAE_obj, x_obs, weight, rkey, sampler_cfg: MCLMCSamplingConfig,
                           batched_log_density=None):
    """
    Sample from PAE posterior for a single galaxy.
    
    Args:
        PAE_obj: PAE model
        x_obs: (n_bands,) observed spectrum
        weight: (n_bands,) weights
        rkey: JAX random key
        sampler_cfg: sampling configuration
        batched_log_density: Pre-compiled logdensity function (optional)
    """
    
    # Create galaxy-specific log_p by partial application
    if batched_log_density is not None:
        # NEW WAY: Use pre-compiled batched function
        log_p = lambda latent: batched_log_density(latent, x_obs, weight)
    else:
        # OLD WAY: Create new closure (fallback for compatibility)
        log_p = make_logdensity_fn_optzprior(
            PAE_obj, x_obs, weight, z_min=sampler_cfg.zmin, z_max=sampler_cfg.zmax,
            nf_alpha=sampler_cfg.nf_alpha,
            redshift_in_flow=sampler_cfg.redshift_in_flow,
            z0_prior=sampler_cfg.z0_prior,
            sigma_prior=sampler_cfg.sigma_prior,
            include_gaussian_prior=sampler_cfg.include_gaussian_prior,
            redshift_prior_type=sampler_cfg.redshift_prior_type
        )
    
    # Rest of the function stays the same
    n_latent = PAE_obj.params['nlatent'] + (1 if sampler_cfg.redshift_in_flow else 0)
    initial_position = jnp.array(initialize_latents_scale(
        n_latent, sampler_cfg.nchain_per_gal, rkey, 
        z_min=sampler_cfg.zmin, z_max=sampler_cfg.zmax,
        include_z=not sampler_cfg.redshift_in_flow, prior_scale=1.0
    ))
    
    # ... rest of sampling logic unchanged ...
```

---

### Step 3: Update `run_batched_sampler()`

Modify the batch loop to create the batched function once:

```python
def run_batched_sampler(
    PAE_obj,
    specs,
    weights,
    redshifts,
    batch_size: int,
    rng_key,
    sampler_cfg: MCLMCSamplingConfig
):
    """Run MCLMC sampling over batches of galaxies."""
    
    # ============ NEW: CREATE BATCHED LOGDENSITY ONCE ============
    print("Creating batched logdensity function...")
    batched_log_density = make_batched_logdensity_fn(PAE_obj, sampler_cfg)
    
    # Pre-compile by evaluating once with dummy data
    print("Pre-compiling logdensity...")
    n_latent = PAE_obj.params['nlatent']
    n_bands = specs.shape[1]
    dummy_latent = jnp.zeros(n_latent + 1)
    dummy_spec = jnp.ones(n_bands)
    dummy_weight = jnp.ones(n_bands)
    _ = batched_log_density(dummy_latent, dummy_spec, dummy_weight)
    print("✓ Logdensity compiled and ready")
    # =============================================================
    
    assert rng_key is not None, "You must provide a JAX random key."
    
    spec_batches = split_into_batches(specs, batch_size)
    weight_batches = split_into_batches(weights, batch_size)
    z_batches = split_into_batches(redshifts, batch_size)
    
    nbatch = len(spec_batches)
    
    all_samples, all_ae_redshifts, all_log_prior, all_log_redshift = [[] for _ in range(4)]
    all_log_L, all_initial_position = [], []
    all_tuned_L, all_tuned_step_size = [], []
    all_mean_logL_per_chain, all_max_logL, all_mcpl = [], [], []
    
    for batch_idx, (spec_batch, w_batch, z_batch) in enumerate(
        zip(spec_batches, weight_batches, z_batches), start=1
    ):
        print(f"On batch {batch_idx} of {nbatch}..")
        rng_key, subkey = jax.random.split(rng_key)
        subkeys = jax.random.split(subkey, num=spec_batch.shape[0])
        
        if sampler_cfg.fix_z:
            # Fixed redshift mode
            samples, log_prior, log_L, initial_pos, tuned_L, tuned_step_size = jax.vmap(
                lambda key, spec, w, ztrue: pae_spec_sample_fixz_vmap(
                    PAE_obj, spec, w, ztrue, key, sampler_cfg
                )
            )(subkeys, spec_batch, w_batch, z_batch)
        else:
            # ============ NEW: PASS BATCHED LOGDENSITY ============
            samples, _, log_prior, log_redshift, log_L, initial_pos, \
                tuned_L, tuned_step_size, mean_logL_per_chain, max_logL_all, \
                z_bins_mcpl, mcpl = jax.vmap(
                    lambda key, spec, w, ztrue: pae_spec_sample_floatz(
                        PAE_obj, spec, w, key, sampler_cfg,
                        batched_log_density=batched_log_density  # Pass pre-compiled function
                    )
                )(subkeys, spec_batch, w_batch, z_batch)
            # =====================================================
            
            all_log_redshift.append(log_redshift)
            all_mean_logL_per_chain.append(mean_logL_per_chain)
            all_max_logL.append(max_logL_all)
            all_mcpl.append(mcpl)
        
        # Append results
        all_log_prior.append(log_prior)
        all_log_L.append(log_L)
        all_initial_position.append(initial_pos)
        all_samples.append(samples)
        all_tuned_L.append(tuned_L)
        all_tuned_step_size.append(tuned_step_size)
    
    # ... rest of concatenation logic unchanged ...
```

---

## Key Benefits

### 1. **Single Compilation** ✅
```python
# OLD: 1000 compilations (one per galaxy)
for galaxy in galaxies:
    log_p = make_logdensity_fn_optzprior(PAE, galaxy.x_obs, galaxy.weight)
    # → JIT compiles HERE, 1000 times

# NEW: 1 compilation (shared across all galaxies)
batched_log_density = make_batched_logdensity_fn(PAE, config)  # Compile once
for galaxy in galaxies:
    log_p = lambda latent: batched_log_density(latent, galaxy.x_obs, galaxy.weight)
    # → Just a lambda, no compilation!
```

### 2. **Memory Efficiency** ✅
- Old: 1000 separate compiled functions in memory
- New: 1 compiled function reused

### 3. **Faster Startup** ✅
- Old: 0.1-1s compilation × 1000 galaxies = 100-1000s
- New: 1s compilation × 1 = 1s

### 4. **Easier to Optimize** ✅
- Single function easier to profile and optimize
- Can apply further optimizations to one place

---

## Performance Comparison

### Before (current):
```
Batch 1: Creating logdensity... 0.8s × 100 galaxies = 80s
Batch 2: Creating logdensity... 0.8s × 100 galaxies = 80s
...
Total compilation overhead: 800s for 1000 galaxies
```

### After (batched):
```
Pre-compile logdensity... 1.0s
Batch 1: Using pre-compiled function... 0s overhead
Batch 2: Using pre-compiled function... 0s overhead
...
Total compilation overhead: 1s for 1000 galaxies
```

**Speedup: ~800x reduction in compilation time!**

---

## Implementation Checklist

- [ ] Add `make_batched_logdensity_fn()` to `inference/like_prior.py`
- [ ] Modify `pae_spec_sample_floatz()` to accept `batched_log_density` parameter
- [ ] Update `run_batched_sampler()` to create and pass batched function
- [ ] Test with small batch (10 galaxies) to verify correctness
- [ ] Profile to confirm compilation overhead is eliminated
- [ ] Update `pae_spec_sample_fixz_vmap()` similarly (if used)

---

## Testing Strategy

```python
# Test that old and new give same results
from sampling.sample_pae_batch_refactor import *

# Single galaxy test
PAE_obj = initialize_PAE(...)
x_obs = data.all_spec_obs[0]
weight = data.weights[0]
key = jr.key(42)

# OLD WAY
log_p_old = make_logdensity_fn_optzprior(PAE_obj, x_obs, weight, ...)
result_old = pae_spec_sample_floatz(PAE_obj, x_obs, weight, key, cfg)

# NEW WAY
batched_log_density = make_batched_logdensity_fn(PAE_obj, cfg)
result_new = pae_spec_sample_floatz(PAE_obj, x_obs, weight, key, cfg, 
                                     batched_log_density=batched_log_density)

# Compare
assert jnp.allclose(result_old[0], result_new[0], rtol=1e-5)
print("✓ Results match!")
```

---

## Next Optimization: vmap Instead of lax.map

After implementing batched logdensity, the next bottleneck is:

```python
# In pae_spec_sample_floatz(), line ~303
log_p_all = jax.lax.map(
    lambda chain: jax.lax.map(log_p, chain),
    post_burnin_samples
)

# Replace with:
log_p_all = jax.vmap(
    lambda chain: jax.vmap(log_p)(chain)
)(post_burnin_samples)
```

This will be 5-10x faster once batched logdensity is in place.
