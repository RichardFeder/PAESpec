# JAX Optimization Analysis: JIT and Parallelism Opportunities

## Current Architecture Analysis

### Overall Flow
```
run_redshifts (redshift_job.py)
    └─> sample_mclmc_wrapper
        └─> run_batched_sampler
            └─> jax.vmap over batch
                └─> pae_spec_sample_floatz (per galaxy)
                    ├─> make_logdensity_fn_optzprior (creates JIT'd log_density)
                    ├─> initialize_latents_scale
                    ├─> Optional: init_reinit with run_mclmc_simp
                    ├─> run_mclmc_simp_with_pretune (JIT'd via vmap)
                    └─> monte_carlo_profile_likelihood_jax
```

## Key Bottlenecks Identified

### 1. **Logdensity Function Creation is Per-Galaxy** ❌
**Location**: `pae_spec_sample_floatz()` line ~245

```python
log_p = make_logdensity_fn_optzprior(
    PAE_obj, x_obs, weight, z_min=sampler_cfg.zmin, z_max=sampler_cfg.zmax,
    nf_alpha=sampler_cfg.nf_alpha, ...
)
```

**Problem**:
- `make_logdensity_fn_optzprior` is called **inside** the vmapped function
- Each galaxy gets its own JIT-compiled `log_density` closure
- This means JIT compilation happens **N_galaxies × N_batches** times
- The closure captures `PAE_obj.push_spec_marg`, `x_obs`, and `w` per galaxy

**Why This Hurts**:
- JIT compilation overhead: ~0.1-1s per galaxy on first call
- For 1000 galaxies: potentially 100-1000s of compilation time
- Can't share compiled code across galaxies even though structure is identical
- Memory overhead from storing N separate compiled functions

**Solution**: Create a **batched** logdensity function once, outside vmap
```python
# OUTSIDE vmap - create once
@jax.jit
def batched_log_density(latent, x_obs, weight):
    """Operates on single galaxy data, but JIT compiled once"""
    # Same logic, but x_obs and weight are parameters, not closed over
    u = latent[:, :-1]
    z_redshift = latent[:, -1]
    log_pz = -0.5 * jnp.sum(u**2) - 0.5 * u.shape[-1] * jnp.log(2 * jnp.pi)
    _, log_px_given_z, z_redshift_out = PAE_obj.push_spec_marg(
        u, z_redshift, observed_flux=x_obs, weight=weight, ...
    )
    return log_pz + log_px_given_z

# INSIDE batch processing
def per_galaxy_sampling(x_obs, weight, key):
    log_p = lambda latent: batched_log_density(latent, x_obs, weight)
    # Now run MCLMC with this log_p
    ...

# vmap over batch
jax.vmap(per_galaxy_sampling)(x_obs_batch, weight_batch, keys)
```

**Expected Speedup**: 10-100x reduction in compilation time for large batches

---

### 2. **MCLMC Initialization Happens Per-Chain Inside vmap** ⚠️
**Location**: `run_mclmc_simp()` and `run_mclmc_simp_with_pretune()`

```python
# Inside mclmc.py
initial_state = blackjax.mcmc.mclmc.init(
    position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
)
```

**Problem**:
- `blackjax.mcmc.mclmc.init()` is called for **every chain of every galaxy**
- For 1000 galaxies × 4 chains = 4000 calls to `init()`
- Each call may compile or do non-trivial setup
- Timing shows "time for mclmc init: 0.X seconds" per chain (from debug prints)

**Why This Hurts**:
- Blackjax's `init()` evaluates the logdensity at initial position
- If logdensity isn't already compiled, triggers compilation
- Even if compiled, there's overhead in state structure creation

**Partial Solution**: Pre-warm the logdensity function
```python
# Before entering vmap, compile the log_density once
dummy_latent = jnp.zeros(n_latent + 1)
_ = log_density(dummy_latent)  # Trigger compilation

# Now when vmap calls init(), log_density is already compiled
```

**Better Solution**: Batch the initialization across chains
```python
# Instead of vmap over chains initializing separately,
# use vmap'd version of init itself
initial_states = jax.vmap(
    lambda pos, key: blackjax.mcmc.mclmc.init(pos, logdensity_fn, key)
)(initial_positions, keys)
```
This allows JAX to optimize the initialization as a batch operation.

---

### 3. **Batch Processing is Sequential** 🔴
**Location**: `run_batched_sampler()` line ~503

```python
for batch_idx, (spec_batch, w_batch, z_batch) in enumerate(...):
    # Process batch sequentially
    samples, ... = jax.vmap(lambda key, spec, w, ztrue: 
        pae_spec_sample_floatz(...))(subkeys, spec_batch, w_batch, z_batch)
    all_samples.append(samples)
```

**Problem**:
- Batches are processed one at a time in a Python for-loop
- Can't parallelize across batches
- CPU sits idle while GPU processes each batch
- No pipelining/overlap between batches

**Why This Hurts**:
- If each batch takes 30s, 10 batches = 300s minimum
- GPU might only be 80% utilized due to CPU-GPU sync points
- Can't take advantage of multi-GPU systems

**Solution**: Use `jax.lax.scan` for batches or pre-process everything
```python
# Option A: Use lax.scan to compile the batch loop
def process_batch(carry, batch_data):
    spec_batch, w_batch, z_batch, key = batch_data
    key, subkey = jax.random.split(key)
    subkeys = jax.random.split(subkey, spec_batch.shape[0])
    
    samples, ... = jax.vmap(pae_spec_sample_floatz_compiled)(
        subkeys, spec_batch, w_batch, z_batch
    )
    return key, samples

# Scan over all batches
_, all_samples = jax.lax.scan(
    process_batch,
    init_key,
    (spec_batches, weight_batches, z_batches, keys)
)

# Option B: Just process all at once if memory allows
# Instead of splitting into batches, do:
all_samples = jax.vmap(pae_spec_sample_floatz)(
    all_keys, all_specs, all_weights, all_z
)
```

**Expected Speedup**: 1.2-2x from better pipelining and less Python overhead

---

### 4. **Profile Likelihood Uses Slow jax.lax.map** 🐌
**Location**: `pae_spec_sample_floatz()` line ~303

```python
log_p_all = jax.lax.map(
    lambda chain: jax.lax.map(log_p, chain),
    post_burnin_samples
)
```

**Problem**:
- `jax.lax.map` is designed for memory efficiency, not speed
- It processes elements sequentially to avoid large intermediate arrays
- Here we're computing log_p for all samples post burn-in
- For 4 chains × 1000 samples = 4000 evaluations per galaxy
- Sequential processing means no parallelism

**Why This Hurts**:
- `jax.lax.map` is ~10x slower than `jax.vmap` for small-medium arrays
- For 1000 galaxies × 4000 samples each = 4M evaluations
- Could be parallelized with vmap

**Solution**: Use `jax.vmap` with chunking if needed
```python
# Direct vmap (if memory allows)
log_p_all = jax.vmap(
    lambda chain: jax.vmap(log_p)(chain)
)(post_burnin_samples)

# Or if memory-constrained, chunk manually
def compute_log_p_chunked(samples, chunk_size=250):
    n_samples = samples.shape[0]
    chunks = [samples[i:i+chunk_size] for i in range(0, n_samples, chunk_size)]
    log_ps = [jax.vmap(log_p)(chunk) for chunk in chunks]
    return jnp.concatenate(log_ps)

log_p_all = jax.vmap(compute_log_p_chunked)(post_burnin_samples)
```

**Expected Speedup**: 5-10x for profile likelihood computation

---

### 5. **PAE Model Calls Not Fully Batched** ⚠️
**Location**: `push_spec_marg()` in `pae_jax.py` line ~257

```python
if nsamp == 1:
    x_interp = jnp.interp(...)
else:
    wave_redshifted = self.central_wavelengths * (1 + redshift[:, None])
    x_interp = jax.vmap(lambda x, y: jnp.interp(...))(wave_redshifted, spec)
```

**Problem**:
- Special-casing `nsamp == 1` means different code paths
- `jnp.interp` inside vmap can be slower than batched operations
- Filter convolution `jnp.dot(self.jax_filters, x_interp.T).T` is not optimal shape

**Why This Hurts**:
- Different code paths mean different compilations
- `vmap` over `interp` doesn't vectorize as well as native JAX ops
- Matrix transpose operations add memory copies

**Solution**: Always use batched path and optimize interpolation
```python
# Always work with batch dimension
redshift = jnp.atleast_1d(redshift)
spec = jnp.atleast_2d(spec)

# Use custom batched interpolation (if available) or optimize vmap
wave_redshifted = self.central_wavelengths * (1 + redshift[:, None])
x_interp = jax.vmap(jnp.interp)(
    self.lam_interp[None, :],  # Broadcast
    wave_redshifted,
    spec
)

# Optimize filter application - no transpose needed
model_flux = jnp.einsum('ij,nj->ni', self.jax_filters, x_interp)
```

**Expected Speedup**: 1.5-2x for likelihood evaluation

---

### 6. **Random Key Splitting in Python Loop** ⚠️
**Location**: `run_batched_sampler()` line ~510

```python
for batch_idx, ... in enumerate(...):
    rng_key, subkey = jax.random.split(rng_key)
    subkeys = jax.random.split(subkey, num=spec_batch.shape[0])
```

**Problem**:
- Key splitting happens in Python loop, not JAX
- Each split creates host-device communication
- Can't be optimized by XLA

**Solution**: Pre-generate all keys before loop
```python
# Before loop
total_samples = len(specs)
all_keys = jax.random.split(rng_key, total_samples)

# Reshape into batches
key_batches = [all_keys[i:i+batch_size] for i in range(0, total_samples, batch_size)]

# In loop (or better, in lax.scan)
for batch_idx, (spec_batch, w_batch, z_batch, key_batch) in enumerate(...):
    # Use key_batch directly, no splitting needed
```

---

## Summary of Optimizations by Impact

### High Impact (10-100x speedup potential)
1. **Create batched logdensity function once, outside vmap** - eliminates repeated JIT compilation
2. **Use vmap instead of lax.map for profile likelihood** - 5-10x faster evaluation

### Medium Impact (2-5x speedup potential)
3. **Eliminate sequential batch processing** - use lax.scan or single vmap
4. **Pre-compile logdensity before MCLMC init** - reduces initialization overhead
5. **Optimize PAE model interpolation** - vectorize better

### Low Impact (1.2-2x speedup potential)
6. **Pre-generate random keys** - reduces host-device communication
7. **Batch MCLMC initialization** - slight reduction in overhead

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 hours)
1. Replace `jax.lax.map` with `jax.vmap` for profile likelihood
2. Pre-generate all random keys before batch loop
3. Remove `nsamp == 1` special case in `push_spec_marg`

### Phase 2: Medium Refactor (1 day)
4. Create batched logdensity function outside per-galaxy vmaps
5. Add pre-compilation step for logdensity
6. Optimize filter application with einsum

### Phase 3: Major Refactor (2-3 days)
7. Replace batch for-loop with `jax.lax.scan` or single vmap
8. Implement pmap for multi-GPU support
9. Profile and optimize MCLMC initialization

---

## Multi-GPU Strategy with pmap

Once single-GPU is optimized, can parallelize across GPUs:

```python
# Split data across devices
n_devices = jax.device_count()
specs_per_device = jnp.reshape(specs, (n_devices, -1, *specs.shape[1:]))
weights_per_device = jnp.reshape(weights, (n_devices, -1, *weights.shape[1:]))

# pmap over devices
@jax.pmap
def process_on_device(spec_chunk, weight_chunk, key):
    # Each device processes its chunk
    return jax.vmap(pae_spec_sample_floatz)(spec_chunk, weight_chunk, key)

# Run on all GPUs in parallel
all_results = process_on_device(specs_per_device, weights_per_device, device_keys)

# Reshape back
all_results = all_results.reshape(-1, *all_results.shape[2:])
```

**Expected Speedup**: Near-linear scaling with number of GPUs (e.g., 4 GPUs = 3.5-4x)

---

## Current Timing Breakdown (Estimated)

Based on debug prints and typical JAX behavior:

| Operation | Time per Galaxy | Parallelized? | Optimization Potential |
|-----------|----------------|---------------|----------------------|
| Logdensity JIT compilation | 0.1-1.0s | ❌ No | **HIGH** - do once |
| MCLMC init | 0.01-0.1s | ⚠️ Partial (vmap) | MEDIUM - pre-compile |
| MCLMC sampling (2000 steps) | 5-15s | ✅ Yes (vmap) | LOW - already good |
| Profile likelihood | 2-5s | ❌ No (lax.map) | **HIGH** - use vmap |
| Total per galaxy | **7-20s** | | |

With optimizations: **3-8s per galaxy** (2-3x speedup)

---

## Memory Considerations

Current memory usage (per galaxy):
- Samples: `(4 chains, 2000 steps, 6 params)` = 4 × 2000 × 6 × 4 bytes = 192 KB
- Profile likelihood: `(4 chains, 1000 post-burnin, 1)` = 16 KB
- Intermediate arrays: ~1-2 MB

With full batch (1000 galaxies):
- Samples: 192 MB
- Profile: 16 MB
- Intermediates: 1-2 GB

**Current batch size of 100-175 is reasonable** for GPU memory. Could potentially increase to 200-250 with optimizations.

---

## Testing Strategy

1. **Create minimal test case**: 10 galaxies, 100 steps
2. **Profile current implementation**: Use `jax.profiler.trace()`
3. **Apply optimizations incrementally**: Measure each change
4. **Verify correctness**: Compare results before/after
5. **Scale up**: Test with full batch sizes

Example profiling code:
```python
import jax.profiler

# Profile a batch run
jax.profiler.start_trace("/tmp/jax-trace")
results = run_batched_sampler(PAE_obj, specs[:10], ...)
jax.profiler.stop_trace()

# View trace at chrome://tracing
```

---

## Questions for Consideration

1. **Memory vs Speed tradeoff**: Would you rather process 175 galaxies at 10s each, or 100 galaxies at 5s each?

2. **Multi-GPU**: Do you have access to multi-GPU nodes? If so, pmap would be valuable.

3. **Recompilation**: Are you running the same configuration multiple times? If so, compilation cost is amortized.

4. **Precision**: Are you using float32 or float64? Float32 is 2x faster for memory bandwidth.

5. **Profile likelihood**: Is the full redshift profile necessary, or just point estimates? Could save significant time.

6. **Burn-in**: Could you reduce nsamp_init (500) or num_steps (2000) based on convergence analysis?

---

## Next Steps

**Without modifying code**, you can:
1. Profile current implementation with `jax.profiler` to confirm bottlenecks
2. Check GPU utilization with `nvidia-smi` during runs
3. Experiment with batch sizes (50, 100, 150, 200) to find memory sweet spot
4. Test if reducing num_steps affects convergence (e.g., 1500 vs 2000)

**When ready to implement**, start with Phase 1 optimizations for immediate gains.
