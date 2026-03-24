# Issues and Fixes Summary

## Issue 1: Gaussian Redshift Prior Not Being Applied

### Problem
Even though `sigma_prior=0.6` is specified in `redshift_job.py` and `include_gaussian_prior=True` is set in the `MCLMCSamplingConfig`, the Gaussian redshift prior was not actually being used during sampling.

### Root Cause
In `inference/like_prior.py`, the `make_logdensity_fn_optzprior` function uses a switch statement with `redshift_prior_type` to determine which prior to apply:
- `redshift_prior_type=0`: No prior (default)
- `redshift_prior_type=1`: Gaussian prior
- `redshift_prior_type=2`: BPZ magnitude-dependent prior

The problem was that `redshift_prior_type` was NOT being passed through from the sampling configuration. It defaulted to 0, meaning no prior was applied even when `include_gaussian_prior=True`.

### Fix Applied
In `sampling/sample_pae_batch_refactor.py`, line 251, added the `redshift_prior_type` parameter:

```python
log_p = make_logdensity_fn_optzprior(
    PAE_obj, x_obs, weight, z_min=sampler_cfg.zmin, z_max=sampler_cfg.zmax,
    nf_alpha=sampler_cfg.nf_alpha,
    redshift_in_flow=sampler_cfg.redshift_in_flow,
    z0_prior=sampler_cfg.z0_prior,
    sigma_prior=sampler_cfg.sigma_prior,
    include_gaussian_prior=sampler_cfg.include_gaussian_prior,
    redshift_prior_type=1 if sampler_cfg.include_gaussian_prior else 0  # NEW LINE
)
```

### Verification
The Gaussian prior with parameters:
- Center: `z0_prior = 0.65` (from MCLMCSamplingConfig default)
- Width: `sigma_prior = 0.6` (set in redshift_job.py)

will now be properly applied when `include_gaussian_prior=True`.

The prior is: `p(z) ∝ exp(-(z - 0.65)² / (2 × 0.6²))`

---

## Issue 2: Burn-in Trimming in Comparison Plots

### Problem
The user asked whether burn-in samples are being excluded when `compare_redshift_results.py` loads the chains for plotting.

### Current Behavior
**NO burn-in trimming occurs** in the comparison plotting workflow:

1. **`diagnostics_jax.py:save_redshift_results()`** (lines 552-641):
   - This function DOES use `burn_in` parameter (default=1000) to compute statistics
   - It excludes burn-in when computing R-hat: `gelman_rubin(all_samples[x, :, burn_in:, -1])`
   - BUT it saves the **full chains** to the `.npz` files without trimming

2. **`visualization/result_plotting_fns.py:prepare_data_for_plotting()`** (line 759):
   - Loads `pae_samples = pae_sample_res['all_samples']`
   - This includes the full chains WITH burn-in samples
   - Added comment: `# NOTE: Full chains, NO burn-in trimming here!`

3. **Other plotting functions**:
   - `compare_PDFs_TF_PAE()` (line 237) hardcodes burn-in=1000: `zsamp_pae = all_zsamp_pae[x,:,1000:,-1].ravel()`
   - But `compare_pae_tf_redshifts()` and `plot_coverage_comparison_grid()` use the full chains

### Implications

**For `compare_pae_tf_redshifts()`:**
- Uses summary statistics (`z_med`, `err_low`, `err_high`) which WERE computed excluding burn-in
- ✅ No issue - the medians/errors already account for burn-in

**For `plot_coverage_comparison_grid()`:**
- Uses `pae_samples` directly to compute PIT values
- ❌ Includes burn-in samples in coverage analysis
- This could slightly bias the coverage estimates if burn-in hasn't converged

### Recommended Fix Options

#### Option 1: Trim burn-in in `prepare_data_for_plotting()`
Add a `burn_in` parameter and trim the samples:

```python
def prepare_data_for_plotting(pae_save_fpath, pae_sample_fpath,
                              tf_results, tf_zpdf_fine_z,
                              nsrc=100, src_idxs=None, burn_in=1000):
    # ... existing code ...
    pae_samples = pae_sample_res['all_samples']
    
    # Trim burn-in if samples are in shape (n_src, n_chain, n_step, n_param)
    if len(pae_samples.shape) == 4:
        pae_samples = pae_samples[:, :, burn_in:, :]
    
    # ... rest of code ...
```

#### Option 2: Save trimmed samples in `save_redshift_results()`
Modify `diagnostics_jax.py` to save both full and trimmed chains:

```python
if sample_fpath is not None:
    print('Saving samples to ', sample_fpath)
    # Trim burn-in from samples before saving
    if len(all_samples.shape) == 4:
        all_samples_trimmed = all_samples[:, :, burn_in:, :]
    else:
        all_samples_trimmed = all_samples
        
    np.savez(sample_fpath, 
             ztrue=redshifts_true, 
             all_samples=all_samples_trimmed,  # Save trimmed version
             sampling_mode=sampling_mode, 
             ae_redshifts=ae_redshifts, 
             z_bins_mcpl=z_bins_mcpl, 
             all_mcpl=all_mcpl,
             burn_in=burn_in)  # Save burn_in value for reference
```

#### Option 3: Add burn-in parameter to `compare_redshift_results.py`
Let the user specify burn-in when running the comparison script.

### Current Burn-in Settings

From `redshift_job.py`:
```python
cfg = MCLMCSamplingConfig(
    burn_in=500,  # burn-in steps
    # ...
)
```

From `save_redshift_results()` default:
```python
def save_redshift_results(fpath, all_samples, ..., burn_in=1000, ...):
```

⚠️ **Note the mismatch**: `redshift_job.py` sets `burn_in=500` but `save_redshift_results()` defaults to `burn_in=1000`.

### Recommendation

I recommend **Option 1** (trim in `prepare_data_for_plotting`) because:
- It's the easiest fix with minimal disruption
- Maintains backward compatibility
- Allows users to control burn-in at plot-time
- The burn-in value (1000 steps) used in existing plotting functions is reasonable

Would you like me to implement Option 1?
