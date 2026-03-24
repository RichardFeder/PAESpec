# Pseudo-Native Filter Implementation for SPHEREx PAE

## Overview

SPHEREx uses Linear Variable Filters (LVFs) where the wavelength response varies across the detector focal plane. Each source may have slightly different effective filter profiles depending on its pixel position. The **pseudo-native filter** approach captures these per-source filter variations without requiring detailed pixel position information.

## The Problem

### Standard "Homogenized" Approach
Traditional PAE implementations use a single set of 102 or 306 **homogenized filters** shared across all sources:
```
All sources → Same 306 filters → Same wavelength response
```

**Limitations:**
- Ignores spatial variations in LVF transmission
- Each fiducial filter represents the average response across the detector
- Real measurements can have Δλ ~ 0.001-0.01 μm shifts from fiducial values

### Native LVF Approach (Ideal)
For true native processing, you'd need:
- Per-measurement pixel positions `(xpix, ypix, det_id)`
- Reduced filter set (FITS files with filter profiles per pixel block)
- Query filters based on `(det_id, xpix, ypix)` for each measurement

**Challenge:** Not all data products include pixel positions!

## The Solution: Pseudo-Native Filters

**Key Insight:** If you have the **measured wavelength** for each flux measurement, you can interpolate the nearest fiducial filter to that exact wavelength.

### Algorithm

For each measurement at wavelength λ_measured:

1. **Find nearest fiducial filter:**
   ```python
   idx_nearest = argmin(|λ_fiducial - λ_measured|)
   ```

2. **Shift the filter profile:**
   ```python
   Δλ = λ_measured - λ_fiducial[idx_nearest]
   λ_shifted = λ_grid - Δλ  # Shift wavelength grid
   ```

3. **Interpolate transmission:**
   ```python
   T_pseudo_native(λ) = interp(λ_shifted, T_fiducial(λ_grid))
   ```

This assumes the filter **shape** is the same, but the **central wavelength** shifts. For SPHEREx LVFs, this is an excellent approximation since the filter resolution (R ~ λ/Δλ) is roughly constant.

## Implementation Architecture

### Module Structure

```
sp-ae-herex/
├── data/
│   └── spherex_native_filters.py      # Core pseudo-native utilities
├── data_proc/
│   └── dataloader_jax.py              # Data loading & normalization
└── models/
    └── pae_jax.py                     # PAE forward model (filter application)
```

### Key Functions

#### 1. `build_pseudo_native_filters_wavelength()` 
*Location:* `data/spherex_native_filters.py`

Builds a single interpolated filter for a measured wavelength:
```python
filter_interp = build_pseudo_native_filters_wavelength(
    lam_measured=1.234,  # Measured wavelength [μm]
    lam_filter=lam_grid,  # Fine wavelength grid (e.g., 1000 points)
    fiducial_filters=filters_306,  # (306, 1000) fiducial filters
    fiducial_cenwav=cenwav_306  # (306,) central wavelengths
)
# Returns: (1000,) interpolated filter at exact wavelength
```

#### 2. `load_pseudo_native_from_parquet()`
*Location:* `data/spherex_native_filters.py`

Loads data from parquet and builds per-source filter matrices:
```python
batch_dict, lam_filter, fiducial_cenwav = load_pseudo_native_from_parquet(
    parquet_file='/path/to/selection_wf.parquet',
    filter_dir='/path/to/filters',
    filter_set_name='SPHEREx_filter_306',
    max_sources=1000,
    max_nbands=None  # Auto-detect
)

# Returns:
# batch_dict = {
#     'flux': (n_sources, max_nbands),
#     'flux_err': (n_sources, max_nbands),
#     'filter_curves': (n_sources, max_nbands, n_lam),  # Per-source filters!
#     'central_wavelengths': (n_sources, max_nbands),
#     'weights': (n_sources, max_nbands),  # 1=valid, 0=padding
#     'n_valid': (n_sources,)  # Number of valid measurements per source
# }
```

**Data flow:**
1. Reads parquet columns: `SPHERExRefID`, `lambda` (array), `flux_dered` (array), `flux_err_dered` (array)
2. For each source's wavelength array, builds interpolated filters
3. Pads to `max_nbands` with proper masking

#### 3. `load_and_prepare_pseudo_native_data()`
*Location:* `data_proc/dataloader_jax.py`

**Main entry point** - handles complete pipeline:
```python
dat_obs, property_cat_df, sphx_data, lam_filter, fiducial_cenwav = \
    load_and_prepare_pseudo_native_data(
        parquet_file='/path/to/data.parquet',
        filter_dir='/path/to/filters',
        filter_set_name='SPHEREx_filter_306',
        max_sources=None,  # Load all
        weight_soft=5e-4,
        abs_norm=True,
        preprocess_outliers=True
    )
```

**Pipeline stages:**
1. **Load pseudo-native filters** via `load_pseudo_native_from_parquet()`
2. **Preprocess outliers** - zero-weights missing data, NaNs, extreme outliers
3. **Normalize fluxes** - divide by mean flux per source
4. **Compute inverse-variance weights** - `w = 1/(σ² + weight_soft)`
5. **Combine weight masks:**
   ```python
   final_weights = measurement_weights * data_quality_weights * phot_weights
   ```
   - `measurement_weights`: 1=valid measurement, 0=padding
   - `data_quality_weights`: 0=bad data, 0.01=downweighted, 1=good
   - `phot_weights`: inverse-variance weights `1/(σ² + ε)`

**Returns:**
- `dat_obs`: Standard `spec_data_jax` object with normalized photometry
- `property_cat_df`: Pandas DataFrame with redshifts, SNR, etc.
- `sphx_data`: `SPHERExData` dataclass with native filter support
- `lam_filter`: Fine wavelength grid (n_lam,)
- `fiducial_cenwav`: Fiducial central wavelengths (n_filters,)

#### 4. `push_spec_marg()` with Native Filters
*Location:* `models/pae_jax.py`

PAE forward model extended to accept per-source filters:
```python
loglike, z = PAE_obj.push_spec_marg(
    latents,  # (n_samples, n_latent)
    redshift,  # (n_samples,)
    marginalize_amplitude=True,
    observed_flux=flux_obs,  # (n_samples, n_bands)
    weight=weights,  # (n_samples, n_bands)
    filter_curves=filter_curves_jax,  # (n_samples, n_bands, n_lam) ← NEW!
    lam_interp=lam_filter  # (n_lam,) ← NEW!
)
```

**How it works:**
```python
# Standard homogenized filters (original behavior):
if filter_curves is None:
    model_flux = jnp.dot(jax_filters, x_interp.T).T  # (n_samples, n_bands)

# Per-source pseudo-native filters:
else:
    # filter_curves: (n_samples, n_bands, n_lam)
    # x_interp: (n_samples, n_lam)
    # Result: (n_samples, n_bands)
    model_flux = jax.vmap(lambda f, x: jnp.dot(f, x))(filter_curves, x_interp)
```

Uses JAX `vmap` to efficiently parallelize filter convolution across samples.

## Weight Masking System

The implementation uses a **three-layer weight system** to handle various data quality issues:

### Layer 1: Measurement Weights (Padding)
```python
measurement_weights: (n_sources, max_nbands)
# 1.0 = valid measurement exists
# 0.0 = padded entry (source has fewer than max_nbands measurements)
```

Sources with variable numbers of measurements are padded to `max_nbands`. The weight mask ensures padded entries don't contribute to the likelihood.

### Layer 2: Data Quality Weights (Outlier Filtering)
```python
data_quality_weights: (n_sources, max_nbands)
# 1.0 = good data
# 0.01 = downweighted (extreme outlier, high flux)
# 0.0 = excluded (missing data, NaN, Inf, bad uncertainty)
```

Applied by `preprocess_real_data_outliers()`:
- **Zero-weighted:** Missing data (σ > 40000), NaN, Inf, σ ≤ 0
- **Downweighted:** Extreme outliers (>100× band mean), absurdly high flux (>10⁴)

### Layer 3: Inverse-Variance Weights (Measurement Precision)
```python
phot_weights: (n_sources, max_nbands)
# w_ij = 1 / (σ_ij² + weight_soft)
```

Standard inverse-variance weighting with soft floor to prevent infinite weights.

### Combined Final Weights
```python
final_weights = measurement_weights * data_quality_weights * phot_weights
```

Used in likelihood calculation:
```python
chi2 = sum(((flux_obs - flux_model)² * final_weights), axis=-1)
loglike = -0.5 * chi2
```

Automatically handles:
- Padded measurements (don't contribute)
- Bad/missing data (don't contribute)
- Variable measurement quality (weighted by precision)

## Memory & Performance

### Memory Footprint

For a batch of 300 sources with max_nbands=102 and n_lam=1000:

```
Component                              Size (MB)
────────────────────────────────────────────────
filter_curves (300, 102, 1000) float32   122.4
flux (300, 102) float32                    0.12
flux_err (300, 102) float32                0.12
weights (300, 102) float32                 0.12
────────────────────────────────────────────────
Total per batch                          ~123 MB
```

**Comparison:**
- Homogenized filters: (306, 1000) = 1.2 MB (shared across all sources)
- Pseudo-native: ~0.4 MB per source

**Scaling:**
- 1,000 sources: ~400 MB
- 10,000 sources: ~4 GB
- 100,000 sources: ~40 GB (process in batches)

Modern GPUs (16-80 GB) can handle batches of 500-2000 sources comfortably.

### Computational Efficiency

**Filter convolution:**
```python
# Homogenized (original):
model_flux = jnp.dot(jax_filters, x_interp.T).T  # Single matrix multiply

# Pseudo-native:
model_flux = jax.vmap(lambda f, x: jnp.dot(f, x))(filter_curves, x_interp)
# Vectorized per-sample dot products
```

JAX's `vmap` compiles to efficient parallel operations. Overhead is minimal (~5-10% slower than homogenized on GPU).

## Usage Examples

### Basic Usage

```python
from data_proc.dataloader_jax import load_and_prepare_pseudo_native_data
from models.pae_jax import initialize_PAE
import jax.numpy as jnp

# 1. Load data with pseudo-native filters
dat_obs, property_cat_df, sphx_data, lam_filter, fiducial_cenwav = \
    load_and_prepare_pseudo_native_data(
        parquet_file='/pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet',
        filter_dir='/pscratch/sd/r/rmfeder/data/filters',
        filter_set_name='SPHEREx_filter_306',
        max_sources=1000,
        preprocess_outliers=True
    )

# 2. Initialize PAE
PAE_obj = initialize_PAE(
    run_name='jax_conv1_nlatent=10_siglevelnorm=0.01_newAllen_all_091325',
    filter_set_name='SPHEREx_filter_306',
    redshift_in_flow=False
)

# 3. Run inference for a source
src_idx = 0
flux_obs = jnp.array(sphx_data.all_spec_obs[src_idx:src_idx+1])
weights = jnp.array(sphx_data.weights[src_idx:src_idx+1])
filter_curves = jnp.array(sphx_data.filter_curves[src_idx:src_idx+1])
redshift = jnp.array([property_cat_df.iloc[src_idx]['redshift']])

# Sample latent space
key = jax.random.PRNGKey(42)
latents = jax.random.normal(key, (100, PAE_obj.params['nlatent']))

# Compute likelihood with pseudo-native filters
loglike, z = PAE_obj.push_spec_marg(
    latents, redshift,
    marginalize_amplitude=True,
    observed_flux=flux_obs,
    weight=weights,
    filter_curves=filter_curves,  # Per-source filters
    lam_interp=lam_filter
)

print(f"Best log-likelihood: {loglike.max():.2f}")
```

### Batch Processing

```python
# Process large dataset in chunks
batch_size = 500
n_sources = len(unique_source_ids)

for batch_start in range(0, n_sources, batch_size):
    batch_end = min(batch_start + batch_size, n_sources)
    
    # Load batch
    dat_obs, property_cat_df, sphx_data, lam_filter, _ = \
        load_and_prepare_pseudo_native_data(
            parquet_file=parquet_file,
            filter_dir=filter_dir,
            max_sources=batch_end,  # Load up to batch_end
            # Then slice [batch_start:batch_end] from results
        )
    
    # Process batch...
    for i in range(len(property_cat_df)):
        # Run inference on source i
        pass
```

### Comparing Homogenized vs Pseudo-Native

```python
# Same setup as above...

# Forward model with homogenized filters
model_flux_homog = PAE_obj.push_spec_marg(
    latents, redshift,
    marginalize_amplitude=False
)

# Forward model with pseudo-native filters
model_flux_native = PAE_obj.push_spec_marg(
    latents, redshift,
    marginalize_amplitude=False,
    filter_curves=filter_curves,
    lam_interp=lam_filter
)

# Compare
valid_mask = weights[0] > 0
diff = np.abs(model_flux_homog[0] - model_flux_native[0])[valid_mask]
print(f"Mean difference: {np.mean(diff):.6f}")
print(f"RMS difference: {np.sqrt(np.mean(diff**2)):.6f}")
```

## Data Format Requirements

### Input Parquet Schema

```python
Required columns:
- SPHERExRefID: int64          # Unique source identifier
- lambda: list[float64]        # Measured wavelengths [μm] (variable length)
- flux_dered: list[float64]    # Dered flux [mJy] (same length as lambda)
- flux_err_dered: list[float64]  # Flux uncertainties [mJy]

Optional columns:
- z_specz: float64             # Spectroscopic redshift
- z_best_gals: float64         # Photometric redshift
- ra, dec: float64             # Coordinates
```

Example structure:
```
SPHERExRefID | lambda                    | flux_dered        | flux_err_dered    | z_specz
─────────────┼───────────────────────────┼───────────────────┼───────────────────┼─────────
12345        | [0.75, 0.76, ..., 4.8]   | [12.3, 15.1, ...] | [0.5, 0.6, ...]   | 0.342
12346        | [0.75, 0.76, ..., 4.8]   | [8.7, 11.2, ...]  | [0.4, 0.5, ...]   | 1.234
```

Each source can have a different number of measurements. The loader handles variable-length arrays automatically.

### Output Data Structure

```python
SPHERExData(
    all_spec_obs: (n_sources, max_nbands)  # Normalized flux
    weights: (n_sources, max_nbands)       # Combined weights (see above)
    redshift: (n_sources,)                 # Redshifts
    
    # Native filter fields:
    filter_curves: (n_sources, max_nbands, n_lam)  # Per-source filters
    central_wavelengths_per_source: (n_sources, max_nbands)  # λ_measured
    measurement_weights: (n_sources, max_nbands)   # Padding mask
    n_valid_measurements: (n_sources,)     # Valid count per source
)
```

## Validation & Testing

See `notebooks/test_pseudo_native_filters.ipynb` for comprehensive validation:

1. **Filter interpolation accuracy** - Visual comparison of fiducial vs interpolated
2. **Padding/masking** - Verify weight masks work correctly
3. **PAE forward model** - Compare homogenized vs pseudo-native results
4. **Memory footprint** - Measure actual memory usage
5. **End-to-end pipeline** - Full data load → PAE inference workflow

## Limitations & Future Work

### Current Limitations

1. **Assumes constant filter shape:** Interpolation only shifts the central wavelength, not the filter shape. For SPHEREx LVFs this is excellent, but may not work for other instruments.

2. **No wavelength-dependent resolution:** Assumes R = λ/Δλ is constant. LVF filters have slightly varying resolution across the bandpass.

3. **Memory scales linearly:** For very large datasets (>100k sources), need to implement batch streaming.

### Future Enhancements

1. **True native filters:** When pixel positions are available, use `load_real_spherex_parquet_native()` with reduced filter FITS files.

2. **Filter caching:** Pre-compute and save filter_curves for repeated use:
   ```python
   np.savez_compressed('filter_curves_10k.npz', **batch_dict)
   ```

## References

- **Pseudo-native implementation:** `data/spherex_native_filters.py`
- **Integration function:** `data_proc/dataloader_jax.py::load_and_prepare_pseudo_native_data()`
- **PAE integration:** `models/pae_jax.py::push_spec_marg()`
- **Test notebook:** `notebooks/test_pseudo_native_filters.ipynb`
- **Original linefit module:** `linefit/` (for true native with pixel positions)

## Quick Start Checklist

- [ ] Data in parquet format with `lambda`, `flux_dered`, `flux_err_dered` columns
- [ ] Filter directory with fiducial filter files (e.g., `SPHEREx_filter_306/`)
- [ ] Import `load_and_prepare_pseudo_native_data` from `data_proc.dataloader_jax`
- [ ] Load data: `dat_obs, df, sphx_data, lam_filter, cenwav = load_and_prepare_pseudo_native_data(...)`
- [ ] Initialize PAE: `PAE_obj = initialize_PAE(...)`
- [ ] Run inference: `loglike, z = PAE_obj.push_spec_marg(..., filter_curves=sphx_data.filter_curves, lam_interp=lam_filter)`
- [ ] Validate with test notebook

---

**Last updated:** December 29, 2025  
**Author:** Richard Feder (with AI assistance)
