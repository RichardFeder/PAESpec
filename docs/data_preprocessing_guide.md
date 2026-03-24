# Real SPHEREx Data Preprocessing Guide

## Overview

The real SPHEREx photometry data requires preprocessing to handle several types of problematic values. This guide explains the preprocessing pipeline and how to use it.

## Data Quality Issues Addressed

### 1. Missing Data
- **Issue**: Missing observations have uncertainties set to ~50,000
- **Solution**: Zero-weighted (completely excluded from likelihood)
- **Threshold**: `err > 40,000` (slightly below 50,000 to catch these)

### 2. Absurdly High Fluxes
- **Issue**: Some flux measurements exceed 10^4, which are unphysical for normalized data
- **Solution**: Downweighted by 100x (1% of normal weight)
- **Threshold**: `flux > 10^4`

### 3. Extreme Outliers
- **Issue**: Individual measurements that are >100x the per-band mean
- **Solution**: Downweighted by 100x (1% of normal weight)
- **Threshold**: `flux / mean(flux_band) > 100`

### 4. NaN Values
- **Issue**: NaN in flux or uncertainty arrays
- **Solution**: Zero-weighted, flux set to 0, uncertainty set to 1.0
- **Common in**: Flux uncertainties

### 5. Inf Values
- **Issue**: Infinite values in flux or uncertainty
- **Solution**: Zero-weighted, flux set to 0, uncertainty set to 1.0

### 6. Non-positive Uncertainties
- **Issue**: Uncertainties ≤ 0 (invalid for likelihood calculation)
- **Solution**: Zero-weighted, flux set to 0, uncertainty set to 1.0

## Usage

### Automatic Preprocessing (Recommended)

The preprocessing is **enabled by default** in `load_real_spherex_parquet`:

```python
from models.pae_jax import load_real_spherex_data

# Preprocessing happens automatically
dat_obs, property_cat_df, _, _, wave_obs = load_real_spherex_data(
    parquet_file='path/to/data.parquet',
    filter_set_name='SPHEREx_filter_306'
)

# Access cleaned data
flux = dat_obs.flux  # Already preprocessed
flux_unc = dat_obs.flux_unc  # Already preprocessed

# Check data quality per source
data_quality = property_cat_df['data_quality_weight']  # Fraction of valid bands
```

### Disable Preprocessing

To disable preprocessing (not recommended for real data):

```python
from data_proc.dataloader_jax import load_real_spherex_parquet

dat_obs, property_cat_df = load_real_spherex_parquet(
    parquet_file='path/to/data.parquet',
    filter_set_name='SPHEREx_filter_306',
    wave_obs=wave_obs,
    preprocess_outliers=False  # Turn off preprocessing
)
```

### Manual Preprocessing

For custom preprocessing or diagnostics:

```python
from data_proc.dataloader_jax import preprocess_real_data_outliers

# Your raw data
flux_raw = ...  # shape (n_sources, n_bands)
err_raw = ...   # shape (n_sources, n_bands)

# Apply preprocessing
flux_clean, err_clean, weights = preprocess_real_data_outliers(
    flux_raw, 
    err_raw,
    missing_data_threshold=40000,  # Adjust thresholds as needed
    high_flux_threshold=1e4,
    extreme_outlier_factor=100,
    downweight_factor=0.01,
    verbose=True  # Print diagnostics
)

# weights array:
#   0.0 = zero-weighted (excluded)
#   0.01 = downweighted (1% weight)
#   1.0 = full weight (good data)
```

## Validation

### Quick Validation (Interactive)

Run the validation notebook:

```bash
jupyter notebook notebooks/quick_validation.ipynb
```

This notebook:
- Loads filters and checks coverage
- Initializes PAE model
- Loads real photometry with preprocessing
- Shows detailed diagnostics of data quality issues
- Plots weight distributions and valid data coverage
- Compares synthetic vs real photometry

### Full Validation (Command-line)

Run the validation script on your full dataset:

```bash
cd /global/homes/r/rmfeder/sed_vae/sp-ae-herex
python scripts/validate_parquet_pipeline.py --n-sources 1000
```

This generates diagnostic plots in `figures/validation_MMDDYY/`:
- Filter coverage plot
- Synthetic photometry examples
- Real photometry examples

## Interpreting Results

### Data Quality Metrics

After preprocessing, check these metrics:

```python
# Fraction of data with issues
total_flagged = np.sum(weights == 0)
fraction_flagged = total_flagged / weights.size

# Per-source data quality
valid_bands_per_source = np.sum(weights > 0, axis=1)
fraction_valid = valid_bands_per_source / weights.shape[1]

# Typical values for SPHEREx L3 data:
# - Missing data: 5-15% (varies by field)
# - High flux outliers: <1%
# - Extreme outliers: <1%
# - NaN in uncertainties: 1-5%
```

### Good vs Problematic Sources

```python
# Sources with >90% valid data
good_sources = fraction_valid > 0.9

# Sources with <50% valid data (may want to exclude entirely)
bad_sources = fraction_valid < 0.5

print(f"Good sources: {np.sum(good_sources)} ({100*np.mean(good_sources):.1f}%)")
print(f"Bad sources: {np.sum(bad_sources)} ({100*np.mean(bad_sources):.1f}%)")
```

## Impact on Likelihood

The preprocessing weights are incorporated into the likelihood calculation:

```
chi^2 = sum_i w_i * (f_obs_i - f_model_i)^2 / sigma_i^2

where:
  w_i = preprocessing weight (0, 0.01, or 1.0)
  f_obs_i = observed flux
  f_model_i = model flux
  sigma_i = flux uncertainty
```

- Zero-weighted data (w=0): No contribution to chi^2
- Downweighted data (w=0.01): Minimal contribution (allows extreme values without dominating fit)
- Full weight data (w=1.0): Normal contribution

## Customizing Thresholds

If you need different thresholds based on your data characteristics:

```python
from data_proc.dataloader_jax import preprocess_real_data_outliers

# Example: More conservative (flag less data)
flux_clean, err_clean, weights = preprocess_real_data_outliers(
    flux_raw, err_raw,
    missing_data_threshold=60000,    # Higher threshold
    high_flux_threshold=5e4,         # Much higher threshold
    extreme_outlier_factor=1000,     # Only flag really extreme outliers
    downweight_factor=0.1,           # Less aggressive downweighting (10%)
    verbose=True
)

# Example: More aggressive (flag more data)
flux_clean, err_clean, weights = preprocess_real_data_outliers(
    flux_raw, err_raw,
    missing_data_threshold=20000,    # Lower threshold
    high_flux_threshold=1e3,         # Lower threshold
    extreme_outlier_factor=10,       # Flag moderate outliers
    downweight_factor=0.001,         # More aggressive downweighting (0.1%)
    verbose=True
)
```

## Troubleshooting

### "Too much data being flagged"

If >30% of your data is being flagged:
1. Check the diagnostic output to see which category dominates
2. Adjust thresholds accordingly
3. Consider if your dataset has systematic issues

### "Data still has unrealistic values"

If you see unrealistic values after preprocessing:
1. Check if preprocessing is enabled: `preprocess_outliers=True`
2. Lower the thresholds for high flux and outliers
3. Add custom preprocessing for your specific issues

### "Fits are poor"

If your fits are poor after preprocessing:
1. Make sure you're not flagging too much good data
2. Check if downweighting is too aggressive (increase `downweight_factor`)
3. Verify the preprocessing isn't introducing biases

## Technical Details

The preprocessing function is in:
- **Location**: `data_proc/dataloader_jax.py`
- **Function**: `preprocess_real_data_outliers()`
- **Integration**: Called automatically in `load_real_spherex_parquet()`
- **Output**: Returns cleaned flux, cleaned uncertainties, and weights array

The weights are currently **not** propagated through to the final likelihood weights (which are computed from uncertainties), but the flagged data is set to zero flux, effectively removing it from fits.

## References

- Main data loading: [data_proc/dataloader_jax.py](../data_proc/dataloader_jax.py)
- Validation notebook: [notebooks/quick_validation.ipynb](../notebooks/quick_validation.ipynb)
- Validation script: [scripts/validate_parquet_pipeline.py](../scripts/validate_parquet_pipeline.py)
- Batch processing: [scripts/redshift_job_batched.py](../scripts/redshift_job_batched.py)
