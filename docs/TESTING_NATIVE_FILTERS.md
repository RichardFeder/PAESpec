# Testing Pseudo-Native Filters

This directory contains scripts for testing the pseudo-native filter implementation on real SPHEREx data.

## Quick Start

### 1. Simple Forward Model Test

Test that pseudo-native filters work correctly without full MCLMC sampling:

```bash
cd /global/homes/r/rmfeder/sed_vae/sp-ae-herex

# Test on 10 sources (fast, ~1-2 minutes)
python scripts/test_native_filters_simple.py --max-sources 10 --save-results

# Test on more sources
python scripts/test_native_filters_simple.py --max-sources 100 --nsamp 200
```

**What it does:**
- Loads data with pseudo-native filters
- Runs PAE forward model with both homogenized and pseudo-native filters
- Compares likelihoods and timing
- Saves results to CSV

**Expected output:**
- Log-likelihood differences between approaches
- Timing comparison (pseudo-native should be ~1.05-1.10x slower)
- Memory footprint (~0.4 MB per source for filter curves)

### 2. Full Inference Test (Small Batch)

**Note:** The batched inference script (`redshift_job_batched_native.py`) requires modifications to the sampling wrapper to pass `filter_curves` through the MCLMC sampler. This is more complex and requires:

1. Updating `sample_mclmc_wrapper()` to accept `filter_curves` and `lam_interp` parameters
2. Modifying `run_batched_sampler()` to pass these through to the logdensity function
3. Updating `pae_spec_sample_floatz()` to use per-source filters

For now, use the simple test above to validate the implementation.

## Test Results to Look For

### ✅ Success Indicators

1. **Data Loading:**
   ```
   Loaded 10 sources in X.XX s
     Filter curves shape: (10, 102, 1000)
     Filter curves memory: Y.Y MB
   ```

2. **Forward Model:**
   ```
   Source 0:
     Homogenized filters:
       Best log-likelihood: -XXX.XX
     Pseudo-native filters:
       Best log-likelihood: -XXX.XX
     Difference: ~small (< 10)
   ```

3. **Weight Masking:**
   ```
   WEIGHT COMBINATION
     measurement_weights (padding): nonzero: ~95-100%
     data_quality_weights (outliers): nonzero: ~85-95%
     final_weights (combined): nonzero: ~80-95%
   ```

### ⚠️ Issues to Watch For

1. **Memory errors:** Reduce `--max-sources` or `--batch-size`
2. **Large log-likelihood differences:** Check filter interpolation
3. **NaN values:** Check weight masking and data quality
4. **Very slow performance:** Check JAX compilation (first run is slow)

## Files

- `test_native_filters_simple.py`: Simple forward model test (recommended starting point)
- `redshift_job_batched_native.py`: Full batched inference (requires sampling wrapper updates)
- `test_pseudo_native_filters.ipynb`: Interactive validation notebook

## Data Requirements

The parquet file must contain:
- `SPHERExRefID`: Source IDs
- `lambda`: Arrays of measured wavelengths per source
- `flux_dered`: Dereddened flux measurements
- `flux_err_dered`: Flux uncertainties
- `z_specz` or `z_best_gals`: Redshifts (optional but recommended)

## Next Steps

After validating with the simple test:

1. **Profile memory usage** for larger batches (100, 500, 1000 sources)
2. **Benchmark speed** vs homogenized filters
3. **Integrate with MCLMC sampler** (requires sampling wrapper modifications)
4. **Run on full dataset** in batches

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the sp-ae-herex directory
cd /global/homes/r/rmfeder/sed_vae/sp-ae-herex
export PYTHONPATH=/global/homes/r/rmfeder/sed_vae/sp-ae-herex:$PYTHONPATH
```

### GPU Memory Issues
Reduce batch sizes:
- `--max-sources 50` (for simple test)
- `--batch-size 200` (for batched inference)

### Filter Loading Errors
Check filter directory exists and contains files:
```bash
ls /pscratch/sd/r/rmfeder/data/filters/SPHEREx_filter_306/
# Should show: SPHEREx_band1_ch*.dat files
```

## Contact

For issues or questions, see:
- Documentation: `docs/PSEUDO_NATIVE_FILTERS.md`
- Test notebook: `notebooks/test_pseudo_native_filters.ipynb`
