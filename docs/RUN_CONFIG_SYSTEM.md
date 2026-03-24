# Run Configuration System

## Overview

As of December 2025, the redshift pipeline automatically saves all run parameters to a `run_params.npz` file. This makes the system self-documenting and eliminates parameter mismatch errors in downstream analysis.

## How It Works

### 1. During Redshift Runs

When running `redshift_job_batched.py`, a `run_params.npz` file is automatically saved to the results directory with all configuration:

```python
# Saved automatically in: 
# {scratch_basepath}/data/pae_sample_results/MCLMC/batched/{datestr}/run_params.npz
```

**Parameters saved:**
- **PAE Model Config**: `run_name`, `filter_set_name`, `nlatent`, `sig_level_norm`, `filename_flow`
- **Batch Processing**: `batch_size`, `sampling_batch_size`, `max_batches`
- **Sampling Config**: `num_steps`, `nsamp_init`, `nchain_per_gal`, `burn_in`, `chi2_red_threshold`, `gr_threshold`, `init_reinit`
- **Priors**: `redshift_prior_type`, `z0_prior`, `sigma_prior`
- **Filtering**: `filter_specz`, `z_min`, `z_max`, `chi2_red_max`
- **Other**: `fix_z`, `parquet_file`, `datestr`, `use_multicore`, `n_devices_per_node`

### 2. In Plotting Scripts

Plotting scripts (`generate_source_reconstructions.py`, etc.) automatically detect and load `run_params.npz`:

```bash
# Just provide the datestr - configuration is auto-loaded!
./scripts/run_source_plots.sh multicore_test_16k_wf_123025
```

**Auto-loading behavior:**
1. Check for `run_params.npz` in results directory
2. If found: Load all PAE config automatically
3. If not found: Fall back to command-line arguments (with warning)

**What gets loaded:**
- ✅ Correct `run_name` for the PAE model
- ✅ Correct `filter_set_name` (e.g., SPHEREx_filter_306)
- ✅ Correct `nlatent` and `sig_level_norm`
- ✅ Correct `batch_size` for sample loading
- ✅ Correct `fix_z` flag

## Benefits

### Before (Manual Configuration)
```bash
# Had to remember/guess all these parameters!
python scripts/generate_source_reconstructions.py \
    --datestr multicore_test_16k_wf_123025 \
    --run-name jax_conv1_nlatent=5_siglevelnorm=0.01_newAllen_all_091325 \
    --filter-set SPHEREx_filter_306 \
    --nlatent 5 \
    --sig-level-norm 0.01 \
    --batch-size 800
```

**Problems:**
- ❌ Easy to use wrong parameters
- ❌ No record of what parameters were actually used
- ❌ Filter set mismatch common (102 vs 306)
- ❌ Manual parameter tracking required

### After (Automatic Configuration)
```bash
# Configuration auto-loaded from run_params.npz!
./scripts/run_source_plots.sh multicore_test_16k_wf_123025
```

**Advantages:**
- ✅ Parameters guaranteed to match original run
- ✅ Self-documenting (just look at run_params.npz)
- ✅ Perfect reproducibility
- ✅ No more parameter mismatch errors
- ✅ Easy auditing of what parameters produced which results

## Inspecting Saved Configuration

```python
import numpy as np

# Load config
config = np.load('run_params.npz', allow_pickle=True)

# View all parameters
print("PAE Model:")
print(f"  run_name: {config['run_name']}")
print(f"  filter_set: {config['filter_set_name']}")
print(f"  nlatent: {config['nlatent']}")
print(f"  sig_level_norm: {config['sig_level_norm']}")

print("\nBatch Processing:")
print(f"  batch_size: {config['batch_size']}")
print(f"  sampling_batch_size: {config['sampling_batch_size']}")
print(f"  max_batches: {config['max_batches']}")

print("\nSampling:")
print(f"  num_steps: {config['num_steps']}")
print(f"  nchain_per_gal: {config['nchain_per_gal']}")
print(f"  burn_in: {config['burn_in']}")
```

## Backward Compatibility

For runs before this system was implemented (no `run_params.npz`):

1. Plotting scripts print a warning:
   ```
   ⚠ No run_params.npz found - using command-line arguments
   (This may cause issues if parameters don't match the original run)
   ```

2. You can still use command-line arguments:
   ```bash
   python scripts/generate_source_reconstructions.py \
       --datestr old_run_without_config \
       --run-name ... \
       --filter-set ... \
       --nlatent ... \
       --sig-level-norm ...
   ```

## Implementation Details

### redshift_job_batched.py

```python
# Save configuration immediately after creating output directory
run_params = {
    'run_name': run_name,
    'filter_set_name': filter_set_name,
    'nlatent': nlatent,
    'sig_level_norm': sig_level_norm,
    # ... all other parameters
}

config_file = save_dir / 'run_params.npz'
np.savez(config_file, **run_params)
```

### generate_source_reconstructions.py

```python
# Auto-load if available
config_file = result_dir / 'run_params.npz'
if config_file.exists():
    saved_config = np.load(config_file, allow_pickle=True)
    
    pae_config = {
        'run_name': str(saved_config['run_name']),
        'filter_set_name': str(saved_config['filter_set_name']),
        'nlatent': int(saved_config['nlatent']),
        'sig_level_norm': float(saved_config['sig_level_norm']),
    }
    
    # Also auto-set batch_size, fix_z, etc.
```

## Future Enhancements

Potential additions to `run_params.npz`:

- ✨ Git commit hash (for code versioning)
- ✨ Timestamp (start/end times)
- ✨ System info (node names, GPU types)
- ✨ Software versions (JAX, NumPy, etc.)
- ✨ Data provenance (parquet file checksums)

## Summary

The `run_params.npz` system makes the pipeline robust and self-documenting. Just provide the `datestr` and everything else is automatically configured correctly!

**Key principle:** Parameters should be saved once during production and loaded automatically everywhere else.
