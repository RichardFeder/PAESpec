# Batched Parquet Loading for Large Datasets

## Overview

For processing 100k+ sources from parquet files, batched loading is essential to:
- **Manage memory efficiently**: Load only what's needed (~10 MB per 10k sources)
- **Enable incremental processing**: Process and save results batch-by-batch
- **Support filtering**: Pre-filter sources before loading (e.g., spec-z only)

## Memory Analysis

**Your dataset: `/pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet`**
- Total sources: **141,440**
- Total memory: **~90 MB** (0.7 KB/source)
- With 306 bands × float32 arrays

**Memory per batch:**
- 1,000 sources: ~1 MB
- 10,000 sources: ~10 MB
- 50,000 sources: ~50 MB

**Recommendation**: Use batch_size=10,000 for optimal balance between:
- Memory efficiency (only 10 MB in RAM at once)
- I/O efficiency (not too many small reads)
- Processing throughput (enough work per batch)

## Quick Start

### 1. Simple batched processing

```bash
# Process all 141k sources in batches of 10k
python scripts/redshift_job_batched.py \
    --parquet-file /pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet \
    --batch-size 10000 \
    --datestr 122725_test
```

### 2. Filter for sources with spec-z

```bash
# Only process sources with spectroscopic redshift
python scripts/redshift_job_batched.py \
    --parquet-file /pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet \
    --batch-size 10000 \
    --filter-specz \
    --datestr 122725_specz_only
```

### 3. Redshift range + quality cuts

```bash
# Process 0.5 < z < 1.5 with good SED fits
python scripts/redshift_job_batched.py \
    --parquet-file /pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet \
    --batch-size 10000 \
    --z-min 0.5 \
    --z-max 1.5 \
    --chi2-red-max 2.0 \
    --datestr 122725_z05_15
```

### 4. Test run (first 2 batches only)

```bash
# Quick test on first 20k sources
python scripts/redshift_job_batched.py \
    --parquet-file /pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet \
    --batch-size 10000 \
    --max-batches 2 \
    --datestr 122725_test
```

### 5. Multi-core parallelization

```bash
# Use 4 GPU cores for faster processing
python scripts/redshift_job_batched.py \
    --parquet-file /pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet \
    --batch-size 10000 \
    --use-multicore \
    --n-devices 4 \
    --datestr 122725_multicore
```

## Python API

### Basic usage

```python
from data_proc.batch_parquet_loader import ParquetBatchLoader

# Create loader
loader = ParquetBatchLoader(
    '/pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet',
    batch_size=10000
)

# Iterate over batches
for batch_df, start_idx, end_idx in loader:
    print(f"Processing sources {start_idx}-{end_idx}")
    # Process batch_df...
```

### With filtering

```python
from data_proc.batch_parquet_loader import (
    ParquetBatchLoader,
    filter_has_specz,
    filter_redshift_range,
    combine_filters
)

# Combine filters
filter_fn = combine_filters(
    filter_has_specz,
    filter_redshift_range(0.5, 1.5)
)

# Only loads sources passing filter
loader = ParquetBatchLoader(
    parquet_file,
    batch_size=10000,
    filter_fn=filter_fn
)
```

### Direct integration with pipeline

```python
from data_proc.batch_parquet_loader import load_fiducial_fluxes_batch

# Load specific range
dat_obs, property_cat_df, central_waves, wave_obs = load_fiducial_fluxes_batch(
    parquet_file='/pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet',
    filter_set_name='SPHEREx_filter_306',
    start_idx=0,
    end_idx=10000,
    filter_fn=lambda df: df['z_specz'] > 0  # Only spec-z sources
)

# Use with existing pipeline
from data_proc.dataloader_jax import SPHERExData
spherex_dat = SPHERExData.from_prep(dat_obs, property_cat_df, None)
```

## Available Filters

Predefined filter functions in `batch_parquet_loader.py`:

```python
# 1. Has spectroscopic redshift
filter_has_specz(df)

# 2. Redshift range (uses spec-z if available, else photo-z)
filter_redshift_range(z_min, z_max)

# 3. Good SED fits
filter_good_fits(chi2_red_max=3.0)

# 4. Combine multiple filters (AND logic)
combine_filters(filter1, filter2, ...)

# 5. Custom filter (any function returning boolean mask)
def my_filter(df):
    return (df['frac_nonzero'] > 0.5) & (df['Nsamples'] > 100)
```

## Output Structure

Results are saved per-batch:

```
$SCRATCH/data/pae_sample_results/MCLMC/batched/122725_test/
├── PAE_results_batch0_start0_122725_test.npz
├── PAE_results_batch1_start10000_122725_test.npz
├── PAE_results_batch2_start20000_122725_test.npz
└── ...
```

Each file contains results for one batch (e.g., 10k sources).

## Performance Tips

### Batch Size Guidelines

| Sources | Memory | Recommended batch_size |
|---------|--------|------------------------|
| < 10k   | < 10 MB | Load all at once      |
| 10-50k  | 10-50 MB | 10,000               |
| 50-200k | 50-200 MB | 10,000 - 20,000     |
| > 200k  | > 200 MB | 20,000               |

### When to use batching

**Use batching when:**
- Total sources > 50k
- Running on shared nodes (limit memory footprint)
- Want incremental results (save after each batch)
- Need to filter/subset before full load

**Load all at once when:**
- Total sources < 10k
- Have plenty of memory (> 1 GB available)
- Need to operate on full dataset (e.g., global statistics)

### Multi-core optimization

For large batches, combine with multi-core:

```bash
# 100k sources, 4 cores, 25k per core
python scripts/redshift_job_batched.py \
    --batch-size 25000 \
    --use-multicore \
    --n-devices 4
```

Each core processes ~6k sources (25k / 4), staying below saturation (600 chains/core).

## Troubleshooting

### Memory issues

**Problem**: "Out of memory" error

**Solution**: Reduce batch_size
```bash
--batch-size 5000  # Instead of 10000
```

### Slow I/O

**Problem**: Parquet loading is slow

**Solution**: 
1. Increase batch_size (fewer I/O operations)
2. Use filtering to reduce data volume
3. Consider copying to faster storage (local SSD)

### Empty batches after filtering

**Problem**: "Batch is empty after filtering"

**Solution**: Check your filter criteria:
```python
# Count how many sources pass filter
df = pd.read_parquet(parquet_file)
n_pass = sum(filter_fn(df))
print(f"{n_pass} sources pass filter")
```

## Comparison: Batch vs Full Load

### Example: 141k sources

**Full load (original `redshift_job.py`):**
```bash
# Loads all 141k sources at once (~90 MB)
python scripts/redshift_job.py --ngal 141440 --batch-size 100
```

**Batched load:**
```bash
# Loads 10k at a time (~10 MB), processes 15 batches
python scripts/redshift_job_batched.py --batch-size 10000
```

**Advantages of batching:**
- Lower peak memory (10 MB vs 90 MB)
- Incremental results (can restart from any batch)
- Better for filtering (only load what you need)
- More predictable memory usage

**Disadvantages:**
- Slightly more overhead per batch
- More output files to manage

## Integration with Existing Code

Your existing pipeline remains unchanged:

```python
# Old way (still works)
from models.pae_jax import load_real_spherex_data
dat_obs, property_cat_df, _, _, wave_obs = load_real_spherex_data(
    parquet_file='/path/to/file.parquet',
    filter_set_name='SPHEREx_filter_306'
)

# New way (batched)
from data_proc.batch_parquet_loader import load_fiducial_fluxes_batch
dat_obs, property_cat_df, _, wave_obs = load_fiducial_fluxes_batch(
    parquet_file='/path/to/file.parquet',
    filter_set_name='SPHEREx_filter_306',
    start_idx=0,
    end_idx=10000
)

# Both produce the same data structures
```

## Summary

**For your 141k source dataset:**

1. **Recommended approach**: Use batched loading with batch_size=10,000
   - Memory efficient: Only 10 MB in RAM at once
   - Processing time: ~15 batches total
   - Incremental results: Can stop/restart anytime

2. **Command to run:**
```bash
python scripts/redshift_job_batched.py \
    --parquet-file /pscratch/sd/r/rmfeder/data/l3_data/selection_wf.parquet \
    --batch-size 10000 \
    --filter-specz \
    --datestr 122725_production
```

3. **Expected output**: 15 result files, each with ~10k sources, total runtime ~2-4 hours (depends on sampling config)
