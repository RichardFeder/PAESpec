# Multi-Core GPU Parallelization Guide

## Overview

The multi-core parallelization feature distributes MCMC sampling work across multiple GPU cores on a single node, providing near-linear speedup for large batches of galaxies.

## Performance Characteristics

- **Single core saturation**: ~600 chains (150 sources × 4 chains)
- **Optimal configuration**: 4 GPU cores processing 600 sources (2400 total chains)
- **Expected speedup**: ~4x with 4 cores (near-linear scaling)

## Usage

### Basic Configuration

In `redshift_job.py`, set:

```python
# Multi-core parallelization
use_multicore = True
n_devices_per_node = 4  # Number of GPU cores to use
```

### Recommended Settings by Scale

| Galaxies | Chains/gal | Total chains | Cores | Chains/core | Mode |
|----------|------------|--------------|-------|-------------|------|
| 50       | 4          | 200          | 1     | 200         | Single |
| 150      | 4          | 600          | 1     | 600         | Single |
| 300      | 4          | 1200         | 2     | 600         | Multi |
| 600      | 4          | 2400         | 4     | 600         | Multi |
| 1000     | 4          | 4000         | 4     | 1000        | Multi |

**Rule of thumb**: Use multi-core when total chains > 800 (above single-core saturation)

### Command Line Example

```bash
# Single-core (default)
python scripts/redshift_job.py --ngal 150 --batch-size 50

# Multi-core (4 cores)
python scripts/redshift_job.py --ngal 600 --batch-size 150
```

Note: Edit `use_multicore = True` in the script before running.

## Implementation Details

### How It Works

1. **Data splitting**: Galaxies are evenly divided across available GPU cores
2. **Parallel execution**: Each core processes its subset independently using `jax.pmap`
3. **Result gathering**: Results are concatenated back into a single array
4. **Logdensity sharing**: The batched logdensity function is compiled once and shared

### Architecture

```
Main Process
    |
    ├─ Device 0: Process galaxies [0:150]    (600 chains)
    ├─ Device 1: Process galaxies [150:300]  (600 chains)
    ├─ Device 2: Process galaxies [300:450]  (600 chains)
    └─ Device 3: Process galaxies [450:600]  (600 chains)
    |
Results gathered → Combined output
```

## Testing

Run the test script to verify setup:

```bash
python scripts/test_multicore.py
```

This will:
- Detect available devices
- Run a small test on 20 galaxies
- Compare single-core vs multi-core timing
- Verify result consistency

## Troubleshooting

### "Only 1 device detected"

**Cause**: JAX is not configured to see multiple GPU cores

**Solutions**:
1. Check GPU allocation: `nvidia-smi`
2. Set JAX to use multiple devices:
   ```python
   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
   ```
3. Verify with: `jax.devices()`

### Slower than single-core

**Cause**: Overhead from data transfer dominates for small batches

**Solution**: Only use multi-core for large batches (>400 galaxies)

### OOM (Out of Memory) errors

**Cause**: Too many chains per core

**Solution**: Reduce `batch_size` or `nchain_per_gal`, or use fewer cores

### Results differ between runs

**Cause**: Device placement affects numerical precision slightly

**Solution**: This is expected - differences should be small (<1% in log-L)

## Performance Tips

1. **Batch size**: Set to ~(total_galaxies / n_devices)
2. **Chain distribution**: Keep chains/core ≤ 600 for optimal performance
3. **Data locality**: Pre-compile logdensity to avoid per-device compilation
4. **Memory**: Monitor `nvidia-smi` to avoid OOM

## Compatibility

- **Compatible with**: `use_batched_logdensity = True` (recommended)
- **Compatible with**: All redshift prior types
- **Not needed for**: Small runs (<200 galaxies)
- **Node types**: Requires multi-GPU node (e.g., Perlmutter GPU nodes)

## Example Configurations

### Development/Testing (small)
```python
use_multicore = False
ngal = 50
batch_size = 25
```

### Production (medium)
```python
use_multicore = True
n_devices_per_node = 2
ngal = 300
batch_size = 150
```

### Production (large)
```python
use_multicore = True
n_devices_per_node = 4
ngal = 1000
batch_size = 250
```

## Summary

Multi-core parallelization provides significant speedup for large-scale redshift estimation:
- Enable with `use_multicore = True`
- Optimal for 400+ galaxies
- Near-linear scaling up to 4 cores
- Maintains result quality and consistency
