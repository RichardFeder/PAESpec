# Batch Redshift Analysis Workflow

This document describes the streamlined workflow for running batch redshift estimation jobs and analyzing the combined results.

## Overview

The workflow has been optimized so that:
1. **Multiple batch jobs** can run in parallel with different `START_IDX` values
2. **Results are automatically combined** using glob patterns
3. **Chain processing** (burn-in trimming) happens right after loading
4. **Comparison plots** are generated against template fitting results

## Workflow Steps

### 1. Configure Priors (Optional)

Edit `scripts/redshift_job.py` lines 30-42 to set your prior configuration:

```python
# Prior configuration
redshift_prior_type = 1  # 0=none, 1=Gaussian, 2=BPZ (not implemented)

# Gaussian prior parameters (only used if redshift_prior_type == 1)
z0_prior = 0.65
sigma_prior = 0.6
```

### 2. Run Batch Jobs

#### Option A: Interactive (for testing)
```bash
# Process 6 batches of 175 galaxies each (1050 total)
for i in {0..5}; do
    START_IDX=$((i * 175))
    python scripts/redshift_job.py --ngal 175 --batch-size 175 --start-idx $START_IDX
done
```

#### Option B: SLURM Job Array (for production)
```bash
# Edit submit_redshift_job.sh to set NGAL_PER_JOB and array size
# Then submit:
sbatch scripts/submit_redshift_job.sh
```

This creates multiple output files:
```
data/redshift_results/PAE_results_175_srcs_*_start0.npz
data/redshift_results/PAE_results_175_srcs_*_start175.npz
data/redshift_results/PAE_results_175_srcs_*_start350.npz
...
data/redshift_results/PAE_samples_175_srcs_*_start0.npz
data/redshift_results/PAE_samples_175_srcs_*_start175.npz
...
```

### 3. Analyze Combined Results

#### Quick Analysis (using helper script):
```bash
bash scripts/analyze_batch_runs.sh
```

This automatically:
- Finds all matching batch files
- Combines them into a single dataset
- Trims burn-in (default: 1000 steps)
- Generates comparison plots
- Saves results to `figures/batch_comparison/`

#### Manual Analysis (more control):
```bash
python scripts/compare_redshift_results.py \
    --pae_results "data/redshift_results/PAE_results_*_start*.npz" \
    --pae_samples "data/redshift_results/PAE_samples_*_start*.npz" \
    --plot_comparison \
    --plot_coverage \
    --tf_load_zpdf \
    --burn_in 1000 \
    --output_dir figures/my_analysis/
```

**Note**: The glob patterns must be in quotes to prevent shell expansion.

### 4. Load Results in Python

For custom analysis, you can load the combined results directly:

```python
from data_proc import load_combined_pae_results

# Load all batch results
results, samples = load_combined_pae_results(
    results_pattern="data/redshift_results/PAE_results_*_start*.npz",
    samples_pattern="data/redshift_results/PAE_samples_*_start*.npz",
    verbose=True
)

# Full chains are preserved
print(f"Total galaxies: {len(results['source_ids'])}")
print(f"MAP redshifts: {results['z_map'].shape}")
print(f"Full samples: {samples['z_samples'].shape}")

# Apply burn-in ONLY when needed for specific analyses (e.g., coverage)
burn_in = 1000
z_samples_trimmed = samples['z_samples'][:, :, burn_in:]
print(f"Trimmed for coverage: {z_samples_trimmed.shape}")

# For PIT analysis, use full chains (no burn-in trimming)
z_samples_full = samples['z_samples']  # Keep all samples for PIT
```

See `scripts/load_batch_results_example.py` for more examples.

## Key Features

### Automatic Batch Detection
`compare_redshift_results.py` automatically detects whether you're using:
- **Single file**: Traditional mode with one result file
- **Multiple files**: Batch mode using glob patterns

### Chain Processing
Full chains are preserved after loading:
- Burn-in trimming is applied **per-analysis** (e.g., for coverage statistics)
- Full chains remain available for PIT and other PDF-based analyses
- Configurable via `--burn_in` parameter (default: 1000)
- Different analyses can use different burn-in values from the same loaded data

### Resource Optimization
- **GPU memory**: Each job processes 175 galaxies (100-200 × 4 chains fits in one GPU)
- **Parallel execution**: SLURM job arrays enable simultaneous processing
- **Automatic indexing**: `START_IDX` correctly indexes into full dataset

## File Organization

```
data/redshift_results/
├── PAE_results_175_srcs_gaussprior_start0.npz      # Batch 0 results
├── PAE_samples_175_srcs_gaussprior_start0.npz      # Batch 0 samples
├── PAE_results_175_srcs_gaussprior_start175.npz    # Batch 1 results
├── PAE_samples_175_srcs_gaussprior_start175.npz    # Batch 1 samples
└── ...

figures/batch_comparison/
├── pae_tf_comparison.png          # Redshift comparison plot
└── coverage_comparison_grid.png   # Coverage analysis
```

## Prior Configuration Reference

```python
# No prior (uniform in redshift)
redshift_prior_type = 0

# Gaussian prior centered at z0 with width sigma
redshift_prior_type = 1
z0_prior = 0.65
sigma_prior = 0.6

# BPZ-style prior (not yet implemented)
# redshift_prior_type = 2
# alpha_prior = 2.46
# beta_prior = 1.81
# m0_prior = 20.0
# kz_prior = 0.0091
```

## Advanced: PIT Analysis and Full PDFs

The workflow preserves full chains (without burn-in trimming) so you can perform analyses that require complete posterior distributions:

### Probability Integral Transform (PIT)
```python
from data_proc import load_combined_pae_results
import numpy as np

# Load full chains
results, samples = load_combined_pae_results(
    results_pattern="PAE_results_*_start*.npz",
    samples_pattern="PAE_samples_*_start*.npz"
)

# For PIT, use ALL samples (no burn-in trimming)
z_samples_full = samples['z_samples']  # (n_gal, n_chain, n_step)
z_true = results['z_true']

# Compute PIT values: P(z < z_true | data)
pit_values = []
for i in range(len(z_true)):
    # Flatten all chains and samples for this galaxy
    z_posterior = z_samples_full[i].flatten()
    # PIT = fraction of posterior samples below true value
    pit = (z_posterior < z_true[i]).mean()
    pit_values.append(pit)

pit_values = np.array(pit_values)

# A well-calibrated posterior should have uniform PIT distribution
import matplotlib.pyplot as plt
plt.hist(pit_values, bins=20, range=(0, 1), density=True)
plt.axhline(1.0, color='r', linestyle='--', label='Uniform (ideal)')
plt.xlabel('PIT value')
plt.ylabel('Density')
plt.legend()
plt.savefig('pit_distribution.png')
```

### Coverage vs Summary Statistics
- **Coverage plots**: Use `--burn_in 1000` (trimmed chains for statistical inference)
- **PIT analysis**: Use full chains (no burn-in) to capture full posterior
- **Comparison plots**: Use MAP/median values (burn-in doesn't matter)

The same loaded data can serve both purposes since full chains are preserved.

## Troubleshooting

### No files found
```bash
# Check if result files exist
ls data/redshift_results/PAE_results_*_start*.npz

# Make sure you're in the correct directory
cd /global/homes/r/rmfeder/sed_vae/sp-ae-herex
```

### Glob pattern not working
Make sure to quote the pattern:
```bash
# CORRECT:
python compare_redshift_results.py --pae_results "PAE_results_*_start*.npz" ...

# WRONG (shell expands the pattern):
python compare_redshift_results.py --pae_results PAE_results_*_start*.npz ...
```

### Memory issues
Reduce batch size in `submit_redshift_job.sh`:
```bash
NGAL_PER_JOB=100  # Instead of 175
BATCH_SIZE=100
```

## Examples

### Example 1: Quick test with 350 galaxies (2 batches)
```bash
# Run 2 batches
python scripts/redshift_job.py --ngal 175 --batch-size 175 --start-idx 0
python scripts/redshift_job.py --ngal 175 --batch-size 175 --start-idx 175

# Analyze combined results
bash scripts/analyze_batch_runs.sh
```

### Example 2: Production run with 10 batches (1750 galaxies)
```bash
# Submit job array
sbatch --array=0-9 scripts/submit_redshift_job.sh

# Wait for jobs to complete, then analyze
bash scripts/analyze_batch_runs.sh
```

### Example 3: Compare different priors
```bash
# Run with no prior
# (edit redshift_job.py: redshift_prior_type = 0)
bash scripts/submit_redshift_job.sh

# Run with Gaussian prior  
# (edit redshift_job.py: redshift_prior_type = 1)
bash scripts/submit_redshift_job.sh

# Compare results
python scripts/compare_redshift_results.py \
    --pae_results "data/redshift_results/PAE_results_*_noprior_start*.npz" \
    --pae_samples "data/redshift_results/PAE_samples_*_noprior_start*.npz" \
    --output_dir figures/no_prior/ --burn_in 1000

python scripts/compare_redshift_results.py \
    --pae_results "data/redshift_results/PAE_results_*_gaussprior_start*.npz" \
    --pae_samples "data/redshift_results/PAE_samples_*_gaussprior_start*.npz" \
    --output_dir figures/gauss_prior/ --burn_in 1000
```
