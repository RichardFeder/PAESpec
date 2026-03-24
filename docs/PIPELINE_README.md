# Analysis Pipeline Overview

Complete workflow for PAE redshift estimation with automated diagnostics and visualization.

## Pipeline Stages

### 1. Redshift Estimation
Run PAE-based redshift inference on your data:

```bash
# Production data (from parquet)
./scripts/run_redshift_job_production.sh

# Mock/simulated data
./scripts/run_redshift_job_mock.sh
```

**Outputs:**
- `PAE_results_combined_*.npz`: Redshift estimates and uncertainties
- `PAE_samples_combined_*.npz`: Full MCMC posterior samples

**Documentation:** [MULTINODE_README.md](MULTINODE_README.md)

---

### 2. Summary Diagnostics (Automatic)
Generates 15 aggregate plots analyzing performance across all sources:

```bash
# Usually runs automatically after Stage 1
# Or run manually:
./scripts/run_plot_afterburner.sh <datestr>
```

**Outputs (in `/pscratch/sd/r/rmfeder/figures/redshift_validation/<datestr>/`):**
- Z-score distributions by uncertainty bin (6 panels)
- Redshift distributions by uncertainty bin (6 panels)
- Overall comparisons (4 panels: scatter, bias, uncertainties)
- Detailed threshold analysis (12 figures across 4 thresholds)

**Documentation:** [PLOTTING_README.md](PLOTTING_README.md)

---

### 3. Per-Source Diagnostics (Manual)
Generates detailed plots for individual sources, automatically categorized as good/bad fits:

```bash
./scripts/run_source_plots.sh <datestr> [--n-good 50] [--n-bad 50]
```

**Outputs (in `<datestr>/good_fits/` and `<datestr>/bad_fits/`):**
- Spectral reconstructions with posterior credible intervals
- Corner plots showing latent space and redshift posteriors
- Redshift posterior distributions

**Documentation:** [SOURCE_PLOTS_README.md](SOURCE_PLOTS_README.md)

---

## Typical Workflow

### Quick Analysis
```bash
# 1. Run redshift estimation (plots generated automatically)
./scripts/run_redshift_job_production.sh

# 2. Generate per-source diagnostics
./scripts/run_source_plots.sh <datestr>

# Done! Review figures in /pscratch/sd/r/rmfeder/figures/redshift_validation/<datestr>/
```

### Detailed Investigation
```bash
# 1. Run estimation with specific date identifier
./scripts/run_redshift_job_production.sh
# (edits datestr in script first)

# 2. Review summary plots
ls /pscratch/sd/r/rmfeder/figures/redshift_validation/<datestr>/*.png

# 3. Generate many per-source plots for thorough review
./scripts/run_source_plots.sh <datestr> --n-good 100 --n-bad 100

# 4. Focus on most problematic sources
./scripts/run_source_plots.sh <datestr> \
    --n-bad 200 --n-good 0 \
    --chi2-percentile 98 --zscore-threshold 5.0
```

### Publication Workflow
```bash
# 1. Run on final dataset
./scripts/run_redshift_job_production.sh

# 2. Generate summary figures (already done automatically)
./scripts/run_plot_afterburner.sh <datestr>

# 3. Select best examples for paper
./scripts/run_source_plots.sh <datestr> \
    --n-good 5 --n-bad 5 \
    --chi2-percentile 99

# 4. Manually curate from good_fits/ and bad_fits/ directories
```

---

## Output Organization

```
/pscratch/sd/r/rmfeder/
├── data/pae_sample_results/MCLMC/
│   └── batched/<datestr>/
│       ├── PAE_results_combined_*.npz       # Stage 1 output
│       ├── PAE_samples_combined_*.npz       # Stage 1 output
│       └── batch*/                          # Individual batch outputs
└── figures/redshift_validation/<datestr>/
    ├── *_zscore_by_bin.png                  # Stage 2: Summary plots
    ├── *_redshift_by_bin.png                #   (15 total)
    ├── *_scatter_dz*.png                    #
    ├── good_fits/                           # Stage 3: Per-source
    │   ├── source_*_reconstruction.png      #   Good fits
    │   ├── source_*_corner.png              #
    │   └── source_*_zposterior.png          #
    └── bad_fits/                            # Stage 3: Per-source
        ├── source_*_reconstruction.png      #   Bad fits
        ├── source_*_corner.png              #
        └── source_*_zposterior.png          #
```

---

## Configuration Files

### Redshift Estimation Parameters
Edit in shell scripts (`run_redshift_job_*.sh`):
- `BATCH_SIZE`: Sources per batch
- `PARQUET_FILE`: Input data location
- `FILTER_SET`: SPHEREx filter configuration
- `DATESTR`: Run identifier
- `GENERATE_PLOTS`: Enable/disable automatic plotting

### Plotting Parameters
Edit in Python scripts or pass as arguments:
- Summary plots (`generate_redshift_plots.py`):
  - `zscore_range`: Range for z-score histograms
  - Uncertainty bins and thresholds (in code)
  
- Per-source plots (`generate_source_reconstructions.py`):
  - `--n-good`, `--n-bad`: Number of sources per category
  - `--chi2-percentile`: Threshold for high chi²
  - `--zscore-threshold`: Threshold for bad z-scores
  - `--nlatent`: Model latent dimensions

---

## Performance Summary

| Stage | Time (10k sources) | Disk Space | Memory |
|-------|-------------------|------------|--------|
| 1. Estimation | ~2-6 hours | ~5 GB | 40-80 GB |
| 2. Summary plots | ~30 seconds | ~5 MB | <1 GB |
| 3. Per-source (100) | ~5-10 min | ~50 MB | ~2 GB |

**Notes:**
- Stage 1 time varies with sampling parameters and hardware
- Stage 3 scales linearly with number of sources selected
- All outputs at 300 DPI for publication quality

---

## Troubleshooting

### Common Issues

**1. "Could not find results directory"**
- Check datestr matches actual directory name
- Verify results in: `/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/`

**2. "No sample file found"**
- Ensure redshift run completed successfully
- Check for `PAE_samples_combined_*.npz` in results directory

**3. "Model not found"**
- Verify `--run-name` matches trained model in `modl_runs/`
- Default: `jax_conv1_nlatent=5_siglevelnorm=0.01_newAllen_all_091325`

**4. "Data sources mismatch"**
- Per-source plotting uses data from PAE model setup
- May differ from actual run if filters/SNR cuts were different
- Some sources may fail; this is expected

**5. Plots look wrong**
- Check burn-in settings (default: 1000 samples)
- Verify sufficient MCMC samples in chains
- Check data normalization

### Getting Help

1. Check relevant documentation:
   - Estimation: [MULTINODE_README.md](MULTINODE_README.md)
   - Summary plots: [PLOTTING_README.md](PLOTTING_README.md)
   - Per-source: [SOURCE_PLOTS_README.md](SOURCE_PLOTS_README.md)

2. Verify file paths and permissions:
   ```bash
   ls -lh /pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/<datestr>/
   ```

3. Check script logs for detailed error messages

4. For memory issues:
   - Reduce `--n-good` and `--n-bad` in per-source plotting
   - Use `--batch-size` to limit simultaneous GPU processing

---

## Advanced Topics

### Custom Filter Sets
To use different SPHEREx filters:
```bash
# In run script, set:
FILTER_SET="SPHEREx_filter_102"  # or SPHEREx_filter_408
```

### Different Priors
Modify sampling configuration in Python script:
```python
cfg = MCLMCSamplingConfig(
    redshift_prior_type='gaussian',  # or 'uniform'
    z0_prior=1.5,
    sigma_prior=1.0,
    # ... other parameters
)
```

### Parallel Per-Source Plotting
For very large analyses:
```bash
# Plot different source ranges on different nodes
./scripts/run_source_plots.sh run1 --n-good 100 --n-bad 0 &
./scripts/run_source_plots.sh run1 --n-good 0 --n-bad 100 &
```

### Customizing Plots
Both Python scripts support modification:
- Edit functions directly for custom analysis
- Add new plot types by following existing patterns
- Adjust figure sizes, colors, layouts in function calls

---

## Quick Reference

| Task | Command |
|------|---------|
| Run estimation | `./scripts/run_redshift_job_production.sh` |
| Summary plots | `./scripts/run_plot_afterburner.sh <datestr>` |
| Per-source plots | `./scripts/run_source_plots.sh <datestr>` |
| Check outputs | `ls /pscratch/sd/r/rmfeder/figures/redshift_validation/<datestr>/` |
| View results | `ls /pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/<datestr>/` |

---

## Related Documentation

- **[MULTINODE_README.md](MULTINODE_README.md)**: Multi-node job submission and batched processing
- **[PLOTTING_README.md](PLOTTING_README.md)**: Summary plot generation and customization
- **[SOURCE_PLOTS_README.md](SOURCE_PLOTS_README.md)**: Per-source diagnostic plots
- Main notebook: `sp-ae-herex/notebooks/redshift_tests.ipynb`
