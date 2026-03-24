# Per-Source Reconstruction Plots

Generate detailed diagnostic plots for individual sources from PAE redshift estimation runs.

## Overview

The per-source plotting system automatically:
1. Identifies "good" and "bad" fits based on chi-squared and z-score metrics
2. Generates comprehensive diagnostic plots for selected sources
3. Organizes outputs into `good_fits/` and `bad_fits/` subdirectories

Each source gets three types of plots:
- **Spectral reconstruction**: Observed vs. reconstructed spectrum with uncertainty bands
- **Corner plot**: Posterior distributions for latent variables (and redshift if not fixed)
- **Redshift posterior**: Distribution of inferred redshift vs. true redshift

## Quick Start

### Basic Usage

```bash
# Generate plots for an existing run
./scripts/run_source_plots.sh multicore_test_16k_wf_123025
```

This will:
- Auto-detect result and sample files from the datestr
- Select 50 good fits and 50 bad fits
- Generate 3 plots per source (150 plots per category = 300 total)
- Save to `/pscratch/sd/r/rmfeder/figures/redshift_validation/<datestr>/good_fits/` and `.../bad_fits/`

### Customized Selection

```bash
# More sources
./scripts/run_source_plots.sh multicore_test_16k_wf_123025 --n-good 100 --n-bad 100

# Stricter criteria for bad fits
./scripts/run_source_plots.sh multicore_test_16k_wf_123025 \
    --chi2-percentile 95 \
    --zscore-threshold 2.5

# More relaxed criteria (more bad fits)
./scripts/run_source_plots.sh multicore_test_16k_wf_123025 \
    --chi2-percentile 80 \
    --zscore-threshold 4.0
```

## Selection Criteria

### Good Fits
Sources are classified as "good" if they meet BOTH criteria:
- **Low chi-squared**: χ² < 25th percentile
- **Low z-score**: |z-score| < 1.0

Where z-score = (z_inferred - z_true) / σ_z

### Bad Fits
Sources are classified as "bad" if they meet EITHER criterion:
- **High chi-squared**: χ² > specified percentile (default: 90th)
- **High z-score**: |z-score| > threshold (default: 3.0)

### Random Selection
If more sources meet the criteria than requested (--n-good, --n-bad), a random subset is selected with fixed random seed for reproducibility.

## Output Structure

```
/pscratch/sd/r/rmfeder/figures/redshift_validation/<datestr>/
├── good_fits/
│   ├── source_00042_reconstruction.png
│   ├── source_00042_corner.png
│   ├── source_00042_zposterior.png
│   ├── source_00137_reconstruction.png
│   ├── ...
└── bad_fits/
    ├── source_01523_reconstruction.png
    ├── source_01523_corner.png
    ├── source_01523_zposterior.png
    └── ...
```

## Plot Types Explained

### 1. Spectral Reconstruction (`*_reconstruction.png`)
**Size**: 7×4 inches

Shows:
- Black points with error bars: Observed spectrum
- Black line: True (noiseless) spectrum (if available)
- Red/C3 line: Posterior mean reconstruction
- Red/C3 shaded regions: 68% and 95% credible intervals
- Blue dashed line: Residual (mean - truth)
- Text annotation: χ² value, true redshift, inferred redshift with uncertainties

**Key diagnostic**: 
- Good fits show tight uncertainty bands around truth
- Bad fits show large residuals or poor overlap with data

### 2. Corner Plot (`*_corner.png`)
**Size**: 7×7 inches

Shows:
- Posterior distributions for all latent variables (u₁, u₂, ..., u_n)
- Posterior distribution for redshift (if not fixed)
- Black vertical/horizontal lines: True redshift (if applicable)
- Contours: 68% and 95% credible regions

**Key diagnostic**:
- Good fits show concentrated posteriors
- Bad fits may show multimodality, broad distributions, or biases

### 3. Redshift Posterior (`*_zposterior.png`)
**Size**: 4×3 inches  
*Only generated when redshift is not fixed*

Shows:
- Histogram: Posterior distribution of redshift
- Black vertical line: True redshift
- Annotations: Percentiles, mean, median

**Key diagnostic**:
- Good fits show narrow posterior centered on truth
- Bad fits show broad posteriors or significant offsets

## Command-Line Options

### File Specification
```bash
# Auto-detect from datestr (recommended)
--datestr <string>

# Or specify files explicitly
--result-file <path>     # PAE_results_combined_*.npz
--sample-file <path>     # PAE_samples_combined_*.npz
--data-file <path>       # Observed data .npz
```

### PAE Model Configuration
```bash
--nlatent <int>              # Number of latent dimensions (default: 5)
--sig-level-norm <float>     # Sigma level normalization (default: 0.01)
--run-name <string>          # Model run name (auto-detected if using datestr)
--sel-str <string>           # Selection string (default: 'all')
--phot-snr-min <float>       # Min photometric SNR (default: 50)
--phot-snr-max <float>       # Max photometric SNR (default: 300)
```

### Selection Criteria
```bash
--n-good <int>               # Number of good sources to plot (default: 50)
--n-bad <int>                # Number of bad sources to plot (default: 50)
--chi2-percentile <float>    # Percentile for chi2 threshold (default: 90)
--zscore-threshold <float>   # Z-score threshold for bad fits (default: 3.0)
```

### Output Options
```bash
--output-dir <path>          # Custom output directory
--verbose                    # Print detailed progress
--fix-z                      # Redshift was fixed during sampling
```

## Advanced Usage

### Explicit File Paths
When not using standard directory structure:

```bash
python scripts/generate_source_reconstructions.py \
    --result-file /path/to/PAE_results_combined.npz \
    --sample-file /path/to/PAE_samples_combined.npz \
    --data-file /path/to/observed_data.npz \
    --output-dir /path/to/figures \
    --run-name "jax_conv1_nlatent=5_siglevelnorm=0.01_newAllen_all_091325" \
    --n-good 75 --n-bad 75 \
    --verbose
```

### Different Model Configurations
For models with different latent dimensions:

```bash
./scripts/run_source_plots.sh my_run_datestr \
    --nlatent 10 \
    --sig-level-norm 0.005 \
    --run-name "jax_conv1_nlatent=10_siglevelnorm=0.005_newAllen_all"
```

### Analyzing Only Extreme Outliers
To focus on the worst fits:

```bash
./scripts/run_source_plots.sh multicore_test_16k_wf_123025 \
    --n-bad 200 \
    --n-good 0 \
    --chi2-percentile 98 \
    --zscore-threshold 5.0
```

## Integration with Redshift Pipeline

### Standalone (Current Approach)
Run after redshift estimation completes:

```bash
# 1. Run redshift estimation
./scripts/run_redshift_job_production.sh

# 2. Generate summary plots
./scripts/run_plot_afterburner.sh <datestr>

# 3. Generate per-source plots
./scripts/run_source_plots.sh <datestr>
```

### Future: Automatic Integration
You could add per-source plotting to the run scripts:

```bash
# In run_redshift_job_production.sh, after plotting section:
if [ "$GENERATE_PLOTS" = "true" ]; then
    echo "Generating summary plots..."
    python scripts/generate_redshift_plots.py --datestr "$DATESTR" --output-dir "$FIG_DIR"
    
    echo "Generating per-source plots..."
    python scripts/generate_source_reconstructions.py \
        --datestr "$DATESTR" \
        --n-good 50 --n-bad 50 \
        --verbose
fi
```

## Performance Notes

- **Time per source**: ~3-5 seconds (depends on number of samples, latent dimensions)
- **Total time**: For 50 good + 50 bad = ~5-8 minutes
- **Disk space**: ~300-500 KB per source (3 plots) = ~50-75 MB for 100 sources
- **Memory**: Loads all samples into memory; ~1-2 GB typical

For large runs (1000s of sources), be selective with `--n-good` and `--n-bad`.

## Troubleshooting

### Issue: "No result file found"
**Solution**: Verify the datestr matches the directory name:
```bash
ls /pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/
```

### Issue: "Error loading PAE model"
**Solution**: Check that model configuration matches the run:
- Verify `--nlatent` matches your model
- Verify `--run-name` points to correct trained model
- Check that model checkpoint exists in `modl_runs/`

### Issue: "No noiseless spectrum found"
**Solution**: This is expected for real data (only mock data has noiseless spectra). Plots will still be generated but without the "truth" line in reconstruction plots.

### Issue: Plots look wrong or have large errors
**Solution**: 
- Check burn-in settings (currently hardcoded to 1000)
- Verify sample file has sufficient samples
- Check that data normalization is correct

### Issue: Takes too long
**Solution**: 
- Reduce `--n-good` and `--n-bad`
- Run in parallel by selecting different source subsets
- Use `--chi2-percentile 99` to focus on most extreme cases

## Example Workflows

### Standard Diagnostic Review
```bash
# Generate standard set
./scripts/run_source_plots.sh multicore_test_16k_wf_123025

# Review outputs
ls /pscratch/sd/r/rmfeder/figures/redshift_validation/multicore_test_16k_wf_123025/good_fits/
ls /pscratch/sd/r/rmfeder/figures/redshift_validation/multicore_test_16k_wf_123025/bad_fits/
```

### Focused Failure Analysis
```bash
# Only bad fits, more stringent
./scripts/run_source_plots.sh multicore_test_16k_wf_123025 \
    --n-good 0 \
    --n-bad 100 \
    --chi2-percentile 95 \
    --zscore-threshold 2.5 \
    --verbose
```

### Publication Figure Generation
```bash
# Small set of best examples
./scripts/run_source_plots.sh multicore_test_16k_wf_123025 \
    --n-good 10 \
    --n-bad 10 \
    --chi2-percentile 98
```

## Technical Details

### Data Flow
1. Load result file → compute chi² and z-scores
2. Apply selection criteria → get good/bad indices
3. Load PAE model → initialize with trained weights
4. Load samples → shape (n_sources, n_chains, n_samples, n_params)
5. For each source:
   - Process posterior samples through PAE decoder
   - Compute reconstruction statistics
   - Generate and save 3 plots

### Dependencies
- JAX/Flax for model inference
- NumPy for array operations
- Matplotlib for plotting
- corner for corner plots
- All PAE/data processing infrastructure from sp-ae-herex

### File Requirements
**Result file** (`PAE_results_combined_*.npz`):
- `ztrue`: True redshifts
- `z_med`: Inferred median redshifts
- `err_low`, `err_high`: Redshift uncertainties
- Optional: `chi2` values

**Sample file** (`PAE_samples_combined_*.npz`):
- `all_samples`: MCMC chains, shape (n_sources, n_chains, n_samples, n_params)

**Data file** (format from `SPHERExData`):
- Observed spectra, uncertainties, source IDs
- Optionally: noiseless spectra for mock data

## See Also

- [PLOTTING_README.md](PLOTTING_README.md) - Summary plot documentation
- [MULTINODE_README.md](MULTINODE_README.md) - Multi-node execution
- Main notebook: `sp-ae-herex/notebooks/redshift_tests.ipynb`
