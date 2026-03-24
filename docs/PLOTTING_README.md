# Redshift Results Plotting Guide

This directory contains scripts for generating comprehensive summary plots and diagnostics from PAE redshift estimation results.

## Two Types of Analysis

1. **Summary Plots** (this document): Aggregate statistics and distributions across all sources
2. **Per-Source Plots** ([SOURCE_PLOTS_README.md](SOURCE_PLOTS_README.md)): Detailed diagnostics for individual sources with automatic good/bad fit categorization

## Quick Start

### Automatic Plotting (Recommended)

Plots are automatically generated after redshift estimation by default:

```bash
# Run redshift estimation (plots generated automatically at end)
./scripts/run_redshift_job_production.sh

# For mock data
./scripts/run_redshift_job_mock.sh
```

To disable automatic plotting, edit the shell script and set:
```bash
GENERATE_PLOTS="false"
```

### Manual/Afterburner Plotting

Generate plots for a completed run:

```bash
# Using datestr (auto-locates result file)
./scripts/run_plot_afterburner.sh multicore_test_16k_wf_123025

# Using explicit file path
./scripts/run_plot_afterburner.sh --result-file /path/to/PAE_results_combined_XXX.npz

# With custom output directory
./scripts/run_plot_afterburner.sh multicore_test_16k_wf_123025 --output-dir ./my_figures
```

## Scripts

### `generate_redshift_plots.py`
Core plotting script that generates three summary figures:

**Figure 1: Z-score distributions by uncertainty bin**
- Shows z-score histograms for 5 different uncertainty bins
- Includes CDF of absolute z-scores
- Compares PAE vs template fitting (TF) performance

**Figure 2: Redshift distributions by uncertainty bin**
- Shows how estimated redshifts distribute across uncertainty bins
- Useful for identifying systematic biases

**Figure 3: Redshift comparison plots**
- Scatter plots: spec-z vs photo-z for PAE and TF
- Bias histograms
- Uncertainty distributions

**Usage:**
```bash
# From datestr
python scripts/generate_redshift_plots.py --datestr multicore_test_16k_wf_123025

# From file
python scripts/generate_redshift_plots.py --result-file /path/to/results.npz

# Show plots interactively (default is save-only)
python scripts/generate_redshift_plots.py --datestr XXX --show

# Custom z-score range
python scripts/generate_redshift_plots.py --datestr XXX --zscore-range -5 5
```

### `run_plot_afterburner.sh`
Convenient wrapper for generating plots after a run completes.

**Usage:**
```bash
./scripts/run_plot_afterburner.sh <datestr>
./scripts/run_plot_afterburner.sh --result-file <path>
```

### `run_redshift_job_production.sh` (updated)
Production redshift estimation script with integrated plotting.

**Configuration:**
```bash
GENERATE_PLOTS="true"   # Set to "false" to skip plotting
DATESTR="your_run_name"  # Used for both redshift run and plot directory
```

### `run_redshift_job_mock.sh` (updated)
Mock data redshift estimation script with integrated plotting.

## Output Directories

### Real Data (Parquet)
- **Results:** `/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched/<datestr>/`
- **Figures:** `/pscratch/sd/r/rmfeder/figures/redshift_validation/<datestr>/`

### Mock Data
- **Results:** `/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/mock_<datestr>/`
- **Figures:** `/global/homes/r/rmfeder/sed_vae/figures/redshift_validation/mock_<datestr>/`

## Generated Figures

Each run produces three PNG files (300 DPI):

1. `<result_name>_zscore_by_uncertainty.png`
   - 6 panels showing z-score distributions
   - Includes statistics: median, NMAD

2. `<result_name>_redshift_by_uncertainty.png`
   - 6 panels showing redshift distributions
   - Separated by uncertainty bins

3. `<result_name>_redshift_comparison.png`
   - 4 panels: PAE scatter, TF scatter, bias histogram, uncertainty distribution

## Examples

### Typical Workflow

```bash
# 1. Run redshift estimation (plots auto-generated)
./scripts/run_redshift_job_production.sh

# 2. If you need to regenerate plots with different settings
./scripts/run_plot_afterburner.sh multicore_test_16k_wf_123025 --zscore-range -5 5

# 3. For interactive viewing
python scripts/generate_redshift_plots.py --datestr multicore_test_16k_wf_123025 --show
```

### Mock Data Workflow

```bash
# 1. Run mock data redshift estimation
./scripts/run_redshift_job_mock.sh

# 2. Regenerate plots if needed
RESULT_FILE="/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/mock_validation_123025/PAE_results_combined_validation_123025.npz"
./scripts/run_plot_afterburner.sh --result-file "$RESULT_FILE"
```

### Batch Processing Multiple Runs

```bash
#!/bin/bash
# Generate plots for multiple completed runs
RUNS="run1_123025 run2_123025 run3_123025"

for run in $RUNS; do
    echo "Processing $run..."
    ./scripts/run_plot_afterburner.sh "$run"
done
```

## Customization

### Modify Plot Settings

Edit `generate_redshift_plots.py` to customize:
- Color schemes
- Bin definitions
- Figure sizes
- Additional diagnostics

### Add New Plots

Add new plotting functions to `generate_redshift_plots.py` following the pattern:

```python
def plot_custom_diagnostic(res, save_dir, result_name):
    """Your custom plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Your plotting code here
    
    if save_dir:
        fig.savefig(Path(save_dir) / f'{result_name}_custom.png', 
                   dpi=300, bbox_inches='tight')
    plt.close(fig)
```

Then call it from `plot_pae_summary()`.

## Troubleshooting

### "Could not find result file"
- Check that `--collate-results` was used during redshift estimation
- Verify the datestr matches your run
- Use explicit `--result-file` path if auto-detection fails

### "No module named matplotlib"
```bash
module load python
# or activate your conda environment
```

### Plots don't show up
- Default behavior is save-only (non-interactive backend)
- Use `--show` flag for interactive display
- Check figure output directory for saved PNGs

### Memory issues with large datasets
- Plotting script loads full `.npz` file into memory
- For very large runs (>100k sources), consider subsampling in the plotting code

## Performance

- Typical runtime: 10-30 seconds for 10k-20k sources
- Memory usage: ~500 MB - 2 GB depending on dataset size
- All figures generated at 300 DPI for publication quality

## Related Documentation

- **[SOURCE_PLOTS_README.md](SOURCE_PLOTS_README.md)**: Generate detailed per-source reconstruction plots with automatic good/bad fit categorization
- **[MULTINODE_README.md](MULTINODE_README.md)**: Multi-node job submission for large datasets
