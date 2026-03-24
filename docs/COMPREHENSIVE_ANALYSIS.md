# Comprehensive Redshift Analysis

## Overview

The `comprehensive_redshift_analysis.py` script automates the generation of a complete suite of analysis plots for PAE redshift estimation results. It consolidates the various analyses typically performed in `paper_plots.ipynb` into a single automated workflow with systematic file naming.

## Features

### Generated Plots

The script generates the following analyses:

#### Basic Comparison (`--plot_groups basic`)
- **comparison_pae_tf.png**: Side-by-side PAE vs TF redshift scatter plots
- **comparison_detailed.png**: Detailed comparison with binned correlation analysis

#### Coverage Analysis (`--plot_groups coverage`)
- **coverage_grid.png**: Coverage probability grid binned by uncertainty levels
  - Evaluates calibration of uncertainty estimates
  - Compares PAE and TF coverage across different σz bins

#### Diagnostic Plots (`--plot_groups diagnostics`)
- **zscore_comparison.png**: Z-score distributions for PAE and TF
- **rhat_histogram.png**: Gelman-Rubin R-hat convergence diagnostic

#### PDF Comparisons (`--plot_groups pdf`)
- **pdf_examples.png**: Grid of PDF comparisons for selected examples
  - Shows PAE posterior vs TF posterior for individual sources
  - Includes true redshift markers

#### Convergence Analysis (`--plot_groups convergence`)
- **pit_histogram.png**: Probability Integral Transform (PIT) histogram
  - Tests posterior calibration
  - Well-calibrated posteriors should be uniform
- **pit_qq.png**: Q-Q plot for PIT uniformity test

#### Summary Report
- **summary.txt**: Text file with key statistics
  - Bias, NMAD, outlier fractions for PAE and TF
  - Z-score statistics
  - Convergence statistics (mean R-hat, fraction > 1.1)
  - Sample configuration details

## Usage

### Basic Usage

```bash
python scripts/comprehensive_redshift_analysis.py \
    --pae_results <path_or_glob> \
    --pae_samples <path_or_glob> \
    --run_name <descriptive_name>
```

### Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--pae_results` | Yes | - | Path or glob pattern for PAE results file(s) |
| `--pae_samples` | Yes | - | Path or glob pattern for PAE samples file(s) |
| `--run_name` | Yes | - | Name for this run (used in figure filenames) |
| `--output_dir` | No | `figures/analysis/` | Directory to save output figures |
| `--burn_in` | No | `1000` | Number of burn-in steps to trim from chains |
| `--nsrc` | No | All | Number of sources to analyze |
| `--tf_load_zpdf` | No | `True` | Load template fitting PDFs |
| `--plot_groups` | No | `all` | Which plot groups to generate |
| `--dpi` | No | `300` | DPI for saved figures |

### Plot Groups

You can selectively generate specific plot groups:

- `all`: Generate all plots (default)
- `basic`: Basic PAE vs TF comparison plots
- `coverage`: Coverage analysis
- `diagnostics`: Diagnostic plots (z-scores, convergence)
- `pdf`: PDF comparison examples
- `convergence`: PIT analysis and Q-Q plots

Example:
```bash
python scripts/comprehensive_redshift_analysis.py \
    --pae_results "data/results.npz" \
    --pae_samples "data/samples.npz" \
    --run_name "quick_check" \
    --plot_groups basic diagnostics
```

## File Naming Scheme

All output files follow a systematic naming pattern:

```
<output_dir>/<run_name>_<plot_type>.png
```

For example, with `--run_name nlatent5_gaussprior_175srcs`:
- `nlatent5_gaussprior_175srcs_comparison_pae_tf.png`
- `nlatent5_gaussprior_175srcs_coverage_grid.png`
- `nlatent5_gaussprior_175srcs_zscore_comparison.png`
- etc.

This makes it easy to organize and compare results from different configurations.

## Recommended Naming Convention

Use descriptive run names that encode key parameters:

```
nlatent<N>_<prior_type>_<ngal>srcs[_<additional_info>]
```

Examples:
- `nlatent5_gaussprior_175srcs`
- `nlatent5_noprior_1000srcs_combined`
- `nlatent10_gaussprior_175srcs_highz`
- `nlatent5_bpzprior_500srcs_cosmos`

## Example Workflows

### 1. Single Batch Analysis

Analyze a single batch of results:

```bash
python scripts/comprehensive_redshift_analysis.py \
    --pae_results "data/tf_results/PAE_results_175_srcs_nlatent5_gaussprior_start0.npz" \
    --pae_samples "data/tf_results/PAE_samples_175_srcs_nlatent5_gaussprior_start0.npz" \
    --run_name "nlatent5_gaussprior_175srcs" \
    --output_dir "figures/comprehensive/nlatent5_gaussprior/" \
    --burn_in 1000
```

### 2. Combined Batch Analysis

Combine multiple batch files using glob patterns:

```bash
python scripts/comprehensive_redshift_analysis.py \
    --pae_results "data/tf_results/PAE_results_175_srcs_*_start*.npz" \
    --pae_samples "data/tf_results/PAE_samples_175_srcs_*_start*.npz" \
    --run_name "nlatent5_gaussprior_1000srcs_combined" \
    --output_dir "figures/comprehensive/nlatent5_gaussprior_combined/" \
    --burn_in 1000
```

### 3. Comparing Different Configurations

Compare different priors:

```bash
# Gaussian prior
python scripts/comprehensive_redshift_analysis.py \
    --pae_results "data/tf_results/PAE_results_*_gaussprior_*.npz" \
    --pae_samples "data/tf_results/PAE_samples_*_gaussprior_*.npz" \
    --run_name "nlatent5_gaussprior_combined" \
    --output_dir "figures/comprehensive/comparison/"

# No prior
python scripts/comprehensive_redshift_analysis.py \
    --pae_results "data/tf_results/PAE_results_*_noprior_*.npz" \
    --pae_samples "data/tf_results/PAE_samples_*_noprior_*.npz" \
    --run_name "nlatent5_noprior_combined" \
    --output_dir "figures/comprehensive/comparison/"
```

Compare different latent dimensions:

```bash
for nlatent in 3 5 7 10; do
    python scripts/comprehensive_redshift_analysis.py \
        --pae_results "data/tf_results/PAE_results_*_nlatent${nlatent}_*.npz" \
        --pae_samples "data/tf_results/PAE_samples_*_nlatent${nlatent}_*.npz" \
        --run_name "nlatent${nlatent}_gaussprior_combined" \
        --output_dir "figures/comprehensive/latent_comparison/"
done
```

### 4. Quick Diagnostic Check

Generate only diagnostic plots for a quick convergence check:

```bash
python scripts/comprehensive_redshift_analysis.py \
    --pae_results "data/tf_results/PAE_results_*.npz" \
    --pae_samples "data/tf_results/PAE_samples_*.npz" \
    --run_name "quick_diagnostic" \
    --plot_groups diagnostics convergence
```

## Integration with Batch Workflow

This tool integrates seamlessly with the batch processing workflow:

1. **Run batches**: Use `redshift_job.py` with SLURM job arrays
   ```bash
   sbatch --array=0-5 scripts/redshift_job.sh
   ```

2. **Combine and analyze**: Use glob patterns to automatically combine results
   ```bash
   python scripts/comprehensive_redshift_analysis.py \
       --pae_results "data/tf_results/PAE_results_175_srcs_*_start*.npz" \
       --pae_samples "data/tf_results/PAE_samples_175_srcs_*_start*.npz" \
       --run_name "production_run_$(date +%Y%m%d)" \
       --output_dir "figures/production/"
   ```

3. **Compare configurations**: Run analysis on different parameter sets
   ```bash
   bash scripts/example_comprehensive_analysis.sh
   ```

## Output Directory Organization

Recommended directory structure:

```
figures/
├── comprehensive/
│   ├── nlatent5_gaussprior/
│   │   ├── nlatent5_gaussprior_175srcs_comparison_pae_tf.png
│   │   ├── nlatent5_gaussprior_175srcs_coverage_grid.png
│   │   ├── nlatent5_gaussprior_175srcs_summary.txt
│   │   └── ...
│   ├── nlatent5_noprior/
│   │   └── ...
│   ├── comparison/
│   │   ├── nlatent5_gaussprior_combined_*.png
│   │   ├── nlatent5_noprior_combined_*.png
│   │   └── ...
│   └── latent_comparison/
│       ├── nlatent3_*.png
│       ├── nlatent5_*.png
│       ├── nlatent7_*.png
│       └── ...
```

## Statistics Computed

The script computes and reports:

### PAE Statistics
- **Bias**: Median of Δz/(1+z)
- **NMAD**: Normalized Median Absolute Deviation
  - NMAD = 1.48 × median(|Δz/(1+z) - median(Δz/(1+z))|)
- **Outlier fraction**: Fraction with |Δz/(1+z)| > 0.15
- **Z-score statistics**: Mean and std of (zout - ztrue) / σz
- **Convergence**: Mean R-hat and fraction > 1.1

### Template Fitting Statistics
- **Bias**: Median of Δz/(1+z)
- **NMAD**: Normalized Median Absolute Deviation
- **Outlier fraction**: Fraction with |Δz/(1+z)| > 0.15

### PIT Analysis
- **Kolmogorov-Smirnov statistic**: Tests uniformity of PIT distribution
  - Values < 0.05 indicate well-calibrated posteriors

## Key Differences from compare_redshift_results.py

| Feature | `compare_redshift_results.py` | `comprehensive_redshift_analysis.py` |
|---------|-------------------------------|--------------------------------------|
| Number of plots | 2 (basic + coverage) | 8+ (full suite) |
| Summary statistics | Printed only | Printed + saved to file |
| PIT analysis | No | Yes |
| Diagnostic plots | Limited | Comprehensive |
| PDF comparisons | No | Yes (examples grid) |
| Organization | Simple | Modular with plot groups |
| Use case | Quick checks | Publication-ready analysis |

## Performance

Typical runtime for 1000 sources:
- Basic plots: ~5 seconds
- Coverage analysis: ~10 seconds (with PDF loading)
- Diagnostics: ~3 seconds
- PDF examples: ~5 seconds
- PIT analysis: ~15 seconds (computing PIT values)
- **Total**: ~40-60 seconds for all plots

Memory usage scales with number of sources and chains preserved.

## Troubleshooting

### "TF PDFs not loaded"
- Ensure `--tf_load_zpdf=True` (default)
- Coverage and PDF plots require TF PDFs

### "Failed to load data"
- Check file paths and glob patterns
- Ensure PAE results and samples files match
- Verify files contain expected arrays

### Memory issues with large samples
- Use `--nsrc` to limit number of sources
- Consider analyzing subsets separately

### Convergence warnings
- Check R-hat histogram output
- If many sources have R-hat > 1.1, consider:
  - Increasing number of MCLMC steps
  - Adjusting step size parameters
  - Checking for multimodal posteriors

## See Also

- [BATCH_WORKFLOW.md](BATCH_WORKFLOW.md): Complete batch processing workflow
- [STREAMLINED_WORKFLOW.md](STREAMLINED_WORKFLOW.md): Overall workflow overview
- `scripts/compare_redshift_results.py`: Simpler comparison tool
- `scripts/example_comprehensive_analysis.sh`: Example usage patterns
