# Quick Start: Comprehensive Redshift Analysis

## What It Does

Automates the generation of publication-quality analysis plots for PAE redshift estimation, including:

✅ Basic PAE vs TF comparison  
✅ Coverage analysis (calibration testing)  
✅ Diagnostic plots (z-scores, convergence)  
✅ PDF comparisons  
✅ PIT analysis (posterior calibration)  
✅ Summary statistics report  

## Quick Usage

```bash
# Single batch analysis
python scripts/comprehensive_redshift_analysis.py \
    --pae_results "data/tf_results/PAE_results_175_srcs_nlatent5_start0.npz" \
    --pae_samples "data/tf_results/PAE_samples_175_srcs_nlatent5_start0.npz" \
    --run_name "nlatent5_gaussprior_175srcs"

# Combined batches (automatic detection)
python scripts/comprehensive_redshift_analysis.py \
    --pae_results "data/tf_results/PAE_results_*_start*.npz" \
    --pae_samples "data/tf_results/PAE_samples_*_start*.npz" \
    --run_name "nlatent5_gaussprior_1000srcs_combined"
```

## Output Files

All files follow the pattern: `<run_name>_<plot_type>.png`

Example with `--run_name nlatent5_gaussprior_175srcs`:
- `nlatent5_gaussprior_175srcs_comparison_pae_tf.png`
- `nlatent5_gaussprior_175srcs_comparison_detailed.png`
- `nlatent5_gaussprior_175srcs_coverage_grid.png`
- `nlatent5_gaussprior_175srcs_zscore_comparison.png`
- `nlatent5_gaussprior_175srcs_rhat_histogram.png`
- `nlatent5_gaussprior_175srcs_pdf_examples.png`
- `nlatent5_gaussprior_175srcs_pit_histogram.png`
- `nlatent5_gaussprior_175srcs_pit_qq.png`
- `nlatent5_gaussprior_175srcs_summary.txt`

## Recommended Naming

Encode key parameters in run name:

```
nlatent<N>_<prior_type>_<ngal>srcs[_<additional_info>]
```

Examples:
- `nlatent5_gaussprior_175srcs`
- `nlatent10_noprior_1000srcs_combined`
- `nlatent5_gaussprior_500srcs_highz`

## Common Workflows

### 1. Analyze production run
```bash
python scripts/comprehensive_redshift_analysis.py \
    --pae_results "data/tf_results/PAE_results_*_start*.npz" \
    --pae_samples "data/tf_results/PAE_samples_*_start*.npz" \
    --run_name "production_$(date +%Y%m%d)" \
    --output_dir "figures/production/"
```

### 2. Compare different priors
```bash
# Gaussian prior
python scripts/comprehensive_redshift_analysis.py \
    --pae_results "data/tf_results/*_gaussprior_*.npz" \
    --pae_samples "data/tf_results/*_gaussprior_*.npz" \
    --run_name "gaussprior_combined" \
    --output_dir "figures/comparison/"

# No prior
python scripts/comprehensive_redshift_analysis.py \
    --pae_results "data/tf_results/*_noprior_*.npz" \
    --pae_samples "data/tf_results/*_noprior_*.npz" \
    --run_name "noprior_combined" \
    --output_dir "figures/comparison/"
```

### 3. Quick diagnostic check
```bash
python scripts/comprehensive_redshift_analysis.py \
    --pae_results "data/results.npz" \
    --pae_samples "data/samples.npz" \
    --run_name "quick_check" \
    --plot_groups diagnostics convergence
```

### 4. Compare latent dimensions
```bash
for nlatent in 3 5 7 10; do
    python scripts/comprehensive_redshift_analysis.py \
        --pae_results "data/tf_results/*_nlatent${nlatent}_*.npz" \
        --pae_samples "data/tf_results/*_nlatent${nlatent}_*.npz" \
        --run_name "nlatent${nlatent}_combined" \
        --output_dir "figures/latent_comparison/"
done
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output_dir` | `figures/analysis/` | Where to save plots |
| `--burn_in` | `1000` | Burn-in steps to trim |
| `--plot_groups` | `all` | `all`, `basic`, `coverage`, `diagnostics`, `pdf`, `convergence` |
| `--dpi` | `300` | Figure resolution |
| `--nsrc` | All sources | Limit number of sources |

## Key Statistics Reported

### PAE Performance
- Bias: median(Δz/(1+z))
- NMAD: 1.48 × median(\|Δz/(1+z) - median(Δz/(1+z))\|)
- Outlier fraction: fraction with \|Δz/(1+z)\| > 0.15
- Z-score statistics: mean and std of (zout - ztrue) / σz
- Convergence: mean R-hat, fraction > 1.1
- PIT KS statistic: uniformity test (< 0.05 is good)

### Template Fitting Performance
- Bias, NMAD, outlier fraction

All statistics are saved to `<run_name>_summary.txt`

## Integration with Workflow

```
1. Run batches
   ↓
   sbatch --array=0-5 scripts/redshift_job.sh
   
2. Analyze results
   ↓
   python scripts/comprehensive_redshift_analysis.py \
       --pae_results "data/tf_results/PAE_results_*_start*.npz" \
       --pae_samples "data/tf_results/PAE_samples_*_start*.npz" \
       --run_name "production_20241215"
       
3. Compare configurations
   ↓
   bash scripts/example_comprehensive_analysis.sh
```

## Full Documentation

See [COMPREHENSIVE_ANALYSIS.md](COMPREHENSIVE_ANALYSIS.md) for:
- Complete plot descriptions
- Detailed usage examples
- Performance notes
- Troubleshooting

## Example Script

Run pre-configured examples:
```bash
bash scripts/example_comprehensive_analysis.sh
```

This demonstrates:
- Single batch analysis
- Combined batch analysis
- Selective plot generation
- Prior comparison
- Latent dimension comparison
