# Per-Source Plots Quick Start

## What You Need

The per-source plotting script requires:

1. **Result file**: `PAE_results_combined_*.npz` (redshift estimates)
2. **Sample file**: `PAE_samples_combined_*.npz` (MCMC posterior samples)  
3. **Model configuration**: Must match the model used during redshift estimation

## Model Configuration

The script uses `initialize_PAE()` (same as redshift scripts) with these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--run-name` | `jax_conv1_nlatent=5_siglevelnorm=0.01_newAllen_all_091325` | Trained model checkpoint |
| `--filter-set` | `SPHEREx_filter_102` | Filter configuration (102 or 408 band) |
| `--filename-flow` | `flow_model_iaf_092225` | Normalizing flow model |
| `--nlatent` | `5` | Number of latent dimensions |
| `--sig-level-norm` | `0.01` | Sigma level for normalization |
| `--sel-str` | `all` | Data selection string |

**CRITICAL**: These parameters must match what was used in your redshift run!

## Basic Usage

```bash
# Use defaults (for standard 102-band runs)
./scripts/run_source_plots.sh multicore_test_16k_wf_123025
```

## If Your Run Used Different Settings

### 408-band filters
```bash
./scripts/run_source_plots.sh <datestr> --filter-set SPHEREx_filter_408
```

### Different latent dimensions
```bash
./scripts/run_source_plots.sh <datestr> --nlatent 10 --run-name "jax_conv1_nlatent=10_siglevelnorm=0.01_newAllen_all"
```

### Different sigma level
```bash
./scripts/run_source_plots.sh <datestr> --sig-level-norm 0.005
```

## Full Example with All Parameters

```bash
python scripts/generate_source_reconstructions.py \
    --datestr multicore_test_16k_wf_123025 \
    --run-name "jax_conv1_nlatent=5_siglevelnorm=0.01_newAllen_all_091325" \
    --filter-set SPHEREx_filter_102 \
    --filename-flow flow_model_iaf_092225 \
    --nlatent 5 \
    --sig-level-norm 0.01 \
    --sel-str all \
    --n-good 50 \
    --n-bad 50 \
    --verbose
```

## How It Works

1. **Loads PAE model**: Uses `initialize_PAE()` with your configuration
2. **Loads COSMOS data**: Uses `load_spherex_data()` to get training/validation data
3. **Matches samples to sources**: Uses source indices from result files
4. **Generates plots**: For good/bad fits based on chi² and z-scores

## What Gets Loaded

```
Model:      modl_runs/<run_name>/
Flow:       modl_runs/<run_name>/<filename_flow>.pkl  
Data:       /global/cfs/cdirs/desi/science/td/pfs/COSMOS_xmatch/cosmos2020_final_*.fits
            (via load_spherex_data)
Filters:    data/filters/<filter_set>/
```

## Common Configurations

### Standard 102-band (default)
```bash
--run-name "jax_conv1_nlatent=5_siglevelnorm=0.01_newAllen_all_091325"
--filter-set SPHEREx_filter_102
--filename-flow flow_model_iaf_092225
--nlatent 5
--sig-level-norm 0.01
```

### 408-band
```bash
--run-name "jax_conv1_nlatent=5_siglevelnorm=0.01_newAllen_all_408band"
--filter-set SPHEREx_filter_408
--filename-flow flow_model_iaf_092225
--nlatent 5
--sig-level-norm 0.01
```

## Troubleshooting

### "Model checkpoint not found"
- Check `--run-name` matches directory in `modl_runs/`
- Verify checkpoint files exist: `ls modl_runs/<run_name>/`

### "Flow model not found"
- Check `--filename-flow` matches file in `modl_runs/<run_name>/`
- Should be `.pkl` file (extension added automatically)

### "Filter files not found"
- Check `--filter-set` matches directory in `data/filters/`
- Common: `SPHEREx_filter_102` or `SPHEREx_filter_408`

### "Wrong number of latent dimensions"
- Check `--nlatent` matches model training (usually 5, sometimes 10)
- Must match the model checkpoint `run_name`

### "Data mismatch"
- The script loads COSMOS mock data for visualization
- This is expected - it matches samples by index to training data
- For real SPHEREx data: modification needed to load actual parquet

## Advanced: Custom Model

If you trained a custom model:

```bash
python scripts/generate_source_reconstructions.py \
    --datestr <your_run> \
    --run-name "custom_model_name" \
    --filter-set SPHEREx_filter_102 \
    --filename-flow custom_flow_model \
    --nlatent 8 \
    --sig-level-norm 0.005 \
    --verbose
```

## Output

Figures saved to:
```
/pscratch/sd/r/rmfeder/figures/redshift_validation/<datestr>/
├── good_fits/
│   ├── source_XXXXX_reconstruction.png
│   ├── source_XXXXX_corner.png
│   └── source_XXXXX_zposterior.png
└── bad_fits/
    ├── source_YYYYY_reconstruction.png
    ├── source_YYYYY_corner.png
    └── source_YYYYY_zposterior.png
```

See [SOURCE_PLOTS_README.md](SOURCE_PLOTS_README.md) for full documentation.
