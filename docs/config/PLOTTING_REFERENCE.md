# Plotting Argument Reference

Script: `scripts/generate_redshift_plots.py`

## Input selection

- `--result-file`: Explicit combined results NPZ file path
- `--datestr`: Run tag used to auto-locate combined results
- `--output-dir`: Output figure directory

## Display/output mode

- `--show`: Show plots interactively
- `--no-show`: Save plots only (batch-safe)

## Quality and filtering controls

- `--zscore-range`: z-score plotting range
- `--rhat_max`: Maximum R-hat filter
- `--chi2_max`: Maximum chi2 filter
- `--chain_std_max`: Maximum chain standard-deviation filter
- `--quality_tier_max`: Maximum quality-tier filter
- `--tuning_cv_min`: Minimum tuning-CV threshold
- `--frac_sampled_min`: Minimum sampled-fraction threshold
- `--snr_min`: Minimum SNR threshold

## Plot feature toggles

- `--hexbin`, `--no-hexbin`: Enable/disable hexbin comparisons
- `--mclmc-diagnostics`, `--no-mclmc-diagnostics`: Enable/disable MCLMC diagnostics panels
- `--snr-diagnostics`, `--no-snr-diagnostics`: Enable/disable SNR diagnostics
- `--compare-datestr`: Optional second run tag for comparison plots
