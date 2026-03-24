# Streamlined Batch Workflow Summary

## What Changed

I've integrated `load_combined_pae_results` with `compare_redshift_results.py` to create a streamlined workflow for batch redshift analysis.

## Key Improvements

### 1. Automatic Batch Detection
`compare_redshift_results.py` now automatically detects and handles multiple batch files:

```bash
# OLD: Required manual combining of files first
python combine_files.py ...
python compare_redshift_results.py --pae_results combined.npz ...

# NEW: Directly use glob patterns
python compare_redshift_results.py \
    --pae_results "PAE_results_*_start*.npz" \
    --pae_samples "PAE_samples_*_start*.npz" \
    ...
```

### 2. Flexible Chain Processing
Full chains are preserved after loading:
- Burn-in trimming applied **per-analysis** as needed
- Enables PIT and other PDF-based analyses (need full chains)
- Enables coverage analysis (needs burn-in trimming)
- Same data serves multiple analysis types

### 3. Single Command Workflow
```bash
# One command does everything:
python scripts/compare_redshift_results.py \
    --pae_results "data/redshift_results/PAE_results_*_start*.npz" \
    --pae_samples "data/redshift_results/PAE_samples_*_start*.npz" \
    --plot_comparison --plot_coverage --tf_load_zpdf \
    --burn_in 1000 --output_dir figures/
```

This automatically:
1. Finds all matching batch files
2. Loads and combines them (sorted by start index)
3. Preserves full chains for flexible analysis
4. Compares against template fitting results
5. Generates comparison and coverage plots (applying burn-in per-plot as needed)

## Files Modified

1. **scripts/compare_redshift_results.py**
   - Added `glob` import and `load_combined_pae_results`
   - Updated argument help text to mention glob patterns
   - Modified `load_pae_and_tf_data()` to detect and handle multiple files
   - Burn-in trimming happens immediately after combining files

2. **data_proc/__init__.py**
   - Exported `load_combined_pae_results` function

## Files Created

1. **scripts/analyze_batch_runs.sh**
   - Helper script that wraps the entire workflow
   - Configurable burn-in and output directory
   - Error checking for missing files

2. **scripts/load_batch_results_example.py**
   - Standalone example for loading batch results
   - Shows how to access combined data in Python
   - Demonstrates burn-in trimming and statistics

3. **docs/BATCH_WORKFLOW.md**
   - Complete workflow documentation
   - Examples for interactive and SLURM execution
   - Troubleshooting guide
   - Prior configuration reference

## Usage Examples

### Quick Analysis (using helper script):
```bash
bash scripts/analyze_batch_runs.sh
```

### Custom Analysis:
```bash
python scripts/compare_redshift_results.py \
    --pae_results "data/redshift_results/PAE_results_*_gaussprior_start*.npz" \
    --pae_samples "data/redshift_results/PAE_samples_*_gaussprior_start*.npz" \
    --plot_comparison --plot_coverage \
    --tf_load_zpdf --burn_in 1000 \
    --output_dir figures/gauss_prior_analysis/
```

### Python API:
```python
from data_proc import load_combined_pae_results

results, samples = load_combined_pae_results(
    results_pattern="PAE_results_*_start*.npz",
    samples_pattern="PAE_samples_*_start*.npz"
)

# Trim burn-in
z_samples_trimmed = samples['z_samples'][:, :, 1000:]
```

## Workflow Diagram

```
Batch Jobs (SLURM or interactive)
    ↓
Multiple .npz files (different START_IDX)
    ↓
compare_redshift_results.py (with glob patterns)
    ├─ Automatic detection of multiple files
    ├─ load_combined_pae_results()
    ├─ Sort by start index
    ├─ Concatenate arrays
    ├─ Trim burn-in
    └─ Format for plotting
    ↓
Comparison with template fitting
    ↓
Figures: pae_tf_comparison.png, coverage_comparison_grid.png
```

## Benefits

1. **Fewer Steps**: No manual file combining needed
2. **Less Error-Prone**: Automatic sorting and concatenation
3. **Flexible**: Works with both single files and batch files
4. **Consistent**: Same burn-in trimming everywhere
5. **Documented**: Complete workflow guide in docs/BATCH_WORKFLOW.md

## Next Steps

You can now:
1. Run your batch jobs: `bash scripts/submit_redshift_job.sh`
2. Analyze results: `bash scripts/analyze_batch_runs.sh`
3. Or use the Python API for custom analysis

See [docs/BATCH_WORKFLOW.md](docs/BATCH_WORKFLOW.md) for complete documentation.
