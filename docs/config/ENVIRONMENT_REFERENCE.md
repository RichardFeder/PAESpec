# Environment and Path Reference

This page documents shell and environment variables used by public PAESpec workflow scripts.

## Shell wrapper variables

Script: `scripts/run_redshift_job_mock.sh`

- `RESULTS_BASE_DIR` or `SPAE_RESULTS_BASE_DIR`: Base directory for inference outputs
- `FIGURES_BASE_DIR` or `SPAE_FIGURES_BASE_DIR`: Base directory for figures
- `COLLATE_RESULTS`: `true` or `false`
- `GENERATE_PLOTS`: `true` or `false`
- `SKIP_REDSHIFT_RUN`: `true` or `false`
- `REPROCESS_BURNIN`: `true` or `false`
- `REPROCESS_BURNIN_VALUE`: Burn-in used by reprocessing path
- `MODEL_RUN_NAME`: Trained model run name to load
- `NF_ALPHA`: Flow-prior strength
- `BPZ_PRIOR_ENABLED`: Enable BPZ prior mode
- `BPZ_PRIOR_JSON`: BPZ prior fit JSON path
- `SAMPLE_LOG_REDSHIFT`: `true` or `false`
- `DATESTR`: Output run tag

## Python path overrides

Module: `config.py`

- `SPAE_BASE_PATH`
- `SPAE_PROJECT_PATH`
- `SPAE_SCRATCH_BASEPATH`
- `SPAE_DATA_PATH`
- `SPAE_MODEL_RUNS_PATH`

Use these when local data or output locations differ from defaults.
