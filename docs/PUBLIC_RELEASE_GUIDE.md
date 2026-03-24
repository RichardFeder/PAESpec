# Public Release Guide

This guide describes the supported public workflow for PAESpec.

## Scope

This public repo is code-first and does not ship large private/mock datasets.
You provide input data paths in your local environment.

## Canonical entrypoints

- `scripts/train_pae_autoencoder.py`
- `scripts/redshift_job_mock_batched.py`
- `scripts/generate_redshift_plots.py`

## YAML-first usage

Both training and mock redshift scripts support `--config-yaml`.

```bash
python scripts/train_pae_autoencoder.py \
  --config-yaml configs/public_mock_template.yaml

python scripts/redshift_job_mock_batched.py \
  --config-yaml configs/public_mock_template.yaml
```

YAML behavior:
- `common` values are shared across scripts.
- `training` section applies to training script.
- `redshift_mock` section applies to mock redshift script.
- Unknown keys are ignored with a warning.

## Required keys

- Training: `run_name`
- Mock redshift: `sources_per_task`

These can be set in YAML or provided on CLI.

## Portable path configuration

### Python path config

`config.py` supports environment-variable overrides:

- `SPAE_BASE_PATH`
- `SPAE_PROJECT_PATH`
- `SPAE_SCRATCH_BASEPATH`
- `SPAE_DATA_PATH`
- `SPAE_MODEL_RUNS_PATH`

### Shell wrapper outputs

`scripts/run_redshift_job_mock.sh` supports:

- `RESULTS_BASE_DIR` or `SPAE_RESULTS_BASE_DIR`
- `FIGURES_BASE_DIR` or `SPAE_FIGURES_BASE_DIR`

Defaults are local repo paths under `results/mock_runs` and `figures/redshift_validation`.

## Dependency note

`--config-yaml` requires PyYAML:

```bash
pip install pyyaml
```
