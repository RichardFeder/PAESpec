# YAML Schema Reference

This page documents the public YAML layout used by `--config-yaml`.

## Top-level sections

- `common`: shared defaults used by both training and mock-redshift scripts
- `training`: defaults for `scripts/train_pae_autoencoder.py`
- `redshift_mock`: defaults for `scripts/redshift_job_mock_batched.py`

## Required fields

- `training.run_name`
- `redshift_mock.run_name`
- `redshift_mock.sources_per_task`

If required fields are missing, scripts will fail with a clear error.

## Minimal example

```yaml
common:
  filter_set: spherex_filters102/
  sig_level_norm: 0.01

training:
  run_name: my_first_paespec_run

redshift_mock:
  run_name: my_first_paespec_run
  sources_per_task: 1000
  datestr: my_first_paespec_eval
```

## Behavior notes

- `common` keys are applied first, then script-specific section keys.
- Unknown keys are ignored with a warning.
- CLI flags still override YAML defaults when explicitly passed.
