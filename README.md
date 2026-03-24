# PAESpec

Public release of the Probabilistic Autoencoder (PAE) workflow for SPHEREx-style
mock-data experiments.

This repository focuses on code and reproducible scripts. Large data products are
not bundled and must be provided locally by users.

## Canonical entrypoints

- `scripts/train_pae_autoencoder.py`
- `scripts/redshift_job_mock_batched.py`
- `scripts/generate_redshift_plots.py`

## YAML-first configuration

Both canonical Python entrypoints support `--config-yaml`.

Example:

```bash
python scripts/train_pae_autoencoder.py \
	--config-yaml configs/public_mock_template.yaml

python scripts/redshift_job_mock_batched.py \
	--config-yaml configs/public_mock_template.yaml
```

Use `configs/public_mock_template.yaml` as a starting point. Required knobs such
as `run_name` and `sources_per_task` can be set in YAML instead of on CLI.

## Portable output paths

The shell wrapper `scripts/run_redshift_job_mock.sh` now supports configurable
output roots via environment variables:

- `RESULTS_BASE_DIR` or `SPAE_RESULTS_BASE_DIR`
- `FIGURES_BASE_DIR` or `SPAE_FIGURES_BASE_DIR`

Defaults are local repo paths under `results/` and `figures/`.

## Docs

- `docs/PUBLIC_RELEASE_GUIDE.md` for public workflow guidance
- `configs/public_mock_template.yaml` for config schema template
