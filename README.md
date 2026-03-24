# PAESpec

Probabilistic Autoencoder (PAE) implementation for SED modeling and redshift estimation of SPHEREx spectrophotometry. This software release corresponds to Feder+2026 (arXiv:xx).

This repository focuses on the main code to train and execute PAESpec on multi-band photometry. Large data products are
not bundled and must be provided locally by users.

## Installation

We recommend creating a dedicated Python environment for PAESpec rather than
installing into your base environment. This avoids version conflicts with other
JAX/ML projects.

```bash
# Choose any environment directory name you like.
ENV_NAME=.venv-paespec
python -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"
pip install -r requirements.txt
```

The main dependency groups are:
- JAX ecosystem: `jax`, `flax`, `optax`, `blackjax`, `jaxopt`
- Flow and model tooling: `flowjax`, `equinox`, `distrax`, `paramax`
- Scientific Python stack: `numpy`, `scipy`, `pandas`, `astropy`, `matplotlib`
- Utilities: `PyYAML` (YAML config support)

## Main workflow scripts

- `scripts/train_pae_autoencoder.py`
- `scripts/redshift_job_mock_batched.py`
- `scripts/generate_redshift_plots.py`

## YAML-first configuration

The training and mock-redshift scripts support `--config-yaml`.

Example:

```bash
python scripts/train_pae_autoencoder.py \
	--config-yaml configs/public_mock_template.yaml

python scripts/redshift_job_mock_batched.py \
	--config-yaml configs/public_mock_template.yaml
```

Use `configs/public_mock_template.yaml` as a starting point. Required knobs such
as `run_name` and `sources_per_task` can be set in YAML instead of on CLI.

Minimal working YAML example:

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

If you are starting from scratch, use this flow:

1. Copy `configs/public_mock_template.yaml` and edit it for your run.
2. Set required fields:
	- `training.run_name`
	- `redshift_mock.sources_per_task`
	- `redshift_mock.run_name` (usually the same model run name as training)
3. Optionally set `redshift_mock.datestr` to control output naming.
4. Run the two commands above with your edited YAML file.

For details on what each section means, see `docs/PUBLIC_RELEASE_GUIDE.md`
(especially "YAML-first usage" and "Required keys").
For full parameter definitions, start at `docs/CONFIG_REFERENCE.md`.

## Portable output paths

The shell wrapper `scripts/run_redshift_job_mock.sh` now supports configurable
output roots via environment variables:

- `RESULTS_BASE_DIR` or `SPAE_RESULTS_BASE_DIR`
- `FIGURES_BASE_DIR` or `SPAE_FIGURES_BASE_DIR`

Defaults are local repo paths under `results/` and `figures/`.

## Docs

- `docs/PUBLIC_RELEASE_GUIDE.md` for public workflow guidance
- `docs/CONFIG_REFERENCE.md` as the reference hub for all parameter definitions
- `docs/config/README.md` index for split config docs
- `docs/config/YAML_SCHEMA.md` for YAML sections and required keys
- `docs/config/TRAINING_REFERENCE.md` for training arguments
- `docs/config/INFERENCE_REFERENCE.md` for mock redshift arguments
- `docs/config/PLOTTING_REFERENCE.md` for plotting and diagnostics arguments
- `docs/config/ENVIRONMENT_REFERENCE.md` for shell/environment variables
- `configs/public_mock_template.yaml` for config schema template
