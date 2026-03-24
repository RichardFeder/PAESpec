# Configuration Docs Index

This folder contains the split parameter-definition docs for PAESpec.

## Start here

- If you are setting up the pipeline for the first time, begin with `docs/PUBLIC_RELEASE_GUIDE.md`.
- If you need exact flag definitions, use the pages below.

## Pages

- `docs/config/YAML_SCHEMA.md`: YAML sections and required fields
- `docs/config/TRAINING_REFERENCE.md`: training script arguments
- `docs/config/INFERENCE_REFERENCE.md`: mock redshift script arguments
- `docs/config/PLOTTING_REFERENCE.md`: plotting script arguments
- `docs/config/ENVIRONMENT_REFERENCE.md`: environment variables and path configuration

## Script mapping

- Training: `scripts/train_pae_autoencoder.py`
- Inference: `scripts/redshift_job_mock_batched.py`
- Plotting: `scripts/generate_redshift_plots.py`
- Shell runner: `scripts/run_redshift_job_mock.sh`
