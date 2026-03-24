# PAESpec Configuration Reference

This page is the entry point for the split configuration reference.

## Reference map

- `docs/config/README.md`: navigation index for config docs
- `docs/config/YAML_SCHEMA.md`: YAML layout, section meanings, and required keys
- `docs/config/TRAINING_REFERENCE.md`: `train_pae_autoencoder.py` argument reference
- `docs/config/INFERENCE_REFERENCE.md`: `redshift_job_mock_batched.py` argument reference
- `docs/config/PLOTTING_REFERENCE.md`: `generate_redshift_plots.py` argument reference
- `docs/config/ENVIRONMENT_REFERENCE.md`: shell wrapper variables and path overrides

## Quick orientation

- Training entrypoint: `scripts/train_pae_autoencoder.py`
- Mock redshift entrypoint: `scripts/redshift_job_mock_batched.py`
- Plotting entrypoint: `scripts/generate_redshift_plots.py`
- YAML template: `configs/public_mock_template.yaml`

If you are new to PAESpec, start with `docs/PUBLIC_RELEASE_GUIDE.md`, then use the
pages above to find exact argument definitions.
