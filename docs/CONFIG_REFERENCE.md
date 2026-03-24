# PAESpec Configuration Reference

This document collects the main user-facing parameters across the public PAESpec workflow.
It is intended as a foundation for future hosted docs.

## How to use this reference

- Training is controlled by `scripts/train_pae_autoencoder.py`.
- Mock redshift inference is controlled by `scripts/redshift_job_mock_batched.py`.
- Post-run diagnostics and figures are controlled by `scripts/generate_redshift_plots.py`.
- YAML defaults can be passed to training/inference with `--config-yaml`.

## YAML layout

Current public YAML structure:

- `common`: shared defaults for scripts that load YAML
- `training`: defaults for `train_pae_autoencoder.py`
- `redshift_mock`: defaults for `redshift_job_mock_batched.py`

Required keys:

- `training.run_name`
- `redshift_mock.run_name`
- `redshift_mock.sources_per_task`

## Training script parameters

Script: `scripts/train_pae_autoencoder.py`

### High-priority run control

- `--config-yaml`: Path to YAML defaults file.
- `--run-name`: Name of model output directory under model-runs path. Required.
- `--skip-ae-training`: Skip autoencoder fitting.
- `--extract-latents`: Extract/save latents from a trained model.
- `--train-flow`: Train flow on saved latents.
- `--skip-flow-training`: Disable flow training stage.

### Core model architecture

- `--nlatent`: Latent dimensionality.
- `--filters`: Convolution channel counts for encoder/decoder blocks.
- `--filter-sizes`: Kernel sizes for convolution blocks.
- `--n-hidden-encoder`: Hidden layer sizes for dense encoder/decoder portions.

### Optimization and training schedule

- `--epochs`: Number of AE epochs.
- `--batch-size`: AE minibatch size.
- `--lr`: AE learning rate.
- `--plot-interval`: Epoch interval for saving plots.

### Similarity and consistency loss terms

- `--lambda-sim`: Weight for similarity regularization.
- `--lambda-consistency`: Weight for consistency augmentation term.
- `--sim-k0`, `--sim-k1`: Similarity kernel/shape controls.
- `--sigma-s`: Similarity scale.
- `--similarity-subsample-size`: Subsampling for similarity computations.
- `--similarity-eps`: Numerical stability epsilon for similarity term.
- `--consistency-aug-scale`: Augmentation scale for consistency term.

### Reconstruction scaling

- `--recon-scale-mode`: Reconstruction amplitude/scaling mode.
- `--amp-eps`: Epsilon for amplitude normalization.
- `--amp-clip-min`, `--amp-clip-max`: Optional clipping of amplitude estimates.

### Data and preprocessing

- `--nbands`: Number of rest-frame wavelength bins.
- `--use-log-wavelength`: Use log-spaced wavelength interpolation.
- `--sig-level-norm`: Input noise level used for augmentation/normalization.
- `--train-frac`: Training fraction in train/validation split.
- `--data-file`: Optional explicit input data file.
- `--z-max`: Optional maximum redshift filter.
- `--scratch-base`: Base data path used by data loaders.

### Flow training stage

- `--lr-flow`: Flow optimizer learning rate.
- `--nepoch-flow`: Number of flow epochs.
- `--batch-size-flow`: Flow minibatch size.
- `--max-sources-flow`: Cap on latent samples used for flow fit.

### Parallel execution

- `--use-multicore`: Enable pmap-style multi-device mode.
- `--n-devices`: Number of devices to use when multicore is enabled.

## Mock redshift inference parameters

Script: `scripts/redshift_job_mock_batched.py`

### High-priority run control

- `--config-yaml`: Path to YAML defaults file.
- `--run-name`: Trained model run directory name.
- `--datestr`: Output run tag used in saved filenames/directories.
- `--collate-results`: Combine per-batch files after run.
- `--stop-on-error`: Abort remaining tasks on first batch/task failure.

### Data/model selection

- `--filter-set`: Filter-set subdirectory name.
- `--nlatent`: Latent dimension expected by model.
- `--sig-level-norm`: Noise/normalization level used to match trained model assumptions.
- `--sel-str`: Selection label passed into loader logic.
- `--with-ext-phot`: Include external photometry channels if available.
- `--data-fpath`: Optional explicit data file override.
- `--filename-flow`: Flow model filename stem.

### Task partitioning and throughput

- `--sources-per-task`: Number of sources per task. Required.
- `--start-task`: Starting task index.
- `--max-tasks`: Number of tasks to process.
- `--batch-size`: Number of sources per likelihood/model-eval batch.
- `--sampling-batch-size`: Number of chains/sources sampled concurrently.

### Sample filtering

- `--z-min`, `--z-max`: Source redshift filter bounds.
- `--snr-min`, `--snr-max`: Source SNR filter bounds.

### MCLMC controls

- `--num-steps`: Total MCMC steps.
- `--burn-in`: Burn-in steps.
- `--nsamp-init`: Initialization samples.
- `--chi2-red-threshold`: Reduced-chi2 threshold for quality/reinit logic.
- `--gr-threshold`: Gelman-Rubin threshold.
- `--fix-z`: Fix redshift instead of sampling.

### Prior and parameterization controls

- `--nf-alpha`: Flow-prior strength during final sampling.
- `--nf-alpha-burnin`: Flow-prior strength during burn-in.
- `--prior-type`: Redshift prior mode (none/Gaussian/BPZ depending on implementation).
- `--z0-prior`, `--sigma-prior`: Gaussian/BPZ prior shape controls.
- `--bpz-prior-json`: JSON fit file for BPZ prior parameters.
- `--sample-log-redshift`: Sample in log-redshift space.
- `--sample-log-amplitude`: Sample log-amplitude explicitly.
- `--log-amplitude-prior-std`: Prior width for log-amplitude.

### Multi-device execution

- `--use-multicore`: Enable multi-device sampling mode.
- `--n-devices`: Number of devices for multicore mode.

### SNR-prefit initialization

- `--use-snr-prefit-init`: Enable SNR-based initialization from prefit mapping.
- `--snr-prefit-json`: JSON file with SNR-prefit parameters.
- `--snr-prefit-column`: Column name used for SNR lookup.

## Plot generation parameters

Script: `scripts/generate_redshift_plots.py`

### Input selection

- `--result-file`: Explicit combined results NPZ path.
- `--datestr`: Run tag used to auto-locate combined results.
- `--output-dir`: Figure output directory.

### Display/output mode

- `--show`: Show plots interactively.
- `--no-show`: Save plots only (batch-safe).

### Quality and filtering controls

- `--zscore-range`: z-score plotting range.
- `--rhat_max`: Max R-hat filter.
- `--chi2_max`: Max chi2 filter.
- `--chain_std_max`: Max chain-std filter.
- `--quality_tier_max`: Max quality-tier filter.
- `--tuning_cv_min`: Minimum tuning CV threshold.
- `--frac_sampled_min`: Minimum sampled-fraction threshold.
- `--snr_min`: Minimum SNR threshold.

### Plot feature toggles

- `--hexbin`, `--no-hexbin`: Enable/disable hexbin comparisons.
- `--mclmc-diagnostics`, `--no-mclmc-diagnostics`: Enable/disable MCLMC diagnostic panels.
- `--snr-diagnostics`, `--no-snr-diagnostics`: Enable/disable SNR diagnostics.
- `--compare-datestr`: Optional second run tag for comparison plots.

## Shell wrapper environment variables

Script: `scripts/run_redshift_job_mock.sh`

- `RESULTS_BASE_DIR` or `SPAE_RESULTS_BASE_DIR`: Base directory for inference outputs.
- `FIGURES_BASE_DIR` or `SPAE_FIGURES_BASE_DIR`: Base directory for figures.
- `COLLATE_RESULTS`: `true` or `false`.
- `GENERATE_PLOTS`: `true` or `false`.
- `SKIP_REDSHIFT_RUN`: `true` or `false`.
- `REPROCESS_BURNIN`: `true` or `false`.
- `REPROCESS_BURNIN_VALUE`: Burn-in value for reprocessing path.
- `MODEL_RUN_NAME`: Trained run to load.
- `NF_ALPHA`: Flow-prior strength.
- `BPZ_PRIOR_ENABLED`: Enable BPZ prior mode.
- `BPZ_PRIOR_JSON`: BPZ prior parameter file.
- `SAMPLE_LOG_REDSHIFT`: `true` or `false`.
- `DATESTR`: Output run tag.

## Notes for future documentation

This reference is intentionally script-focused and user-facing. If you later build hosted docs,
it can be split into:

- Quick start
- YAML schema and examples
- Training parameter reference
- Inference parameter reference
- Plotting and diagnostics reference
- Environment and deployment guidance
