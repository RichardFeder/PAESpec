# Training Argument Reference

Script: `scripts/train_pae_autoencoder.py`

## High-priority run control

- `--config-yaml`: Path to YAML defaults file
- `--run-name`: Output model run name (required)
- `--skip-ae-training`: Skip autoencoder fitting
- `--extract-latents`: Extract/save latents from trained model
- `--train-flow`: Train flow on saved latents
- `--skip-flow-training`: Disable flow training stage

## Core model architecture

- `--nlatent`: Latent dimensionality
- `--filters`: Encoder/decoder convolution channel counts
- `--filter-sizes`: Encoder/decoder kernel sizes
- `--n-hidden-encoder`: Dense hidden sizes in encoder/decoder blocks

## Optimization and schedule

- `--epochs`: Number of AE epochs
- `--batch-size`: AE minibatch size
- `--lr`: AE learning rate
- `--plot-interval`: Plot/checkpoint interval in epochs

## Similarity and consistency losses

- `--lambda-sim`: Similarity regularization weight
- `--lambda-consistency`: Consistency augmentation weight
- `--sim-k0`, `--sim-k1`: Similarity shape/kernel controls
- `--sigma-s`: Similarity scale parameter
- `--similarity-subsample-size`: Subsample size for similarity term
- `--similarity-eps`: Stability epsilon for similarity calculations
- `--consistency-aug-scale`: Augmentation scale for consistency term

## Reconstruction scaling

- `--recon-scale-mode`: Reconstruction scaling mode
- `--amp-eps`: Amplitude normalization epsilon
- `--amp-clip-min`, `--amp-clip-max`: Optional amplitude clipping bounds

## Data and preprocessing

- `--nbands`: Number of rest-frame wavelength bins
- `--use-log-wavelength`: Enable log-spaced wavelength interpolation
- `--sig-level-norm`: Noise level for augmentation/normalization
- `--train-frac`: Train/validation split fraction
- `--data-file`: Optional explicit input data path
- `--z-max`: Optional redshift upper bound filter
- `--scratch-base`: Base path used by data loaders

## Flow training stage

- `--lr-flow`: Flow optimizer learning rate
- `--nepoch-flow`: Number of flow epochs
- `--batch-size-flow`: Flow training minibatch size
- `--max-sources-flow`: Max latent samples used for flow training

## Parallel execution

- `--use-multicore`: Enable multi-device mode
- `--n-devices`: Number of devices in multi-device mode
