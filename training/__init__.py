"""
Model training utilities and optimization.

This package contains:
- train_ae_jax: Autoencoder training loops, loss functions, state management
"""

from .train_ae_jax import (
    InferenceState,
    EarlyStopper,
    create_train_state,
    train_jax_modl,
    run_ae_sed_fit_jax,
    convert_to_decoder_inferencestate,
    plot_sed_recon_epoch_jax,
    forward_mse_loss,
    forward_logL_loss,
    gaussian_prior_kl,
)

__all__ = [
    # State and configuration
    'InferenceState',
    'EarlyStopper',
    'create_train_state',
    'convert_to_decoder_inferencestate',
    # Training
    'train_jax_modl',
    'run_ae_sed_fit_jax',
    # Loss functions
    'forward_mse_loss',
    'forward_logL_loss',
    'gaussian_prior_kl',
    # Visualization
    'plot_sed_recon_epoch_jax',
]
