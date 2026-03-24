"""
Model training utilities and optimization.

This package contains:
- train_ae_jax: Autoencoder training loops, loss functions, state management
- fine_tune_impl: Fine-tuning utilities and gradient analysis
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

from .fine_tune_impl import (
    fine_tune_manual_gradient,
    deep_inspect_model,
    diagnose_with_proper_manual_update,
    test_decoder_sensitivity,
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
    # Fine-tuning
    'fine_tune_manual_gradient',
    'deep_inspect_model',
    'diagnose_with_proper_manual_update',
    'test_decoder_sensitivity',
]
