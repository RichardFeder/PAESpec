"""
MCMC sampling algorithms and utilities.

This package contains:
- mclmc: MCLMC sampler wrapper functions
- sample_pae: Core sampling functions (PocoMC, initialization)
- sample_pae_batch: Deprecated batch sampling
- sample_pae_batch_refactor: Current batch sampling implementation
- samplers_aux: Auxiliary sampling utilities
"""

from .mclmc import (
    run_mclmc_simp,
)

from .sample_pae import (
    grab_map_spec,
    initialize_latents,
    pae_spec_sample_pocomc,
    sample_pocomc_wrapper,
    postproc_samples,
    compare_chains_logp_indiv,
)

from .sample_pae_batch_refactor import (
    MCLMCSamplingConfig,
    sample_mclmc_wrapper,
    initialize_latents_scale,
    reinit_chains,
    batched_log_likelihood,
    pae_spec_sample_fixz_vmap,
    run_batched_sampler,
)

# Import samplers_aux functions if they exist
try:
    from .samplers_aux import *
except ImportError:
    pass

__all__ = [
    # MCLMC
    'run_mclmc_simp',
    # Configuration
    'MCLMCSamplingConfig',
    # Main sampling functions
    'sample_mclmc_wrapper',
    'pae_spec_sample_pocomc',
    'sample_pocomc_wrapper',
    'pae_spec_sample_fixz_vmap',
    'run_batched_sampler',
    # Initialization
    'initialize_latents',
    'initialize_latents_scale',
    'reinit_chains',
    # Utilities
    'grab_map_spec',
    'postproc_samples',
    'compare_chains_logp_indiv',
    'batched_log_likelihood',
]
