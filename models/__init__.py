"""
Neural network models and architectures.

This package contains:
- nn_modl_jax: Encoder/Decoder architectures (SpectrumEncoder_JAX, SpectrumDecoder_JAX)
- pae_jax: Probabilistic Autoencoder class and setup functions
- flowjax_modl: FlowJAX-based normalizing flow models
- flow_jax: Distrax-based flow models (MAF, IAF)
"""

from .nn_modl_jax import (
    SpectrumEncoder_JAX,
    SpectrumDecoder_JAX,
)

from .pae_jax import (
    PAE_JAX,
    initialize_PAE,
    load_spherex_data,
    set_up_pae_wrapper,
    convert_to_bfloat16_recursive,
)

# Flow models - use wildcard to get all classes
from .flowjax_modl import *
from .flow_jax import *

__all__ = [
    # Core architectures
    'SpectrumEncoder_JAX',
    'SpectrumDecoder_JAX',
    # PAE
    'PAE_JAX',
    'initialize_PAE',
    'load_spherex_data',
    'set_up_pae_wrapper',
    'convert_to_bfloat16_recursive',
]
