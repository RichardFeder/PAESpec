"""
Plotting and visualization utilities.

This package contains:
- result_plotting_fns: Comprehensive plotting functions for results, spectra, posteriors
"""

# Import all plotting functions
from .result_plotting_fns import *

__all__ = [
    # These will be available after moving result_plotting_fns.py
    # Main functions include:
    # - plot_spec_with_incomplete_coverage
    # - plot_bandpass_and_interp
    # - plot_log_phot_weights
    # - plot_norm_phot_fluxes
    # - plot_snr_persource
    # - plot_unnorm_fluxes
    # etc.
]
