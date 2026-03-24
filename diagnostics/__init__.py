"""
Diagnostics, convergence checks, and statistical analysis.

This package contains:
- diagnostics_jax: Convergence diagnostics, chi2, redshift statistics
"""

from .diagnostics_jax import (
    monte_carlo_correlation_null_distribution,
    proc_spec_post,
    monte_carlo_profile_likelihood_jax,
    compute_rho_from_redshift_latents,
    compute_redshift_stats,
    compute_redshift_percentiles,
    hpd_interval,
    hpd_interval2,
    compute_hdpi,
    compute_chi2_perobj,
    compute_mse_perobj_jax,
    cleanup_mask,
    calc_all_gr,
    gelman_rubin,
    save_redshift_results,
    effective_sample_size,
    quick_ess,
    posterior_skewness,
    count_peaks,
    mean_outside_hpd,
    check_posterior_quality,
    grab_encoded_vars_dataset,
    plot_latent_z_params,
    ae_result_fig_wrapper,
    nf_result_fig_wrapper,
    compare_flow_to_latentz,
)

__all__ = [
    # Convergence and quality
    'monte_carlo_correlation_null_distribution',
    'gelman_rubin',
    'calc_all_gr',
    'effective_sample_size',
    'quick_ess',
    'check_posterior_quality',
    'cleanup_mask',
    # Posterior processing
    'proc_spec_post',
    'monte_carlo_profile_likelihood_jax',
    'postproc_samples',
    # Redshift statistics
    'compute_rho_from_redshift_latents',
    'compute_redshift_stats',
    'compute_redshift_percentiles',
    'save_redshift_results',
    # HPD and intervals
    'hpd_interval',
    'hpd_interval2',
    'compute_hdpi',
    'mean_outside_hpd',
    'posterior_skewness',
    'count_peaks',
    # Chi-squared and metrics
    'compute_chi2_perobj',
    'compute_mse_perobj_jax',
    # Encoding/latent analysis
    'grab_encoded_vars_dataset',
    # Visualization
    'plot_latent_z_params',
    'ae_result_fig_wrapper',
    'nf_result_fig_wrapper',
    'compare_flow_to_latentz'
]
