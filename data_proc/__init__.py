"""
Data loading, preprocessing, and file I/O utilities.

This package contains:
- dataloader_jax: SPHERExData class, dataloaders, filter conversion
- data_file_utils: Model/result saving/loading, file path management
- sphx_data_proc: SPHEREx-specific data preprocessing
"""

from .dataloader_jax import (
    SPHERExData,
    prep_obs_dat,
    convert_filters_to_jax,
    create_train_validation_dataloaders_jax,
    draw_train_validation_idxs_jax,
    grab_train_validation_dat_jax,
    gen_subset_flux_dataclasses,
    subset_with_missing,
    grab_idxs,
    make_bfit_tid_dict,
    update_dict,
    spec_data_jax,
)

# File utilities - import all
from .data_file_utils import (
    create_result_dir_structure,
    check_config_dir,
    save_params,
    grab_fpaths_rf,
    grab_fpaths_traindat,
    load_combined_pae_results,
    load_tf_results,
    load_sphx_filters,
    grab_ext_phot_filt_files,
    load_ext_filters,
    load_indiv_filter,
    save_ae_jax,
    load_jax_state,
    save_train_metrics_jax,
    save_model,
    load_model,
)

# SPHEREx preprocessing
from .sphx_data_proc import (
    zweight_naninf_vals,
    make_crossmatch_property_cat,
)

__all__ = [
    # Core data class
    'SPHERExData',
    # Data preparation
    'prep_obs_dat',
    'convert_filters_to_jax',
    'gen_subset_flux_dataclasses',
    'subset_with_missing',
    # Dataloaders
    'create_train_validation_dataloaders_jax',
    'draw_train_validation_idxs_jax',
    'grab_train_validation_dat_jax',
    'update_dict',
    'spec_data_jax',
    # Utilities
    'grab_idxs',
    'make_bfit_tid_dict',
    # File I/O
    'create_result_dir_structure',
    'check_config_dir',
    'save_params',
    'grab_fpaths_rf',
    'grab_fpaths_traindat',
    'load_combined_pae_results',
    'load_tf_results',
    'load_sphx_filters',
    'grab_ext_phot_filt_files',
    'load_ext_filters',
    'load_indiv_filter',
    'save_ae_jax',
    'load_jax_state',
    'save_train_metrics_jax',
    'save_model',
    'load_model',
    # SPHEREx preprocessing
    'zweight_naninf_vals',
    'make_crossmatch_property_cat',
]
