import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
import jax
from astropy.io import ascii
from dataclasses import dataclass
from typing import Any, Optional, Sequence
import pandas as pd

import config
from visualization.result_plotting_fns import plot_bandpass_and_interp
from data_proc.sphx_data_proc import *


# import astropy
# import matplotlib.pyplot as plt


def preprocess_real_data_outliers(flux, flux_unc, missing_data_threshold=40000,
                                  high_flux_threshold=1e4, low_flux_threshold=-1e3, extreme_outlier_factor=100,
                                  downweight_factor=0.01, verbose=True):
    """
    Preprocess real SPHEREx data to handle outliers, missing data, and bad values.
    
    This function identifies and handles several types of problematic data:
    1. Missing data (uncertainties set to ~50000)
    2. Absurdly high flux values (>10^4)
    3. Extreme outliers (>100x the per-band mean)
    4. NaN values in flux or uncertainties
    5. Inf values in flux or uncertainties
    6. Non-positive uncertainties
    
    Parameters
    ----------
    flux : np.ndarray
        Flux array with shape (n_sources, n_bands)
    flux_unc : np.ndarray
        Flux uncertainty array with shape (n_sources, n_bands)
    missing_data_threshold : float
        Threshold for missing data flag (default 40000, catches uncertainties set to 50000)
    high_flux_threshold : float
        Threshold for absurdly high flux values (default 1e4)
    extreme_outlier_factor : float
        Factor for extreme outliers relative to per-band mean (default 100)
    downweight_factor : float
        Downweighting factor for flagged data (default 0.01 = 1%)
    verbose : bool
        Print diagnostic information
        
    Returns
    -------
    flux_clean : np.ndarray
        Cleaned flux array (bad values set to 0)
    flux_unc_clean : np.ndarray
        Cleaned uncertainty array (bad values set to 1.0)
    weights : np.ndarray
        Data weights (0 = excluded, 0.01 = downweighted, 1 = full weight)
    """
    flux_clean = flux.copy()
    flux_unc_clean = flux_unc.copy()
    weights = np.ones_like(flux)
    
    n_total = flux.size
    
    if verbose:
        print(f"\n{'='*60}")
        print("PREPROCESSING REAL DATA FOR OUTLIERS")
        print(f"{'='*60}")
    
    # 1. Zero-weight missing data (large uncertainties ~50000)
    mask_missing = flux_unc_clean > missing_data_threshold
    n_missing = np.sum(mask_missing)
    weights[mask_missing] = 0.0
    flux_clean[mask_missing] = 0.0
    if verbose:
        print(f"1. Missing data (err > {missing_data_threshold}): {n_missing} ({100*n_missing/n_total:.2f}%)")
    
    # 2. Zero-weight absurdly high fluxes
    mask_high_flux = flux_clean > high_flux_threshold
    n_high = np.sum(mask_high_flux)
    weights[mask_high_flux] = 0.0
    if verbose:
        print(f"2. High flux (> {high_flux_threshold:.0e}): {n_high} ({100*n_high/n_total:.2f}%) - downweighted")
    
    # 2b. Zero-weight negative SNR outliers

    snr = flux_clean / (flux_unc_clean + 1e-10)
    mask_negsnr = np.logical_or(snr < -5.0, flux_clean < low_flux_threshold)
    weights[mask_negsnr] = 0.0
    if verbose:
        print(f"3. Extreme negative SNR (< -5): {np.sum(mask_negsnr)} ({100*np.sum(mask_negsnr)/n_total:.2f}%) - zero-weighted")

    # 3. Zero-weight extreme outliers (relative to per-band mean)
    flux_mean_per_band = np.mean(flux, axis=0)
    flux_ratio = flux / (flux_mean_per_band[None, :] + 1e-10)
    mask_extreme = flux_ratio > extreme_outlier_factor
    n_extreme = np.sum(mask_extreme)
    weights[mask_extreme] = 0.0
    if verbose:
        print(f"3. Extreme outliers (>{extreme_outlier_factor}x band mean): {n_extreme} ({100*n_extreme/n_total:.2f}%) - downweighted")
    
    # 4. Zero-weight NaN values
    mask_nan = np.isnan(flux_clean) | np.isnan(flux_unc_clean)
    n_nan = np.sum(mask_nan)
    weights[mask_nan] = 0.0
    flux_clean[mask_nan] = 0.0
    flux_unc_clean[mask_nan] = 1.0  # dummy value to avoid division by zero
    if verbose:
        print(f"4. NaN values: {n_nan} ({100*n_nan/n_total:.2f}%)")
    
    # 5. Zero-weight Inf values
    mask_inf = np.isinf(flux_clean) | np.isinf(flux_unc_clean)
    n_inf = np.sum(mask_inf)
    weights[mask_inf] = 0.0
    flux_clean[mask_inf] = 0.0
    flux_unc_clean[mask_inf] = 1.0
    if verbose:
        print(f"5. Inf values: {n_inf} ({100*n_inf/n_total:.2f}%)")
    
    # 6. Zero-weight bad uncertainties (≤ 0)
    mask_bad_err = flux_unc_clean <= 0
    n_bad_err = np.sum(mask_bad_err)
    weights[mask_bad_err] = 0.0
    flux_clean[mask_bad_err] = 0.0
    flux_unc_clean[mask_bad_err] = 1.0
    if verbose:
        print(f"6. Bad uncertainties (≤ 0): {n_bad_err} ({100*n_bad_err/n_total:.2f}%)")
    
    # Summary
    n_zero = np.sum(weights == 0)
    n_downweighted = np.sum((weights > 0) & (weights < 1))
    n_full = np.sum(weights == 1)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Zero-weighted: {n_zero} ({100*n_zero/n_total:.2f}%)")
        print(f"  Downweighted: {n_downweighted} ({100*n_downweighted/n_total:.2f}%)")
        print(f"  Full weight: {n_full} ({100*n_full/n_total:.2f}%)")
        print(f"{'='*60}\n")
    
    return flux_clean, flux_unc_clean, weights


@dataclass
class SPHERExData:
    all_spec_obs: np.ndarray        # Observed spectra (e.g. N_sources × N_channels)
    all_noiseless_spec: np.ndarray  # Corresponding noiseless rest-frame spectra
    all_flux_unc: np.ndarray        # Photometric uncertainties
    weights: np.ndarray             # Sample weights or inverse variances
    src_idxs: np.ndarray            # Index mapping between catalogs
    redshift_rf: np.ndarray         # Redshifts for rest-frame catalog
    srcid_rf: np.ndarray            # Source IDs (rest-frame)
    redshift: np.ndarray            # Redshifts for observed catalog
    srcid_obs: np.ndarray           # Source IDs (observed)
    phot_snr: np.ndarray            # Per-object photometric SNR
    norms: np.ndarray               # Normalizations applied to spectra
    srcids_noiseless: np.ndarray    # source IDs for noiseless ground truth spectra
    log_amplitude: np.ndarray       # log10 of weighted mean flux (for amplitude-dependent priors)
    
    # Native filter support (optional, for variable-length per-source filters)
    filter_curves: Optional[np.ndarray] = None  # (N_sources, max_nbands, n_lam) per-source filters
    central_wavelengths_per_source: Optional[np.ndarray] = None  # (N_sources, max_nbands) per-measurement wavelengths
    measurement_weights: Optional[np.ndarray] = None  # (N_sources, max_nbands) 1=valid, 0=padding
    n_valid_measurements: Optional[np.ndarray] = None  # (N_sources,) number of valid measurements per source

    @classmethod
    def from_prep(cls, dat_obs: Any, df_rf: Any, df_obs: Any, **kwargs):
        """Create SPHERExData from `prep_obs_dat` output."""
        outputs = prep_obs_dat(dat_obs, df_rf, df_obs, **kwargs)
        return cls(*outputs)

def _compute_snr_quad(row):
    """Compute sqrt(sum((F/sigma)^2)) for a single source row from the parquet DataFrame.

    Only bands with flux > 0 AND 0 < flux_err < 5e4 (sentinel for missing) are included.
    Returns np.nan if no valid bands exist or flux columns are absent.
    """
    try:
        f = np.array(row.get('flux_dered_fiducial', []), dtype=np.float64)
        s = np.array(row.get('flux_err_dered_fiducial', []), dtype=np.float64)
        ok = (f > 0) & (s > 0) & (s < 5e4)
        return float(np.sqrt(np.sum((f[ok] / s[ok]) ** 2))) if np.any(ok) else np.nan
    except Exception:
        return np.nan


def load_real_spherex_parquet(parquet_file=None, filter_set_name=None, wave_obs=None, 
                              weight_soft=5e-4, abs_norm=True, max_normflux=100, df=None,
                              preprocess_outliers=True, use_weighted_mean=False,
                              channel_mask=None):
    """
    Load real SPHEREx data from parquet file and convert to spec_data_jax format.
    
    Parameters
    ----------
    parquet_file : str, optional
        Path to parquet file with columns: SPHERExRefID, ra, dec, lambda, 
        flux_dered_fiducial, flux_err_dered_fiducial, z_specz, etc.
        Not needed if df is provided.
    filter_set_name : str
        Name of filter set (e.g., 'SPHEREx_filter_306')
    wave_obs : array
        Central wavelengths of the filters
    weight_soft : float
        Soft weighting parameter
    abs_norm : bool
        Whether to use absolute normalization
    max_normflux : float
        Maximum normalized flux value
    df : DataFrame, optional
        Pre-loaded dataframe. If provided, parquet_file is ignored.
    preprocess_outliers : bool
        Whether to apply outlier preprocessing (missing data, high fluxes, NaNs, etc.)
        Default True. Preprocessing zero-weights missing data and downweights extreme values.
    channel_mask : np.ndarray or None
        Boolean array of length nbands. Where True, the channel is excluded from
        the likelihood by setting flux_unc to 1e10 (making its inverse-variance
        weight effectively zero). Applied after outlier preprocessing.
        
    Returns
    -------
    dat_obs : spec_data_jax
        Observed data object with loaded photometry
    property_cat_df : DataFrame
        Property catalog with source IDs, redshifts, etc.
    """
    # Load parquet file or use provided dataframe
    if df is None:
        if parquet_file is None:
            raise ValueError("Must provide either parquet_file or df")
        df = pd.read_parquet(parquet_file)
    
    # Get unique sources
    unique_sources = df['SPHERExRefID'].unique()
    nsrc = len(unique_sources)
    nbands = len(wave_obs)
    
    print(f"Loading {nsrc} sources with {nbands} bands")
    print(f"wave_obs shape: {wave_obs.shape}, type: {type(wave_obs)}")
    
    # Initialize arrays
    flux = np.zeros((nsrc, nbands))
    flux_unc = np.zeros((nsrc, nbands))
    
    # Load data: Each source has ONE row containing arrays of all flux values
    # NOTE: fiducial fluxes (flux_dered_fiducial) are on a FIXED wavelength grid
    # specified by the central wavelength file and should already be in the correct order.
    
    for i, src_id in enumerate(unique_sources):
        src_row = df[df['SPHERExRefID'] == src_id].iloc[0]
        
        # Extract flux arrays - stored as arrays within the dataframe
        src_flux = np.array(src_row['flux_dered_fiducial'])
        src_flux_err = np.array(src_row['flux_err_dered_fiducial'])
        
        # NOTE: Flags are NOT applied to fiducial fluxes because:
        # - Flags correspond to native band measurements (~146 bands, varies by source)
        # - Fiducial fluxes are interpolated to a fixed grid (306 bands)
        # - Flag filtering happens at the native measurement level before interpolation
        # - The fiducial flux uncertainties already reflect measurement quality
        
        # Debug first source
        if i == 0:
            print(f"First source ({src_id}):")
            print(f"  src_flux type: {type(src_flux)}, shape: {src_flux.shape}, dtype: {src_flux.dtype}")
            print(f"  src_flux_err type: {type(src_flux_err)}, shape: {src_flux_err.shape}, dtype: {src_flux_err.dtype}")
            print(f"  Number of bands in data: {len(src_flux)}")
            print(f"  Expected bands from filter file: {nbands}")
            if len(src_flux) >= 5:
                print(f"  First 5 flux values: {src_flux[:5]}")
                print(f"  First 5 flux_err values: {src_flux_err[:5]}")
        
        # Check if the number of bands matches
        if len(src_flux) != nbands:
            if i == 0:
                print(f"  WARNING: Source {src_id} has {len(src_flux)} bands, expected {nbands}")
                print(f"           Padding or truncating as needed.")
            # Assign what we have, leave rest as zeros
            n_to_copy = min(len(src_flux), nbands)
            flux[i, :n_to_copy] = src_flux[:n_to_copy]
            flux_unc[i, :n_to_copy] = src_flux_err[:n_to_copy]
        else:
            # Direct assignment - fiducial fluxes are already on the correct wavelength grid
            flux[i, :] = src_flux
            flux_unc[i, :] = src_flux_err
        
        # Progress indicator for large datasets
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{nsrc} sources...")
    
    # Create property catalog DataFrame
    # Get one row per source for metadata
    property_cat_data = []
    for src_id in unique_sources:
        src_rows = df[df['SPHERExRefID'] == src_id]
        first_row = src_rows.iloc[0]
        
        # Priority: spectroscopic redshift (z_specz) if available, else photometric (z_best_gals)
        redshift_val = first_row.get('z_specz', first_row.get('z_best_gals', 0.0))
        
        property_cat_data.append({
            'SPHERExRefID': src_id,
            'Tractor_ID': src_id,  # Use SPHERExRefID as Tractor_ID for compatibility
            'ra': first_row['ra'],
            'dec': first_row['dec'],
            'redshift': redshift_val,  # Fixed: was 'redshift_tf', should be 'redshift'
            'z_specz': first_row.get('z_specz', np.nan),  # Store spec-z separately if available
            'z_phot': first_row.get('z_best_gals', np.nan),  # Store photo-z separately if available
            'z_err': first_row.get('z_err_std_gals', 0.0),
            'minchi2_gals': first_row.get('minchi2_minchi2_gals', 0.0),
            'Nsamples': first_row.get('Nsamples', 0),
            'frac_sampled_102': first_row.get('frac_sampled_102', 1.0),
            'snr_quad': _compute_snr_quad(first_row),
        })
    
    property_cat_df = pd.DataFrame(property_cat_data)
    
    # Print redshift statistics
    n_specz = property_cat_df['z_specz'].notna().sum()
    n_phot = property_cat_df['z_phot'].notna().sum()
    print(f"\nRedshift statistics:")
    print(f"  Sources with spec-z: {n_specz} ({100*n_specz/nsrc:.1f}%)")
    print(f"  Sources with photo-z: {n_phot} ({100*n_phot/nsrc:.1f}%)")
    print(f"  Mean redshift: {property_cat_df['redshift'].mean():.3f}")
    print(f"  Redshift range: {property_cat_df['redshift'].min():.3f} - {property_cat_df['redshift'].max():.3f}")
    
    # Apply outlier preprocessing if requested
    if preprocess_outliers:
        flux, flux_unc, data_weights = preprocess_real_data_outliers(
            flux, flux_unc, verbose=True
        )
        
        # CRITICAL FIX: Encode zero-weighted data into flux_unc so downstream code respects it
        # Where data_weights == 0, set flux_unc to a very large value so 1/flux_unc^2 → 0
        # This ensures parse_input_sphx_phot respects our preprocessing weights
        flux_unc = np.where(data_weights == 0, 1e10, flux_unc)
        
        # Store preprocessing weights for later use
        property_cat_df['data_quality_weight'] = np.sum(data_weights > 0, axis=1) / nbands

    # Apply explicit channel mask (e.g. exclude all of SPHEREx band 4)
    # Set flux_unc to 1e10 for masked channels so their inverse-variance weight → 0.
    if channel_mask is not None:
        channel_mask = np.asarray(channel_mask, dtype=bool)
        if channel_mask.shape != (nbands,):
            raise ValueError(
                f"channel_mask has shape {channel_mask.shape}, expected ({nbands},)"
            )
        n_masked = channel_mask.sum()
        if n_masked > 0:
            flux_unc[:, channel_mask] = 1e10
            print(f"Channel mask applied: {n_masked}/{nbands} channels zero-weighted")

    # Create spec_data_jax object
    dat_obs = spec_data_jax(nbands)
    dat_obs.flux = flux
    dat_obs.flux_unc = flux_unc
    dat_obs.sed_um_wave = wave_obs
    dat_obs.catgrid_flux_noiseless = None  # No ground truth for real data
    dat_obs.srcids_noiseless = None
    
    # Parse photometry (normalize and compute weights)
    dat_obs.phot_dict = parse_input_sphx_phot(
        flux, flux_unc, 
        weight_soft=weight_soft, 
        abs_norm=abs_norm, 
        max_normflux=max_normflux,
        use_weighted_mean=use_weighted_mean
    )
    
    dat_obs.phot_proc = dat_obs.phot_dict['phot_fluxes']
    property_cat_df['phot_snr'] = dat_obs.phot_dict['phot_snr']
    
    # Set up for dataloaders (no train/val split for real data)
    dat_obs.data_train = dat_obs.phot_proc
    dat_obs.data_valid = None
    
    return dat_obs, property_cat_df


def load_and_prepare_pseudo_native_data(parquet_file, filter_dir, filter_set_name='SPHEREx_filter_306',
                                        max_sources=None, max_nbands=None,
                                        weight_soft=5e-4, abs_norm=True, max_normflux=100,
                                        preprocess_outliers=True):
    """
    Load real SPHEREx data with pseudo-native (wavelength-based) filters and prepare for inference.
    
    This is the main entry point for loading data with per-source filter variations when 
    pixel positions (xpix, ypix) are not available. It:
    1. Loads wavelength-based pseudo-native filters (interpolated from fiducial filters)
    2. Applies standard normalization (same as load_real_spherex_parquet)
    3. Combines all weight masks (padding, outliers, inverse-variance)
    4. Returns SPHERExData with native filter support
    
    Parameters
    ----------
    parquet_file : str
        Path to parquet file with columns: SPHERExRefID, lambda (array), 
        flux_dered (array), flux_err_dered (array), z_specz, etc.
    filter_dir : str
        Base directory containing filter files (e.g., '/path/to/filters')
    filter_set_name : str
        Filter subdirectory name (e.g., 'SPHEREx_filter_306')
    max_sources : int, optional
        Maximum number of sources to load. If None, loads all.
    max_nbands : int, optional
        Maximum number of bands to pad to. If None, uses max in dataset.
    weight_soft : float
        Soft weighting parameter for inverse-variance weighting
    abs_norm : bool
        Whether to use absolute normalization (divide by mean flux)
    max_normflux : float
        Maximum normalized flux value (clips extremes)
    preprocess_outliers : bool
        Whether to apply outlier preprocessing (missing data, high fluxes, NaNs, etc.)
        
    Returns
    -------
    dat_obs : spec_data_jax
        Observed data object with normalized photometry
    property_cat_df : DataFrame
        Property catalog with source IDs, redshifts, etc.
    sphx_data : SPHERExData
        Complete data structure including native filter support
    lam_filter : np.ndarray
        Fine wavelength grid for filters (n_lam,)
    fiducial_cenwav : np.ndarray
        Central wavelengths of fiducial filters (n_filters,)
    """
    from data.spherex_native_filters import load_pseudo_native_from_parquet
    
    print(f"\n{'='*70}")
    print("LOADING PSEUDO-NATIVE FILTER DATA")
    print(f"{'='*70}")
    
    # Load pseudo-native filter data
    batch_dict, lam_filter, fiducial_cenwav = load_pseudo_native_from_parquet(
        parquet_file=parquet_file,
        filter_dir=filter_dir,
        filter_set_name=filter_set_name,
        max_sources=max_sources,
        max_nbands=max_nbands
    )
    
    # Extract arrays
    flux = batch_dict['flux']
    flux_err = batch_dict['flux_err']
    filter_curves = batch_dict['filter_curves']
    central_wavelengths_per_source = batch_dict['central_wavelengths']
    measurement_weights = batch_dict['weights']  # 1=valid, 0=padding
    n_valid_measurements = batch_dict['n_valid']
    
    nsrc, nbands = flux.shape
    n_lam = filter_curves.shape[2]
    
    print(f"\nLoaded data shapes:")
    print(f"  flux: {flux.shape}")
    print(f"  flux_err: {flux_err.shape}")
    print(f"  filter_curves: {filter_curves.shape} ({filter_curves.nbytes/1e6:.1f} MB)")
    print(f"  measurement_weights: {measurement_weights.shape}")
    print(f"  lam_filter: {lam_filter.shape}")
    
    # Apply outlier preprocessing if requested
    if preprocess_outliers:
        print(f"\n{'='*70}")
        print("APPLYING OUTLIER PREPROCESSING")
        print(f"{'='*70}")
        flux, flux_err, data_quality_weights = preprocess_real_data_outliers(
            flux, flux_err, verbose=True
        )
        # Combine with measurement weights (padding mask)
        # If measurement is padding (weight=0) OR bad data (weight=0), final weight is 0
        combined_data_weights = measurement_weights * data_quality_weights
    else:
        combined_data_weights = measurement_weights.copy()
        data_quality_weights = np.ones_like(flux)
    
    # Build property catalog from parquet metadata
    import pandas as pd
    df = pd.read_parquet(parquet_file)
    if max_sources is not None:
        unique_ids = df['SPHERExRefID'].unique()[:max_sources]
        df = df[df['SPHERExRefID'].isin(unique_ids)]
    
    unique_sources = df['SPHERExRefID'].unique()
    property_cat_data = []
    
    for src_id in unique_sources:
        src_rows = df[df['SPHERExRefID'] == src_id]
        first_row = src_rows.iloc[0]
        
        # Priority: spectroscopic redshift (z_specz) if available, else photometric
        redshift_val = first_row.get('z_specz', first_row.get('z_best_gals', 0.0))
        
        property_cat_data.append({
            'SPHERExRefID': src_id,
            'Tractor_ID': src_id,  # Use SPHERExRefID as Tractor_ID for compatibility
            'ra': first_row.get('ra', 0.0),
            'dec': first_row.get('dec', 0.0),
            'redshift': redshift_val,
            'z_specz': first_row.get('z_specz', np.nan),
            'z_phot': first_row.get('z_best_gals', np.nan),
            'z_err': first_row.get('z_err_std_gals', 0.0),
            'minchi2_gals': first_row.get('minchi2_minchi2_gals', 0.0),
            'Nsamples': first_row.get('Nsamples', 0),
            'frac_sampled_102': first_row.get('frac_sampled_102', 1.0),
            'n_valid_bands': n_valid_measurements[property_cat_data.__len__()]
        })
    
    property_cat_df = pd.DataFrame(property_cat_data)
    
    # Print redshift statistics
    n_specz = property_cat_df['z_specz'].notna().sum()
    n_phot = property_cat_df['z_phot'].notna().sum()
    print(f"\n{'='*70}")
    print("REDSHIFT STATISTICS")
    print(f"{'='*70}")
    print(f"  Sources with spec-z: {n_specz} ({100*n_specz/nsrc:.1f}%)")
    print(f"  Sources with photo-z: {n_phot} ({100*n_phot/nsrc:.1f}%)")
    print(f"  Mean redshift: {property_cat_df['redshift'].mean():.3f}")
    print(f"  Redshift range: [{property_cat_df['redshift'].min():.3f}, {property_cat_df['redshift'].max():.3f}]")
    
    # Create spec_data_jax object (standard format)
    dat_obs = spec_data_jax(nbands)
    dat_obs.flux = flux
    dat_obs.flux_unc = flux_err
    dat_obs.sed_um_wave = fiducial_cenwav  # Use fiducial central wavelengths as reference
    dat_obs.catgrid_flux_noiseless = None
    dat_obs.srcids_noiseless = None
    
    # Apply standard normalization and compute inverse-variance weights
    print(f"\n{'='*70}")
    print("APPLYING NORMALIZATION")
    print(f"{'='*70}")
    dat_obs.phot_dict = parse_input_sphx_phot(
        flux, flux_err,
        weight_soft=weight_soft,
        abs_norm=abs_norm,
        max_normflux=max_normflux
    )
    
    # Get normalized data and SNR
    phot_fluxes = dat_obs.phot_dict['phot_fluxes']
    phot_weights = dat_obs.phot_dict['phot_weights']  # Inverse-variance weights
    phot_norms = dat_obs.phot_dict['phot_norms']
    phot_snr = dat_obs.phot_dict['phot_snr']
    
    property_cat_df['phot_snr'] = phot_snr
    property_cat_df['phot_norm'] = phot_norms.flatten()
    
    # Combine all weight masks:
    # final_weights = measurement_weights (padding) * data_quality_weights (outliers) * phot_weights (inverse-variance)
    final_weights = combined_data_weights * phot_weights
    
    print(f"\n{'='*70}")
    print("WEIGHT COMBINATION")
    print(f"{'='*70}")
    print(f"  measurement_weights (padding): {measurement_weights.shape}, "
          f"nonzero: {(measurement_weights > 0).sum()}/{measurement_weights.size} "
          f"({100*(measurement_weights > 0).sum()/measurement_weights.size:.1f}%)")
    print(f"  data_quality_weights (outliers): {data_quality_weights.shape}, "
          f"nonzero: {(data_quality_weights > 0).sum()}/{data_quality_weights.size} "
          f"({100*(data_quality_weights > 0).sum()/data_quality_weights.size:.1f}%)")
    print(f"  phot_weights (inverse-variance): {phot_weights.shape}, "
          f"nonzero: {(phot_weights > 0).sum()}/{phot_weights.size} "
          f"({100*(phot_weights > 0).sum()/phot_weights.size:.1f}%)")
    print(f"  final_weights (combined): {final_weights.shape}, "
          f"nonzero: {(final_weights > 0).sum()}/{final_weights.size} "
          f"({100*(final_weights > 0).sum()/final_weights.size:.1f}%)")
    
    per_source_valid_fraction = np.sum(final_weights > 0, axis=1) / nbands
    print(f"\n  Per-source valid data fraction:")
    print(f"    Min: {per_source_valid_fraction.min():.1%}")
    print(f"    Median: {np.median(per_source_valid_fraction):.1%}")
    print(f"    Max: {per_source_valid_fraction.max():.1%}")
    
    property_cat_df['valid_band_fraction'] = per_source_valid_fraction
    
    # Set up for dataloaders
    dat_obs.phot_proc = phot_fluxes
    dat_obs.data_train = phot_fluxes
    dat_obs.data_valid = None
    
    # Create SPHERExData structure with native filter support
    sphx_data = SPHERExData(
        all_spec_obs=phot_fluxes,
        all_noiseless_spec=None,
        all_flux_unc=dat_obs.phot_dict['phot_flux_unc'],
        weights=final_weights,  # Combined weights
        src_idxs=np.arange(nsrc),
        redshift_rf=None,
        srcid_rf=None,
        redshift=np.array(property_cat_df['redshift']),
        srcid_obs=np.array(property_cat_df['Tractor_ID']),
        phot_snr=phot_snr,
        norms=phot_norms,
        srcids_noiseless=None,
        # Native filter fields
        filter_curves=filter_curves,
        central_wavelengths_per_source=central_wavelengths_per_source,
        measurement_weights=measurement_weights,
        n_valid_measurements=n_valid_measurements
    )
    
    print(f"\n{'='*70}")
    print("DATA LOADING COMPLETE")
    print(f"{'='*70}\n")
    
    return dat_obs, property_cat_df, sphx_data, lam_filter, fiducial_cenwav


def load_real_spherex_parquet_native(parquet_file=None, filter_file_path=None, lam_fine=None,
                                     weight_soft=5e-4, abs_norm=True, max_normflux=100, df=None,
                                     preprocess_outliers=True, max_nbands=None, Dy=4, Dx=128):
    """
    Load real SPHEREx data with native (per-source) filter profiles.
    
    This function loads per-measurement pixel positions (xpix, ypix, det_id) and queries
    the reduced filter set to build per-source filter matrices. Sources are padded to
    max_nbands with proper masking.
    
    Parameters
    ----------
    parquet_file : str, optional
        Path to parquet file with columns: SPHERExRefID, ra, dec, lambda, x_image, y_image, 
        det_id, flux (native), flux_err (native), z_specz, etc.
        Not needed if df is provided.
    filter_file_path : str
        Path to reduced filter FITS file (from linefit module)
    lam_fine : array
        Fine wavelength grid for filter standardization
    weight_soft : float
        Soft weighting parameter
    abs_norm : bool
        Whether to use absolute normalization
    max_normflux : float
        Maximum normalized flux value
    df : DataFrame, optional
        Pre-loaded dataframe. If provided, parquet_file is ignored.
    preprocess_outliers : bool
        Whether to apply outlier preprocessing
    max_nbands : int, optional
        Maximum number of bands to pad to. If None, uses max in batch.
    Dy, Dx : int
        Pixel block sizes for reduced filter set (default: 4, 128)
        
    Returns
    -------
    dat_obs : spec_data_jax
        Observed data object with native filter support
    property_cat_df : DataFrame
        Property catalog with source IDs, redshifts, etc.
    """
    from data.spherex_native_filters import ReducedFilterSet, pad_and_mask_batch
    
    # Load parquet file or use provided dataframe
    if df is None:
        if parquet_file is None:
            raise ValueError("Must provide either parquet_file or df")
        df = pd.read_parquet(parquet_file)
    
    # Check required columns for native processing
    required_cols = ['SPHERExRefID', 'x_image', 'y_image', 'det_id', 'lambda']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Native filter loading requires column '{col}' in parquet file")
    
    # Initialize reduced filter set
    print(f"Loading reduced filter set from {filter_file_path}")
    filter_set = ReducedFilterSet(filter_file_path, Dy=Dy, Dx=Dx, standardize=True)
    
    if lam_fine is None:
        lam_fine = filter_set.lam_fine
    
    # Get unique sources
    unique_sources = df['SPHERExRefID'].unique()
    nsrc = len(unique_sources)
    
    print(f"Loading {nsrc} sources with native (variable-length) measurements")
    print(f"Fine wavelength grid: {len(lam_fine)} points from {lam_fine[0]:.3f} to {lam_fine[-1]:.3f} μm")
    
    # Collect per-source data
    flux_list = []
    flux_err_list = []
    filter_matrix_list = []
    cenwav_list = []
    n_meas_list = []
    
    for i, src_id in enumerate(unique_sources):
        src_rows = df[df['SPHERExRefID'] == src_id]
        
        # Extract per-measurement data
        det_ids = np.array(src_rows['det_id'], dtype=np.int32)
        xpix = np.array(src_rows['x_image'], dtype=np.float32)
        ypix = np.array(src_rows['y_image'], dtype=np.float32)
        flux_native = np.array(src_rows['flux'])  # Native flux (not fiducial)
        flux_err_native = np.array(src_rows['flux_err'])
        lambdas = np.array(src_rows['lambda'])
        
        n_meas = len(flux_native)
        n_meas_list.append(n_meas)
        
        # Build filter matrix for this source
        _, filter_matrix = filter_set.build_source_filter_matrix(
            det_ids, xpix, ypix, lam_interp=lam_fine
        )
        
        flux_list.append(flux_native)
        flux_err_list.append(flux_err_native)
        filter_matrix_list.append(filter_matrix)
        cenwav_list.append(lambdas)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{nsrc} sources...")
    
    print(f"\nMeasurement statistics:")
    print(f"  Min measurements per source: {min(n_meas_list)}")
    print(f"  Max measurements per source: {max(n_meas_list)}")
    print(f"  Mean measurements per source: {np.mean(n_meas_list):.1f}")
    
    # Pad and mask batch
    if max_nbands is None:
        max_nbands = max(n_meas_list)
    
    print(f"\nPadding to max_nbands = {max_nbands}")
    batch_dict = pad_and_mask_batch(
        flux_list, flux_err_list, filter_matrix_list, cenwav_list, max_nbands=max_nbands
    )
    
    flux = batch_dict['flux']
    flux_unc = batch_dict['flux_err']
    filter_curves = batch_dict['filter_curves']
    central_wavelengths_per_source = batch_dict['central_wavelengths']
    measurement_weights = batch_dict['weights']
    n_valid_measurements = batch_dict['n_valid']
    
    # Create property catalog DataFrame
    property_cat_data = []
    for src_id in unique_sources:
        src_rows = df[df['SPHERExRefID'] == src_id]
        first_row = src_rows.iloc[0]
        
        # Priority: spectroscopic redshift if available, else photometric
        redshift_val = first_row.get('z_specz', first_row.get('z_best_gals', 0.0))
        
        property_cat_data.append({
            'SPHERExRefID': src_id,
            'Tractor_ID': src_id,
            'ra': first_row['ra'],
            'dec': first_row['dec'],
            'redshift': redshift_val,
            'z_specz': first_row.get('z_specz', np.nan),
            'z_phot': first_row.get('z_best_gals', np.nan),
            'z_err': first_row.get('z_err_std_gals', 0.0),
        })
    
    property_cat_df = pd.DataFrame(property_cat_data)
    
    # Print redshift statistics
    n_specz = property_cat_df['z_specz'].notna().sum()
    n_phot = property_cat_df['z_phot'].notna().sum()
    print(f"\nRedshift statistics:")
    print(f"  Sources with spec-z: {n_specz} ({100*n_specz/nsrc:.1f}%)")
    print(f"  Sources with photo-z: {n_phot} ({100*n_phot/nsrc:.1f}%)")
    
    # Apply outlier preprocessing if requested (with measurement weights)
    if preprocess_outliers:
        flux, flux_unc, data_weights = preprocess_real_data_outliers(
            flux, flux_unc, verbose=True
        )
        # Combine with measurement weights (valid/padding mask)
        measurement_weights = measurement_weights * data_weights
    
    # Create spec_data_jax object with native filter support
    dat_obs = spec_data_jax(max_nbands)
    dat_obs.flux = flux
    dat_obs.flux_unc = flux_unc
    dat_obs.sed_um_wave = lam_fine  # Use fine wavelength grid
    dat_obs.catgrid_flux_noiseless = None
    dat_obs.srcids_noiseless = None
    
    # Store native filter information
    dat_obs.filter_curves_native = filter_curves
    dat_obs.central_wavelengths_per_source = central_wavelengths_per_source
    dat_obs.measurement_weights = measurement_weights
    dat_obs.n_valid_measurements = n_valid_measurements
    
    # Parse photometry with proper weighting
    dat_obs.phot_dict = parse_input_sphx_phot(
        flux, flux_unc,
        weight_soft=weight_soft,
        abs_norm=abs_norm,
        max_normflux=max_normflux
    )
    
    # Apply measurement weights (valid/padding mask) to computed photometric weights
    if 'phot_weights' in dat_obs.phot_dict:
        dat_obs.phot_dict['phot_weights'] = dat_obs.phot_dict['phot_weights'] * measurement_weights
    else:
        dat_obs.phot_dict['phot_weights'] = measurement_weights
    
    dat_obs.phot_proc = dat_obs.phot_dict['phot_fluxes']
    property_cat_df['phot_snr'] = dat_obs.phot_dict['phot_snr']
    
    # Set up for dataloaders
    dat_obs.data_train = dat_obs.phot_proc
    dat_obs.data_valid = None
    
    return dat_obs, property_cat_df


def gen_subset_flux_dataclasses(spherex_dat, n_retain_sources=10, miss_frac_list=[0.0, 0.25, 0.5, 0.75]):

    spherex_dat = SPHERExData.from_prep(
        dat_obs,
        property_cat_df_restframe,
        property_cat_df_obs,
        phot_snr_min=100,
        phot_snr_max=1000,
        zmin=0.0,
        zmax=3.0
    )

    all_datclass = []

    for miss_frac in miss_frac_list:

        sphx_datclass = subset_with_missing(spherex_dat,
                                            retain_indices = spherex_dat.src_idxs[:n_retain_sources],
                                            missing_fraction=miss_frac)

        all_datclass.append(sphx_datclass)

    return all_datclass

def subset_with_missing(
    data: SPHERExData,
    retain_indices: Optional[Sequence[int]] = None,
    n_retain_sources: Optional[int] = None,
    missing_fraction: float = 1.0,
    nan_miss_flux: bool = False,
    rng: Optional[np.random.Generator] = None
) -> SPHERExData:
    """
    Create a smaller SPHERExData object with only a subset of sources and missing flux measurements.

    Parameters
    ----------
    data : SPHERExData
        Original dataset.
    retain_indices : sequence of int, optional
        Indices of sources to retain. If None, will randomly select `n_retain_sources`.
    n_retain_sources : int, optional
        Number of sources to retain if `retain_indices` is None.
    missing_fraction : float
        Fraction of flux measurements (per source) to drop.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    SPHERExData
        New dataset containing only the retained sources with missing measurements applied.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_sources, n_channels = data.all_spec_obs.shape

    # Determine which sources to keep
    if retain_indices is None:
        if n_retain_sources is None:
            raise ValueError("Must provide either retain_indices or n_retain_sources.")
        retain_indices = rng.choice(n_sources, size=n_retain_sources, replace=False)

    retain_indices = np.array(retain_indices, dtype=int)

    # Slice all arrays in the dataclass to keep only retained sources
    def subset_array(arr):
        return None if arr is None else np.copy(arr[retain_indices])

    flux_obs = subset_array(data.all_spec_obs)
    flux_noiseless = subset_array(data.all_noiseless_spec)
    flux_unc = subset_array(data.all_flux_unc)
    weights = subset_array(data.weights)
    src_idxs = subset_array(data.src_idxs)
    redshift = subset_array(data.redshift)
    srcid_obs = subset_array(data.srcid_obs)
    phot_snr = subset_array(data.phot_snr)
    norms = subset_array(data.norms)

    # Apply missing data mask for retained sources
    for i in range(len(retain_indices)):
        drop_mask = rng.random(n_channels) < missing_fraction
        weights[i, drop_mask] = 0.0
        flux_unc[i, drop_mask] = np.inf
        # Optional: mark flux values as NaN for clarity

        if nan_miss_flux:
            flux_obs[i, drop_mask] = np.nan

    # Create a new dataclass with only the retained subset
    return SPHERExData(
        all_spec_obs=flux_obs,
        all_noiseless_spec=flux_noiseless,
        all_flux_unc=flux_unc,
        weights=weights,
        src_idxs=src_idxs,
        redshift_rf=None,
        srcid_rf=None,
        redshift=redshift,
        srcid_obs=srcid_obs,
        phot_snr=phot_snr,
        norms=norms,
        srcids_noiseless=None
    )


def grab_idxs(fpath_str):
    splitfpath = fpath_str.split('_')
    i0, i1 = int(splitfpath[-3]), int(splitfpath[-2])
    return i0, i1

def make_bfit_tid_dict(list_file):

    ''' Make best fit template ID dictionary object from Bomee's file '''
    
    temps = ascii.read(list_file, format='no_header')
    bfit_tid_dict = dict({})
    
    for _, line in enumerate(temps):
        tempstr = line.__getitem__('col1').replace('.dat', '').replace('.sed', '')        
        bfit_tid_dict[l+1] = tempstr
        
    return bfit_tid_dict

def convert_filters_to_jax(central_wavelengths, bandpass_wavs, bandpass_vals, lam_interp=None, lam_min=0.7, lam_max=5.0, nlam=1000, \
                                 plot=False, zero_out_frac=1e-3, device=None):
    
    n_filter = len(central_wavelengths)
    
    if lam_interp is None:
        lam_interp = jnp.linspace(lam_min, lam_max, nlam)
    else:
        nlam = len(lam_interp)

    lambda_inv = 1.0 / lam_interp

    jax_filters = jnp.zeros((n_filter, nlam))
    
    for fidx in range(n_filter):
        
        bandpass_wav, bandpass_val = jnp.array(bandpass_wavs[fidx]), jnp.array(bandpass_vals[fidx])
        filter_interp = jnp.interp(lam_interp, bandpass_wav, bandpass_val)
        filter_interp = jnp.where((lam_interp > jnp.max(bandpass_wav)) | (lam_interp < jnp.min(bandpass_wav)), 0.0, filter_interp)
        filter_interp = jnp.where(jnp.abs(lam_interp - central_wavelengths[fidx]) > 0.1, 0.0, filter_interp)        
        filter_interp /= jnp.sum(filter_interp) # normalized to unity

        jax_filters = jax_filters.at[fidx, :].set(filter_interp)

        # convert to photon units

        # filter_interp_photon = filter_interp * lam_interp
        # filter_interp_photon /= jnp.sum(filter_interp_photon)

        # filter_interp_photon = filter_interp * lambda_inv
        # filter_interp_photon /= np.sum(filter_interp_photon)
        # jax_filters = jax_filters.at[fidx, :].set(filter_interp_photon)
        
        if fidx>n_filter-3 and plot:
            plot_bandpass_and_interp(np.array(lam_interp), np.array(filter_interp_photon), \
                                     np.array(bandpass_wav), np.array(bandpass_val), \
                                     central_wavelengths[fidx])

    return jax_filters, lam_interp

# sphx_data_proc.py
# def make_crossmatch_property_cat(features, catgrid, linecat_df, srcids=None, tid_idx_catgrid=0, which_set='COSMOS'):
# def calc_phot_mean_weights(fluxes, flux_unc=None, max_normflux=1000, weight_soft=5e-4, plot=False):
# def parse_input_sphx_phot(flux, flux_unc, max_normflux=1000, plot=False, weight_soft=5e-4, flux_noiseless=None):
    
''' Dataloader for training/validation datasets '''

# def grab_train_validation_dat(dat, train_frac, trainidx, valididx=None):

def grab_train_validation_dat_jax(dat, train_frac, trainidx, valididx=None):

    dat_train = jnp.array(dat[trainidx])
    dat_valid = jnp.array(dat[valididx]) if train_frac < 1. else None

    return dat_train, dat_valid
    

def draw_train_validation_idxs_jax(data, train_frac=0.8, key=None):
    """
    Splits dataset indices into training and validation sets using JAX.
    
    Parameters:
    - data (jnp.array or np.array): Dataset to split
    - train_frac (float): Fraction of data to use for training (default: 0.8)
    - key (jax.random.PRNGKey): JAX random key for reproducibility

    Returns:
    - trainidx (jnp.array): Indices for training data
    - valididx (jnp.array): Indices for validation data
    """
    num_samples = data.shape[0]

    if train_frac == 1.0:
        return jnp.arange(num_samples), jnp.array([])

    if key is None:
        key = jrandom.PRNGKey(0)  # Default key if not provided

    # Generate a random permutation of indices
    permutation = jrandom.permutation(key, num_samples)

    ntrain = int(num_samples * train_frac)
    trainidx, valididx = permutation[:ntrain], permutation[ntrain:]

    return trainidx, valididx

def update_dict(dictobj, val_train, val_valid, key):
    dictobj[key+'_train'] = val_train
    dictobj[key+'_valid'] = val_valid
    return dictobj

def prep_obs_dat(dat_obs,
                 property_cat_df_obs,
                 property_cat_df_restframe=None,
                 phot_snr_min=100, phot_snr_max=300,
                 zmin=0., zmax=5.0,
                 apply_subselect=False):

    ''' applies selection to observed data catalog based on SNR, redshift'''
    all_spec_obs = dat_obs.data_train
    all_noiseless_spec = dat_obs.catgrid_flux_noiseless
    all_flux_unc = dat_obs.phot_dict['phot_flux_unc']
    srcid_noiseless = dat_obs.srcids_noiseless

    weights = jnp.array(dat_obs.phot_dict['phot_weights'])
    # print('weights:', weights.shape)
    
    norms, phot_snr = [dat_obs.phot_dict[key] for key in ['phot_norms', 'phot_snr']]

    if property_cat_df_restframe is not None:
        redshift_rf, srcid_rf = [np.array(property_cat_df_restframe[key]) for key in ['redshift', 'Tractor_ID']]
    else:
        redshift_rf, srcid_rf = None, None
        
    redshift, srcid_obs = [np.array(property_cat_df_obs[key]) for key in ['redshift', 'Tractor_ID']]
    
    subselect_mask = np.ones_like(phot_snr)

    if phot_snr_min is not None:
        subselect_mask *= (phot_snr > phot_snr_min)
    if phot_snr_max is not None:
        subselect_mask *= (phot_snr < phot_snr_max)

    if zmin is not None:
        subselect_mask *= (redshift > zmin)
    if zmax is not None:
        subselect_mask *= (redshift < zmax)

    src_idxs = np.where(subselect_mask)[0]
    
    # Compute log10-amplitude for amplitude-dependent priors
    # log_amplitude = log10(weighted mean flux per source)
    flux = dat_obs.flux
    flux_unc = dat_obs.flux_unc

    # if flux_unc is not None:
    #     inv_var = 1.0 / (flux_unc**2 + 1e-20)
    #     weighted_sum = np.sum(flux * inv_var, axis=1)
    #     weight_sum = np.sum(inv_var, axis=1)
    #     weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    #     weighted_mean_flux = weighted_sum / weight_sum
    # else:
    weighted_mean_flux = np.mean(flux, axis=1)

    log_amplitude = np.log10(np.maximum(weighted_mean_flux, 1e-10))

    jax.debug.print("Computed log-amplitudes for {n} sources", n=log_amplitude.shape[0])
    jax.debug.print("  Range: [{min:.3f}, {max:.3f}]", min=float(jnp.min(log_amplitude)), max=float(jnp.max(log_amplitude)))
    jax.debug.print("  Mean: {mean:.3f} ± {std:.3f}", mean=float(jnp.mean(log_amplitude)), std=float(jnp.std(log_amplitude)))

    return all_spec_obs, all_noiseless_spec, all_flux_unc, weights, src_idxs, redshift_rf, srcid_rf, redshift, srcid_obs, phot_snr, norms, srcid_noiseless, log_amplitude
    
    
''' Dataloader for training/validation datasets '''
def create_train_validation_dataloaders_jax(data, val_list, train_frac=0.8, batch_size=128, \
                                        val_names=['weights', 'norms', 'orig', 'total_snr']):

    trainidx, valididx = draw_train_validation_idxs_jax(data, train_frac=train_frac)
    
    data_train = data[trainidx,:]
    if train_frac < 1.:
        data_valid = data[valididx,:]
        
    train_valid_dict = dict({})
    
    vals_valid, vals_train = [], []

    for v, val_array in enumerate(val_list):

        vals_train_indiv, vals_valid_indiv = grab_train_validation_dat_jax(val_array, train_frac, trainidx, valididx=valididx)
        train_valid_dict = update_dict(train_valid_dict, vals_train_indiv, vals_valid_indiv, key=val_names[v])
        vals_train.append(vals_train_indiv)
        vals_valid.append(vals_valid_indiv)
        
    data_train = jnp.array(data_train)
    data_valid = jnp.array(data_valid) if train_frac < 1. else None

    vals_train = [jnp.array(v) for v in vals_train]
    vals_valid = [jnp.array(v) for v in vals_valid] if train_frac < 1. else None

    def batch_generator(data, vals, batch_size):
        """Yields batches of data and values."""
        num_samples = data.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_data = data[batch_idx]
            batch_vals = [v[batch_idx] for v in vals]
            yield (batch_data, *batch_vals)

    trainloader = batch_generator(data_train, vals_train, batch_size)
    validloader = batch_generator(data_valid, vals_valid, batch_size) if train_frac < 1. else None
    
    return trainloader, validloader, data_train, data_valid, trainidx, valididx, np.mean(data), train_valid_dict, vals_train, vals_valid



class spec_data_jax():
    
    result_dir = None
    train_mode='deep'
    
    def __init__(self, nbands, **kwargs):
            
        self.nbands = nbands
        for key, val in kwargs.items():
            if val is not None:
                self.key = val
    
    
    def build_dataloaders(self, fpath_dict=None, load_property_cat=True, train_frac=0.8, weight_soft=5e-4, property_cat_fpath=None, restframe=False, save_property_cat=False, sig_level_norm=None, plot=False, apply_sel=True, sel_str='zlt22.5', abs_norm=True, noiseless_norm=False, sel_cosmos_temp_fits=False, sel_b14_temp_fits=False, pivot_wavelength=None, max_normflux=10):
        
        ''' This loads the input SED datasets, with and/or without noise. 
        Currently "restframe" refers to noiseless, rest frame spectra. '''
                
        self.flux, self.flux_unc, self.sed_um_wave,\
                self.catgrid_flux_noiseless, property_cat_df, self.srcids_noiseless = load_in_sphx_dat(fpath_dict, restframe=restframe, property_cat_fpath=property_cat_fpath, load_property_cat=load_property_cat, save_property_cat=save_property_cat, \
                                                                                                                   apply_sel=apply_sel)

        if noiseless_norm:
            
            print('src ids noiseless has length', self.srcids_noiseless.shape)
            norms_noiseless = np.mean(self.catgrid_flux_noiseless, axis=1)
            print('norms noiseless has shape', norms_noiseless.shape)
            print(norms_noiseless)
            
            srcid_obs = np.array(property_cat_df['Tractor_ID'])
            print('src id obs:', srcid_obs.shape)
            
            idxs_noiseless = np.array([np.where((self.srcids_noiseless==srcid_obs[idx]))[0][0] for idx in range(len(srcid_obs))])
            print('idxs noiseless has length:', idxs_noiseless.shape)
            norms_noiseless_sel = norms_noiseless[idxs_noiseless]
            print('norms noiseless sel has length', len(norms_noiseless_sel))
        
        else:
            norms_noiseless_sel = None
        
        if restframe:
            weight_soft = 0.

        print('abs norm is ', abs_norm)

        if sel_cosmos_temp_fits or sel_b14_temp_fits:

            bfit_temp_id = np.array(property_cat_df['bfit_tid'])

            print('bfit temp id has length', len(bfit_temp_id))

            temp_fpath = config.sphx_dat_path+'brown_cosmos_cont_bomee_order.list'
            all_temp_dict = make_bfit_tid_dict(temp_fpath)
            
            cos_temp_fpath = config.sphx_dat_path+'cosmos_cont_only.list'
            cosmos_temp_dict = make_bfit_tid_dict(cos_temp_fpath)
            cos_names = [cosmos_temp_dict[key] for key in cosmos_temp_dict.keys()]

            bfit_template_names = [all_temp_dict[idx] for idx in bfit_temp_id]

            if sel_cosmos_temp_fits:
                sel_mask = np.array([1 if item in cos_names else 0 for item in bfit_template_names])

            elif sel_b14_temp_fits:
                print('Brown 2014 galaxy selection')
                sel_mask = np.array([0 if item in cos_names else 1 for item in bfit_template_names])

                
            print('sum of downselect mask is ', np.sum(sel_mask))

            print('before downselect, flux has shape', self.flux.shape)
            self.flux = self.flux[np.where(sel_mask)[0]]

            print('flux now has shape', self.flux.shape)

            if self.flux_unc is not None:
                self.flux_unc = self.flux_unc[np.where(sel_mask)[0]]

        if restframe:
            print('pivot wavelength in build dataloaders is ', pivot_wavelength)
            self.phot_dict = parse_input_sphx_phot(self.flux, self.flux_unc, weight_soft=weight_soft, sig_level_norm=sig_level_norm, plot=plot, abs_norm=abs_norm, norms_noiseless=norms_noiseless_sel, wav=self.sed_um_wave, pivot_wavelength=pivot_wavelength, max_normflux=max_normflux)
        else:
            self.phot_dict = parse_input_sphx_phot(self.flux, self.flux_unc, weight_soft=weight_soft, sig_level_norm=sig_level_norm, plot=plot, abs_norm=abs_norm, norms_noiseless=norms_noiseless_sel)
        
        self.phot_proc = self.phot_dict['phot_fluxes']
        
        if not restframe:
            property_cat_df['phot_snr'] = self.phot_dict['phot_snr']
            self.val_list = [self.phot_dict['phot_weights'], self.phot_dict['phot_norms'], self.catgrid_flux_noiseless, self.phot_dict['phot_snr']]

            print('catgrid flux noiseless here has shape', self.catgrid_flux_noiseless.shape)
            
            val_names=['weights', 'norms', 'orig', 'total_snr']
        else:
            if sig_level_norm is not None:
                self.val_list = [self.phot_dict['phot_norms'], self.phot_dict['phot_weights'], self.flux]
                val_names = ['norms', 'weights', 'orig']
            else:
                self.val_list = [self.phot_dict['phot_norms'], self.flux]
                val_names = ['norms', 'orig']
            
         
        print('Creating data loaders for training/validation..')
        self.trainloader, self.validloader, self.data_train, self.data_valid,\
            self.trainidx, self.valididx, self.mean_dat,\
                    self.train_valid_dict, self.vals_train, self.vals_valid = create_train_validation_dataloaders_jax(self.phot_proc, self.val_list, train_frac=train_frac, val_names=val_names)
        
        return property_cat_df


    def build_dataloaders_new(self, 
                              fpath_dict=None,
                              load_property_cat=True,
                              train_frac=0.8,
                              weight_soft=5e-4,
                              property_cat_fpath=None,
                              restframe=False,
                              save_property_cat=False,
                              sig_level_norm=None,
                              plot=False,
                              apply_sel=True,
                              sel_str='zlt22.5',
                              abs_norm=True,
                              noiseless_norm=False,
                              sel_cosmos_temp_fits=False,
                              sel_b14_temp_fits=False,
                              pivot_wavelength=None,
                              max_normflux=100, 
                              z_min=None, z_max=None):
        
        ''' This loads the input SED datasets, with and/or without noise. 
        Currently "restframe" refers to noiseless, rest frame spectra. '''

        # Step 1: Load data
        (
            self.flux,
            self.flux_unc,
            self.sed_um_wave,
            self.catgrid_flux_noiseless,
            property_cat_df,
            self.srcids_noiseless,
        ) = load_in_sphx_dat_new(
            fpath_dict,
            restframe=restframe,
            property_cat_fpath=property_cat_fpath,
            load_property_cat=load_property_cat,
            save_property_cat=save_property_cat,
            apply_sel=apply_sel,
        )

        norms_noiseless_sel = self._compute_noiseless_norm(property_cat_df) if noiseless_norm else None
        
        if restframe:
            weight_soft = 0.

        # Apply template selection filters
        if sel_cosmos_temp_fits or sel_b14_temp_fits:
            self._apply_template_selection(property_cat_df, sel_cosmos_temp_fits, sel_b14_temp_fits)

        if z_min is not None or z_max is not None:
            property_cat_df = self._apply_redshift_selection(property_cat_df, z_min=z_min, z_max=z_max)

        self.phot_dict = self._parse_photometry(
            restframe, weight_soft, sig_level_norm, plot, abs_norm, norms_noiseless_sel, pivot_wavelength, max_normflux
        )
        self.phot_proc = self.phot_dict['phot_fluxes']

        # prep additional values for loader
        val_names, self.val_list = self._prepare_val_list(property_cat_df, restframe, sig_level_norm)

        print('Creating data loaders for training/validation..')

        (
            self.trainloader,
            self.validloader,
            self.data_train,
            self.data_valid,
            self.trainidx,
            self.valididx,
            self.mean_dat,
            self.train_valid_dict,
            self.vals_train,
            self.vals_valid,
        ) = create_train_validation_dataloaders_jax(self.phot_proc, self.val_list, train_frac=train_frac, val_names=val_names)

        return property_cat_df


    def _compute_noiseless_norm(self, property_cat_df):
        """Compute normalization values for noiseless spectra."""
        norms_noiseless = np.mean(self.catgrid_flux_noiseless, axis=1)
        srcid_obs = np.array(property_cat_df["Tractor_ID"])
        idxs_noiseless = np.array(
            [np.where(self.srcids_noiseless == sid)[0][0] for sid in srcid_obs]
        )
        return norms_noiseless[idxs_noiseless]

    def _apply_redshift_selection(self, property_cat_df, z_min=None, z_max=None):
        """
        Apply a redshift range selection to the dataset.
    
        Parameters
        ----------
        property_cat_df : pd.DataFrame
            DataFrame containing at least a 'redshift' column.
        z_min : float or None
            Minimum redshift (inclusive). If None, no lower bound.
        z_max : float or None
            Maximum redshift (inclusive). If None, no upper bound.
    
        Returns
        -------
        property_cat_df : pd.DataFrame
            Filtered property catalog with matching ordering to flux arrays.
        """
        if 'redshift' not in property_cat_df.columns:
            raise ValueError("property_cat_df must contain a 'redshift' column.")
    
        # Build selection mask
        mask = np.ones(len(property_cat_df), dtype=bool)
        if z_min is not None:
            mask &= property_cat_df['redshift'].values >= z_min
        if z_max is not None:
            mask &= property_cat_df['redshift'].values <= z_max
    
        print(f"Applying redshift selection: {np.sum(mask)} / {len(mask)} sources kept.")
    
        # Apply mask to flux arrays
        self.flux = self.flux[mask]
        if self.flux_unc is not None:
            self.flux_unc = self.flux_unc[mask]
    
        # Apply mask to property catalog
        property_cat_df = property_cat_df[mask].reset_index(drop=True)
    
        return property_cat_df

        
    def _apply_template_selection(self, property_cat_df, sel_cosmos, sel_b14):
        """Filter flux arrays based on template selection criteria."""
        bfit_temp_id = np.array(property_cat_df["bfit_tid"])
        all_temp_dict = make_bfit_tid_dict(config.sphx_dat_path + "brown_cosmos_cont_bomee_order.list")
        cosmos_temp_dict = make_bfit_tid_dict(config.sphx_dat_path + "cosmos_cont_only.list")
        cos_names = set(cosmos_temp_dict.values())
        bfit_template_names = [all_temp_dict[idx] for idx in bfit_temp_id]

        if sel_cosmos:
            sel_mask = np.array([name in cos_names for name in bfit_template_names])
        elif sel_b14:
            sel_mask = np.array([name not in cos_names for name in bfit_template_names])

        indices = np.where(sel_mask)[0]
        self.flux = self.flux[indices]
        if self.flux_unc is not None:
            self.flux_unc = self.flux_unc[indices]

    def _parse_photometry(self, restframe, weight_soft, sig_level_norm, plot, abs_norm, norms_noiseless_sel, pivot_wavelength, max_normflux):
        """Run photometry parsing with appropriate parameters."""
        kwargs = dict(
            weight_soft=weight_soft,
            sig_level_norm=sig_level_norm,
            plot=plot,
            abs_norm=abs_norm,
            norms_noiseless=norms_noiseless_sel,
        )
        if restframe:
            kwargs.update(wav=self.sed_um_wave, pivot_wavelength=pivot_wavelength, max_normflux=max_normflux)
        return parse_input_sphx_phot(self.flux, self.flux_unc, **kwargs)
            
    def _prepare_val_list(self, property_cat_df, restframe, sig_level_norm):
        """Prepare value lists for dataloaders."""
        if not restframe:
            property_cat_df["phot_snr"] = self.phot_dict["phot_snr"]
            
            # Check if catgrid_flux_noiseless matches phot array sizes
            # (it may not after redshift filtering)
            expected_size = len(self.phot_dict["phot_weights"])
            if (self.catgrid_flux_noiseless is not None and 
                len(self.catgrid_flux_noiseless) == expected_size):
                val_list = [
                    self.phot_dict["phot_weights"],
                    self.phot_dict["phot_norms"],
                    self.catgrid_flux_noiseless,
                    self.phot_dict["phot_snr"],
                ]
                return ["weights", "norms", "orig", "total_snr"], val_list
            else:
                # Skip noiseless if sizes don't match
                print(f"Skipping catgrid_flux_noiseless (size mismatch: {len(self.catgrid_flux_noiseless) if self.catgrid_flux_noiseless is not None else 'None'} vs {expected_size})")
                val_list = [
                    self.phot_dict["phot_weights"],
                    self.phot_dict["phot_norms"],
                    self.phot_dict["phot_snr"],
                ]
                return ["weights", "norms", "total_snr"], val_list
        else:
            if sig_level_norm is not None:
                val_list = [self.phot_dict["phot_norms"], self.phot_dict["phot_weights"], self.flux]
                return ["norms", "weights", "orig"], val_list
            else:
                val_list = [self.phot_dict["phot_norms"], self.flux]
                return ["norms", "orig"], val_list

        