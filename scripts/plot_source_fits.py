#!/usr/bin/env python
"""
Standalone script to inspect individual PAE SED fits vs observed data.

Selects sources from PAE redshift results based on user-specified criteria
(SNR range, true-redshift range, z-score range, specific source IDs) and
generates per-source SED comparison plots showing:
  - Observed fiducial (306-band) photometry with error bars
  - PAE posterior mean reconstruction with 68%/95% credible intervals
  - Residual panel (chi per channel)
  - Annotations: z_true, z_hat, chi^2, SNR, etc.
  - Optional: native (per-measurement) data overlay or coadded spectrum

Usage:
    # Select 10 random sources with SNR >= 50 and |zscore| < 1:
    python scripts/plot_source_fits.py --datestr multinode_validation_run_022126 \\
        --snr-range 50 1000 --zscore-max 1.0 --n-sources 10

    # Inspect specific sources by SPHERExRefID:
    python scripts/plot_source_fits.py --datestr multinode_validation_run_022126 \\
        --src-ids 1701041318911803397 1703715159677075460

    # Show bad fits (large z-score) with native measurements:
    python scripts/plot_source_fits.py --datestr multinode_validation_run_022126 \\
        --zscore-min 3.0 --n-sources 20 --native
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import jax.numpy as jnp
import jax

from config import scratch_basepath


# ---------------------------------------------------------------------------
# Emission line catalogue (rest-frame Angstroms → μm)
# ---------------------------------------------------------------------------

# Rest wavelengths in Angstroms; grouped for colour + label assignment.
# Lines with nearly the same wavelength are treated as a single tick
# (use the stronger component for the label).
_EMISSION_LINES_AA = [
    # Balmer series
    ('Hα + [NII]', 6562.8,  'C1'),  # H-alpha complex with NII
    ('[OIII]+Hβ',  5006.8,  'C2'),  # OIII + H-beta complex (at OIII 5007)
    # Forbidden oxygen
    ('[OII]λ3727', 3727.1,  'C2'),
    # Paschen series
    ('Paα',        18751.0, 'C0'),
    ('Paβ',        12821.6, 'C0'),
    # Brackett series
    ('Brα',        40522.6, 'C9'),
    ('Brβ',        26258.7, 'C9'),
    ('Brγ',        21661.2, 'C9'),
]
# Convert to μm
EMISSION_LINES = [(name, wav_aa * 1e-4, color)
                  for name, wav_aa, color in _EMISSION_LINES_AA]


def draw_emission_lines(ax, z, wave_min=0.75, wave_max=5.05,
                        fontsize=9, alpha_line=0.45):
    """
    Draw vertical lines for emission lines redshifted to redshift *z*.
    Only lines with observed wavelength in [wave_min, wave_max] μm are drawn.
    Labels are placed alternating above/below the top of the plot area so
    they do not overlap too heavily.
    """
    visible = [(name, lam_rest * (1 + z), color)
               for name, lam_rest, color in EMISSION_LINES
               if wave_min <= lam_rest * (1 + z) <= wave_max]

    # Sort by observed wavelength for alternating label placement
    visible.sort(key=lambda x: x[1])

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin

    for i, (name, lam_obs, color) in enumerate(visible):
        ax.axvline(lam_obs, color=color, lw=0.7, ls='--',
                   alpha=alpha_line, zorder=1)
        # Place labels at middle of y-axis
        y_frac = 0.5
        ax.text(lam_obs, ymin + y_frac * span, name,
                rotation=90, fontsize=fontsize,
                color=color, va='center', ha='right',
                alpha=0.8, zorder=1,
                clip_on=True)

def load_combined_results(result_dir):
    """Load the combined results npz and return as dict."""
    result_files = list(Path(result_dir).glob('PAE_results_combined_*.npz'))
    if not result_files:
        print(f"Error: No combined result file found in {result_dir}")
        sys.exit(1)
    fpath = result_files[0]
    print(f"Loading combined results: {fpath.name}")
    res = np.load(str(fpath), allow_pickle=True)
    # Eagerly load all arrays into a plain dict so the file handle can close
    out = {}
    for k in res.files:
        out[k] = res[k]
    res.close()
    return out


def select_sources(res, snr_range=None, z_range=None,
                   zscore_min=None, zscore_max=None,
                   frac_min=0.7, n_sources=10, src_ids=None,
                   z_tf_range=None, z_pae_range=None,
                   snr_cache_path=None, parquet_path=None,
                   seed=42):
    """
    Select source indices from combined results matching the given criteria.

    Parameters
    ----------
    res : dict
        Combined results (from load_combined_results).
    snr_range : tuple of (lo, hi) or None
    z_range : tuple of (lo, hi) or None
        True-redshift range.
    zscore_min, zscore_max : float or None
        Absolute z-score range.
    frac_min : float
        Minimum frac_sampled_102.
    n_sources : int
        Number of sources to return (ignored if src_ids given).
    src_ids : list of int or None
        Specific SPHERExRefID values to select.
    snr_cache_path : Path or None
        Path to snr_cache.npz (snr_quad, src_id).
    parquet_path : str or None
        Parquet file for computing SNR if cache missing.
    seed : int
        Random seed for reproducible subset selection.

    Returns
    -------
    indices : array of int
        Indices into the combined results arrays.
    """
    n_total = len(res['src_id'])

    # If specific source IDs requested, just find them
    if src_ids is not None:
        src_id_arr = np.array(res['src_id'])
        indices = []
        for sid in src_ids:
            matches = np.where(src_id_arr == int(sid))[0]
            if len(matches) == 0:
                print(f"  Warning: src_id {sid} not found in results")
            else:
                indices.append(matches[0])
        if len(indices) == 0:
            print("Error: None of the requested src_ids found")
            sys.exit(1)
        return np.array(indices)

    # Build mask from criteria
    mask = np.ones(n_total, dtype=bool)

    # frac_sampled_102
    if 'frac_sampled_102' in res and frac_min is not None:
        frac = np.array(res['frac_sampled_102'])
        mask &= (frac >= frac_min)
        print(f"  After frac_sampled_102 >= {frac_min}: {mask.sum():,} / {n_total:,}")

    # z-score
    zscore = np.array(res['zscore'])
    if zscore_min is not None:
        mask &= (np.abs(zscore) >= zscore_min)
        print(f"  After |zscore| >= {zscore_min}: {mask.sum():,}")
    if zscore_max is not None:
        mask &= (np.abs(zscore) <= zscore_max)
        print(f"  After |zscore| <= {zscore_max}: {mask.sum():,}")

    # True redshift
    ztrue = np.array(res['ztrue'])
    if z_range is not None:
        mask &= (ztrue >= z_range[0]) & (ztrue <= z_range[1])
        print(f"  After z_true in [{z_range[0]}, {z_range[1]}]: {mask.sum():,}")

    # Template-fitting redshift range
    if z_tf_range is not None:
        if 'z_TF' in res:
            z_tf = np.array(res['z_TF'])
            mask &= (z_tf >= z_tf_range[0]) & (z_tf <= z_tf_range[1])
            print(f"  After z_TF in [{z_tf_range[0]}, {z_tf_range[1]}]: {mask.sum():,}")
        else:
            print("  Warning: z_TF not in results — skipping z_tf_range filter")

    # PAE median redshift range
    if z_pae_range is not None:
        if 'z_med' in res:
            z_pae = np.array(res['z_med'])
            mask &= (z_pae >= z_pae_range[0]) & (z_pae <= z_pae_range[1])
            print(f"  After z_PAE in [{z_pae_range[0]}, {z_pae_range[1]}]: {mask.sum():,}")
        else:
            print("  Warning: z_med not in results — skipping z_pae_range filter")

    # SNR (broadband quad from cache or parquet)
    if snr_range is not None:
        snr_arr = _get_snr_array(res, snr_cache_path, parquet_path)
        if snr_arr is not None:
            mask &= (snr_arr >= snr_range[0]) & (snr_arr <= snr_range[1])
            print(f"  After SNR in [{snr_range[0]}, {snr_range[1]}]: {mask.sum():,}")
        else:
            print("  Warning: Could not load SNR array — skipping SNR filter")

    # Select
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        print("Error: No sources match the selection criteria")
        sys.exit(1)

    if n_sources >= len(valid_indices):
        selected = valid_indices
    else:
        rng = np.random.default_rng(seed)
        selected = rng.choice(valid_indices, size=n_sources, replace=False)

    # Sort for deterministic ordering
    selected = np.sort(selected)
    print(f"  Selected {len(selected)} sources out of {len(valid_indices)} matching")
    return selected


def _get_snr_array(res, snr_cache_path, parquet_path):
    """Load or compute broadband snr_quad array aligned to combined results."""
    # Try cache first
    if snr_cache_path is not None and Path(snr_cache_path).exists():
        cache = np.load(str(snr_cache_path))
        cache_ids = cache['src_id']
        cache_snr = cache['snr_quad']
        # Build lookup
        id_to_snr = dict(zip(cache_ids.astype(int), cache_snr))
        src_ids = np.array(res['src_id'])
        snr_arr = np.array([id_to_snr.get(int(s), np.nan) for s in src_ids])
        n_valid = np.sum(np.isfinite(snr_arr))
        print(f"  Loaded SNR cache: {n_valid:,} / {len(snr_arr):,} matched")
        return snr_arr

    # Compute from parquet if available
    if parquet_path is not None and Path(parquet_path).exists():
        return _compute_snr_from_parquet(res, parquet_path)

    return None


def _compute_snr_from_parquet(res, parquet_path):
    """Compute snr_quad from parquet fiducial fluxes for each source in results."""
    import pyarrow.parquet as pq
    print(f"  Computing snr_quad from parquet (may take a few minutes)...")

    src_ids = np.array(res['src_id'])
    src_id_set = set(int(s) for s in src_ids)
    id_to_quad = {}

    pf = pq.ParquetFile(parquet_path)
    id_col = 'SPHERExRefID'
    flux_col = 'flux_dered_fiducial'
    err_col = 'flux_err_dered_fiducial'

    for batch in pf.iter_batches(columns=[id_col, flux_col, err_col]):
        ids_b = batch.column(id_col).to_pylist()
        hit = np.array([int(s) in src_id_set for s in ids_b])
        if not hit.any():
            continue
        for j in np.where(hit)[0]:
            sid = int(ids_b[j])
            flux = np.array(batch.column(flux_col)[j].as_py(), dtype=float)
            err = np.array(batch.column(err_col)[j].as_py(), dtype=float)
            good = (flux > 0) & (err > 0) & (err < 5e4)
            if good.any():
                id_to_quad[sid] = float(np.sqrt(np.sum((flux[good] / err[good]) ** 2)))
            else:
                id_to_quad[sid] = 0.0

    snr_arr = np.array([id_to_quad.get(int(s), np.nan) for s in src_ids])
    print(f"  Computed snr_quad for {np.sum(np.isfinite(snr_arr)):,} sources")
    return snr_arr


# ---------------------------------------------------------------------------
# Batch index: map src_id -> (sample_file, local_idx)
# ---------------------------------------------------------------------------

def build_batch_index(result_dir):
    """
    Scan all task directories and build a lookup: src_id -> (sample_file, local_idx).

    Returns
    -------
    index : dict
        Mapping int(src_id) -> (str(sample_file_path), int(local_idx)).
    """
    result_dir = Path(result_dir)
    task_dirs = sorted(result_dir.glob('task*'),
                       key=lambda p: int(p.name.replace('task', '')))
    if not task_dirs:
        print("Error: No task directories found")
        sys.exit(1)

    print(f"Building batch index from {len(task_dirs)} task directories...")
    index = {}
    n_files = 0
    for td in task_dirs:
        # Find matching pairs: results + samples
        res_files = sorted(td.glob('PAE_results_batch*'))
        for rf in res_files:
            # Derive corresponding sample file name
            sf_name = rf.name.replace('PAE_results_', 'PAE_samples_')
            sf = td / sf_name
            if not sf.exists():
                continue
            # Load only src_id from results file
            with np.load(str(rf)) as data:
                sids = data['src_id']
            for local_idx, sid in enumerate(sids):
                index[int(sid)] = (str(sf), local_idx)
            n_files += 1

    print(f"  Indexed {len(index):,} sources across {n_files} batch files")
    return index


def load_source_samples(batch_index, src_id):
    """
    Load MCMC samples for a single source.

    Returns
    -------
    samples : array, shape (n_chains, n_steps, n_dim)
    """
    key = int(src_id)
    if key not in batch_index:
        raise FileNotFoundError(f"Source {src_id} not in batch index")

    sample_file, local_idx = batch_index[key]
    with np.load(sample_file) as data:
        samples = data['all_samples'][local_idx]  # (n_chains, n_steps, n_dim)
    return samples


# ---------------------------------------------------------------------------
# Load observed data for a single source from parquet
# ---------------------------------------------------------------------------

def load_source_flux_from_parquet(parquet_path, src_id, load_native=False):
    """
    Load fiducial flux / flux_err for a single source from parquet.

    Parameters
    ----------
    parquet_path : str
    src_id : int
    load_native : bool
        If True, also return native measurement arrays (lambda, flux, flux_err).

    Returns
    -------
    flux_fid : array (nbands,)
        Fiducial flux in μJy.
    flux_err_fid : array (nbands,)
        Fiducial flux uncertainty in μJy.
    z_specz : float
    native : dict or None
        {'lam': array, 'flux': array, 'flux_err': array} if load_native=True.
    """
    import pyarrow.parquet as pq
    import pyarrow.compute as pc

    pf = pq.ParquetFile(parquet_path)
    cols = ['SPHERExRefID', 'flux_dered_fiducial', 'flux_err_dered_fiducial', 'z_specz']
    if load_native:
        cols.extend(['lambda', 'flux', 'flux_err'])

    for batch in pf.iter_batches(columns=cols):
        ids_b = batch.column('SPHERExRefID').to_pylist()
        for j, sid in enumerate(ids_b):
            if int(sid) == int(src_id):
                flux_fid = np.array(batch.column('flux_dered_fiducial')[j].as_py(), dtype=float)
                flux_err_fid = np.array(batch.column('flux_err_dered_fiducial')[j].as_py(), dtype=float)
                z_specz = float(batch.column('z_specz')[j].as_py())

                native = None
                if load_native:
                    lam_nat = np.array(batch.column('lambda')[j].as_py(), dtype=float)
                    flux_nat = np.array(batch.column('flux')[j].as_py(), dtype=float)
                    err_nat = np.array(batch.column('flux_err')[j].as_py(), dtype=float)
                    native = {'lam': lam_nat, 'flux': flux_nat, 'flux_err': err_nat}

                return flux_fid, flux_err_fid, z_specz, native

    raise ValueError(f"Source {src_id} not found in parquet")


def normalize_source(flux, flux_err, phot_norm, weight_soft=5e-4):
    """
    Normalize flux/flux_err and compute inverse-variance weights,
    matching the training pipeline (parse_input_sphx_phot).

    Returns
    -------
    spec_obs : array  (normalized flux)
    weight : array    (inverse-variance weights)
    flux_unc_norm : array  (normalized uncertainties)
    """
    spec_obs = flux / phot_norm
    flux_unc_norm = flux_err / np.abs(phot_norm)

    # Zero-weight bad data (very large uncertainties from preprocessing)
    bad = (flux_err > 1e9) | (~np.isfinite(flux)) | (~np.isfinite(flux_err))
    flux_unc_norm[bad] = 1e10
    spec_obs[bad] = 0.0

    weight = 1.0 / (flux_unc_norm ** 2 + weight_soft)
    weight[bad] = 0.0
    # Also zero-weight where flux_unc_norm is extreme
    weight[flux_unc_norm > 1e8] = 0.0

    return spec_obs, weight, flux_unc_norm


# ---------------------------------------------------------------------------
# PAE model loading
# ---------------------------------------------------------------------------

def init_pae_model(run_params_path):
    """
    Initialize the PAE model from saved run parameters.

    Returns
    -------
    PAE_obj : PAE_JAX
    wave_obs : array (306 central wavelengths in μm)
    run_params : dict
    """
    from models.pae_jax import initialize_PAE

    rp = np.load(str(run_params_path), allow_pickle=True)
    run_name = str(rp['run_name'])
    filter_set_name = str(rp['filter_set_name'])
    nlatent = int(rp['nlatent'])
    sig_level_norm = float(rp['sig_level_norm'])
    # filename_flow stored without .pkl suffix; default matches initialize_PAE default
    filename_flow = str(rp['filename_flow']) if 'filename_flow' in rp.files else 'flow_model_iaf_50k'

    print(f"\n  Initializing PAE model:")
    print(f"    run_name         = {run_name}")
    print(f"    filter_set_name  = {filter_set_name}")
    print(f"    nlatent          = {nlatent}")
    print(f"    sig_level_norm   = {sig_level_norm}")
    print(f"    filename_flow    = {filename_flow}")

    PAE_obj = initialize_PAE(
        run_name=run_name,
        filter_set_name=filter_set_name,
        filename_flow=filename_flow,
        redshift_in_flow=False,
        inference_dtype=jnp.float32,
    )

    wave_obs = np.array(PAE_obj.wave_obs)

    run_params = {
        'run_name': run_name,
        'filter_set_name': filter_set_name,
        'nlatent': nlatent,
        'sig_level_norm': sig_level_norm,
    }

    print(f"    wave_obs: {len(wave_obs)} bands, {wave_obs[0]:.3f} – {wave_obs[-1]:.3f} μm")
    return PAE_obj, wave_obs, run_params


def auto_label(args):
    """Build a directory-safe label string from selection arguments."""
    parts = []
    if args.src_ids:
        parts.append(f"srcids_{len(args.src_ids)}")
    else:
        if args.snr_range:
            parts.append(f"snr{args.snr_range[0]:.0f}-{args.snr_range[1]:.0f}")
        if args.z_range:
            parts.append(f"z{args.z_range[0]:.2f}-{args.z_range[1]:.2f}")
        if args.z_tf_range:
            parts.append(f"zTF{args.z_tf_range[0]:.2f}-{args.z_tf_range[1]:.2f}")
        if args.z_pae_range:
            parts.append(f"zPAE{args.z_pae_range[0]:.2f}-{args.z_pae_range[1]:.2f}")
        if args.zscore_min is not None and args.zscore_max is None:
            parts.append(f"zs_min{args.zscore_min:.1f}")
        elif args.zscore_max is not None and args.zscore_min is None:
            parts.append(f"zs_max{args.zscore_max:.1f}")
        elif args.zscore_min is not None and args.zscore_max is not None:
            parts.append(f"zs{args.zscore_min:.1f}-{args.zscore_max:.1f}")
        parts.append(f"n{args.n_sources}")
    return "_".join(parts) if parts else "default"


# ---------------------------------------------------------------------------
# Spectrum reconstruction
# ---------------------------------------------------------------------------

def reconstruct_spectrum(PAE_obj, samples, spec_obs, weight,
                         burn_in=1000, thin_fac=1):
    """
    Reconstruct posterior spectrum from MCMC samples using proc_spec_post.

    Parameters
    ----------
    PAE_obj : PAE_JAX
    samples : array, shape (n_chains, n_steps, n_dim)
    spec_obs : array, normalized observed flux
    weight : array, inverse-variance weights
    burn_in : int
    thin_fac : int

    Returns
    -------
    recon_x : array, shape (N_samples_post, nbands) — normalized model fluxes
    redshift_post : array, shape (N_samples_post,) — posterior redshift samples
    logL : array, shape (N_samples_post,) — log-likelihoods
    """
    from diagnostics.diagnostics_jax import proc_spec_post

    # samples must be (n_chains, n_steps, n_dim)
    if samples.ndim != 3:
        raise ValueError(
            f"Expected samples shape (n_chains, n_steps, n_dim), got {samples.shape}. "
            "Load with all_samples[local_idx], not all_samples[local_idx, chain_idx]."
        )
    n_chains, n_steps, n_dim = samples.shape
    n_post = (n_steps - burn_in) // thin_fac * n_chains
    if n_post <= 0:
        raise ValueError(
            f"burn_in={burn_in} discards all {n_steps} steps. "
            "Reduce --burn-in."
        )

    recon_x, logL, redshift_post = proc_spec_post(
        PAE_obj, samples, spec_obs, weight,
        combine_chains=True, burn_in=burn_in, thin_fac=thin_fac,
        redshift_fix=None,
    )

    # proc_spec_post vmaps per-sample with u=latent[None,:] so push_spec_marg
    # returns (1, nbands) per call → vmap stacks to (N, 1, nbands). Squeeze it.
    recon_x = np.array(recon_x).squeeze()
    if recon_x.ndim == 1:          # only 1 post-burn-in sample
        recon_x = recon_x[None, :]

    return recon_x, np.array(redshift_post), np.array(logL)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_source_fit(wave_obs, flux_phys, flux_err_phys, recon_phys,
                    redshift_post, z_true, z_med, err_low, err_high,
                    chi2_full, snr_val, src_id,
                    z_TF=None, z_TF_err=None, chisq_tf=None, native_data=None, coadd_native=True,
                    show_lines=False,
                    figsize=(10, 5), dpi=150):
    """
    Create a two-panel SED fit comparison figure.

    Panel 1 (top): observed photometry + posterior reconstruction (mean + CI).
    Panel 2 (bottom): residuals (chi per channel).

    Parameters
    ----------
    wave_obs : array
        Central wavelengths (μm) for fiducial bands.
    flux_phys : array
        Observed fiducial flux (μJy, physical units).
    flux_err_phys : array
        Observed fiducial flux uncertainties (μJy).
    recon_phys : array, shape (N, nbands)
        Posterior model reconstructions (μJy).
    redshift_post : array
        Posterior redshift samples.
    z_true : float
    z_med : float
    err_low, err_high : float
    chi2_full : float
    snr_val : float or None
    src_id : int
    z_TF : float or None
        Template-fitting photo-z.
    native_data : dict or None
        {'lam': array, 'flux': array, 'flux_err': array}
    coadd_native : bool
        If True, show coadded native spectrum; else raw measurements.
    figsize : tuple
    dpi : int

    Returns
    -------
    fig : matplotlib.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax_main = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1], sharex=ax_main)

    # ---- Band mask: exclude uncovered bands (flux_err >= 5e4 uJy) ----
    covered = flux_err_phys < 5e4
    n_covered = int(covered.sum())
    wave_plot = wave_obs[covered]
    flux_plot = flux_phys[covered]
    err_plot = flux_err_phys[covered]

    # ---- Posterior statistics (all 306 bands for model) ----
    spec_med = np.median(recon_phys, axis=0)
    spec_mean = np.mean(recon_phys, axis=0)
    pct5 = np.percentile(recon_phys, 5, axis=0)
    pct16 = np.percentile(recon_phys, 16, axis=0)
    pct84 = np.percentile(recon_phys, 84, axis=0)
    pct95 = np.percentile(recon_phys, 95, axis=0)

    # ---- Main panel: SED ----
    # Alpha scales down with more bands, clamped to [0.1, 1.0]
    pt_alpha = float(np.clip(0.8 / np.sqrt(max(n_covered, 1) / 102), 0.1, 1.0))

    # Observed data
    ax_main.errorbar(wave_plot, flux_plot, yerr=err_plot,
                     fmt='o', color='k', markersize=2.5, capsize=0,
                     alpha=pt_alpha, label='Observed', zorder=1)

    # 95% CI (all 306 bands)
    ax_main.fill_between(wave_obs, pct5, pct95,
                         alpha=0.15, color='C3', label='95% CI', zorder=2)
    # 68% CI (all 306 bands)
    ax_main.fill_between(wave_obs, pct16, pct84,
                         alpha=0.35, color='C3', label='68% CI', zorder=3)
    # Posterior mean (all 306 bands)
    ax_main.plot(wave_obs, spec_mean, color='C3', linewidth=1.2,
                 label='Posterior mean', zorder=4)

    # Optional: native measurement overlay
    if native_data is not None:
        lam_n = native_data['lam']
        flux_n = native_data['flux']
        err_n = native_data['flux_err']
        # Filter valid
        ok = np.isfinite(lam_n) & np.isfinite(flux_n) & (err_n > 0) & (err_n < 5e4)
        if coadd_native and ok.sum() > 5:
            from specstack.qso_stack_utils import coadd_spectrum_to_grid
            lam_g, f_g, e_g, n_g = coadd_spectrum_to_grid(
                lam_n[ok], flux_n[ok], err_n[ok], R=200)
            good_g = np.isfinite(f_g)
            ax_main.errorbar(lam_g[good_g], f_g[good_g], yerr=e_g[good_g],
                             fmt='s', color='C0', markersize=3, capsize=1.5,
                             alpha=0.6, label='Native (coadd R=200)', zorder=5)
        else:
            ax_main.scatter(lam_n[ok], flux_n[ok], s=4, color='C0',
                            alpha=0.3, label='Native meas.', zorder=5)

    # Annotation text (left side)
    z_mean_post = np.mean(redshift_post)
    snr_str = f"SNR = {snr_val:.0f}  " if snr_val is not None and np.isfinite(snr_val) else ""
    has_TF = z_TF is not None and np.isfinite(z_TF)
    if has_TF:
        if z_TF_err is not None and np.isfinite(z_TF_err):
            tf_str = f"$z_{{\\rm TF}} = {z_TF:.4f} \\pm {z_TF_err:.4f}$"
        else:
            tf_str = f"$z_{{\\rm TF}} = {z_TF:.4f}$"
    else:
        tf_str = ""
    ann_lines = [
        f"SPHERExRefID: {src_id}",
        f"$z_{{\\rm spec}} = {z_true:.4f}$",
        tf_str,
        f"$\\hat{{z}}_{{\\rm PAE}} = {z_med:.4f}"
        f"^{{+{err_high:.4f}}}_{{-{err_low:.4f}}}$",
        f"{snr_str}$\\chi^2_{{\\rm PAE}} = {chi2_full:.1f}$  ({n_covered} bands)"
        + (f"$\\quad \\chi^2_{{\\rm TF}} = {chisq_tf:.1f}$" if chisq_tf is not None and np.isfinite(chisq_tf) else ""),
    ]
    ann_text = "\n".join(line for line in ann_lines if line)
    ax_main.text(0.02, 0.97, ann_text, transform=ax_main.transAxes,
                 fontsize=11, va='top', ha='left',
                 bbox=dict(facecolor='white', alpha=0.85, edgecolor='0.7',
                           boxstyle='round,pad=0.4'))

    # Y-limits: auto with padding — leave top ~45% free for annotation + inset
    good_flux = flux_plot[err_plot < 1e8]
    if len(good_flux) > 0:
        ylo = min(-20, np.percentile(good_flux, 1) - 10)
        yhi = max(80, np.percentile(good_flux, 99) * 2.8)
    else:
        ylo, yhi = -20, 150
    ax_main.set_ylim(ylo, yhi)
    ax_main.set_xlim(0.7, 5.1)
    ax_main.set_ylabel('Flux density [$\\mu$Jy]', fontsize=12)
    ax_main.legend(loc='upper left', fontsize=11, ncol=3,
                   bbox_to_anchor=(0.28, 1.20), framealpha=0.9)
    ax_main.grid(alpha=0.15)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # ---- Optional emission line overlay (drawn before text so text is on top) ----
    if show_lines:
        draw_emission_lines(ax_main, z_true)
    ax_ins = ax_main.inset_axes([0.67, 0.45, 0.31, 0.44])
    # Draw TF span first so histogram renders on top
    if has_TF and z_TF_err is not None and np.isfinite(z_TF_err) and z_TF_err > 0:
        ax_ins.axvspan(z_TF - z_TF_err, z_TF + z_TF_err,
                       color='C0', alpha=0.65, lw=0)
    ax_ins.hist(redshift_post, bins=50, density=True,
                color='C3', alpha=0.35, linewidth=0)
    ax_ins.axvline(z_true, color='k', lw=1.2, ls='--', label='$z_{\\rm spec}$')
    ax_ins.axvline(z_med, color='C3', lw=1.2, ls='-', label='$\\hat{z}_{\\rm PAE}$')
    if has_TF:
        ax_ins.axvline(z_TF, color='C0', lw=1.2, ls='-', label='$z_{\\rm TF}$')
    ax_ins.set_xlabel('$z$', fontsize=10)
    ax_ins.set_ylabel('$p(z)$', fontsize=10)
    ax_ins.xaxis.set_label_position('top')
    ax_ins.xaxis.tick_top()
    ax_ins.tick_params(labelsize=9)
    ax_ins.legend(fontsize=9, loc='upper left', handlelength=1.2,
                  framealpha=0.8, borderpad=0.4)

    # ---- Residual panel (covered bands only) ----
    valid = err_plot > 0
    chi = np.full_like(flux_plot, np.nan)
    # Extract model at covered bands for residual calculation
    spec_mean_covered = spec_mean[covered]
    chi[valid] = (flux_plot[valid] - spec_mean_covered[valid]) / err_plot[valid]

    ax_resid.scatter(wave_plot, chi, s=3, color='k', alpha=pt_alpha, zorder=2)
    ax_resid.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_resid.axhline(2, color='0.6', linestyle=':', linewidth=0.6)
    ax_resid.axhline(-2, color='0.6', linestyle=':', linewidth=0.6)

    chi_lim = max(3, min(8, np.nanmax(np.abs(chi[valid])) * 1.1)) if valid.any() else 3
    ax_resid.set_ylim(-chi_lim, chi_lim)
    ax_resid.set_ylabel('$\\chi$', fontsize=12)
    ax_resid.set_xlabel('$\\lambda_{\\rm obs}$ [$\\mu$m]', fontsize=12)
    ax_resid.grid(alpha=0.15)

    fig.align_ylabels([ax_main, ax_resid])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Inspect individual PAE SED fits vs observed data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('--datestr', type=str, required=True,
                        help='Run datestring (directory name under batched/)')

    # Source selection
    sel = parser.add_argument_group('Source selection')
    sel.add_argument('--snr-range', type=float, nargs=2, default=None,
                     metavar=('LO', 'HI'),
                     help='Broadband SNR range (snr_quad)')
    sel.add_argument('--z-range', type=float, nargs=2, default=None,
                     metavar=('LO', 'HI'),
                     help='True-redshift range')
    sel.add_argument('--z-tf-range', type=float, nargs=2, default=None,
                     metavar=('LO', 'HI'),
                     help='Template-fitting redshift range filter')
    sel.add_argument('--z-pae-range', type=float, nargs=2, default=None,
                     metavar=('LO', 'HI'),
                     help='PAE median redshift range filter')
    sel.add_argument('--zscore-min', type=float, default=None,
                     help='Minimum |z-score| (select bad fits)')
    sel.add_argument('--zscore-max', type=float, default=None,
                     help='Maximum |z-score| (select good fits)')
    sel.add_argument('--frac-min', type=float, default=0.7,
                     help='Minimum frac_sampled_102 (default: 0.7)')
    sel.add_argument('--n-sources', type=int, default=10,
                     help='Number of sources to plot (default: 10)')
    sel.add_argument('--src-ids', type=int, nargs='+', default=None,
                     help='Specific SPHERExRefID(s) to plot')
    sel.add_argument('--seed', type=int, default=42,
                     help='Random seed for source selection (default: 42)')

    # Reconstruction
    recon = parser.add_argument_group('Reconstruction')
    recon.add_argument('--burn-in', type=int, default=None,
                       help='MCMC burn-in samples to discard '
                            '(default: read from run_params.npz)')
    recon.add_argument('--thin-fac', type=int, default=1,
                       help='MCMC thinning factor (default: 1)')

    # Native measurements
    nat = parser.add_argument_group('Native measurements')
    nat.add_argument('--native', action='store_true',
                     help='Overlay native (per-measurement) data')
    nat.add_argument('--native-raw', action='store_true',
                     help='Show raw native points instead of coadded spectrum')
    nat.add_argument('--show-lines', action='store_true',
                     help='Overlay predicted emission line positions at z_PAE')

    # Output
    out = parser.add_argument_group('Output')
    out.add_argument('--outdir', type=str, default=None,
                     help='Root output directory (auto if not set). '
                          'Figures go into --outdir/LABEL/')
    out.add_argument('--label', type=str, default=None,
                     help='Subdirectory label for this selection '
                          '(auto-generated from selection args if not set)')
    out.add_argument('--dpi', type=int, default=150,
                     help='Figure DPI (default: 150)')
    out.add_argument('--figsize', type=float, nargs=2, default=[11, 6],
                     help='Figure size in inches (default: 11 6)')

    args = parser.parse_args()

    # ---- Paths ----
    result_base = Path(scratch_basepath) / 'data' / 'pae_sample_results' / 'MCLMC' / 'batched'
    result_dir = result_base / args.datestr
    if not result_dir.exists():
        print(f"Error: Result directory not found: {result_dir}")
        sys.exit(1)

    parquet_path = str(Path(scratch_basepath) / 'data' / 'l3_data' /
                       'full_validation_sz_0-1000.0_z_0-1000.0.parquet')

    # Mirror the same figures root that generate_redshift_plots.py uses:
    #   scratch_basepath/figures/redshift_validation/{datestr}/snr_diagnostics/snr_cache.npz
    fig_base = Path(scratch_basepath) / 'figures' / 'redshift_validation' / args.datestr
    snr_cache_path = fig_base / 'snr_diagnostics' / 'snr_cache.npz'
    if snr_cache_path.exists():
        print(f"  Found SNR cache: {snr_cache_path}")
    else:
        print(f"  No SNR cache found at {snr_cache_path} — will compute from parquet if SNR filter requested")

    # Output directory: root/source_fits/LABEL/
    root_dir = Path(args.outdir) if args.outdir else (fig_base / 'source_fits')
    label = args.label if args.label else auto_label(args)
    outdir = root_dir / label
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {outdir}")

    # ---- Load combined results ----
    print(f"\n{'='*70}")
    print("PAE SED Fit Inspection")
    print(f"{'='*70}")

    res = load_combined_results(result_dir)

    # ---- Select sources ----
    print("\n[1/4] Selecting sources...")
    indices = select_sources(
        res,
        snr_range=args.snr_range,
        z_range=args.z_range,
        zscore_min=args.zscore_min,
        zscore_max=args.zscore_max,
        frac_min=args.frac_min,
        n_sources=args.n_sources,
        src_ids=args.src_ids,
        z_tf_range=args.z_tf_range,
        z_pae_range=args.z_pae_range,
        snr_cache_path=str(snr_cache_path),
        parquet_path=parquet_path,
        seed=args.seed,
    )

    # Extract per-source metadata
    src_ids = np.array(res['src_id'])[indices]
    z_true_arr = np.array(res['ztrue'])[indices]
    z_med_arr = np.array(res['z_med'])[indices]
    err_low_arr = np.array(res['err_low'])[indices]
    err_high_arr = np.array(res['err_high'])[indices]
    chi2_full_arr = np.array(res['chi2_full'])[indices]
    phot_norms_arr = np.array(res['phot_norms'])[indices]
    zscore_arr = np.array(res['zscore'])[indices]

    # Optional arrays
    z_TF_arr = np.array(res['z_TF'])[indices] if 'z_TF' in res else None
    z_TF_err_arr = np.array(res['z_TF_err'])[indices] if 'z_TF_err' in res else None
    chisq_tf_arr = np.array(res['minchi2_gals'])[indices] if 'minchi2_gals' in res else None

    # Try to get SNR values
    snr_arr = None
    if snr_cache_path and Path(snr_cache_path).exists():
        cache = np.load(str(snr_cache_path))
        id_to_snr = dict(zip(cache['src_id'].astype(int), cache['snr_quad']))
        snr_arr = np.array([id_to_snr.get(int(s), np.nan) for s in src_ids])

    # ---- Build batch index for sample loading ----
    print("\n[2/4] Building batch sample index...")
    batch_index = build_batch_index(result_dir)

    # ---- Initialize PAE model ----
    print("\n[3/4] Initializing PAE model...")
    # Find run_params.npz in any task directory
    task_dirs = sorted(result_dir.glob('task*'),
                       key=lambda p: int(p.name.replace('task', '')))
    run_params_file = None
    for td in task_dirs:
        rp = td / 'run_params.npz'
        if rp.exists():
            run_params_file = rp
            break
    if run_params_file is None:
        print("Error: No run_params.npz found in any task directory")
        sys.exit(1)

    PAE_obj, wave_obs, run_params = init_pae_model(run_params_file)

    # Determine burn-in: explicit arg > run_params.npz value
    if args.burn_in is not None:
        burn_in = args.burn_in
    else:
        rp = np.load(str(run_params_file), allow_pickle=True)
        burn_in = int(rp['burn_in']) if 'burn_in' in rp.files else 200
        print(f"    burn_in = {burn_in} (from run_params.npz)")

    # ---- Generate per-source plots ----
    print(f"\n[4/4] Generating {len(indices)} source fit plots...")
    n_success = 0

    for i, (idx, sid) in enumerate(zip(indices, src_ids)):
        sid_int = int(sid)
        print(f"\n  [{i+1}/{len(indices)}] Source {sid_int} (combined idx={idx})")
        print(f"    z_true={z_true_arr[i]:.4f}  z_med={z_med_arr[i]:.4f}  "
              f"zscore={zscore_arr[i]:.2f}  chi2={chi2_full_arr[i]:.1f}")

        try:
            # Load MCMC samples
            samples = load_source_samples(batch_index, sid_int)
            print(f"    Loaded samples: {samples.shape}")

            # Load observed flux from parquet
            flux_fid, flux_err_fid, z_specz, native_data = load_source_flux_from_parquet(
                parquet_path, sid_int, load_native=args.native
            )
            print(f"    Loaded fiducial flux: {len(flux_fid)} bands")

            # Check band count matches
            if len(flux_fid) != len(wave_obs):
                print(f"    Warning: flux has {len(flux_fid)} bands but model expects {len(wave_obs)}")
                n_use = min(len(flux_fid), len(wave_obs))
                flux_fid = flux_fid[:n_use]
                flux_err_fid = flux_err_fid[:n_use]
                wave_obs_use = wave_obs[:n_use]
            else:
                wave_obs_use = wave_obs

            # Normalize (matching sampling pipeline)
            phot_norm = phot_norms_arr[i]
            spec_obs, weight, _ = normalize_source(flux_fid, flux_err_fid, phot_norm)

            # Reconstruct posterior spectra
            print(f"    Reconstructing posterior spectra...")
            recon_x, redshift_post, logL = reconstruct_spectrum(
                PAE_obj, samples, spec_obs, weight,
                burn_in=burn_in, thin_fac=args.thin_fac,
            )
            # Un-normalize to physical units
            recon_phys = np.array(recon_x) * phot_norm

            # Clip number of bands to match
            n_use = min(recon_phys.shape[1], len(wave_obs_use))
            recon_phys = recon_phys[:, :n_use]
            flux_phys = flux_fid[:n_use]
            flux_err_phys = flux_err_fid[:n_use]
            wave_plot = wave_obs_use[:n_use]

            # SNR for this source
            snr_val = snr_arr[i] if snr_arr is not None else None
            z_TF_val = z_TF_arr[i] if z_TF_arr is not None else None
            z_TF_err_val = z_TF_err_arr[i] if z_TF_err_arr is not None else None
            chisq_tf_val = chisq_tf_arr[i] if chisq_tf_arr is not None else None

            # Plot
            fig = plot_source_fit(
                wave_plot, flux_phys, flux_err_phys, recon_phys,
                redshift_post, z_true_arr[i], z_med_arr[i],
                err_low_arr[i], err_high_arr[i],
                chi2_full_arr[i], snr_val, sid_int,
                z_TF=z_TF_val,
                z_TF_err=z_TF_err_val,
                chisq_tf=chisq_tf_val,
                native_data=native_data,
                coadd_native=not args.native_raw,
                show_lines=args.show_lines,
                figsize=tuple(args.figsize),
                dpi=args.dpi,
            )

            # Save
            fname = outdir / f"fit_src{sid_int}_z{z_true_arr[i]:.3f}.png"
            fig.savefig(str(fname), dpi=args.dpi, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {fname.name}")
            n_success += 1

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"Done: {n_success}/{len(indices)} source plots saved to:")
    print(f"  {outdir}")
    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
