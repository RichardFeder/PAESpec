"""
plot_snr_diagnostics.py
-----------------------
Standalone diagnostic script: plots redshift estimation statistics and
sampling convergence properties as a function of source SNR.

This script loads completed PAE results (npz files) from a batched run
directory alongside the original parquet catalog to retrieve per-source SNR.
The match is done via Tractor_ID (saved as src_id in each results file).

Usage
-----
    python plot_snr_diagnostics.py --datestr multinode_validation_run_022126 \
        [--parquet /pscratch/sd/r/rmfeder/data/l3_data/full_validation_sz_0-1000.0_z_0-1000.0.parquet] \
        [--results-base /pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched] \
        [--outdir /path/to/figures] \
        [--n-snr-bins 20] \
        [--snr-col snr_per_filter_gals] \
        [--tasks 0 1 2 ...]    # default: all task dirs found

Arguments
---------
    --datestr      : name of the run (subdirectory under --results-base)
    --parquet      : path to the parquet catalog
    --results-base : base dir containing batched results; default is scratch
    --outdir       : directory to save figures; defaults to figures/snr_diagnostics/<datestr>
    --n-snr-bins   : number of log-spaced SNR bins (default 20)
    --snr-col      : parquet column to use as SNR (default: snr_per_filter_gals)
    --tasks        : restrict to specific task IDs; default all found

Outputs
-------
    snr_photoz_stats.png       -- NMAD, outlier fraction, bias, uncertainties, z-scores
    snr_convergence_stats.png  -- R-hat, chi2_reduced, autocorr_length, sampling coverage
    snr_sample_counts.png      -- source count per SNR bin, n_measurements_nonzero
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Helper: load all results files from a batched datestr directory
# ---------------------------------------------------------------------------

def load_all_results(results_dir: Path, task_ids=None, allow_pickle=True):
    """Load and concatenate all PAE_results_*.npz files from task subdirectories.

    Parameters
    ----------
    results_dir : Path
        Top-level directory containing task0/, task1/, ... subdirs
    task_ids : list[int] or None
        If given, only load results for these task IDs.
    allow_pickle : bool
        Passed to np.load (needed for object arrays like quality_thresholds).

    Returns
    -------
    dict  mapping array name → concatenated 1-D arrays across all batches.
    """
    # Discover task directories
    if task_ids is not None:
        task_dirs = [results_dir / f'task{i}' for i in task_ids]
        task_dirs = [d for d in task_dirs if d.is_dir()]
    else:
        task_dirs = sorted(
            [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('task')],
            key=lambda d: int(d.name.replace('task', ''))
        )

    if not task_dirs:
        raise FileNotFoundError(f"No task* directories found in {results_dir}")

    print(f"Found {len(task_dirs)} task directories: "
          f"{[d.name for d in task_dirs[:5]]}"
          f"{'...' if len(task_dirs) > 5 else ''}")

    # Scalar / 1-D array keys we want to concatenate
    scalar_keys = [
        'z_mean', 'z_med', 'ztrue', 'zscore',
        'z_TF', 'z_TF_err',
        'err_low', 'err_high',
        'chi2', 'chi2_reduced', 'chi2_full',
        'R_hat', 'autocorr_length',
        'n_measurements_nonzero', 'frac_sampled_102',
        'phot_norms', 'minchi2_gals',
        'data_idx', 'src_id',
    ]

    # Per-chain 2-D keys (shape n_galaxies × n_chains) -- take mean over chains
    chain_keys_mean = [
        'all_log_L', 'all_log_prior', 'all_log_redshift', 'all_mean_logL_per_chain',
        'preinit_final_logL',
    ]
    chain_keys_max = ['all_max_logL']

    arrays = {k: [] for k in scalar_keys + chain_keys_mean + chain_keys_max}
    n_loaded = 0
    n_skipped = 0

    for tdir in task_dirs:
        batch_files = sorted(tdir.glob('PAE_results_batch*.npz'))
        for fpath in batch_files:
            try:
                d = np.load(fpath, allow_pickle=True)
            except Exception as e:
                print(f"  WARNING: could not load {fpath.name}: {e}")
                n_skipped += 1
                continue

            available = set(d.keys())

            for k in scalar_keys:
                if k in available:
                    arr = d[k]
                    # Scalar 0-d arrays or short arrays
                    if arr.ndim == 0:
                        arrays[k].append(arr.reshape(1))
                    else:
                        arrays[k].append(arr)
                else:
                    # Fill with NaN placeholder; we'll handle missing later
                    pass

            for k in chain_keys_mean:
                if k in available:
                    arr = d[k]          # shape (n_gals, n_chains)
                    arrays[k].append(np.mean(arr, axis=-1))

            for k in chain_keys_max:
                if k in available:
                    arr = d[k]          # shape (n_gals,)
                    arrays[k].append(arr)

            n_loaded += 1

    print(f"Loaded {n_loaded} batch files, skipped {n_skipped}")

    # Concatenate
    combined = {}
    for k, lst in arrays.items():
        if lst:
            combined[k] = np.concatenate(lst)

    if 'z_mean' not in combined:
        raise ValueError("No z_mean data found; check results directory.")

    n_total = len(combined['z_mean'])
    print(f"Total sources loaded: {n_total:,}")

    return combined


# ---------------------------------------------------------------------------
# Helper: read SNR from parquet via Tractor_ID join
# ---------------------------------------------------------------------------

def load_snr_from_parquet(parquet_path: str, src_ids: np.ndarray, snr_col: str = 'snr_per_filter_gals'):
    """Read SNR values from parquet for the requested source IDs.

    Parameters
    ----------
    parquet_path : str
        Path to the parquet catalog.
    src_ids : np.ndarray, shape (N,)
        Tractor_IDs from the results files.
    snr_col : str
        Column name to use as SNR (scalar per source).

    Returns
    -------
    snr : np.ndarray, shape (N,)
        SNR values in the same order as src_ids.  NaN where not found.
    phot_chi2_snr : np.ndarray, shape (N,)
        phot_chi2_snr values (also available as a secondary SNR metric).
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow is required; activate the jax-env conda environment.")

    # The pipeline stores SPHERExRefID as 'Tractor_ID' for compatibility;
    # src_id in results files == SPHERExRefID in the parquet.
    id_col = 'SPHERExRefID'

    print(f"\nReading SNR from parquet: {Path(parquet_path).name}")
    print(f"  Columns: {id_col}, {snr_col}, phot_chi2_snr")

    pf = pq.ParquetFile(parquet_path)
    cols = [id_col, snr_col]
    if snr_col != 'phot_chi2_snr':
        cols.append('phot_chi2_snr')

    # Read in batches to avoid memory issues
    tractor_ids_all = []
    snr_all = []
    chi2snr_all = []

    for batch in pf.iter_batches(batch_size=50_000, columns=cols):
        tractor_ids_all.append(np.array(batch.column(id_col).to_pylist(), dtype=np.int64))
        snr_all.append(np.array(batch.column(snr_col).to_pylist(), dtype=np.float64))
        if snr_col != 'phot_chi2_snr' and 'phot_chi2_snr' in cols:
            chi2snr_all.append(np.array(batch.column('phot_chi2_snr').to_pylist(), dtype=np.float64))

    parquet_tractor_ids = np.concatenate(tractor_ids_all)
    parquet_snr = np.concatenate(snr_all)
    if chi2snr_all:
        parquet_chi2snr = np.concatenate(chi2snr_all)
    else:
        parquet_chi2snr = parquet_snr.copy()

    print(f"  Parquet: {len(parquet_tractor_ids):,} rows, "
          f"SNR range [{np.nanmin(parquet_snr):.3f}, {np.nanmax(parquet_snr):.3f}]")

    # Build lookup
    id_to_snr = dict(zip(parquet_tractor_ids, parquet_snr))
    id_to_chi2snr = dict(zip(parquet_tractor_ids, parquet_chi2snr))

    snr_out = np.array([id_to_snr.get(sid, np.nan) for sid in src_ids])
    chi2snr_out = np.array([id_to_chi2snr.get(sid, np.nan) for sid in src_ids])

    n_matched = np.sum(np.isfinite(snr_out))
    print(f"  Matched {n_matched:,}/{len(src_ids):,} sources ({100*n_matched/len(src_ids):.1f}%)")

    return snr_out, chi2snr_out


# ---------------------------------------------------------------------------
# Compute observed broadband SNR from flux arrays
# ---------------------------------------------------------------------------

def compute_observed_snr_from_parquet(parquet_path: str, src_ids: np.ndarray,
                                      batch_size: int = 50_000):
    """Compute per-source broadband SNR metrics from flux/flux_err in the parquet.

    Only bands with flux > 0 AND 0 < flux_err < 5e4 are included in every metric.
    flux_err == 5e4 is the sentinel value for missing/non-detected measurements.

    Metrics
    -------
    snr_quad   : sqrt(sum((F/sigma)^2))  -- combined detection significance;
                  grows with the number of valid bands, best overall proxy
    snr_median : median(F/sigma)         -- typical per-band SNR
    snr_peak   : max(F/sigma)            -- brightest single-band SNR
    n_valid    : int, number of bands satisfying flux>0 & 0<flux_err<5e4

    Parameters
    ----------
    parquet_path : str
    src_ids : np.ndarray, shape (N,)  -- SPHERExRefID values from results files
    batch_size : int

    Returns
    -------
    dict with keys: snr_quad, snr_median, snr_peak, n_valid  (all length-N arrays)
    """
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError("pyarrow required; activate the jax-env conda environment.")

    id_col   = 'SPHERExRefID'
    flux_col = 'flux_dered_fiducial'
    ferr_col = 'flux_err_dered_fiducial'

    print(f"\nComputing observed broadband SNR from flux/flux_err:")
    print(f"  Parquet : {Path(parquet_path).name}")
    print(f"  Sources : {len(src_ids):,} to match")
    print(f"  Metrics : snr_quad=sqrt(sum((F/sigma)^2)),  "
          f"snr_median=median(F/sigma),  snr_peak=max(F/sigma)")

    src_id_set = set(int(s) for s in src_ids)

    id_to_quad   = {}
    id_to_median = {}
    id_to_peak   = {}
    id_to_nvalid = {}

    pf = pq.ParquetFile(parquet_path)
    n_processed = 0

    for batch in pf.iter_batches(batch_size=batch_size, columns=[id_col, flux_col, ferr_col]):
        ids_batch = np.array(batch.column(id_col).to_pylist(), dtype=np.int64)
        hit = np.array([int(sid) in src_id_set for sid in ids_batch])
        if not np.any(hit):
            continue

        # Convert list-of-lists to 2-D arrays; only rows we need
        flux_sub = np.array([batch.column(flux_col)[i].as_py()
                             for i in np.where(hit)[0]], dtype=np.float64)
        ferr_sub = np.array([batch.column(ferr_col)[i].as_py()
                             for i in np.where(hit)[0]], dtype=np.float64)
        ids_sub  = ids_batch[hit]

        for i, sid in enumerate(ids_sub):
            f = flux_sub[i]
            s = ferr_sub[i]
            valid = (f > 0) & (s > 0) & (s < 5e4)  # 5e4 = sentinel for missing bands
            n_v = int(np.sum(valid))
            if n_v == 0:
                id_to_quad[sid]   = np.nan
                id_to_median[sid] = np.nan
                id_to_peak[sid]   = np.nan
            else:
                snr_bands = f[valid] / s[valid]
                id_to_quad[sid]   = float(np.sqrt(np.sum(snr_bands ** 2)))
                id_to_median[sid] = float(np.median(snr_bands))
                id_to_peak[sid]   = float(np.max(snr_bands))
            id_to_nvalid[sid] = n_v
        n_processed += 1

    print(f"  Processed {n_processed} batches, matched all {len(id_to_quad):,} sources")

    snr_quad   = np.array([id_to_quad.get(int(s),   np.nan) for s in src_ids])
    snr_median = np.array([id_to_median.get(int(s), np.nan) for s in src_ids])
    snr_peak   = np.array([id_to_peak.get(int(s),   np.nan) for s in src_ids])
    n_valid    = np.array([id_to_nvalid.get(int(s),  0)     for s in src_ids], dtype=int)

    for name, arr in [('snr_quad', snr_quad), ('snr_median', snr_median), ('snr_peak', snr_peak)]:
        fin = arr[np.isfinite(arr) & (arr > 0)]
        if len(fin):
            print(f"  {name:12s}: median={np.median(fin):.2f}, "
                  f"p10={np.percentile(fin, 10):.2f}, "
                  f"p90={np.percentile(fin, 90):.2f}, "
                  f"p99={np.percentile(fin, 99):.2f}")

    return {'snr_quad': snr_quad, 'snr_median': snr_median,
            'snr_peak': snr_peak, 'n_valid': n_valid}


# ---------------------------------------------------------------------------
# Binning utilities
# ---------------------------------------------------------------------------

def make_snr_bins(snr: np.ndarray, n_bins: int = 20, snr_min: float = None, snr_max: float = None):
    """Create log-spaced SNR bin edges from finite, positive values."""
    valid_snr = snr[np.isfinite(snr) & (snr > 0)]
    lo = snr_min if snr_min is not None else max(np.percentile(valid_snr, 1), 1e-3)
    hi = snr_max if snr_max is not None else np.percentile(valid_snr, 99)
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return edges, centers


def bin_stats(snr: np.ndarray, values: np.ndarray, edges: np.ndarray,
              stat_fn=np.nanmedian, min_count: int = 10):
    """Compute binned statistics along with counts and percentile bands.

    Returns
    -------
    medians, p16, p84, counts
    """
    medians = np.full(len(edges) - 1, np.nan)
    p16 = np.full(len(edges) - 1, np.nan)
    p84 = np.full(len(edges) - 1, np.nan)
    counts = np.zeros(len(edges) - 1, dtype=int)

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (snr >= lo) & (snr < hi) & np.isfinite(values)
        counts[i] = np.sum(mask)
        if counts[i] >= min_count:
            v = values[mask]
            medians[i] = stat_fn(v)
            p16[i] = np.nanpercentile(v, 16)
            p84[i] = np.nanpercentile(v, 84)

    return medians, p16, p84, counts


def compute_nmad(delta_z):
    """NMAD of Δz/(1+z) values."""
    return 1.4826 * np.nanmedian(np.abs(delta_z - np.nanmedian(delta_z)))


def bin_nmad(snr, z_mean, z_true, edges, min_count=10):
    dz = (z_mean - z_true) / (1.0 + z_true)
    nmad_arr = np.full(len(edges) - 1, np.nan)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (snr >= lo) & (snr < hi) & np.isfinite(dz)
        counts[i] = np.sum(mask)
        if counts[i] >= min_count:
            nmad_arr[i] = compute_nmad(dz[mask])
    return nmad_arr, counts


def bin_outlier_frac(snr, z_mean, z_true, edges, threshold=0.06, min_count=10):
    dz = np.abs(z_mean - z_true) / (1.0 + z_true)
    frac_arr = np.full(len(edges) - 1, np.nan)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (snr >= lo) & (snr < hi) & np.isfinite(dz)
        counts[i] = np.sum(mask)
        if counts[i] >= min_count:
            frac_arr[i] = np.mean(dz[mask] > threshold)
    return frac_arr, counts


def bin_outlier_3sigma(snr, z_mean, z_true, err_low, err_high, edges, n_sigma=3.0, min_count=10):
    """η_{3σ}: fraction of sources where |Δz/(1+z)| > n_sigma * σ_{Δz/(1+z)}.

    σ_{Δz/(1+z)} = 0.5*(err_low + err_high) / (1 + z_true) using the reported
    posterior half-width as a 1σ proxy.
    """
    dz_norm = np.abs(z_mean - z_true) / (1.0 + z_true)
    sigma_z = 0.5 * (err_low + err_high) / (1.0 + z_true)
    sigma_z = np.maximum(sigma_z, 1e-6)   # guard against zero
    n_sigma_obs = dz_norm / sigma_z

    frac_arr = np.full(len(edges) - 1, np.nan)
    counts    = np.zeros(len(edges) - 1, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (snr >= lo) & (snr < hi) & np.isfinite(n_sigma_obs)
        counts[i] = np.sum(mask)
        if counts[i] >= min_count:
            frac_arr[i] = np.mean(n_sigma_obs[mask] > n_sigma)
    return frac_arr, counts


def bin_outlier_welldetermined(snr, z_mean, z_true, err_low, err_high, edges,
                               threshold=0.15, sigma_cut=0.05, min_count=10):
    """η_{threshold%, σ<σ_cut}: catastrophic outlier rate for well-constrained sources.

    Selects sources with σ_{Δz/(1+z)} = 0.5*(err_low+err_high)/(1+z_true) < sigma_cut,
    then computes the fraction where |Δz/(1+z)| > threshold.

    Returns
    -------
    frac_arr : outlier fraction per bin (NaN if fewer than min_count sources pass the σ cut)
    counts_all : total sources per bin (denominator before σ cut)
    counts_sel : sources per bin passing the σ cut
    """
    dz_norm = np.abs(z_mean - z_true) / (1.0 + z_true)
    sigma_z = 0.5 * (err_low + err_high) / (1.0 + z_true)
    well_determined = sigma_z < sigma_cut

    frac_arr   = np.full(len(edges) - 1, np.nan)
    counts_all = np.zeros(len(edges) - 1, dtype=int)
    counts_sel = np.zeros(len(edges) - 1, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (snr >= lo) & (snr < hi)
        counts_all[i] = np.sum(in_bin & np.isfinite(dz_norm))
        sel = in_bin & well_determined & np.isfinite(dz_norm)
        counts_sel[i] = np.sum(sel)
        if counts_sel[i] >= min_count:
            frac_arr[i] = np.mean(dz_norm[sel] > threshold)
    return frac_arr, counts_all, counts_sel


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

RHAT_THRESH = 1.1
CHI2_THRESH = 2.0
OUTLIER_THRESH = 0.06


def _plot_binned(ax, centers, medians, p16=None, p84=None, counts=None,
                 color='steelblue', label=None, fill_alpha=0.2,
                 min_count=10):
    """Plot median line + shaded 16-84th percentile band, masking low-count bins."""
    mask = (counts >= min_count) if counts is not None else np.ones(len(centers), dtype=bool)
    x = centers[mask]
    y = medians[mask]

    line, = ax.plot(x, y, 'o-', color=color, lw=1.5, ms=4, label=label)

    if p16 is not None and p84 is not None:
        ax.fill_between(x, p16[mask], p84[mask], color=color, alpha=fill_alpha)

    return line


def plot_photoz_stats(results: dict, snr: np.ndarray, edges: np.ndarray, centers: np.ndarray,
                      outdir: Path):
    """Five standalone photo-z performance figures vs SNR (no dual y-axes).

    Outputs saved to outdir:
      snr_nmad.png         -- NMAD of delta-z/(1+z)
      snr_eta3sigma.png    -- eta_{3sigma} outlier fraction
      snr_eta15pct.png     -- eta_{15%, sigma<5%} catastrophic fraction
      snr_bias.png         -- Median bias delta-z/(1+z) with 16-84th pct band
      snr_zscore_rms.png   -- z-score RMS
    """
    z_mean = results['z_mean']
    z_true = results.get('ztrue')
    err_low = results.get('err_low')
    err_high = results.get('err_high')
    zscore = results.get('zscore')
    has_unc = err_low is not None and err_high is not None
    has_ztrue = z_true is not None and np.any(np.isfinite(z_true))

    colors = plt.cm.tab10.colors
    FS = (5, 4)

    def _save(fig, name):
        p = outdir / name
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {name}')

    def _add_count_twin(ax, snr, edges, label='N sources', color='gray'):
        """Add twin y-axis with source count histogram overlay."""
        counts_per_bin = np.zeros(len(edges) - 1, dtype=int)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            counts_per_bin[i] = np.sum((snr >= lo) & (snr < hi) & np.isfinite(snr))
        ax2 = ax.twinx()
        ax2.fill_between(centers, 0, counts_per_bin, color=color, alpha=0.12, step='mid')
        ax2.set_ylabel(label, fontsize=8, color=color)
        ax2.tick_params(axis='y', labelsize=7, labelcolor=color)
        ax2.set_yscale('log')
        return ax2

    # --- NMAD ---
    fig, ax = plt.subplots(figsize=FS)
    if has_ztrue:
        nmad_vals, counts = bin_nmad(snr, z_mean, z_true, edges)
        mask = counts >= 10
        ax.plot(centers[mask], nmad_vals[mask], 'o-', color=colors[0], lw=1.5, ms=4)
        ax.axhline(0.05, ls='--', color='gray', alpha=0.6, label='0.05')
        ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('SNR')
    ax.set_ylabel(r'$\sigma_\mathrm{NMAD}$')
    ax.set_title('NMAD vs SNR')
    _add_count_twin(ax, snr, edges)
    _save(fig, 'snr_nmad.png')

    # --- eta_{3sigma} ---
    fig, ax = plt.subplots(figsize=FS)
    if has_ztrue and has_unc:
        eta3s, counts = bin_outlier_3sigma(snr, z_mean, z_true, err_low, err_high, edges,
                                           n_sigma=3.0)
        mask = counts >= 10
        ax.plot(centers[mask], eta3s[mask], 'o-', color=colors[1], lw=1.5, ms=4)
        ax.axhline(0.003, ls='--', color='gray', alpha=0.6, label='0.3%')
        ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('SNR')
    ax.set_ylabel(r'$\eta_{3\sigma}$')
    ax.set_title(r'$\eta_{3\sigma}$: frac with $|\Delta z/(1+z)| > 3\,\sigma_z$ vs SNR')
    _add_count_twin(ax, snr, edges)
    _save(fig, 'snr_eta3sigma.png')

    # --- eta_{15%, sigma<5%} ---
    SIGMA_CUT = 0.05
    OUTLIER_WD = 0.15
    fig, ax = plt.subplots(figsize=FS)
    if has_ztrue and has_unc:
        eta_wd, counts_all, counts_sel = bin_outlier_welldetermined(
            snr, z_mean, z_true, err_low, err_high, edges,
            threshold=OUTLIER_WD, sigma_cut=SIGMA_CUT)
        mask_sel = counts_sel >= 10
        ax.plot(centers[mask_sel], eta_wd[mask_sel], 'o-', color=colors[2], lw=1.5, ms=4)
        ax.axhline(0.05, ls='--', color='gray', alpha=0.6, label='5%')
        ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('SNR')
    ax.set_ylabel(r'$\eta_{15\%%}\;[\sigma_z < %.0f\%%]$' % (SIGMA_CUT * 100))
    ax.set_title(r'$\eta_{15\%%}$ for $\sigma_z < %.0f\%%$ vs SNR' % (SIGMA_CUT * 100))
    _add_count_twin(ax, snr, edges)
    _save(fig, 'snr_eta15pct.png')

    # --- Mean bias ---
    fig, ax = plt.subplots(figsize=FS)
    if has_ztrue:
        dz_norm = (z_mean - z_true) / (1.0 + z_true)
        med, p16, p84, counts = bin_stats(snr, dz_norm, edges)
        _plot_binned(ax, centers, med, p16, p84, counts, color=colors[3], label='median')
        ax.axhline(0, ls='--', color='gray', alpha=0.6)
        ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('SNR')
    ax.set_ylabel(r'$\langle\Delta z/(1+z)\rangle$')
    ax.set_title('Mean Bias (median ± 16–84th pct) vs SNR')
    _add_count_twin(ax, snr, edges)
    _save(fig, 'snr_bias.png')

    # --- z-score RMS ---
    fig, ax = plt.subplots(figsize=FS)
    if zscore is not None:
        zs_rms = np.full(len(edges) - 1, np.nan)
        counts_bin = np.zeros(len(edges) - 1, dtype=int)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (snr >= lo) & (snr < hi) & np.isfinite(zscore)
            counts_bin[i] = np.sum(mask)
            if counts_bin[i] >= 10:
                zs_rms[i] = np.sqrt(np.mean(zscore[mask] ** 2))
        valid = counts_bin >= 10
        ax.plot(centers[valid], zs_rms[valid], 'o-', color=colors[6], lw=1.5, ms=4)
        ax.axhline(1.0, ls='--', color='gray', alpha=0.6, label='ideal=1')
        ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('SNR')
    ax.set_ylabel('z-score RMS')
    ax.set_title('z-score RMS (ideal=1) vs SNR')
    _add_count_twin(ax, snr, edges)
    _save(fig, 'snr_zscore_rms.png')

def plot_convergence_stats(results: dict, snr: np.ndarray, edges: np.ndarray, centers: np.ndarray,
                           outdir: Path, obs_snr: dict = None, snr_label: str = 'SNR'):
    """Four standalone convergence-diagnostic figures vs SNR.

    Outputs saved to outdir:
      snr_rhat_median.png    -- Median R-hat (clipped at 10) with 16–84th pct band
      snr_rhat_fraction.png  -- Non-converged fraction (R-hat > threshold)
      snr_autocorr.png       -- Median autocorrelation length with 16–84th pct band
      snr_logL.png           -- Mean log-likelihood per chain with 16–84th pct band
    """
    R_hat = results.get('R_hat')
    autocorr_length = results.get('autocorr_length')
    log_L = results.get('all_log_L')          # already mean over chains

    colors = plt.cm.tab10.colors
    FS = (5, 4)

    def _save(fig, name):
        p = outdir / name
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {name}')

    def _add_count_twin(ax, snr, edges, label='N sources', color='gray'):
        """Add twin y-axis with source count histogram overlay."""
        counts_per_bin = np.zeros(len(edges) - 1, dtype=int)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            counts_per_bin[i] = np.sum((snr >= lo) & (snr < hi) & np.isfinite(snr))
        ax2 = ax.twinx()
        ax2.fill_between(centers, 0, counts_per_bin, color=color, alpha=0.12, step='mid')
        ax2.set_ylabel(label, fontsize=8, color=color)
        ax2.tick_params(axis='y', labelsize=7, labelcolor=color)
        ax2.set_yscale('log')
        return ax2

    # --- Median R-hat ---
    fig, ax = plt.subplots(figsize=FS)
    if R_hat is not None:
        R_hat_clipped = np.where(np.isfinite(R_hat), np.minimum(R_hat, 10.0), np.nan)
        med, p16, p84, counts = bin_stats(snr, R_hat_clipped, edges)
        _plot_binned(ax, centers, med, p16, p84, counts, color=colors[0])
        ax.axhline(RHAT_THRESH, ls='--', color='tomato', alpha=0.8,
                   label=f'threshold={RHAT_THRESH}')
        ax.axhline(1.0, ls='--', color='gray', alpha=0.5, label='ideal=1')
        ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel(snr_label)
    ax.set_ylabel(r'$\hat{R}$')
    ax.set_ylim(0.5, 6)
    ax.set_title(r'Median $\hat{R}$ (\u00b1 16\u201384th pct) vs ' + snr_label)
    _add_count_twin(ax, snr, edges)
    _save(fig, 'snr_rhat_median.png')

    # --- Non-converged fraction ---
    fig, ax = plt.subplots(figsize=FS)
    if R_hat is not None:
        frac_bad = np.full(len(edges) - 1, np.nan)
        counts_bin = np.zeros(len(edges) - 1, dtype=int)
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (snr >= lo) & (snr < hi) & np.isfinite(R_hat)
            counts_bin[i] = np.sum(mask)
            if counts_bin[i] >= 10:
                frac_bad[i] = np.mean(R_hat[mask] > RHAT_THRESH)
        valid = counts_bin >= 10
        ax.plot(centers[valid], frac_bad[valid], 'o-', color=colors[1], lw=1.5, ms=4)
        ax.axhline(0.1, ls='--', color='gray', alpha=0.6, label='10%')
        ax.legend(fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel(snr_label)
    ax.set_ylabel(r'Fraction with $\hat{R} > ' + str(RHAT_THRESH) + '$')
    ax.set_title(f'Non-converged Fraction ($\\hat{{R}} > {RHAT_THRESH}$) vs ' + snr_label)
    _add_count_twin(ax, snr, edges)
    _save(fig, 'snr_rhat_fraction.png')

    # --- Median autocorrelation length ---
    fig, ax = plt.subplots(figsize=FS)
    if autocorr_length is not None:
        valid_ac = np.isfinite(autocorr_length) & (autocorr_length > 0)
        med, p16, p84, counts = bin_stats(snr, np.where(valid_ac, autocorr_length, np.nan), edges)
        _plot_binned(ax, centers, med, p16, p84, counts, color=colors[3])
    ax.set_xscale('log')
    ax.set_xlabel(snr_label)
    ax.set_ylabel('Autocorrelation length')
    ax.set_title('Median Autocorrelation Length (\u00b1 16\u201384th pct) vs ' + snr_label)
    _add_count_twin(ax, snr, edges)
    _save(fig, 'snr_autocorr.png')

    # --- Mean log-likelihood per chain ---
    fig, ax = plt.subplots(figsize=FS)
    if log_L is not None:
        med, p16, p84, counts = bin_stats(snr, log_L, edges)
        _plot_binned(ax, centers, med, p16, p84, counts, color=colors[5])
    ax.set_xscale('log')
    ax.set_xlabel(snr_label)
    ax.set_ylabel('Mean log-likelihood')
    ax.set_ylim(-800, 0)
    ax.set_title('Mean log-likelihood per chain (\u00b1 16\u201384th pct) vs ' + snr_label)
    _add_count_twin(ax, snr, edges)
    _save(fig, 'snr_logL.png')


def plot_sample_counts(results: dict, snr: np.ndarray, edges: np.ndarray, centers: np.ndarray,
                       outpath: Path, snr_col_name: str, obs_snr: dict = None,
                       use_hexbin: bool = True):
    """Figure 3: Source counts, data coverage, and cross-diagnostics.

    When use_hexbin=True, point-cloud panels are rendered as log-scaled hexbin
    density maps instead of scatter plots:  plasma cmap for z-space panels,
    viridis for convergence/chi2 panels.
    """
    frac_102 = results.get('frac_sampled_102')
    R_hat = results.get('R_hat')
    chi2_reduced = results.get('chi2_reduced')
    z_mean = results['z_mean']
    z_true = results.get('ztrue')
    has_ztrue = z_true is not None and np.any(np.isfinite(z_true))
    # n_valid comes from obs_snr computation (bands with F>0 and sigma>0)
    n_valid = obs_snr['n_valid'] if obs_snr is not None else None

    fig = plt.figure(figsize=(24, 10))
    fig.suptitle('Data Coverage & Cross-diagnostics vs Source SNR', fontsize=16, y=0.98)
    gs = GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.32)
    axs = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]

    colors = plt.cm.tab10.colors

    # Count per bin
    counts_per_bin = np.zeros(len(edges) - 1, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        counts_per_bin[i] = np.sum((snr >= lo) & (snr < hi) & np.isfinite(snr))

    # --- Panel 0: Source count histogram ---
    ax = axs[0]
    ax.bar(centers, counts_per_bin, width=np.diff(edges), color=colors[0],
           alpha=0.7, align='center', log=True)
    ax.set_xscale('log')
    ax.set_xlabel(snr_col_name)
    ax.set_ylabel('N sources')
    ax.set_title('Source Count per SNR Bin')

    # --- Panel 1: Valid bands (flux>0 & flux_err>0), from obs SNR computation ---
    ax = axs[1]
    if n_valid is not None:
        med, p16, p84, counts = bin_stats(snr, n_valid.astype(float), edges)
        _plot_binned(ax, centers, med, p16, p84, counts, color=colors[1])
        ax.set_title('Valid Bands per Source (F>0 & σ>0, median, 16-84th pct)')
    else:
        ax.text(0.5, 0.5, 'Run with --obs-snr to show\nn_valid_bands',
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
        ax.set_title('Valid Bands (not computed)')
    ax.set_xscale('log')
    ax.set_xlabel(snr_col_name)
    ax.set_ylabel('n_valid_bands')

    # --- Panel 2: frac_sampled_102 ---
    ax = axs[2]
    if frac_102 is not None:
        med, p16, p84, counts = bin_stats(snr, frac_102, edges)
        _plot_binned(ax, centers, med, p16, p84, counts, color=colors[2])
        ax.axhline(1.0, ls='--', color='gray', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel(snr_col_name)
    ax.set_ylabel('frac_sampled_102')
    ax.set_title('Fraction Sampled (102 bands, median, 16-84th pct)')

    # --- Panel 3: |Δz/(1+z)| vs SNR ---
    ax = axs[3]
    if has_ztrue:
        dz = (z_mean - z_true) / (1.0 + z_true)
        valid = np.isfinite(dz) & np.isfinite(snr) & (snr > 0) & (np.abs(dz) > 0)
        if np.any(valid):
            if use_hexbin:
                hb = ax.hexbin(snr[valid], np.abs(dz[valid]),
                               xscale='log', yscale='log',
                               bins='log', cmap='plasma',
                               mincnt=1, gridsize=50)
                plt.colorbar(hb, ax=ax, label='log10(count)')
            else:
                idx = np.where(valid)[0]
                if len(idx) > 15000:
                    idx = np.random.default_rng(42).choice(idx, 15000, replace=False)
                ax.scatter(snr[idx], np.abs(dz[idx]), s=3, alpha=0.3, color='steelblue')
                ax.set_xscale('log'); ax.set_yscale('log')
            ax.axhline(OUTLIER_THRESH, ls='--', color='tomato', alpha=0.8,
                       label=f'outlier={OUTLIER_THRESH}')
            ax.set_xscale('log')
            ax.set_yscale('log')
    ax.set_xlabel(snr_col_name)
    ax.set_ylabel(r'$|\Delta z/(1+z)|$')
    ax.set_title(r'$|\Delta z/(1+z)|$ vs SNR' + (' (hexbin)' if use_hexbin else ' (scatter)'))
    ax.legend(fontsize=8)

    # --- Panel 4: R_hat vs SNR ---
    ax = axs[4]
    if R_hat is not None:
        valid = np.isfinite(R_hat) & np.isfinite(snr) & (snr > 0)
        if np.any(valid):
            if use_hexbin:
                hb = ax.hexbin(snr[valid], np.minimum(R_hat[valid], 10.0),
                               xscale='log',
                               bins='log', cmap='viridis',
                               mincnt=1, gridsize=50)
                plt.colorbar(hb, ax=ax, label='log10(count)')
            else:
                idx = np.where(valid)[0]
                if len(idx) > 15000:
                    idx = np.random.default_rng(42).choice(idx, 15000, replace=False)
                ax.scatter(snr[idx], np.minimum(R_hat[idx], 10.0), s=3, alpha=0.3, color='steelblue')
            ax.axhline(RHAT_THRESH, ls='--', color='tomato', alpha=0.8,
                       label=f'R-hat={RHAT_THRESH}')
    ax.set_xscale('log')
    ax.set_ylim(0, 10)
    ax.set_xlabel(snr_col_name)
    ax.set_ylabel(r'$\hat{R}$ (clipped at 10)')
    ax.set_title(r'R-hat vs SNR' + (' (hexbin)' if use_hexbin else ' (scatter)'))
    ax.legend(fontsize=8)

    # --- Panel 5: chi2_reduced distribution by SNR quartile ---
    ax = axs[5]
    if chi2_reduced is not None:
        valid_snr = np.isfinite(snr) & (snr > 0)
        quartiles = np.percentile(snr[valid_snr], [25, 50, 75])
        quartile_edges = [snr[valid_snr].min(), quartiles[0], quartiles[1], quartiles[2],
                          snr[valid_snr].max()]
        q_labels = ['Q1 (low SNR)', 'Q2', 'Q3', 'Q4 (high SNR)']
        q_colors = [colors[6], colors[7], colors[8], colors[9 % len(colors)]]
        chi2_plot_bins = np.linspace(0, 5, 50)
        for qi in range(4):
                mask = ((snr >= quartile_edges[qi]) & (snr < quartile_edges[qi + 1])
                        & np.isfinite(chi2_reduced))
                if np.sum(mask) > 10:
                    ax.hist(chi2_reduced[mask], bins=chi2_plot_bins, density=True,
                            alpha=0.4, color=q_colors[qi], label=q_labels[qi])
        ax.axvline(1.0, ls='--', color='gray', alpha=0.6, label='ideal=1')
        ax.axvline(CHI2_THRESH, ls='--', color='tomato', alpha=0.6, label=f'thresh={CHI2_THRESH}')
        ax.legend(fontsize=8)
    ax.set_xlabel(r'$\chi^2_\nu$')
    ax.set_ylabel('Density')
    ax.set_title(r'$\chi^2_\nu$ Distribution by SNR Quartile')

    # --- Panel 6: z_phot vs z_spec hexbin ---
    ax = axs[6]
    if has_ztrue:
        valid = np.isfinite(z_true) & np.isfinite(z_mean) & (z_true > 0) & (z_mean > 0)
        if np.any(valid):
            z_max_plot = np.percentile(z_true[valid], 99.5)
            if use_hexbin:
                hb = ax.hexbin(z_true[valid], z_mean[valid],
                               bins='log', cmap='plasma',
                               mincnt=1, gridsize=60,
                               extent=[0, z_max_plot, 0, z_max_plot])
                plt.colorbar(hb, ax=ax, label='log10(count)')
            else:
                idx = np.where(valid)[0]
                if len(idx) > 20000:
                    idx = np.random.default_rng(42).choice(idx, 20000, replace=False)
                ax.scatter(z_true[idx], z_mean[idx], s=2, alpha=0.2, color='steelblue')
            z_line = np.linspace(0, z_max_plot, 100)
            ax.plot(z_line, z_line, 'w--', lw=1.2, alpha=0.8, label='1:1')
            ax.set_xlim(0, z_max_plot)
            ax.set_ylim(0, z_max_plot)
        ax.set_xlabel(r'$z_\mathrm{spec}$')
        ax.set_ylabel(r'$z_\mathrm{phot}$')
        ax.set_title(r'$z_\mathrm{phot}$ vs $z_\mathrm{spec}$'
                     + (' (hexbin)' if use_hexbin else ' (scatter)'))
        ax.legend(fontsize=8)
    else:
        ax.set_visible(False)

    # Hide unused panel 7
    axs[7].set_visible(False)

    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {outpath.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datestr', required=True,
                   help='Run directory name (e.g. multinode_validation_run_022126)')
    p.add_argument('--parquet',
                   default='/pscratch/sd/r/rmfeder/data/l3_data/'
                           'full_validation_sz_0-1000.0_z_0-1000.0.parquet',
                   help='Path to parquet catalog')
    p.add_argument('--results-base',
                   default='/pscratch/sd/r/rmfeder/data/pae_sample_results/MCLMC/batched',
                   help='Base directory containing batched results')
    p.add_argument('--outdir', default=None,
                   help='Output directory for figures (default: figures/snr_diagnostics/<datestr>)')
    p.add_argument('--n-snr-bins', type=int, default=20,
                   help='Number of log-spaced SNR bins')
    p.add_argument('--tasks', type=int, nargs='+', default=None,
                   help='Restrict to specific task IDs (default: all)')
    p.add_argument('--snr-min', type=float, default=None,
                   help='Minimum SNR for binning (default: 1st percentile of chosen metric)')
    p.add_argument('--snr-max', type=float, default=None,
                   help='Maximum SNR for binning (default: 99th percentile of chosen metric)')
    # --- SNR source options ---
    snr_grp = p.add_argument_group('SNR source (mutually exclusive options)')
    snr_grp.add_argument('--obs-snr', action='store_true',
                         help='Compute SNR from flux/flux_err in the parquet '
                              '(snr_quad, snr_median, snr_peak). '
                              'Slower first run; combine with --save-snr-cache.')
    snr_grp.add_argument('--snr-metric',
                         choices=['snr_quad', 'snr_median', 'snr_peak', 'catalog'],
                         default='catalog',
                         help='Which SNR metric to use as the x-axis for binning. '
                              '"catalog" uses --snr-col from the parquet scalar column. '
                              'The other options require --obs-snr or --snr-cache. '
                              '(default: catalog)')
    snr_grp.add_argument('--snr-col', default='snr_per_filter_gals',
                         help='Parquet scalar column for catalog SNR '
                              '(used when --snr-metric=catalog, default: snr_per_filter_gals)')
    snr_grp.add_argument('--snr-cache', default=None,
                         help='Path to pre-computed SNR cache npz '
                              '(from a previous run with --save-snr-cache). '
                              'Skips recomputation.')
    snr_grp.add_argument('--save-snr-cache', default=None,
                         help='Save computed obs SNR to this npz path for future reuse.')
    p.add_argument('--hexbin', action='store_true', default=True,
                   help='Use log-scaled hexbin density maps instead of scatter plots '
                        '(plasma for z-space, viridis for convergence). Default: True.')
    p.add_argument('--no-hexbin', dest='hexbin', action='store_false',
                   help='Use scatter plots instead of hexbins.')
    return p.parse_args()


def main():
    args = parse_args()

    # Validate SNR argument combination
    if args.snr_metric != 'catalog' and not args.obs_snr and args.snr_cache is None:
        sys.exit("ERROR: --snr-metric requires either --obs-snr or --snr-cache.")

    results_dir = Path(args.results_base) / args.datestr
    if not results_dir.is_dir():
        sys.exit(f"ERROR: Results directory not found: {results_dir}")

    # Output directory
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        repo_root = Path(__file__).resolve().parent.parent
        outdir = repo_root / 'figures' / 'snr_diagnostics' / args.datestr
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}")

    # 1. Load results
    print(f"\n{'='*60}")
    print(f"Loading results from: {results_dir}")
    results = load_all_results(results_dir, task_ids=args.tasks)

    if 'src_id' not in results:
        sys.exit("ERROR: src_id not found in results files; cannot match to parquet.")

    src_ids = results['src_id']

    # 2. Obtain SNR
    obs_snr = None  # dict with snr_quad/snr_median/snr_peak/n_valid, or None

    if args.snr_metric == 'catalog':
        # Fast path: read scalar SNR column from parquet
        snr, _ = load_snr_from_parquet(args.parquet, src_ids, args.snr_col)
        snr_label = args.snr_col
    else:
        # Observed broadband SNR from flux arrays
        if args.snr_cache is not None:
            cache_path = Path(args.snr_cache)
            if not cache_path.exists():
                sys.exit(f"ERROR: SNR cache not found: {cache_path}")
            print(f"\nLoading SNR cache: {cache_path}")
            cache = np.load(cache_path)
            # Be tolerant of older caches that may only contain 'snr_quad' + 'src_id'
            obs_snr = {}
            if 'snr_quad' in cache:
                obs_snr['snr_quad'] = np.array(cache['snr_quad'])
            else:
                sys.exit(f"ERROR: snr_quad missing from cache: {cache_path}")

            # Derive or load optional statistics
            if 'snr_median' in cache:
                obs_snr['snr_median'] = np.array(cache['snr_median'])
            else:
                # fallback: median across channels per source if snr_quad is 2D, else global median
                try:
                    if obs_snr['snr_quad'].ndim > 1:
                        obs_snr['snr_median'] = np.nanmedian(obs_snr['snr_quad'], axis=1)
                    else:
                        obs_snr['snr_median'] = np.nanmedian(obs_snr['snr_quad'])
                except Exception:
                    obs_snr['snr_median'] = np.nanmedian(obs_snr['snr_quad'])

            if 'snr_peak' in cache:
                obs_snr['snr_peak'] = np.array(cache['snr_peak'])
            else:
                try:
                    if obs_snr['snr_quad'].ndim > 1:
                        obs_snr['snr_peak'] = np.nanmax(obs_snr['snr_quad'], axis=1)
                    else:
                        obs_snr['snr_peak'] = np.nanmax(obs_snr['snr_quad'])
                except Exception:
                    obs_snr['snr_peak'] = np.nanmax(obs_snr['snr_quad'])

            if 'n_valid' in cache:
                # n_valid may be scalar or array
                obs_snr['n_valid'] = int(cache['n_valid']) if np.isscalar(cache['n_valid']) else np.array(cache['n_valid'])
            else:
                obs_snr['n_valid'] = len(obs_snr['snr_quad'])

            print(f"  Loaded {len(obs_snr['snr_quad']):,} entries from cache.")
        else:
            # --obs-snr: compute from parquet
            obs_snr = compute_observed_snr_from_parquet(args.parquet, src_ids)
            if args.save_snr_cache:
                save_path = Path(args.save_snr_cache)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(save_path, **obs_snr, src_id=src_ids)
                print(f"  SNR cache saved to: {save_path}")

        snr = obs_snr[args.snr_metric]
        snr_label = args.snr_metric

    # 3. Create bins
    # Default range for snr_quad: [0.1, 1000] (overridden by explicit CLI args)
    snr_min = args.snr_min
    snr_max = args.snr_max
    if args.snr_metric == 'snr_quad':
        if snr_min is None:
            snr_min = 5.0
        if snr_max is None:
            snr_max = 5000.0

    print(f"\nCreating {args.n_snr_bins} log-spaced SNR bins (metric={snr_label})...")
    edges, centers = make_snr_bins(snr, n_bins=args.n_snr_bins,
                                   snr_min=snr_min, snr_max=snr_max)
    print(f"  SNR range: [{edges[0]:.4f}, {edges[-1]:.4f}]")

    valid = np.isfinite(snr) & (snr > 0)
    print(f"  Sources with valid SNR: {np.sum(valid):,}/{len(snr):,}")
    print(f"  SNR median={np.nanmedian(snr[valid]):.3f}, "
          f"p10={np.percentile(snr[valid], 10):.3f}, "
          f"p90={np.percentile(snr[valid], 90):.3f}")

    # 4. Make plots
    print(f"\n{'='*60}")
    print("Generating figures...")

    plot_photoz_stats(results, snr, edges, centers, outdir)

    plot_convergence_stats(results, snr, edges, centers, outdir,
                           obs_snr=obs_snr, snr_label=snr_label)

    plot_sample_counts(results, snr, edges, centers,
                       outdir / 'snr_sample_counts.png',
                       snr_col_name=snr_label,
                       obs_snr=obs_snr,
                       use_hexbin=args.hexbin)

    print(f"\nAll figures saved to: {outdir}")


if __name__ == '__main__':
    main()
