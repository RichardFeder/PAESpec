#!/usr/bin/env python
"""
Compare MCLMC and pocoMC samplers on a subset of sources.

Loads mock (or real) data, initializes the PAE model, then runs both
MCLMC (GPU-batched) and pocoMC (sequential) on the same set of sources.
Outputs summary statistics and saves results for further comparison.

Usage:
    # Quick test on 5 sources
    python scripts/compare_samplers.py --ngal 5

    # Specify samplers and prior
    python scripts/compare_samplers.py --ngal 20 --samplers mclmc pocomc \
        --nf-alpha 1.0 --prior-type 1 --sigma-prior 0.4

    # Only pocoMC (skip MCLMC)
    python scripts/compare_samplers.py --ngal 10 --samplers pocomc

    # Custom output
    python scripts/compare_samplers.py --ngal 10 --datestr sampler_comparison_test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import time
from pathlib import Path

from models.pae_jax import initialize_PAE, load_spherex_data
from training.train_ae_jax import param_dict_gen
from sampling.sample_pae_batch_refactor import MCLMCSamplingConfig, sample_mclmc_wrapper
from sampling.sample_pae import sample_pocomc_wrapper
from data_proc.dataloader_jax import SPHERExData
from diagnostics.diagnostics_jax import compute_redshift_stats
from config import scratch_basepath


def load_data_and_model(args):
    """Load mock SPHEREx data and initialize PAE model.
    
    Returns
    -------
    spherex_dat : SPHERExData
    property_cat_df_obs : DataFrame
    PAE_model : PAE object
    valid_indices : array of valid source indices
    """
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")
    t0 = time.time()

    dat_obs, property_cat_df_obs, property_cat_df_restframe, \
        central_wavelengths, wave_obs = load_spherex_data(
            sig_level_norm=args.sig_level_norm,
            sel_str=args.sel_str,
            abs_norm=True,
            with_ext_phot=args.with_ext_phot,
            load_rf_dat=False,
            load_obs_dat=True,
            weight_soft=5e-4,
            data_fpath=args.data_fpath
        )

    spherex_dat = SPHERExData.from_prep(
        dat_obs,
        property_cat_df_obs,
        property_cat_df_restframe,
        phot_snr_min=None,
        phot_snr_max=None,
        zmin=None,
        zmax=None
    )
    print(f"  Total sources: {len(spherex_dat.redshift)}")

    # Apply filters
    nsrc = len(spherex_dat.redshift)
    valid_mask = np.ones(nsrc, dtype=bool)

    if args.z_min is not None:
        valid_mask &= spherex_dat.redshift >= args.z_min
    if args.z_max is not None:
        valid_mask &= spherex_dat.redshift <= args.z_max
    if hasattr(spherex_dat, 'phot_snr') and spherex_dat.phot_snr is not None:
        if args.snr_min is not None:
            valid_mask &= spherex_dat.phot_snr >= args.snr_min
        if args.snr_max is not None:
            valid_mask &= spherex_dat.phot_snr <= args.snr_max

    valid_indices = np.where(valid_mask)[0]
    print(f"  Sources after filtering: {len(valid_indices)}")
    print(f"  Data loaded in {time.time()-t0:.1f}s")

    # Initialize PAE model
    print(f"\n{'='*70}")
    print("INITIALIZING PAE MODEL")
    print(f"{'='*70}")
    t0 = time.time()

    run_name = f'jax_conv1_nlatent={args.nlatent}_siglevelnorm={args.sig_level_norm}_newAllen_all_091325'

    PAE_model = initialize_PAE(
        run_name,
        filter_set_name=args.filter_set,
        with_ext_phot=args.with_ext_phot,
        inference_dtype=jnp.float32,
        lam_min_rest=0.15,
        lam_max_rest=5.0,
        nlam_rest=500,
        filename_flow=args.filename_flow
    )
    print(f"  PAE model initialized in {time.time()-t0:.1f}s")

    return spherex_dat, property_cat_df_obs, PAE_model, valid_indices


def select_sources(valid_indices, args, spherex_dat):
    """Select source indices for the comparison."""
    ngal = min(args.ngal, len(valid_indices))

    if args.src_idxs is not None:
        # Manual selection
        src_idxs = np.array(args.src_idxs)
        src_idxs = src_idxs[src_idxs < len(valid_indices)]
        selected = valid_indices[src_idxs]
    elif args.random_seed is not None:
        rng = np.random.default_rng(args.random_seed)
        selected = rng.choice(valid_indices, size=ngal, replace=False)
    else:
        selected = valid_indices[:ngal]

    selected = np.sort(selected)
    print(f"\nSelected {len(selected)} sources for comparison")
    for i, idx in enumerate(selected[:10]):
        print(f"  [{i}] idx={idx}, z={spherex_dat.redshift[idx]:.3f}")
    if len(selected) > 10:
        print(f"  ... and {len(selected)-10} more")

    return selected


def run_mclmc(PAE_model, spherex_dat, property_cat_df_obs, src_idxs, args, save_dir):
    """Run MCLMC sampling on selected sources.
    
    Returns dict with keys: samples, z_med, z_true, sigmaz, time_elapsed
    """
    print(f"\n{'='*70}")
    print("RUNNING MCLMC SAMPLER")
    print(f"{'='*70}")

    ngal = len(src_idxs)

    cfg = MCLMCSamplingConfig(
        num_steps=args.num_steps_mclmc,
        burn_in=args.burn_in,
        nsamp_init=args.nsamp_init,
        fix_z=False,
        nf_alpha=args.nf_alpha,
        init_reinit=True,
        redshift_prior_type=args.prior_type,
        z0_prior=args.z0_prior,
        sigma_prior=args.sigma_prior,
        chi2_red_threshold=args.chi2_red_threshold,
        gr_threshold=args.gr_threshold,
        use_batched_logdensity=True,
        use_multicore=args.use_multicore,
        n_devices_per_node=args.n_devices if args.use_multicore else 1,
    )

    save_fpath = str(save_dir / f'MCLMC_results_{args.datestr}.npz')
    sample_fpath = str(save_dir / f'MCLMC_samples_{args.datestr}.npz')

    print(f"  Sources: {ngal}")
    print(f"  Steps: {args.num_steps_mclmc}, Burn-in: {args.burn_in}")
    print(f"  NF alpha: {args.nf_alpha}")
    print(f"  Prior type: {args.prior_type}")
    print(f"  Batch size: {args.batch_size}")

    t0 = time.time()

    results = sample_mclmc_wrapper(
        PAE_model,
        spherex_dat,
        cfg,
        ngal=ngal,
        batch_size=args.batch_size,
        save_results=True,
        save_fpath=save_fpath,
        sample_fpath=sample_fpath,
        return_results=True,
        do_cleanup=False,
        src_idxs_sub=src_idxs,
        property_cat_df=property_cat_df_obs
    )

    elapsed = time.time() - t0

    # Load saved results to extract summary stats
    res = np.load(save_fpath, allow_pickle=True)
    samp = np.load(sample_fpath, allow_pickle=True)

    z_true = np.array(res['ztrue'])
    z_med = np.array(res['z_med'])
    err_low = np.array(res['err_low'])
    err_high = np.array(res['err_high'])
    sigmaz = 0.5 * (err_low + err_high)

    all_samples = np.array(samp['all_samples'])

    print(f"\n  ✓ MCLMC completed in {elapsed:.1f}s ({elapsed/ngal:.2f}s per source)")
    print(f"    Samples shape: {all_samples.shape}")

    return {
        'sampler': 'MCLMC',
        'z_true': z_true,
        'z_med': z_med,
        'sigmaz': sigmaz,
        'err_low': err_low,
        'err_high': err_high,
        'all_samples': all_samples,
        'time_elapsed': elapsed,
        'time_per_source': elapsed / ngal,
        'save_fpath': save_fpath,
        'sample_fpath': sample_fpath,
    }


def run_pocomc(PAE_model, spherex_dat, src_idxs, args, save_dir):
    """Run pocoMC sampling on selected sources.
    
    Note: pocoMC runs sequentially (not GPU-batched).
    
    Returns dict with keys: samples, z_med, z_true, sigmaz, time_elapsed
    """
    print(f"\n{'='*70}")
    print("RUNNING pocoMC SAMPLER")
    print(f"{'='*70}")

    ngal = len(src_idxs)

    # Build a redshift prior if requested
    from scipy.stats import norm as scipy_norm, uniform as scipy_uniform
    z_prior = None
    if args.prior_type == 1:
        z_prior = scipy_norm(loc=args.z0_prior, scale=args.sigma_prior)
        print(f"  Using Gaussian z prior: N({args.z0_prior}, {args.sigma_prior})")
    else:
        print(f"  Using uniform z prior")

    print(f"  Sources: {ngal}")
    print(f"  n_total: {args.n_total_pocomc}")
    print(f"  n_active: {args.n_active_pocomc}")
    print(f"  n_effective: {args.n_effective_pocomc}")
    print(f"  NF alpha: {args.nf_alpha}")
    print(f"  Dynamic: {args.pocomc_dynamic}")

    save_fpath = str(save_dir / f'pocoMC_results_{args.datestr}.npz')
    sample_fpath = str(save_dir / f'pocoMC_samples_{args.datestr}.npz')

    t0 = time.time()

    all_samp_min, redshifts_true, failed_idx = sample_pocomc_wrapper(
        PAE_model,
        spherex_dat.all_spec_obs,
        spherex_dat.weights,
        spherex_dat.redshift,
        src_idxs,
        ngal,
        nf_alpha=args.nf_alpha,
        n_total=args.n_total_pocomc,
        n_active=args.n_active_pocomc,
        n_effective=args.n_effective_pocomc,
        dynamic=args.pocomc_dynamic,
        z_prior=z_prior,
        save_results=True,
        save_fpath=save_fpath,
        sample_fpath=sample_fpath,
        return_results=True,
    )

    elapsed = time.time() - t0

    # Compute summary statistics from samples
    # pocoMC samples shape: (ngal, n_samples, nlatent+1), redshift is last dim
    z_samples = all_samp_min[:, :, -1]
    z_med = np.median(z_samples, axis=1)
    pct16 = np.percentile(z_samples, 16, axis=1)
    pct84 = np.percentile(z_samples, 84, axis=1)
    err_low = z_med - pct16
    err_high = pct84 - z_med
    sigmaz = 0.5 * (err_low + err_high)

    print(f"\n  ✓ pocoMC completed in {elapsed:.1f}s ({elapsed/ngal:.2f}s per source)")
    print(f"    Samples shape: {all_samp_min.shape}")
    if len(failed_idx) > 0:
        print(f"    ⚠ Failed sources: {failed_idx}")

    return {
        'sampler': 'pocoMC',
        'z_true': redshifts_true,
        'z_med': z_med,
        'sigmaz': sigmaz,
        'err_low': err_low,
        'err_high': err_high,
        'all_samples': all_samp_min,
        'time_elapsed': elapsed,
        'time_per_source': elapsed / ngal,
        'failed_idx': failed_idx,
        'save_fpath': save_fpath,
        'sample_fpath': sample_fpath,
    }


def print_comparison(results_list):
    """Print side-by-side comparison of sampler results."""
    print(f"\n{'='*70}")
    print("SAMPLER COMPARISON RESULTS")
    print(f"{'='*70}")

    for res in results_list:
        name = res['sampler']
        z_true = res['z_true']
        z_med = res['z_med']
        sigmaz = res['sigmaz']

        # Filter valid values
        valid = np.isfinite(z_med) & np.isfinite(z_true) & np.isfinite(sigmaz)
        sigz_oneplusz = sigmaz[valid] / (1.0 + z_med[valid])

        print(f"\n  {name}:")
        print(f"    N sources: {len(z_true)} ({np.sum(valid)} valid)")
        print(f"    Time: {res['time_elapsed']:.1f}s total, {res['time_per_source']:.2f}s/source")

        if np.sum(valid) == 0:
            print(f"    ⚠ No valid sources for statistics")
            continue

        # Standard redshift statistics
        try:
            arg_bias, arg_std, bias, NMAD, cond_outl, outl_rate, outl_rate_15pct = compute_redshift_stats(
                z_med[valid], z_true[valid],
                sigma_z_select=sigz_oneplusz,
                nsig_outlier=3,
                outlier_pct=15
            )

            mean_sigz = np.mean(sigz_oneplusz)
            print(f"    Mean σ_z/(1+z): {mean_sigz:.5f}")
            print(f"    Bias [median(Δz/(1+z))]: {bias:.5f}")
            print(f"    NMAD: {NMAD:.5f}")
            print(f"    Outlier rate (3σ): {outl_rate:.4f} ({int(outl_rate*100)}%)")
            print(f"    Outlier rate (15%): {outl_rate_15pct:.4f} ({int(outl_rate_15pct*100)}%)")
        except Exception as e:
            print(f"    ⚠ Error computing stats: {e}")

    # Per-source comparison if both samplers ran
    if len(results_list) == 2:
        r0, r1 = results_list
        n = min(len(r0['z_true']), len(r1['z_true']))

        print(f"\n  Per-source comparison ({r0['sampler']} vs {r1['sampler']}):")
        print(f"  {'idx':>4s}  {'z_true':>7s}  {'z_med_0':>7s}  {'z_med_1':>7s}  "
              f"{'σz_0':>7s}  {'σz_1':>7s}  {'|Δz0|':>7s}  {'|Δz1|':>7s}")
        print(f"  {'-'*65}")

        for i in range(min(n, 20)):
            dz0 = abs(r0['z_med'][i] - r0['z_true'][i])
            dz1 = abs(r1['z_med'][i] - r1['z_true'][i])
            print(f"  {i:4d}  {r0['z_true'][i]:7.4f}  {r0['z_med'][i]:7.4f}  {r1['z_med'][i]:7.4f}  "
                  f"{r0['sigmaz'][i]:7.4f}  {r1['sigmaz'][i]:7.4f}  {dz0:7.4f}  {dz1:7.4f}")

        if n > 20:
            print(f"  ... ({n-20} more sources)")


def main():
    parser = argparse.ArgumentParser(
        description='Compare MCLMC and pocoMC samplers on PAE redshift estimation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data configuration
    parser.add_argument('--filter-set', type=str, default='spherex_filters102/')
    parser.add_argument('--nlatent', type=int, default=5)
    parser.add_argument('--sig-level-norm', type=float, default=0.01)
    parser.add_argument('--sel-str', type=str, default='zlt22.5')
    parser.add_argument('--with-ext-phot', action='store_true')
    parser.add_argument('--data-fpath', type=str, default=None)
    parser.add_argument('--filename-flow', type=str, default='flow_model_iaf_092225')

    # Source selection
    parser.add_argument('--ngal', type=int, default=10,
                        help='Number of sources to sample')
    parser.add_argument('--src-idxs', type=int, nargs='+', default=None,
                        help='Specific source indices to use')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Seed for random source selection')

    # Filtering
    parser.add_argument('--z-min', type=float, default=None)
    parser.add_argument('--z-max', type=float, default=None)
    parser.add_argument('--snr-min', type=float, default=None)
    parser.add_argument('--snr-max', type=float, default=None)

    # Sampler selection
    parser.add_argument('--samplers', type=str, nargs='+', default=['mclmc', 'pocomc'],
                        choices=['mclmc', 'pocomc'],
                        help='Which samplers to run')

    # Shared sampling parameters
    parser.add_argument('--nf-alpha', type=float, default=1.0,
                        help='Normalizing flow prior strength')
    parser.add_argument('--prior-type', type=int, default=0,
                        help='Redshift prior: 0=none, 1=Gaussian, 2=BPZ')
    parser.add_argument('--z0-prior', type=float, default=0.65,
                        help='Gaussian prior center')
    parser.add_argument('--sigma-prior', type=float, default=0.4,
                        help='Gaussian prior width')

    # MCLMC-specific parameters
    parser.add_argument('--num-steps-mclmc', type=int, default=2000,
                        help='MCLMC: number of MCMC steps')
    parser.add_argument('--burn-in', type=int, default=1000,
                        help='MCLMC: burn-in steps')
    parser.add_argument('--nsamp-init', type=int, default=500,
                        help='MCLMC: initial samples for reinitialization')
    parser.add_argument('--chi2-red-threshold', type=float, default=1.5,
                        help='MCLMC: chi-squared threshold')
    parser.add_argument('--gr-threshold', type=float, default=1.5,
                        help='MCLMC: Gelman-Rubin threshold')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='MCLMC: sampling batch size')
    parser.add_argument('--use-multicore', action='store_true',
                        help='MCLMC: use multiple GPUs')
    parser.add_argument('--n-devices', type=int, default=4,
                        help='MCLMC: number of GPUs for multicore')

    # pocoMC-specific parameters
    parser.add_argument('--n-total-pocomc', type=int, default=2048,
                        help='pocoMC: total number of samples')
    parser.add_argument('--n-active-pocomc', type=int, default=256,
                        help='pocoMC: number of active walkers')
    parser.add_argument('--n-effective-pocomc', type=int, default=512,
                        help='pocoMC: number of effective samples')
    parser.add_argument('--pocomc-dynamic', action='store_true',
                        help='pocoMC: enable dynamic mode')

    # Output
    parser.add_argument('--datestr', type=str, default='sampler_comparison',
                        help='Date/run string for output files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: scratch_basepath/data/sampler_comparison/DATESTR)')

    args = parser.parse_args()

    # Set up output directory
    if args.output_dir is None:
        save_dir = Path(scratch_basepath) / 'data' / 'sampler_comparison' / args.datestr
    else:
        save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {save_dir}")

    # Save run configuration
    run_params = {k: str(v) for k, v in vars(args).items()}
    np.savez(save_dir / 'run_config.npz', **run_params)

    # Load data and model
    spherex_dat, property_cat_df_obs, PAE_model, valid_indices = load_data_and_model(args)

    # Select sources
    src_idxs = select_sources(valid_indices, args, spherex_dat)
    ngal = len(src_idxs)

    # Save selected source indices
    np.savez(save_dir / 'source_selection.npz',
             src_idxs=src_idxs,
             z_true=spherex_dat.redshift[src_idxs])

    # Run samplers
    results_list = []

    if 'mclmc' in args.samplers:
        mclmc_dir = save_dir / 'mclmc'
        mclmc_dir.mkdir(parents=True, exist_ok=True)
        try:
            mclmc_results = run_mclmc(
                PAE_model, spherex_dat, property_cat_df_obs,
                src_idxs, args, mclmc_dir
            )
            results_list.append(mclmc_results)
        except Exception as e:
            print(f"\n  ✗ MCLMC failed: {e}")
            import traceback
            traceback.print_exc()

    if 'pocomc' in args.samplers:
        pocomc_dir = save_dir / 'pocomc'
        pocomc_dir.mkdir(parents=True, exist_ok=True)
        try:
            pocomc_results = run_pocomc(
                PAE_model, spherex_dat,
                src_idxs, args, pocomc_dir
            )
            results_list.append(pocomc_results)
        except Exception as e:
            print(f"\n  ✗ pocoMC failed: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison
    if len(results_list) > 0:
        print_comparison(results_list)

        # Save combined comparison
        comparison = {}
        for res in results_list:
            name = res['sampler'].lower()
            for key in ['z_true', 'z_med', 'sigmaz', 'err_low', 'err_high',
                        'time_elapsed', 'time_per_source']:
                comparison[f'{name}_{key}'] = res[key]
        comparison['src_idxs'] = src_idxs

        comparison_path = save_dir / f'comparison_summary_{args.datestr}.npz'
        np.savez(comparison_path, **comparison)
        print(f"\n✓ Comparison saved to: {comparison_path}")
    else:
        print("\n✗ No samplers completed successfully")
        return 1

    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
