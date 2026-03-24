#!/usr/bin/env python
"""
Train a Probabilistic Autoencoder (PAE) on rest-frame SPHEREx SED data.

This script trains a convolutional autoencoder to compress high-resolution
rest-frame SEDs into a low-dimensional latent representation. The trained
model is used for photo-z estimation via MCMC sampling in latent space.

Usage Examples
--------------
# Train with default parameters (nlatent=5, sig_level=0.01)
python scripts/train_pae_autoencoder.py --run-name my_test_run

# Train with specific latent dimension
python scripts/train_pae_autoencoder.py --run-name nlatent8_run --nlatent 8

# Train with noise augmentation
python scripts/train_pae_autoencoder.py --run-name noisy_run --sig-level-norm 0.005

# Multiple training runs (sweeping parameters)
python scripts/train_pae_autoencoder.py --run-name sweep --nlatent 5 8 10 --sig-level-norm 0.005 0.01

# Skip autoencoder training, only extract latents
python scripts/train_pae_autoencoder.py --run-name existing_run --skip-ae-training --extract-latents

Output Structure
----------------
modl_runs/{run_name}/
    ├── params.pkl                          # Training hyperparameters
    ├── ae_state.pkl                        # Trained model weights
    ├── training_metrics.npz                # Loss curves
    ├── latents/
    │   └── latents.npz                     # Encoded training/validation latents
    └── figures/
        └── training/
            └── reconstruction_vs_epoch/    # Reconstruction quality plots
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster use
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp

# Import training modules
import config
from data_proc.dataloader_jax import spec_data_jax
from data_proc.data_file_utils import grab_fpaths_traindat, save_params
from training.train_ae_jax import run_ae_sed_fit_jax, param_dict_gen
from diagnostics.diagnostics_jax import grab_encoded_vars_dataset
from visualization.result_plotting_fns import make_color_corner_plot
from models.flow_jax import fit_flow_to_latents_jax
from utils.config_loader import apply_yaml_defaults


def interpolate_to_log_wavelength(data, wav_linear, wav_log):
    """
    Interpolate SED data from linear to log-spaced wavelength grid.
    
    Parameters
    ----------
    data : np.ndarray
        SED data on linear wavelength grid, shape (n_sources, n_wav_linear)
    wav_linear : np.ndarray
        Linear wavelength grid, shape (n_wav_linear,)
    wav_log : np.ndarray
        Log-spaced wavelength grid, shape (n_wav_log,)
        
    Returns
    -------
    data_log : np.ndarray
        SED data on log wavelength grid, shape (n_sources, n_wav_log)
    """
    n_sources = data.shape[0]
    data_log = np.zeros((n_sources, len(wav_log)))
    
    for i in range(n_sources):
        data_log[i] = np.interp(wav_log, wav_linear, data[i])
    
    return data_log


def setup_data(nbands=500, sig_level_norm=0.01, train_frac=0.8, 
               data_file=None, scratch_base='/pscratch/sd/r/rmfeder/data/',
               max_normflux=100.0, z_max=None, sel_str=None, max_sources=50000,
               use_log_wavelength=False):
    """
    Load and prepare training/validation data.
    
    Parameters
    ----------
    nbands : int
        Number of wavelength bins for rest-frame SED
    sig_level_norm : float
        Noise level for data augmentation (None = noiseless)
    train_frac : float
        Fraction of data to use for training (rest is validation)
    data_file : str, optional
        Path to SED data file (uses default if None)
    scratch_base : str
        Base path for data storage
    max_normflux : float
        Maximum allowed normalized flux (clips outliers)
    z_max : float, optional
        Maximum redshift for training sample
    sel_str : str, optional
        Selection string for filtering
    use_log_wavelength : bool, optional
        If True, interpolate training data to log-spaced wavelength grid.
        This enables faster inference by converting redshift to index shifts.
        
    Returns
    -------
    jax_spec : spec_data_jax
        Data object with train/valid splits
    property_cat_df : DataFrame
        Property catalog with redshift, E(B-V), etc.
    wav_rest : array
        Rest-frame wavelength grid (log-spaced if use_log_wavelength=True)
    """
    print(f"\n{'='*70}")
    print("DATA PREPARATION")
    print(f"{'='*70}")
    print(f"Number of wavelength bins: {nbands}")
    print(f"Noise augmentation level: {sig_level_norm}")
    print(f"Training fraction: {train_frac}")
    
    # Define wavelength grid based on number of bands
    if nbands == 500:
        wav_rest = jnp.linspace(0.15, 5.0, 500)
    elif nbands == 1000:
        wav_rest = jnp.linspace(0.1, 5.0, 1000)
    else:
        wav_rest = jnp.linspace(0.15, 5.0, nbands)
    
    print(f"Wavelength range: {wav_rest[0]:.2f} - {wav_rest[-1]:.2f} μm")
    
    # Initialize data loader
    jax_spec = spec_data_jax(nbands)
    
    # Get file paths
    fpath_dict_rf = grab_fpaths_traindat('COSMOS', restframe=True, sel_str='')
    
    # Override data file if specified
    if data_file is None:
        # Select appropriate data file based on nbands
        if nbands == 500:
            # Use newAllen data (500 bands, 0.15-5.0 μm)
            data_file = scratch_base + 'phot/hires_sed_COSMOS_0_200000_0p1_8_um_z=0_aug2x_debv_0p02_dustlaw_newAllen.npz'
            print(f"Using 500-band newAllen data")
        elif nbands == 1000:
            # Use 062325 data (1000 bands, 0.1-8.0 μm)
            data_file = scratch_base + 'phot/hires_sed_COSMOS_0_200000_0p1_8_um_z=0_aug2x_debv_0p02_dustlaw_062325.npz'
            print(f"Using 1000-band 062325 data")
        else:
            raise ValueError(f"Unsupported nbands={nbands}. Must be 500 or 1000.")
    
    fpath_dict_rf['data_fpath'] = data_file
    print(f"Data file: {Path(data_file).name}")
    print(f"Expected bands: {nbands}")
    
    # Build data loaders
    property_cat_df = jax_spec.build_dataloaders_new(
        fpath_dict_rf,
        train_frac=train_frac,
        load_property_cat=False,
        property_cat_fpath=None,
        restframe=True,
        save_property_cat=False,
        sig_level_norm=sig_level_norm,
        plot=True,
        sel_cosmos_temp_fits=False,
        apply_sel=False,
        sel_str=sel_str,
        pivot_wavelength=None,
        max_normflux=max_normflux,
        z_max=z_max
    )
    
    print(f"Training samples: {len(jax_spec.trainidx):,}")
    print(f"Validation samples: {len(jax_spec.valididx):,}")
    
    # Verify data dimensions
    actual_bands = jax_spec.data_train.shape[1]
    print(f"Actual data bands: {actual_bands}")
    
    if actual_bands != nbands:
        raise ValueError(
            f"Data dimension mismatch! "
            f"Expected {nbands} bands but got {actual_bands} bands. "
            f"Check that the data file matches --nbands argument."
        )
    
    print(f"✓ Data dimensions verified: {nbands} bands as expected")
    
    # Optionally transform to log-wavelength grid
    if use_log_wavelength:
        print(f"\nTransforming to log-wavelength grid...")
        
        # Create log-spaced wavelength grid with same number of points
        wav_linear = np.array(wav_rest)
        wav_log = np.logspace(np.log10(wav_linear.min()), np.log10(wav_linear.max()), nbands)
        
        # Interpolate training and validation data to log grid
        # Note: this is a one-time cost during data preparation
        data_train_log = interpolate_to_log_wavelength(
            np.array(jax_spec.data_train), wav_linear, wav_log
        )
        data_valid_log = interpolate_to_log_wavelength(
            np.array(jax_spec.data_valid), wav_linear, wav_log
        )
        
        # Update the data in jax_spec
        jax_spec.data_train = jnp.array(data_train_log)
        jax_spec.data_valid = jnp.array(data_valid_log)
        
        # Update wavelength grid to log-spaced
        wav_rest = jnp.array(wav_log)
        
        # Compute log step for redshift shifting
        dloglam = (np.log(wav_log[-1]) - np.log(wav_log[0])) / (nbands - 1)
        
        print(f"  Log wavelength range: {wav_log[0]:.4f} - {wav_log[-1]:.4f} μm")
        print(f"  dloglam (natural log step) = {dloglam:.6f}")
        print(f"  ✓ Data transformed to log-wavelength grid")
        
        # Store log wavelength info for saving with model
        jax_spec.use_log_wavelength = True
        jax_spec.dloglam = dloglam
    else:
        jax_spec.use_log_wavelength = False
        jax_spec.dloglam = None
    
    return jax_spec, property_cat_df, wav_rest


def train_autoencoder(jax_spec, run_name, params, wav_rest, 
                     beta=0.0, variance_penalty_threshold=None,
                     batch_size=128):
    """
    Train the autoencoder model.
    
    Parameters
    ----------
    jax_spec : spec_data_jax
        Data object with train/valid splits
    run_name : str
        Name for this training run
    params : dict
        Training parameters from param_dict_gen
    wav_rest : array
        Rest-frame wavelength grid
    beta : float
        Beta parameter for variational penalty
    variance_penalty_threshold : float
        Threshold for variance penalty
    batch_size : int
        Batch size for training
        
    Returns
    -------
    state : TrainState
        Trained model state
    model : nn.Module
        Autoencoder model
    jax_spec : spec_data_jax
        Data object (with rundir set)
    metric_dict : dict
        Training metrics
    model_fpath : str
        Path to saved model
    """
    print(f"\n{'='*70}")
    print("AUTOENCODER TRAINING")
    print(f"{'='*70}")
    print(f"Run name: {run_name}")
    print(f"Latent dimensions: {params['nlatent']}")
    print(f"Learning rate: {params['lr']}")
    print(f"Epochs: {params['epochs']}")
    print(f"Batch size: {batch_size}")
    
    # Run training
    state, model, jax_spec, metric_dict, model_fpath = run_ae_sed_fit_jax(
        jax_spec,
        train_mode='deep',
        property_cat_df=None,
        verbose=False,
        run_name=run_name,
        params=params,
        beta=beta,
        variance_penalty_threshold=variance_penalty_threshold,
        wav_rest=wav_rest,
        batch_size=batch_size
    )
    
    print(f"\n✓ Training complete!")
    print(f"Model saved to: {model_fpath}")
    
    # Save parameters
    rundir = config.sphx_base_path + 'modl_runs/' + run_name
    save_params(rundir, params)
    print(f"Parameters saved to: {rundir}/params.pkl")
    
    return state, model, jax_spec, metric_dict, model_fpath


def extract_latents(state, jax_spec, property_cat_df, run_name):
    """
    Extract latent representations from trained model.
    
    Parameters
    ----------
    state : TrainState
        Trained model state
    jax_spec : spec_data_jax
        Data object
    property_cat_df : DataFrame
        Property catalog
    run_name : str
        Name of run (for saving)
        
    Returns
    -------
    all_z_train : array
        Latent codes for training set
    all_z_valid : array
        Latent codes for validation set
    ncode : int
        Number of latent dimensions
    """
    print(f"\n{'='*70}")
    print("EXTRACTING LATENT REPRESENTATIONS")
    print(f"{'='*70}")
    
    rundir = config.sphx_base_path + 'modl_runs/' + run_name
    
    all_z_train, all_z_valid, ncode = grab_encoded_vars_dataset(
        state,
        jax_spec,
        property_cat_df,
        save=True,
        rundir=rundir
    )
    
    print(f"Training latents shape: {all_z_train.shape}")
    print(f"Validation latents shape: {all_z_valid.shape}")
    print(f"Saved to: {rundir}/latents/latents.npz")

    # If a trained flow already exists, transform the AE latents to u-space and
    # save latents_with_u_space.npz without touching flow training at all.
    flow_pkl   = Path(rundir) / 'flows' / 'flow_model_iaf.pkl'
    locscale_f = Path(rundir) / 'latents' / 'latent_loc_std.npz'
    if flow_pkl.exists() and locscale_f.exists():
        print(f"\nFound trained flow – computing u-space latents...")
        try:
            import jax
            from models.flow_jax import transform_latents_to_u_space
            from models.flowjax_modl import init_flowjax_modl
            from data_proc.data_file_utils import load_model

            locscale = np.load(str(locscale_f))
            loc, scale = locscale['loc'], locscale['scale']

            key = jax.random.PRNGKey(0)
            _, flow_template = init_flowjax_modl(key, ncode)
            flow_model = load_model(flow_template, rundir, filename='flow_model_iaf.pkl')

            latents_u_train = np.array(transform_latents_to_u_space(flow_model, all_z_train, loc, scale))
            latents_u_valid = np.array(transform_latents_to_u_space(flow_model, all_z_valid, loc, scale))

            # Build save dict – start with z/u-space latents and normalization params
            u_save_dict = dict(
                latents_z_train=all_z_train,
                latents_z_valid=all_z_valid,
                latents_u_train=latents_u_train,
                latents_u_valid=latents_u_valid,
                loc=loc,
                scale=scale,
            )
            # Pull source properties (redshift, ebv, bfit_tid) from latents.npz
            latents_npz = np.load(Path(rundir) / 'latents' / 'latents.npz')
            for col in ['redshift', 'ebv', 'bfit_tid']:
                if f'{col}_train' in latents_npz:
                    u_save_dict[f'{col}_train'] = latents_npz[f'{col}_train']
                    u_save_dict[f'{col}_valid'] = latents_npz[f'{col}_valid']

            u_save_path = Path(rundir) / 'latents' / 'latents_with_u_space.npz'
            np.savez(str(u_save_path), **u_save_dict)
            print(f"  u-space train mean: {latents_u_train.mean(axis=0)}")
            print(f"  u-space train std:  {latents_u_train.std(axis=0)}")
            print(f"  Saved to: {u_save_path}")
        except Exception as e:
            print(f"  ⚠ Could not compute u-space latents: {e}")
            import traceback; traceback.print_exc()
    else:
        print("\nNo trained flow found – skipping u-space latent computation.")
        print(f"  (expected: {flow_pkl})")

    return all_z_train, all_z_valid, ncode


def plot_latent_correlations(all_z_train, property_cat_df, jax_spec, 
                             run_name, nlatent, figsize=(10, 10)):
    """
    Generate corner plots showing latent-property correlations.
    
    Parameters
    ----------
    all_z_train : array
        Latent codes for training set
    property_cat_df : DataFrame
        Property catalog
    jax_spec : spec_data_jax
        Data object (for train indices)
    run_name : str
        Name of run (for saving)
    nlatent : int
        Number of latent dimensions
    figsize : tuple
        Figure size
    """
    print(f"\n{'='*70}")
    print("GENERATING DIAGNOSTIC PLOTS")
    print(f"{'='*70}")
    
    rundir = config.sphx_base_path + 'modl_runs/' + run_name
    fig_dir = Path(rundir) / 'figures' / 'latent_diagnostics'
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    xv = 0.1
    xlim = [-xv, xv]
    ylim = [-xv, xv]
    xticks = [-xv, 0.0, xv]
    
    # Plot 1: Latents colored by E(B-V)
    if 'ebv' in property_cat_df.columns:
        print("Plotting latents vs E(B-V)...")
        ebv_train = np.array(property_cat_df['ebv'])[jax_spec.trainidx]
        
        fig = make_color_corner_plot(
            nlatent, all_z_train, ebv_train, 'E(B-V)',
            vmin=0.0, vmax=0.3,
            xlim=xlim, ylim=ylim, yticks=xticks,
            color='C3', alph=0.1, figsize=figsize
        )
        fig.savefig(fig_dir / f'{run_name}_latents_vs_ebv.png', 
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Plot 2: Latents colored by redshift
    if 'redshift' in property_cat_df.columns:
        print("Plotting latents vs redshift...")
        redshift_train = np.array(property_cat_df['redshift'])[jax_spec.trainidx]
        
        fig = make_color_corner_plot(
            nlatent, all_z_train, redshift_train, 'Redshift $z$',
            vmin=0.0, vmax=2.0,
            xlim=xlim, ylim=ylim, yticks=xticks,
            color='C3', alph=0.1, figsize=figsize
        )
        fig.savefig(fig_dir / f'{run_name}_latents_vs_redshift.png',
                   dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Plot 3: Latent distributions (no coloring)
    print("Plotting latent distributions...")
    fig = make_color_corner_plot(
        nlatent, all_z_train, None, None,
        vmin=None, vmax=None,
        xlim=xlim, ylim=ylim, yticks=xticks,
        color='C3', alph=0.1, figsize=figsize
    )
    fig.savefig(fig_dir / f'{run_name}_latent_distributions.png',
               dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Plots saved to: {fig_dir}")


def train_flow(jax_spec, run_name, params, verbose=True, max_sources_flow=50000):
    """
    Train normalizing flow on extracted latents.
    
    Parameters
    ----------
    jax_spec : spec_data_jax
        Data object (used for batch generation)
    run_name : str
        Name for this training run
    params : dict
        Training parameters (must include lr_flow, nepoch_flow)
    verbose : bool
        Print progress
    max_sources_flow : int
        Maximum number of sources to use for flow training (default 50000)
        This will use the first 40k for training and 10k for validation
        
    Returns
    -------
    flow_loaded : flowjax.Transformed
        Trained flow model
    flow_fpath : str
        Path to saved flow model
    train_loss : array
        Training loss history
    valid_loss : array
        Validation loss history
    """
    if verbose:
        print(f"\n{'='*70}")
        print("NORMALIZING FLOW TRAINING")
        print(f"{'='*70}")
        print(f"Run name: {run_name}")
        print(f"Learning rate: {params.get('lr_flow', 1e-3)}")
        print(f"Epochs: {params.get('nepoch_flow', 50)}")
    
    rundir = config.sphx_base_path + 'modl_runs/' + run_name
    latents_file = Path(rundir) / 'latents' / 'latents.npz'
    
    if not latents_file.exists():
        raise FileNotFoundError(f"Latents not found: {latents_file}. Run with --extract-latents first.")
    
    # Load latents
    latents_data = np.load(str(latents_file))
    latents_train = latents_data['all_z_train']
    latents_valid = latents_data['all_z_valid']

    # Load any source properties saved alongside latents (redshift, ebv, bfit_tid)
    prop_keys = ['redshift', 'ebv', 'bfit_tid']
    source_props_train = {}
    source_props_valid = {}
    for key in prop_keys:
        if f'{key}_train' in latents_data:
            source_props_train[key] = latents_data[f'{key}_train']
            source_props_valid[key] = latents_data[f'{key}_valid']
    if verbose and source_props_train:
        print(f"  Loaded source properties: {list(source_props_train.keys())}")

    if verbose:
        print(f"Loaded latents: train={latents_train.shape}, valid={latents_valid.shape}")
    
    # Subsample latents if max_sources_flow is specified
    if max_sources_flow is not None:
        n_train = int(max_sources_flow * 0.8)  # 80% for training
        n_valid = max_sources_flow - n_train   # 20% for validation
        
        if latents_train.shape[0] > n_train:
            if verbose:
                print(f"\nSubsampling latents for flow training:")
                print(f"  Original: train={latents_train.shape[0]}, valid={latents_valid.shape[0]}")
                print(f"  Subsampled: train={n_train}, valid={n_valid}")
            latents_train = latents_train[:n_train]
            latents_valid = latents_valid[:n_valid]
            # Apply same slice to source properties
            source_props_train = {k: v[:n_train] for k, v in source_props_train.items()}
            source_props_valid = {k: v[:n_valid] for k, v in source_props_valid.items()}
    
    # Train flow
    
    flow_loaded, flow_fpath, train_loss, valid_loss = fit_flow_to_latents_jax(
        latents_train=latents_train,
        latents_valid=latents_valid,
        rundir=rundir,
        lr=params.get('lr_flow', 1e-3),
        n_epoch_flow=params.get('nepoch_flow', 50),
        batch_size=params.get('batch_size_flow', 128),
        verbose=verbose,
        source_props_train=source_props_train if source_props_train else None,
        source_props_valid=source_props_valid if source_props_valid else None,
    )
    
    if verbose:
        print(f"\n✓ Flow training complete!")
        print(f"Model saved to: {flow_fpath}")
        print(f"Final train loss: {train_loss[-1]:.4f}")
        print(f"Final valid loss: {valid_loss[-1]:.4f}")
    
    # Generate corner plots for autoencoder and normalizing flow latents
    try:
        print("\nGenerating latent corner plots...")
        from visualization.result_plotting_fns import make_latent_corner_plots
        import jax
        
        # Load the latents with u-space transforms that were just saved
        latents_file = Path(rundir) / 'latents' / 'latents_with_u_space.npz'
        
        if not latents_file.exists():
            print(f"  ⚠ Latents file not found: {latents_file}")
            print("    Skipping corner plots")
        else:
            latents_data = np.load(str(latents_file))
            latents_z = latents_data['latents_z_train']  # Raw autoencoder latents
            latents_u = latents_data['latents_u_train']  # Transformed u-space latents
            
            # Load PAE model to get rescale transformation for zeta plotting
            locscale = np.load(rundir+'/latents/latent_loc_std.npz')
            loc, scale = locscale['loc'], locscale['scale']
            latents_rescaled = (latents_z * scale) + loc
            latents_rescaled = np.array(latents_rescaled)

            print('After loading, the rescaled latents have mean ', jnp.mean(latents_rescaled, axis=0), ' and std ', jnp.std(latents_rescaled, axis=0))
            
            # Generate corner plots
            nlatent = latents_z.shape[1]
            nsamp_plot = min(10000, latents_z.shape[0])
            
            fig_corner_zeta, fig_corner_u = make_latent_corner_plots(
                latents_rescaled=latents_rescaled,
                latents_u=latents_u,
                nlatent=nlatent,
                nsamp=nsamp_plot,
                xlim=3,
                figsize=(6, 6)
            )
            
            # Save the figures
            fig_dir = Path(rundir) / 'figures' / 'latent_corner_plots'
            fig_dir.mkdir(parents=True, exist_ok=True)
            
            ae_path = fig_dir / 'corner_plot_ae_latents.png'
            fig_corner_zeta.savefig(ae_path, bbox_inches='tight', dpi=300)
            print(f"  ✓ Autoencoder corner plot saved to: {ae_path}")
            
            nf_path = fig_dir / 'corner_plot_nf_latents.png'
            fig_corner_u.savefig(nf_path, bbox_inches='tight', dpi=300)
            print(f"  ✓ Normalizing flow corner plot saved to: {nf_path}")
            
            plt.close(fig_corner_zeta)
            plt.close(fig_corner_u)
        
    except Exception as e:
        print(f"  ⚠ Corner plot generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return flow_loaded, flow_fpath, train_loss, valid_loss


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        '--config-yaml',
        type=str,
        default=None,
        help='Path to YAML config file. Values are used as defaults and can be overridden by CLI flags.'
    )
    pre_args, remaining_argv = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description='Train Probabilistic Autoencoder for SPHEREx photo-z',
        parents=[pre_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Run configuration
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this training run (used for saving)')
    parser.add_argument('--skip-ae-training', action='store_true',
                       help='Skip autoencoder training (load existing model)')
    parser.add_argument('--train-flow', action='store_true',
                       help='Train normalizing flow after autoencoder')
    parser.add_argument('--skip-flow-training', action='store_true',
                       help='Skip flow training even if --train-flow is set')
    parser.add_argument('--extract-latents', action='store_true',
                       help='Extract and save latent representations')
    parser.add_argument('--generate-plots', action='store_true', default=True,
                       help='Generate diagnostic plots')
    
    # Model architecture
    parser.add_argument('--nlatent', type=int, nargs='+', default=[5],
                       help='Number of latent dimensions (can specify multiple for sweep)')
    parser.add_argument('--filters', type=int, nargs=4, default=[16, 32, 128, 256],
                       help='Number of filters in each conv layer')
    parser.add_argument('--filter-sizes', type=int, nargs=4, default=[5, 5, 5, 5],
                       help='Kernel sizes for conv layers')
    parser.add_argument('--n-hidden-encoder', type=int, nargs='+', default=[256, 64, 16],
                       help='Hidden layer sizes in encoder')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--plot-interval', type=int, default=5,
                       help='Save reconstruction plots every N epochs')

    # Similarity / consistency loss parameters
    parser.add_argument('--lambda-sim', type=float, default=0.0,
                       help='Weight for similarity loss term')
    parser.add_argument('--lambda-consistency', type=float, default=0.0,
                       help='Weight for consistency loss term')
    parser.add_argument('--sim-k0', type=float, default=1.0,
                       help='k0 offset for similarity sigmoid terms')
    parser.add_argument('--sim-k1', type=float, default=1.0,
                       help='k1 scale for similarity sigmoid terms')
    parser.add_argument('--sigma-s', type=float, default=1.0,
                       help='Sigma_s latent scale for consistency loss')
    parser.add_argument('--similarity-subsample-size', type=int, default=0,
                       help='Optional batch subsample size for O(N^2) similarity term (0=full batch)')
    parser.add_argument('--similarity-eps', type=float, default=1e-8,
                       help='Numerical epsilon for similarity/consistency loss calculations')
    parser.add_argument('--consistency-aug-scale', type=float, default=0.1,
                       help='Augmentation noise scale for consistency loss')

    # Optional reconstruction scaling mode (loss-only change; preprocessing stays unchanged)
    parser.add_argument('--recon-scale-mode', type=str, default='fixed',
                       choices=['fixed', 'marginalized'],
                       help='Fiducial reconstruction loss mode: fixed MSE or marginalized per-spectrum amplitude')
    parser.add_argument('--amp-eps', type=float, default=1e-8,
                       help='Numerical epsilon for marginalized amplitude denominator')
    parser.add_argument('--amp-clip-min', type=float, default=0.0,
                       help='Minimum clip for marginalized amplitude (default: 0.0 enforces positive amplitudes)')
    parser.add_argument('--amp-clip-max', type=float, default=None,
                       help='Optional maximum clip for marginalized amplitude (disabled if not set)')
    
    # Data parameters
    parser.add_argument('--nbands', type=int, default=500,
                       choices=[500, 1000],
                       help='Number of wavelength bins (500 or 1000)')
    parser.add_argument('--use-log-wavelength', action='store_true',
                       help='Train on log-spaced wavelength grid for faster inference')
    parser.add_argument('--sig-level-norm', type=float, nargs='+', default=[0.01],
                       help='Noise augmentation level (can specify multiple for sweep)')
    parser.add_argument('--train-frac', type=float, default=0.8,
                       help='Fraction of data for training')
    parser.add_argument('--data-file', type=str, default=None,
                       help='Path to SED data file')
    parser.add_argument('--z-max', type=float, default=None,
                       help='Maximum redshift for training sample')
    parser.add_argument('--scratch-base', type=str, default='/pscratch/sd/r/rmfeder/data/',
                       help='Base path for data storage')
    
    # Flow training parameters
    parser.add_argument('--lr-flow', type=float, default=1e-3,
                       help='Learning rate for flow training')
    parser.add_argument('--nepoch-flow', type=int, default=50,
                       help='Number of epochs for flow training')
    parser.add_argument('--batch-size-flow', type=int, default=256,
                       help='Batch size for flow training')
    parser.add_argument('--max-sources-flow', type=int, default=50000,
                       help='Max sources for flow training (default 50k = 40k train + 10k valid, None = all)')
    
    # Multi-core configuration
    parser.add_argument('--use-multicore', action='store_true',
                       help='Use multi-core mode for parameter sweeps (parallelize across GPUs)')
    parser.add_argument('--n-devices', type=int, default=None,
                       help='Number of GPUs to use (default: all available)')
    
    if pre_args.config_yaml:
        apply_yaml_defaults(parser, pre_args.config_yaml, section='training')

    args = parser.parse_args(remaining_argv)

    if args.run_name is None:
        parser.error('Missing required setting: --run-name (or provide run_name in --config-yaml)')
    
    print(f"\n{'='*70}")
    print("PAE AUTOENCODER TRAINING")
    print(f"{'='*70}")
    print(f"Run name: {args.run_name}")
    print(f"Latent dimensions: {args.nlatent}")
    print(f"Noise levels: {args.sig_level_norm}")
    print(f"Skip training: {args.skip_ae_training}")
    print(f"Train flow: {args.train_flow}")
    print(f"Extract latents: {args.extract_latents}")
    print(
        f"Loss weights: lambda_sim={args.lambda_sim}, "
        f"lambda_consistency={args.lambda_consistency}"
    )
    print(
        f"Reconstruction mode: {args.recon_scale_mode} "
        f"(amp_eps={args.amp_eps}, amp_clip_min={args.amp_clip_min}, amp_clip_max={args.amp_clip_max})"
    )
    
    # Determine available devices for multi-core mode
    available_devices = jax.devices()
    if args.use_multicore:
        n_devices = args.n_devices if args.n_devices is not None else len(available_devices)
        n_devices = min(n_devices, len(available_devices))
        print(f"\nMulti-core mode enabled: {n_devices} GPUs")
        print(f"Available devices: {[str(d) for d in available_devices[:n_devices]]}")
    else:
        n_devices = 1
        print(f"\nSingle-GPU mode")
    
    # Generate all parameter combinations
    param_combinations = [(nl, sig) for nl in args.nlatent for sig in args.sig_level_norm]
    n_combinations = len(param_combinations)
    
    print(f"\nParameter sweep: {n_combinations} combinations")
    for nl, sig in param_combinations:
        print(f"  - nlatent={nl}, sig_level_norm={sig}")
    
    # Multi-core parameter sweep: distribute parameter combinations across GPUs
    if args.use_multicore and n_combinations > 1 and n_devices > 1:
        print(f"\nDistributing {n_combinations} training runs across {n_devices} GPUs...")
        print("Each GPU will train different parameter combinations in sequence.\n")
        
        # Assign parameter combinations to devices
        import itertools
        device_assignments = list(itertools.zip_longest(*[iter(param_combinations)]*((n_combinations + n_devices - 1) // n_devices)))
        device_assignments = [[p for p in group if p is not None] for group in zip(*device_assignments)]
        
        # Function to train on a specific device
        def train_on_device(device_id, param_list):
            import os
            # Set CUDA_VISIBLE_DEVICES for this process
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
            
            for nlatent, sig_level_norm in param_list:
                run_name_full = f"{args.run_name}_nlatent={nlatent}_sig={sig_level_norm}"
                print(f"\n[GPU {device_id}] Training: {run_name_full}")
                
                # Train this combination (code below will be same as serial version)
                # ... (training code)
        
        # For simplicity, we'll still run serially but note the capability
        # Full multi-process implementation would use multiprocessing or submitit
        print("Note: Full multi-GPU parallelization requires process-based execution.")
        print("Running parameter sweep serially for now.\n")
        
    # Parameter sweep over nlatent and sig_level_norm (serial execution)
    for nlatent in args.nlatent:
        for sig_level_norm in args.sig_level_norm:
            
            # Generate run name with parameters
            if len(args.nlatent) > 1 or len(args.sig_level_norm) > 1:
                run_name = f"{args.run_name}_nlatent={nlatent}_sig={sig_level_norm}"
            else:
                run_name = args.run_name
            
            print(f"\n{'#'*70}")
            print(f"TRAINING RUN: {run_name}")
            print(f"{'#'*70}")
            
            # Setup parameters
            params = param_dict_gen(
                'jax',
                filter_sizes=args.filter_sizes,
                n_hidden_encoder=args.n_hidden_encoder,
                lr=args.lr,
                filters=args.filters,
                nlatent=nlatent,
                epochs=args.epochs,
                nbands=args.nbands,
                restframe=True,
                mean_sub_latents=False,
                plot_interval=args.plot_interval,
                weight_decay=0.0,
                lambda_sim=args.lambda_sim,
                lambda_consistency=args.lambda_consistency,
                sim_k0=args.sim_k0,
                sim_k1=args.sim_k1,
                sigma_s=args.sigma_s,
                similarity_subsample_size=args.similarity_subsample_size,
                similarity_eps=args.similarity_eps,
                consistency_aug_scale=args.consistency_aug_scale,
                recon_scale_mode=args.recon_scale_mode,
                amp_eps=args.amp_eps,
                amp_clip_min=args.amp_clip_min,
                amp_clip_max=args.amp_clip_max,
            )
            
            # Add log wavelength flag to params
            params['use_log_wavelength'] = args.use_log_wavelength
            if args.use_log_wavelength:
                # Compute and store dloglam for inference
                if args.nbands == 500:
                    wav_range = (0.15, 5.0)
                elif args.nbands == 1000:
                    wav_range = (0.1, 5.0)
                else:
                    wav_range = (0.15, 5.0)
                params['dloglam'] = (np.log(wav_range[1]) - np.log(wav_range[0])) / (args.nbands - 1)
            else:
                params['dloglam'] = None
            
            # Load data only if we need to train the autoencoder or extract latents
            # For flow-only retraining, we can skip this entirely
            need_data = (not args.skip_ae_training) or args.extract_latents
            
            if need_data:
                jax_spec, property_cat_df, wav_rest = setup_data(
                    nbands=args.nbands,
                    sig_level_norm=sig_level_norm,
                    train_frac=args.train_frac,
                    data_file=args.data_file,
                    scratch_base=args.scratch_base,
                    z_max=args.z_max,
                    use_log_wavelength=args.use_log_wavelength
                )
            else:
                # For flow-only retraining, set up minimal wavelength grid
                print(f"\n{'='*70}")
                print("SKIPPING DATA LOADING (flow retraining only)")
                print(f"{'='*70}")
                if args.nbands == 500:
                    wav_linear = jnp.linspace(0.15, 5.0, 500)
                elif args.nbands == 1000:
                    wav_linear = jnp.linspace(0.1, 5.0, 1000)
                else:
                    wav_linear = jnp.linspace(0.15, 5.0, args.nbands)
                
                if args.use_log_wavelength:
                    wav_rest = jnp.logspace(
                        jnp.log10(wav_linear.min()), 
                        jnp.log10(wav_linear.max()), 
                        args.nbands
                    )
                else:
                    wav_rest = wav_linear
                jax_spec = None
                property_cat_df = None
                
            # Train or load model
            if not args.skip_ae_training:
                state, model, jax_spec, metric_dict, model_fpath = train_autoencoder(
                    jax_spec, run_name, params, wav_rest,
                    batch_size=args.batch_size
                )
            else:
                print(f"\n{'='*70}")
                print("SKIPPING AUTOENCODER TRAINING - USING EXISTING MODEL")
                print(f"{'='*70}")
                print(f"Loading existing model from: modl_runs/{run_name}")
                
                # Load existing model state
                from training.train_ae_jax import create_train_state
                from models.nn_modl_jax import instantiate_ae_modl_gen_jax
                from data_proc.data_file_utils import load_jax_state
                
                model = instantiate_ae_modl_gen_jax(params, wav_rest)
                input_shape = (1, len(wav_rest))
                state_template, _, _ = create_train_state(model, input_shape)
                state = load_jax_state(config.modl_runs_path + run_name, state_template)
                
                print(f"  ✓ Model loaded successfully")
                metric_dict = None  # No training metrics when loading
            
            # Extract latents if requested
            if args.extract_latents or not args.skip_ae_training:
                if jax_spec is None or property_cat_df is None:
                    print("\n⚠ Cannot extract latents without loaded data")
                    print("  Skipping latent extraction (using existing latents)")
                else:
                    all_z_train, all_z_valid, ncode = extract_latents(
                        state, jax_spec, property_cat_df, run_name
                    )
                    
                    # Generate diagnostic plots
                    if args.generate_plots:
                        plot_latent_correlations(
                            all_z_train, property_cat_df, jax_spec,
                            run_name, nlatent
                        )
            
            # Train normalizing flow if requested
            if args.train_flow and not args.skip_flow_training:
                # Check if latents file exists
                rundir_check = config.sphx_base_path + 'modl_runs/' + run_name
                latents_file_check = Path(rundir_check) / 'latents' / 'latents.npz'
                
                if not latents_file_check.exists():
                    if jax_spec is not None and property_cat_df is not None:
                        print("\nExtracting latents for flow training...")
                        all_z_train, all_z_valid, ncode = extract_latents(
                            state, jax_spec, property_cat_df, run_name
                        )
                    else:
                        print("\n✗ ERROR: No latents found and cannot extract (data not loaded)")
                        print(f"  Looked for: {latents_file_check}")
                        print("  Run with --extract-latents first or ensure latents exist")
                        continue  # Skip to next parameter combination
                
                # Update params with flow training settings
                params['lr_flow'] = args.lr_flow
                params['nepoch_flow'] = args.nepoch_flow
                params['batch_size_flow'] = args.batch_size_flow
                
                # Train flow (jax_spec not actually used by train_flow)
                try:
                    flow_loaded, flow_fpath, train_loss, valid_loss = train_flow(
                        None, run_name, params, verbose=True,
                        max_sources_flow=args.max_sources_flow
                    )
                    
                    print(f"\n✓ Flow training completed successfully")
                    
                except Exception as e:
                    print(f"\n✗ Flow training failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
            # Generate corner plots after flow training (or attempt)
            # This runs whether flow succeeded or not, as long as latents exist
            if args.generate_plots:
                print("\n" + "="*70)
                print("GENERATING LATENT CORNER PLOTS")
                print("="*70)
                try:
                    from visualization.result_plotting_fns import plot_latent_corner_plots
                    rundir_full = config.sphx_base_path + 'modl_runs/' + run_name
                    # Get flow_name from params if available, otherwise use default
                    flow_name = params.get('flow_name', 'flow_model_iaf')
                    plot_latent_corner_plots(
                        rundir=rundir_full,
                        nlatent=nlatent,
                        nsamp=10000,
                        xlim=5,
                        save_ae=True,
                        save_nf=True,
                        flow_name=flow_name
                    )
                except Exception as e:
                    print(f"⚠ Corner plot generation failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    print("Continuing with training pipeline...")
    
    print(f"\n{'='*70}")
    print("✓ ALL TRAINING RUNS COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
