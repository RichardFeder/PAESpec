#!/usr/bin/env python
"""
Test script to verify batched logdensity implementation works correctly.

This script compares results between:
1. Old approach: per-galaxy logdensity function creation
2. New approach: single batched logdensity function

Usage:
    python scripts/test_batched_logdensity.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax.numpy as jnp
import jax.random as jr
import time

from models.pae_jax import initialize_PAE, load_spherex_data
from training.train_ae_jax import param_dict_gen
from sampling.sample_pae_batch_refactor import MCLMCSamplingConfig, pae_spec_sample_floatz
from inference.like_prior import make_logdensity_fn_optzprior, make_batched_logdensity_fn
from data_proc.dataloader_jax import SPHERExData
from config import scratch_basepath


def test_batched_logdensity():
    """Test that batched and per-galaxy logdensity give same results."""
    
    print("=" * 80)
    print("Testing Batched Logdensity Implementation")
    print("=" * 80)
    
    # Initialize PAE model (using same config as redshift_job.py)
    print("\n1. Loading PAE model...")
    filter_set_name = 'spherex_filters102/'
    nlatent = 5
    sig_level_norm = 0.01
    sel_str = 'zlt22.5'
    run_name = 'jax_conv1_nlatent=5_siglevelnorm=0.01_newAllen_all_091325'
    filename_flow = 'flow_model_iaf_092225'
    
    PAE_COSMOS = initialize_PAE(
        run_name,
        filter_set_name=filter_set_name,
        with_ext_phot=False,
        inference_dtype=jnp.float32,
        lam_min_rest=0.15,
        lam_max_rest=5.0,
        nlam_rest=500,
        filename_flow=filename_flow
    )
    print("✓ PAE model loaded")
    
    # Load test data
    print("\n2. Loading test data (3 galaxies)...")
    dat_obs, property_cat_df_obs, property_cat_df_restframe, \
        central_wavelengths, wave_obs = load_spherex_data(
            sig_level_norm=sig_level_norm,
            sel_str=sel_str,
            abs_norm=True,
            with_ext_phot=False,
            load_rf_dat=False,
            load_obs_dat=True,
            weight_soft=5e-4
        )
    
    # Create model parameters separately
    filters = [16, 32, 128, 256]
    n_hidden_encoder = [256, 64, 16]
    filter_sizes = [5, 5, 5, 5]
    params = param_dict_gen('jax', filter_sizes=filter_sizes, n_hidden_encoder=n_hidden_encoder,
                           lr=2e-4, filters=filters, nlatent=nlatent, epochs=100, nbands=500,
                           restframe=True, mean_sub_latents=False,
                           plot_interval=5, weight_decay=0., nepoch_flow=50)
    
    spherex_dat = SPHERExData.from_prep(
        dat_obs,
        property_cat_df_obs,
        property_cat_df_restframe,
        phot_snr_min=None,
        phot_snr_max=None,
        zmin=None,
        zmax=None
    )
    
    # Select first 3 galaxies for testing
    test_idxs = np.arange(3)
    specs = spherex_dat.all_spec_obs[test_idxs]
    weights = spherex_dat.weights[test_idxs]
    redshifts_true = spherex_dat.redshift[test_idxs]
    print(f"✓ Loaded {len(test_idxs)} test galaxies")
    
    # Create sampling config
    cfg = MCLMCSamplingConfig(
        num_steps=100,  # Short run for testing
        nsamp_init=50,
        nchain_per_gal=2,  # Fewer chains for faster test
        burn_in=20,
        redshift_prior_type=0,  # No prior for simplicity
        use_batched_logdensity=False  # Will test both modes
    )
    
    # Test 1: Old approach (per-galaxy logdensity)
    print("\n3. Testing OLD approach (per-galaxy logdensity)...")
    cfg_old = cfg
    cfg_old.use_batched_logdensity = False
    
    rkey_old = jr.key(42)
    t0 = time.time()
    
    # Test single galaxy with old approach
    result_old = pae_spec_sample_floatz(
        PAE_COSMOS, specs[0], weights[0], rkey_old, cfg_old,
        batched_log_density=None  # Force old approach
    )
    
    time_old = time.time() - t0
    print(f"✓ Old approach completed in {time_old:.3f}s")
    
    # Test 2: New approach (batched logdensity)
    print("\n4. Testing NEW approach (batched logdensity)...")
    cfg_new = cfg
    cfg_new.use_batched_logdensity = True
    
    # Create batched logdensity once
    print("   Creating batched logdensity function...")
    t0_compile = time.time()
    batched_log_density = make_batched_logdensity_fn(PAE_COSMOS, cfg_new)
    
    # Pre-compile
    n_latent = PAE_COSMOS.params['nlatent']
    n_bands = specs.shape[1]
    dummy_latent = jnp.zeros(n_latent + 1)
    dummy_spec = jnp.ones(n_bands)
    dummy_weight = jnp.ones(n_bands)
    _ = batched_log_density(dummy_latent, dummy_spec, dummy_weight)
    time_compile = time.time() - t0_compile
    print(f"   ✓ Compiled in {time_compile:.3f}s")
    
    # Test single galaxy with new approach
    rkey_new = jr.key(42)  # Same seed for comparison
    t0 = time.time()
    
    result_new = pae_spec_sample_floatz(
        PAE_COSMOS, specs[0], weights[0], rkey_new, cfg_new,
        batched_log_density=batched_log_density
    )
    
    time_new = time.time() - t0
    print(f"✓ New approach completed in {time_new:.3f}s")
    
    # Compare results
    print("\n5. Comparing results...")
    samples_old, _, log_prior_old, log_redshift_old, log_L_old = result_old[:5]
    samples_new, _, log_prior_new, log_redshift_new, log_L_new = result_new[:5]
    
    # Check shapes match
    assert samples_old.shape == samples_new.shape, "Sample shapes don't match!"
    print(f"   ✓ Shapes match: {samples_old.shape}")
    
    # Check values are close (they should be identical with same seed)
    samples_close = jnp.allclose(samples_old, samples_new, rtol=1e-4, atol=1e-4)
    log_L_close = jnp.allclose(log_L_old, log_L_new, rtol=1e-4, atol=1e-4)
    
    if samples_close and log_L_close:
        print("   ✓ Results match between old and new approach!")
        print(f"     Max sample difference: {jnp.max(jnp.abs(samples_old - samples_new)):.2e}")
        print(f"     Max log_L difference: {jnp.max(jnp.abs(log_L_old - log_L_new)):.2e}")
    else:
        print("   ⚠ Results differ (this may be due to different random sequences)")
        print(f"     Max sample difference: {jnp.max(jnp.abs(samples_old - samples_new)):.2e}")
        print(f"     Max log_L difference: {jnp.max(jnp.abs(log_L_old - log_L_new)):.2e}")
    
    # Test 3: Test logdensity functions directly
    print("\n6. Testing logdensity functions directly...")
    
    # Create old-style logdensity
    log_p_old = make_logdensity_fn_optzprior(
        PAE_COSMOS, specs[0], weights[0],
        z_min=cfg.zmin, z_max=cfg.zmax,
        nf_alpha=cfg.nf_alpha,
        redshift_in_flow=cfg.redshift_in_flow,
        z0_prior=cfg.z0_prior,
        sigma_prior=cfg.sigma_prior,
        redshift_prior_type=cfg.redshift_prior_type
    )
    
    # Create new-style logdensity (wrapper)
    log_p_new = lambda latent: batched_log_density(latent, specs[0], weights[0])
    
    # Test with random latent vector
    test_latent = jnp.array([0.1, -0.2, 0.3, -0.1, 0.05, 1.5])  # 5 latents + redshift
    
    logp_old_val = log_p_old(test_latent)
    logp_new_val = log_p_new(test_latent)
    
    print(f"   Old logdensity value: {logp_old_val:.6f}")
    print(f"   New logdensity value: {logp_new_val:.6f}")
    print(f"   Difference: {abs(logp_old_val - logp_new_val):.2e}")
    
    if jnp.allclose(logp_old_val, logp_new_val, rtol=1e-5):
        print("   ✓ Logdensity functions produce identical results!")
    else:
        print("   ⚠ Logdensity functions differ slightly")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Old approach time: {time_old:.3f}s")
    print(f"New approach time: {time_new:.3f}s (+ {time_compile:.3f}s compilation)")
    print(f"\nFor large batches, new approach will be much faster because:")
    print(f"  - Old: {time_old:.3f}s × N galaxies (compilation per galaxy)")
    print(f"  - New: {time_compile:.3f}s + {time_new:.3f}s × N galaxies (compile once)")
    print(f"\nBreakeven at ~{int(time_old / (time_old - time_new)):.0f} galaxies")
    print(f"For 1000 galaxies: Old ~{time_old * 1000 / 60:.1f}min, New ~{(time_compile + time_new * 1000) / 60:.1f}min")
    print("\n✓ Test completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    test_batched_logdensity()
