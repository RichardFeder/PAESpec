from flax.core import freeze, unfreeze
import jax
import jax.numpy as jnp
# import optax
# from tqdm import trange


def fine_tune_manual_gradient(
    pae,
    latents,
    z_spec, 
    observed_flux,
    weight,
    lr=1e-5,
    n_epochs=50,
    layers_to_train=['convT_layers_3'],
    epsilon=1e-5,
    sample_fraction=0.1,
    verbose=True,
    per_layer_lr=None,
    loss_type='mse',  # New: 'mse', 'huber', 'percentile', 'weighted'
    huber_delta=1.0,  # For Huber loss
    percentile_cutoff=75,  # For percentile loss (e.g., only use best 75%)
    adaptive_weighting=False  # Dynamically reweight based on current performance
):
    """Fine-tune using finite differences with robust loss options"""
    
    # [Previous setup code stays the same...]
    if isinstance(layers_to_train, str):
        layers_to_train = [layers_to_train]
    
    if per_layer_lr is None:
        per_layer_lr = {layer: lr for layer in layers_to_train}
    else:
        for layer in layers_to_train:
            if layer not in per_layer_lr:
                per_layer_lr[layer] = lr
    
    decoder_params = unfreeze(pae.model.params['decoder'])
    
    def compute_loss(params, return_individual=False):
        """Compute loss with different robust options"""
        model_flux, _, _ = pae.push_spec_marg(
            latents, z_spec,
            decoder_params=params,
            observed_flux=observed_flux,
            weight=weight,
            marginalize_amplitude=True,
            return_rescaled_flux_and_loglike=True,
            use_jit=True
        )
        
        # Individual chi2 per galaxy
        chi2_per_galaxy = jnp.sum(weight * (model_flux - observed_flux) ** 2, axis=-1)
        
        if return_individual:
            return chi2_per_galaxy
        
        # Apply different loss strategies
        if loss_type == 'mse':
            loss = float(jnp.mean(chi2_per_galaxy))
            
        elif loss_type == 'huber':
            # Huber loss - quadratic for small errors, linear for large
            mask = chi2_per_galaxy < huber_delta
            huber_loss = jnp.where(
                mask,
                0.5 * chi2_per_galaxy,
                huber_delta * (jnp.sqrt(chi2_per_galaxy) - 0.5 * jnp.sqrt(huber_delta))
            )
            loss = float(jnp.mean(huber_loss))
            
        elif loss_type == 'percentile':
            # Only use the best performing galaxies
            threshold = jnp.percentile(chi2_per_galaxy, percentile_cutoff)
            mask = chi2_per_galaxy <= threshold
            loss = float(jnp.mean(chi2_per_galaxy[mask]))
            
        elif loss_type == 'weighted':
            # Inverse weighting - give less weight to high chi2
            weights = 1.0 / (1.0 + chi2_per_galaxy)
            loss = float(jnp.sum(weights * chi2_per_galaxy) / jnp.sum(weights))
            
        return loss
    
    loss_history = []
    
    # Pre-compute sampling indices
    sampling_indices = {}
    for layer_name in layers_to_train:
        if layer_name in decoder_params:
            sampling_indices[layer_name] = {}
            for param_name in ['kernel', 'bias']:
                if param_name in decoder_params[layer_name]:
                    param_shape = decoder_params[layer_name][param_name].shape
                    total_params = np.prod(param_shape[:2] if len(param_shape) > 2 else param_shape)
                    n_samples = max(1, int(total_params * sample_fraction))
                    
                    all_indices = list(np.ndindex(*param_shape[:2] if len(param_shape) > 2 else param_shape))
                    sampled_indices = np.random.choice(len(all_indices), size=n_samples, replace=False)
                    sampling_indices[layer_name][param_name] = [all_indices[i] for i in sampled_indices]
    
    # Track performance for adaptive weighting
    galaxy_weights = jnp.ones(len(latents))
    
    for epoch in range(n_epochs):
        # Optionally update galaxy weights based on current performance
        if adaptive_weighting and epoch > 0:
            chi2_individual = compute_loss(freeze(decoder_params), return_individual=True)
            # Exponential moving average of weights
            new_weights = 1.0 / (1.0 + chi2_individual)
            galaxy_weights = 0.9 * galaxy_weights + 0.1 * new_weights
            galaxy_weights = galaxy_weights / jnp.mean(galaxy_weights)  # Normalize
        
        current_loss = compute_loss(freeze(decoder_params))
        loss_history.append(current_loss)
        
        # Compute gradients
        grads = {}
        
        for layer_name in layers_to_train:
            if layer_name not in decoder_params:
                if verbose:
                    print(f"Warning: Layer {layer_name} not found in decoder params")
                continue
                
            grads[layer_name] = {}
            
            for param_name in ['kernel', 'bias']:
                if param_name in decoder_params[layer_name]:
                    param_shape = decoder_params[layer_name][param_name].shape
                    grad = jnp.zeros(param_shape)
                    
                    for idx in sampling_indices[layer_name][param_name]:
                        temp_params = unfreeze(decoder_params)
                        original_val = temp_params[layer_name][param_name][idx]
                        temp_params[layer_name][param_name] = temp_params[layer_name][param_name].at[idx].set(original_val + epsilon)
                        
                        loss_plus = compute_loss(freeze(temp_params))
                        
                        grad = grad.at[idx].set((loss_plus - current_loss) / (epsilon * sample_fraction))
                    
                    grads[layer_name][param_name] = grad
        
        # Apply gradient updates
        for layer_name in grads:
            layer_lr = per_layer_lr[layer_name]
            for param_name in grads[layer_name]:
                decoder_params[layer_name][param_name] -= layer_lr * grads[layer_name][param_name]
        
        if verbose and (epoch % 5 == 0 or epoch == 0):
            # Also report statistics on individual chi2 values
            chi2_individual = compute_loss(freeze(decoder_params), return_individual=True)
            grad_info = []
            for layer_name in grads:
                if 'kernel' in grads[layer_name]:
                    grad_norm = float(jnp.linalg.norm(grads[layer_name]['kernel']))
                    grad_info.append(f"{layer_name}:{grad_norm:.2f}")
            
            print(f"Epoch {epoch+1}: Loss={current_loss:.3f}, "
                  f"Chi2 median={float(jnp.median(chi2_individual)):.3f}, "
                  f"Chi2 95%={float(jnp.percentile(chi2_individual, 95)):.3f}, "
                  f"Grad=[{', '.join(grad_info)}]")
    
    return freeze(decoder_params), loss_history


def deep_inspect_model(pae, latents, observed_flux, weight):
    """Thoroughly inspect the model to understand parameter structure"""
    
    print("=== FULL PARAMETER STRUCTURE ===")
    params = pae.model.params
    
    def print_tree(d, indent=0):
        for k, v in d.items():
            if isinstance(v, dict):
                print("  " * indent + f"{k}:")
                print_tree(v, indent + 1)
            else:
                print("  " * indent + f"{k}: shape={v.shape}, dtype={v.dtype}")
    
    print_tree(params)
    
    print("\n=== DECODER INITIALIZATION TEST ===")
    # Try to understand what the decoder expects
    dummy_latent = jnp.ones((2, latents.shape[1]))  # Single sample
    dummy_z = jnp.array([0.5, 0.5])
    
    # Test with original params
    try:
        output = pae.decoder.apply({'params': params['decoder']}, dummy_latent)
        print(f"Direct decoder output shape: {output.shape}")
    except Exception as e:
        print(f"Direct decoder call failed: {e}")
    
    # Test the push_spec_marg
    try:
        output = pae.push_spec_marg(
            dummy_latent, dummy_z,
            decoder_params=params['decoder'],
            observed_flux=observed_flux[:2],
            weight=weight[:2],
            marginalize_amplitude=True,
            return_rescaled_flux_and_loglike=True,
            use_jit=False
        )
        print("push_spec_marg works with original params")
    except Exception as e:
        print(f"push_spec_marg failed: {e}")
    
    return params

def diagnose_with_proper_manual_update(
    pae,
    latents,
    z_spec,
    observed_flux,
    weight,
    lr=1e-6,
    n_epochs=50,
    layername ='convT_layers_3'
):
    """Properly handle JAX immutability in manual updates"""
    
    decoder_params = unfreeze(pae.model.params['decoder'])
    # original_kernel = decoder_params['convT_final']['kernel'].copy()
    original_kernel = decoder_params[layername]['kernel'].copy()
    
    def loss_fn(params):
        model_flux, _, _ = pae.push_spec_marg(
            latents, z_spec,
            decoder_params=params,
            observed_flux=observed_flux,
            weight=weight,
            marginalize_amplitude=True,
            return_rescaled_flux_and_loglike=True
        )
        chi2 = jnp.sum(weight * (model_flux - observed_flux) ** 2, axis=-1)
        return jnp.mean(chi2)
    
    loss_grad_fn = jax.value_and_grad(loss_fn)
    
    print(f"Initial kernel mean: {float(jnp.mean(original_kernel)):.10f}")
    print(f"LR = {lr}")
    print()
    
    for epoch in range(n_epochs):
        loss_val, grads = loss_grad_fn(freeze(decoder_params))
        
        # PROPERLY create new params dict with updated values
        grad_kernel = grads[layername]['kernel']
        new_kernel = decoder_params[layername]['kernel'] - lr * grad_kernel
        
        # Create completely new params dict
        decoder_params = unfreeze(pae.model.params['decoder'])  # Fresh copy
        decoder_params[layername]['kernel'] = new_kernel    # Update with new array
        
        # Check change with higher precision
        param_change = float(jnp.max(jnp.abs(new_kernel - original_kernel)))
        param_mean = float(jnp.mean(new_kernel))
        grad_max = float(jnp.max(jnp.abs(grad_kernel)))
        
        print(f"Epoch {epoch+1}:")
        print(f"  Loss: {float(loss_val):.2f}")
        print(f"  Max gradient: {grad_max:.2e}")
        print(f"  Expected max change this step: {lr * grad_max:.2e}")
        print(f"  Actual max param change from start: {param_change:.2e}")
        print(f"  Current param mean: {param_mean:.10f}")
        print()
        
        if float(loss_val) > 1e4:
            break
    
    return decoder_params

def test_decoder_sensitivity(pae, latents, z_spec, observed_flux, weight):
    """Test if convT_final parameters actually affect the output"""
    
    decoder_params = unfreeze(pae.model.params['decoder'])
    
    def get_output(scale_factor):
        params = unfreeze(pae.model.params['decoder'])
        params['convT_final']['kernel'] = params['convT_final']['kernel'] * scale_factor
        
        model_flux, _, _ = pae.push_spec_marg(
            latents, z_spec,  # Just use 10 examples
            decoder_params=freeze(params),
            observed_flux=observed_flux,
            weight=weight,
            marginalize_amplitude=True,
            return_rescaled_flux_and_loglike=True
        )
        return model_flux
    
    flux_original = get_output(1.0)
    flux_scaled = get_output(1.1)  # Scale weights by 10%
    
    max_diff = float(jnp.max(jnp.abs(flux_scaled - flux_original)))
    mean_diff = float(jnp.mean(jnp.abs(flux_scaled - flux_original)))
    
    print(f"Max flux difference when scaling convT_final by 10%: {max_diff:.2e}")
    print(f"Mean flux difference: {mean_diff:.2e}")
    print(f"Original flux mean: {float(jnp.mean(jnp.abs(flux_original))):.2e}")

# def fine_tune_decoder_last_layers(
#     pae,
#     latents,
#     z_spec,
#     observed_flux,
#     weight,
#     lr=1e-5,                  # Small LR for stability
#     n_epochs=50,
#     batch_size=None,           # Default: full dataset per step
#     layers_to_train=('convT_layers_3', 'convT_final'),
#     grad_clip_norm=0.5,        # Clip gradients to avoid blow-up
#     grad_scale=0.1,            # Scale gradients down
#     l2_reg_strength=1e-2,      # L2 regularization toward original weights
#     verbose=True
# ):
#     """
#     Fine-tunes only selected layers of the decoder in a PAE model with stability measures.

#     Args:
#         pae: PAE model instance
#         latents: (N, latent_dim) latent codes
#         z_spec: (N,) redshifts
#         observed_flux: (N, n_wave) observed fluxes
#         weight: (N, n_wave) weights for loss
#         lr: learning rate
#         n_epochs: number of epochs
#         batch_size: number of samples per batch; None = full dataset
#         layers_to_train: tuple of decoder layer names to update
#         grad_clip_norm: gradient clipping norm
#         grad_scale: scale factor for gradients
#         l2_reg_strength: L2 regularization strength toward original weights
#         verbose: print progress

#     Returns:
#         pae: updated PAE instance
#         loss_history: list of loss values per epoch
#     """

#     # --- 1. Get full decoder params ---
#     decoder_params_full = unfreeze(pae.model.params['decoder'])
#     orig_params = freeze(pae.model.params['decoder'])  # For L2 regularization

#     # --- 2. Create optimizer mask ---
#     def mask_fn(params):
#         """Mask so that only layers_to_train are updated."""
#         mask = {}
#         for lname, lparams in params.items():
#             mask[lname] = {}
#             for pname in lparams.keys():
#                 # Train all params in selected layers
#                 mask[lname][pname] = (lname in layers_to_train)
#         return freeze(mask)

#     mask = mask_fn(decoder_params_full)

#     # --- 3. Create optimizer with stability measures ---
#     tx = optax.chain(
#         optax.scale(grad_scale),                        # Scale gradients
#         optax.clip_by_global_norm(grad_clip_norm),      # Clip gradients
#         optax.masked(optax.adam(lr), mask)              # Adam with masking
#     )

#     opt_state = tx.init(freeze(decoder_params_full))

#     # --- 4. Loss function ---
#     def loss_fn(params, latents_batch, z_batch, flux_batch, w_batch):
#         model_flux, _, _ = pae.push_spec_marg(
#             latents_batch,
#             z_batch,
#             decoder_params=params,
#             observed_flux=flux_batch,
#             weight=w_batch,
#             marginalize_amplitude=True,
#             return_rescaled_flux_and_loglike=True
#         )
#         chi2 = jnp.sum(w_batch * (model_flux - flux_batch) ** 1.5, axis=-1)

#         # chi2_indiv = w_batch * (model_flux - flux_batch) ** 2
#         # print('min/max chi2:', np.min(chi2_indiv), np.max(chi2_indiv))

        
#         # print('chi2 has shape', chi2)
#         # print(np.min(chi2), np.max(chi2))
        
#         # Safe L2 regularization toward original weights
#         l2_reg = 0.0
#         for layer in layers_to_train:
#             if layer in params and layer in orig_params:
#                 for pname in params[layer]:
#                     if pname in orig_params[layer]:
#                         l2_reg += jnp.sum((params[layer][pname] - orig_params[layer][pname]) ** 2)
#         l2_reg *= l2_reg_strength
    
#         return jnp.mean(chi2)

#     loss_grad_fn = jax.value_and_grad(loss_fn)

#     # --- 5. Training loop ---
#     N = latents.shape[0]
#     loss_history = []

#     for epoch in trange(n_epochs, disable=not verbose):
#         perm = jax.random.permutation(jax.random.PRNGKey(epoch), N)

#         if batch_size is None:
#             # Full dataset in one pass
#             idx = perm
#             batch_loss, grads = loss_grad_fn(
#                 freeze(decoder_params_full),
#                 latents[idx],
#                 z_spec[idx],
#                 observed_flux[idx],
#                 weight[idx]
#             )
#             updates, opt_state = tx.update(grads, opt_state, freeze(decoder_params_full))
#             decoder_params_full = unfreeze(optax.apply_updates(freeze(decoder_params_full), updates))
#             loss_history.append(float(batch_loss))
#         else:
#             # Mini-batch training
#             epoch_loss = 0.0
#             n_batches = N // batch_size + int(N % batch_size != 0)
#             for i in range(n_batches):
#                 idx = perm[i * batch_size: (i + 1) * batch_size]
#                 batch_loss, grads = loss_grad_fn(
#                     freeze(decoder_params_full),
#                     latents[idx],
#                     z_spec[idx],
#                     observed_flux[idx],
#                     weight[idx]
#                 )
#                 updates, opt_state = tx.update(grads, opt_state, freeze(decoder_params_full))
#                 decoder_params_full = unfreeze(optax.apply_updates(freeze(decoder_params_full), updates))
#                 epoch_loss += float(batch_loss)
#             loss_history.append(epoch_loss / n_batches)

#         if verbose:
#             print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss_history[-1]:.6f}")

#     # --- 6. Save updated params back into model ---
#     pae.model = pae.model.replace(
#         params=freeze({
#             **pae.model.params,
#             'decoder': freeze(decoder_params_full)
#         })
#     )
#     pae.decoder_params = pae.model.params['decoder']

#     return pae, loss_history
