import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import distrax
from flax.training import train_state

import os
from utils.utils_jax import *
from data_proc.data_file_utils import *


class ZScoreTransform(nn.Module):
    """Z-score normalization transform: (x - mean) / std (forward) and inverse."""
    mean: jnp.ndarray  # Precomputed mean of the dataset
    std: jnp.ndarray   # Precomputed standard deviation of the dataset

    def __call__(self, x, inverse=False):
        if inverse:
            return x * self.std + self.mean  # Undo normalization
        else:
            return (x - self.mean) / self.std  # Apply normalization

class ActNorm(nn.Module):
    """Activation Normalization Layer (ActNorm)"""
    num_features: int
    init_stats: dict = None  # Dictionary with 'mean' and 'std' as keys

    def setup(self):

        mean_init = self.init_stats.get('mean', jnp.zeros(self.num_features)) if self.init_stats else jnp.zeros(self.num_features)
        std_init = self.init_stats.get('std', jnp.ones(self.num_features)) if self.init_stats else jnp.ones(self.num_features)

        self.loc = self.param('loc', lambda rng, shape: mean_init, (self.num_features,))
        self.scale = self.param('scale', lambda rng, shape: std_init, (self.num_features,))

    def __call__(self, x, inverse=False):
        if inverse:
            return (x - self.loc) / self.scale
        else:
            return x * self.scale + self.loc
            
class MADE(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        """Computes autoregressive parameters (mean and log std)."""
        h = nn.relu(nn.Dense(self.hidden_dim)(x))
        h = nn.relu(nn.Dense(self.hidden_dim)(h))
        out = nn.Dense(self.output_dim * 2)(h)  # Output both mean and log std
        mean, log_std = jnp.split(out, 2, axis=-1)  # Split into mean & log std
        return mean, jnp.tanh(log_std)  # Tanh ensures controlled log_std range



class MaskedAutoregressiveFlow(nn.Module):
    """MAF transformation for density estimation."""
    hidden_dim: int

    @nn.compact
    def __call__(self, x, inverse=False):
        made = MADE(self.hidden_dim, x.shape[-1])
        mean, log_std = made(x)

        if inverse:
            # Inverse transformation (sampling)
            z = mean + jnp.exp(log_std) * x
        else:
            # Forward transformation (density estimation)
            z = (x - mean) / jnp.exp(log_std)

        return z, log_std

class MAFModel(nn.Module):
    """Full MAF Model with multiple flow steps."""
    hidden_dim: int
    num_flows: int
    mean: jnp.ndarray
    std: jnp.ndarray

    @nn.compact
    def __call__(self, x, inverse=False):
        zscore = ZScoreTransform(self.mean, self.std)

        if not inverse:
            x = zscore(x, inverse=False)

        log_q = jnp.zeros(x.shape[0])  # Log likelihood tracker
        
        for n in range(self.num_flows):
            maf = MaskedAutoregressiveFlow(self.hidden_dim)
            x, log_std = maf(x, inverse=inverse)

            if not inverse:
                log_q -= jnp.sum(log_std, axis=-1)  # Track log likelihood

        if inverse:
            x = zscore(x, inverse=True)

        return x, log_q if not inverse else x


class Permutation(nn.Module):
    """Fixed random permutation of input features."""
    num_features: int

    def setup(self):

        key = jax.random.PRNGKey(42)
        perm = jax.random.permutation(key, self.num_features)  # Generate permutation
        self.perm = perm  # Store directly as an attribute
        self.inv_perm = jnp.argsort(perm)  # Compute inverse permutation

    def __call__(self, x, inverse=False):
        if inverse:
            return x[..., self.inv_perm]  # Apply inverse permutation
        else:
            return x[..., self.perm]  # Apply forward permutation

class InverseAutoregressiveFlow(nn.Module):
    hidden_dim: int

    init_stats: dict = None  # Optional dictionary for ActNorm initialization

    @nn.compact
    def __call__(self, x, inverse=False):

        actnorm = ActNorm(num_features=x.shape[-1])  # Initialize ActNorm layer
        if inverse==False:
            x = actnorm(x, inverse=False)
            
        made = MADE(self.hidden_dim, x.shape[-1])
        mean, log_std = made(x)

        if inverse:
            z = (x - mean) / jnp.exp(log_std)
        else:
            z = mean + jnp.exp(log_std) * x  # Inverse autoregressive transformation

        if inverse:
            z = actnorm(z, inverse=True)
            
        return z, log_std


''' Inverse autoregressive flow for fast sampling '''
class IAFModel(nn.Module):
    hidden_dim: int
    num_flows: int

    mean: jnp.ndarray
    std: jnp.ndarray

    init_stats: dict = None  # Optional dictionary for ActNorm initialization


    @nn.compact
    def __call__(self, x, inverse=False):

        zscore = ZScoreTransform(self.mean, self.std)

        if not inverse:
            x = zscore(x, inverse=False)
            # print('x has variance', jnp.std(x, axis=0), jnp.mean(x ,axis=0))

        # actnorm = ActNorm(num_features=x.shape[-1], init_stats=self.init_stats)  # Initialize ActNorm layer

        # if inverse==False:
        #     x = actnorm(x, inverse=False)

        log_q = jnp.zeros(x.shape[0])  # Log likelihood tracker
        
        for n in range(self.num_flows):

            perm = Permutation(x.shape[-1])  # Create permutation layer
            iaf = InverseAutoregressiveFlow(self.hidden_dim)

            if not inverse:

                x = perm(x, inverse=False)  # Apply forward permutation
                x, log_std = iaf(x, inverse=False)
                log_q -= jnp.sum(log_std, axis=-1)  # Track log likelihood

            else:
                x, log_std = iaf(x, inverse=True)
                x = perm(x, inverse=True)


        if inverse:
            x = zscore(x, inverse=True)

        return x, log_q if not inverse else x



# ---------------------

# class MADE(nn.Module):
#     hidden_dim: int
#     output_dim: int

#     @nn.compact
#     def __call__(self, x):
#         """Computes autoregressive parameters (mean and log std)."""
#         h = nn.relu(nn.Dense(self.hidden_dim)(x))
#         h = nn.relu(nn.Dense(self.hidden_dim)(h))
#         out = nn.Dense(self.output_dim * 2)(h)  # Output both mean and log std
#         mean, log_std = jnp.split(out, 2, axis=-1)  # Split into mean & log std
#         return mean, jnp.tanh(log_std)  # Tanh ensures controlled log_std range


# class InverseAutoregressiveFlow(nn.Module):
#     hidden_dim: int

#     @nn.compact
#     def __call__(self, x):
#         made = MADE(self.hidden_dim, x.shape[-1])
#         mean, log_std = made(x)
#         z = mean + jnp.exp(log_std) * x  # Inverse autoregressive transformation
#         return z, log_std

# ''' Inverse autoregressive flow for fast sampling '''
# class IAFModel(nn.Module):
#     hidden_dim: int
#     num_flows: int

#     @nn.compact
#     def __call__(self, x):
#         log_q = jnp.zeros(x.shape[0])  # Log likelihood tracker
#         for _ in range(self.num_flows):
#             iaf = InverseAutoregressiveFlow(self.hidden_dim)
#             x, log_std = iaf(x)
#             log_q -= jnp.sum(log_std, axis=-1)  # Track log likelihood
#         return x, log_q

def loss_fn_flow(params, state, x):
    """Negative log-likelihood loss."""

    z, log_q = state.apply_fn({'params': params}, x)
    log_p = distrax.Normal(0., 1.).log_prob(z).sum(axis=-1)  # Base Gaussian log prob

    return -jnp.mean(log_p - log_q), log_q  # Maximize likelihood

def loss_fn(params, state, x):
    """Negative log-likelihood loss."""
    z, log_q = state.apply_fn({'params': params}, x)
    log_p = distrax.Normal(0, 1).log_prob(z).sum(axis=-1)  # Base Gaussian log prob
    return -jnp.mean(log_p - log_q)  # Maximize likelihood

def one_cycle_lr_schedule(max_lr, total_steps, pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
    """Implements OneCycleLR learning rate schedule using Optax."""

    initial_lr = jnp.float32(max_lr / div_factor)
    final_lr = max_lr / final_div_factor

    warmup_steps = int(pct_start * total_steps)
    anneal_steps = total_steps - warmup_steps

    warmup_fn = optax.linear_schedule(
        init_value=initial_lr, end_value=max_lr, transition_steps=warmup_steps
    )

    anneal_fn = optax.cosine_decay_schedule(
        init_value=max_lr, decay_steps=anneal_steps, alpha=final_lr / max_lr
    )
    return optax.join_schedules([warmup_fn, anneal_fn], boundaries=[warmup_steps])


def init_flow_model(nlatent, mean, std, init_stats, lr=1e-3, batch_size=128, hidden_features=50, num_transforms=4, print_summary=False, steps_per_epoch=20, n_epoch=100, \
                   flow_type='maf'):
    ''' mean and std are computed  over dataset for z-score rescaling before feeding to IAF '''
    
    key = jax.random.PRNGKey(42)

    # Initialize model & training state
    # model = MaskedAutoregressiveFlow(nlatent, hidden_features=hidden_features, num_transforms=num_transforms, init_stats=init_stats)
    # model = NeuralDensityEstimator(nlatent, hidden_features=hidden_features, num_transforms=num_transforms, num_bins=10, initial_pos=initial_pos)
    if flow_type=='maf':
        model = MAFModel(hidden_dim=hidden_features, num_flows=num_transforms, mean=mean, std=std)

    elif flow_type=='iaf':
        model = IAFModel(hidden_dim=hidden_features, num_flows=num_transforms, mean=mean, std=std, init_stats=init_stats)

    x_init = jax.random.normal(key, (batch_size, nlatent))  # 5D input
    params = model.init(key, x_init)['params']

    lr_schedule = one_cycle_lr_schedule(max_lr=1e-2, total_steps=steps_per_epoch*n_epoch)
    
    # tx = optax.adam(learning_rate=lr_schedule)
    tx = optax.adam(learning_rate=lr)

        # Initialize training state
    class TrainState(train_state.TrainState):
        pass  # Extend if needed (e.g., add batch statistics)

    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    if print_summary:
        print(model.tabulate(jax.random.PRNGKey(0), x_init))

    return state

def fit_flow_to_latents_jax(latents_train, latents_valid, rundir=None, lr=1e-3, n_epoch_flow=100, batch_size=128, \
                           interval=4, knots=8, verbose=False, flow_name='flow_model_iaf', invert=False,
                           source_props_train=None, source_props_valid=None):
    """
    source_props_train / source_props_valid : dict or None
        Optional dicts of source properties (e.g. 'all_redshift', 'ebv', 'bfit_tid')
        that are indexed to the same rows as latents_train/valid and will be saved
        alongside the u-space latents in latents_with_u_space.npz.
    """
    
    
    flow_dir = rundir+'/flows/'
    if not os.path.isdir(flow_dir):
        os.makedirs(flow_dir)

    # Use the standard filename format for compatibility
    filename_save = f'{flow_name}.pkl'
    flow_fpath = flow_dir + '/' + filename_save
                    
    # Store original latents before any normalization for saving later
    latents_train_orig = jnp.array(latents_train)
    latents_valid_orig = jnp.array(latents_valid)
    
    latents_train = jnp.array(latents_train)
    latents_valid = jnp.array(latents_valid)
    
    if verbose:
        print('latents have type', latents_train.dtype)

    print('Training normalizing flow using flowjax...')
    print('knots, interval, lr, batch_size, invert:', knots, interval, lr, batch_size, invert)
    
    # Use the flowjax training function (returns equinox-compatible model)
    from models.flowjax_modl import train_flow_flowjax, init_flowjax_modl
    flow_iaf, loc, scale, losses = train_flow_flowjax(
        latents_train, 
        invert=invert, 
        knots=knots, 
        interval=interval, 
        learning_rate=lr,
        n_epoch_flow=n_epoch_flow,
        batch_size=batch_size
    )

    # Save loc and scale for later use
    latent_stats_path = rundir + '/latents/latent_loc_std.npz'
    print(f"Saving latent normalization statistics to: {latent_stats_path}")
    np.savez(latent_stats_path, loc=loc, scale=scale)
    
    # Save using equinox serialization (compatible with load_model)
    save_model(flow_iaf, rundir, filename=filename_save)
    
    # Load using a fresh template (important: use untrained flow as template, not the trained one)
    # This ensures we load the same structure as when loading in notebooks
    key_load = jax.random.PRNGKey(42)
    _, flow_template = init_flowjax_modl(key_load, latents_train.shape[1], invert=invert, knots=knots, interval=interval)
    flow_loaded = load_model(flow_template, rundir, filename=filename_save)
    
    print(f"\n✓ Flow model saved to: {flow_fpath}")

    # Save loc/scale for later use - this is the canonical source of normalization params
    latent_stats_path = rundir + '/latents/latent_loc_std.npz'
    print('loc and scale are ', loc, scale)
    print(f"Saving latent normalization statistics to: {latent_stats_path}")
    np.savez(latent_stats_path, loc=loc, scale=scale)
    
    print("\nTransforming latents to u-space...")
    print(f"  Input latents - train mean: {jnp.mean(latents_train, axis=0)}")
    print(f"  Input latents - train std:  {jnp.std(latents_train, axis=0)}")
    
    # Use the saved loc and scale to ensure consistency with training
    # Apply two-step transformation: z-score normalization -> flow.bijection.inverse
    # latents_rescaled_train = (latents_train + loc) * scale
    latents_rescaled_train = (latents_train * scale) + loc
    latents_rescaled_valid = (latents_valid * scale) + loc

    print(f"  Rescaled latents - train mean: {jnp.mean(latents_rescaled_train, axis=0)}")
    print(f"  Rescaled latents - train std:  {jnp.std(latents_rescaled_train, axis=0)}")

    print("\nTransforming latents to u-space...")

    latents_u_train = jax.vmap(flow_loaded.bijection.inverse)(latents_rescaled_train)
    latents_u_valid = jax.vmap(flow_loaded.bijection.inverse)(latents_rescaled_valid)

    
    # Verify the u-space latents are approximately N(0,1)
    print("\nVerifying u-space transformation:")
    print(f"  u-space train - mean: {jnp.mean(latents_u_train, axis=0)}")
    print(f"  u-space train - std:  {jnp.std(latents_u_train, axis=0)}")
    print(f"  Expected: mean≈0, std≈1 for each dimension")
    
    # Save both the original latents (z-space) and transformed latents (u-space)
    latent_save_path = rundir + '/latents/latents_with_u_space.npz'
    print(f"\nSaving latents with u-space transforms to: {latent_save_path}")
    
    save_dict = dict(
        latents_z_train=np.array(latents_train_orig),
        latents_z_valid=np.array(latents_valid_orig),
        latents_u_train=np.array(latents_u_train),
        latents_u_valid=np.array(latents_u_valid),
        loc=loc,
        scale=scale,
    )
    # Persist source properties (redshift, ebv, bfit_tid, …) with matching row order
    if source_props_train is not None:
        for k, v in source_props_train.items():
            save_dict[f'{k}_train'] = np.array(v)
    if source_props_valid is not None:
        for k, v in source_props_valid.items():
            save_dict[f'{k}_valid'] = np.array(v)
    np.savez(latent_save_path, **save_dict)
    
    # Extract loss history - flowjax returns a dict with various keys
    # Check what keys are available and use the appropriate one
    if isinstance(losses, dict):
        # Try common keys
        if 'loss' in losses:
            train_loss_vs_epoch = np.array(losses['loss'])
        elif 'train_loss' in losses:
            train_loss_vs_epoch = np.array(losses['train_loss'])
        else:
            # If losses dict exists but has unexpected keys, use the first array value
            print(f"Warning: losses dict has keys {list(losses.keys())}, using first array value")
            for v in losses.values():
                if hasattr(v, '__len__'):
                    train_loss_vs_epoch = np.array(v)
                    break
            else:
                train_loss_vs_epoch = np.array([0.0])  # Fallback
    else:
        # If losses is already an array
        train_loss_vs_epoch = np.array(losses)
    
    valid_loss_vs_epoch = np.zeros_like(train_loss_vs_epoch)  # flowjax doesn't track validation separately
    
    # Save flow training metrics
    flow_metrics_path = rundir + '/flow_metrics.npz'
    print(f"\nSaving flow training metrics to: {flow_metrics_path}")
    np.savez(flow_metrics_path, trainloss=train_loss_vs_epoch, validloss=valid_loss_vs_epoch)
            
    return flow_loaded, flow_fpath, train_loss_vs_epoch, valid_loss_vs_epoch


def transform_latents_to_u_space(flow_model, latents, loc, scale):
    """
    Transform latents to normalizing flow space (u-space) using trained flow.
    
    This applies the complete transformation matching the user's workflow:
    1. Z-score normalization: (latents + loc) * scale  [equivalent to rescale.inverse]
    2. Flow bijection inverse transformation  [equivalent to flow.bijection.inverse]
    
    The result should be approximately N(0,1) distributed.
    
    Parameters:
    -----------
    flow_model : flowjax.Transformed
        Trained normalizing flow model (loaded via load_model)
    latents : jnp.ndarray
        Input latents in z-space (raw autoencoder outputs), shape (n_samples, n_latent)
    loc : jnp.ndarray
        Location parameter for z-score normalization (from latent_loc_std.npz)
    scale : jnp.ndarray
        Scale parameter for z-score normalization (from latent_loc_std.npz)
        
    Returns:
    --------
    latents_u : jnp.ndarray  
        Transformed latents in u-space (should be ~N(0,1) distributed)
    """
    from flowjax.bijections import Affine
    
    # Create the preprocessing Affine transform from saved loc/scale
    rescale_transform = Affine(loc, scale)
    
    # Apply two-step transformation: rescale.transform -> flow.bijection.inverse
    latents_rescaled = jax.vmap(rescale_transform.transform)(latents)
    latents_u = jax.vmap(flow_model.bijection.inverse)(latents_rescaled)
    
    return latents_u


# def save_training_latents_with_u_transform(PAE_obj, dat_obj, rundir, save_u_transform=True):
#     """
#     Save training latents along with their normalizing flow transformations.
#     Call this after training both the autoencoder and normalizing flow.
    
#     Parameters:
#     -----------
#     PAE_obj : PAE_JAX
#         Trained PAE model with flow loaded
#     dat_obj : spec_data_jax
#         Data object containing training/validation data 
#     rundir : str
#         Run directory to save files
#     save_u_transform : bool
#         Whether to compute and save u-space transforms
#     """
    
#     print("Extracting and saving training latents with u-space transforms...")
    
#     # Get encoded latents from autoencoder
#     latents_train = PAE_obj.encoder.apply({'params': PAE_obj.encoder_params}, dat_obj.data_train)
#     latents_valid = PAE_obj.encoder.apply({'params': PAE_obj.encoder_params}, dat_obj.data_valid)
    
#     save_dict = {
#         'latents_z_train': np.array(latents_train),
#         'latents_z_valid': np.array(latents_valid),
#         'train_indices': dat_obj.trainidx,
#         'valid_indices': dat_obj.valididx
#     }
    
#     if save_u_transform:
#         # Transform to u-space using the normalizing flow
#         print("Computing u-space transforms...")
#         latents_u_train = PAE_obj.get_encoded_u(dat_obj.data_train) 
#         latents_u_valid = PAE_obj.get_encoded_u(dat_obj.data_valid)
        
#         save_dict.update({
#             'latents_u_train': np.array(latents_u_train), 
#             'latents_u_valid': np.array(latents_u_valid)
#         })
        
#         print(f"u-space latents stats - train mean: {np.mean(latents_u_train):.4f}, std: {np.std(latents_u_train):.4f}")
#         print(f"u-space latents stats - valid mean: {np.mean(latents_u_valid):.4f}, std: {np.std(latents_u_valid):.4f}")
    
#     # Save all latents
#     save_path = rundir + '/latents/complete_latents.npz'
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     np.savez(save_path, **save_dict)
#     print(f"Saved complete latents to: {save_path}")
    
#     return save_path
    