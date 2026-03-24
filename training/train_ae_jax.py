import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import optax
from functools import partial
from flax.training import train_state  # Utility for managing model state
from flax.struct import dataclass

import config


from models.nn_modl_jax import *
from utils.utils_jax import *
# from load_phot_data import *
from data_proc.data_file_utils import *

# Define your autoencoder model
@jax.jit
def apply_autoencoder(variables, x):
    return model.apply(variables, x)

@jax.jit
def forward_mse_loss(variables, x):
    recon_x, _ = model.apply(variables, x) # call function also returns latent variables
    loss = jnp.mean((recon_x - x) **2) # MSE loss
    return loss

@jax.jit
def compute_gradients_noiseless(variables, x):
    # Compute the loss and gradients
    loss = forward_mse_loss(variables, x)
    grads = jax.grad(forward_mse_loss)(variables, x)  # Compute gradients of the loss
    return grads, loss


@jax.jit
def forward_logL_loss(variables, x, w):
    recon_x = model.apply(variables, x)
    loss = jnp.mean(0.5 * w *(recon_x - x) **2)
    return loss


@jax.jit
def compute_gradients_logP_train(variables, x):
    # Compute the loss and gradients
    loss = forward_logL_loss(variables, x)
    grads = jax.grad(forward_logL_loss)(variables, x)  # Compute gradients of the loss
    return grads, loss

@dataclass
class InferenceState:
    """A lightweight TrainState-like container for inference only."""
    apply_fn: callable
    params: dict


class EarlyStopper:
    def __init__(self, precision=1e-3, patience=10):
        self.precision = precision
        self.patience = patience
        self.badepochs = 0
        self.min_valid_loss = jnp.inf  # Use jnp.inf for JAX compatibility
        
    def step(self, valid_loss):
        if valid_loss < self.min_valid_loss * (1 - self.precision):
            self.badepochs = 0
            self.min_valid_loss = valid_loss
        else:
            self.badepochs += 1
            
        return self.badepochs < self.patience  # Returns False when stopping is needed


def create_train_state(model, input_shape, lr=1e-3, patience_ReduceLR=5, patience_stopper=10, verbose=True, key=0, decay_rate=0.9):
    if verbose:
        print('Initializing optimizer and scheduler..')
        print(f'lr = {lr}, patience_ReduceLR = {patience_ReduceLR}, patience_stopper = {patience_stopper}')

    # Define the optimizer with weight decay
    optimizer = optax.adam(learning_rate=lr)

    # Initialize training state
    class TrainState(train_state.TrainState):
        pass  # Extend if needed (e.g., add batch statistics)

    params = model.init(jax.random.PRNGKey(0), jnp.ones(input_shape))['params']

    params = jax.device_put(params, jax.devices('gpu')[0])
    
    state = TrainState.create(
        apply_fn=model.apply,  # Function to apply the model
        params=params,  # Initialize params (adjust shape)
        tx=optimizer,  # Set optimizer
    )

    # Custom learning rate scheduler (ReduceLROnPlateau alternative)
    scheduler = optax.exponential_decay(init_value=lr, transition_steps=patience_ReduceLR, decay_rate=decay_rate)

    # Early stopper
    stopper = EarlyStopper(patience=patience_stopper)

    return state, scheduler, stopper

def convert_to_decoder_inferencestate(train_state, decoder, central_wavelengths, nlatent, key=None):
    """Extracts the decoder from a trained TrainState and creates an InferenceState."""
    
    # Extract only the decoder parameters
    decoder_params = train_state.params['decoder']

    # Initialize decoder separately to ensure parameters match
    if key is None:
        key = jax.random.PRNGKey(0)
        
    dummy_z = jnp.zeros((1, nlatent))  # Shape should match the expected latent space size
    init_params = decoder.init(key, dummy_z)['params']
    
    # Ensure the extracted decoder params match the expected structure
    if decoder_params.keys() != init_params.keys():
        raise ValueError("Decoder params do not match the expected structure after initialization!")

    # Store the decoder in an inference state
    infer_state = InferenceState(
        apply_fn=decoder.apply,
        params=decoder_params
    )
    
    return infer_state


def train_jax_modl(model, state, train_loader, valid_loader, params=None, verbose=False, beta=0.1, variance_penalty_threshold=0.1, \
                   sig_level=None):
    if params is None:
        params = {}

    lambda_sim = float(params.get('lambda_sim', 0.0))
    lambda_consistency = float(params.get('lambda_consistency', 0.0))
    sim_k0 = float(params.get('sim_k0', 1.0))
    sim_k1 = float(params.get('sim_k1', 1.0))
    sigma_s = float(params.get('sigma_s', 1.0))
    similarity_subsample_size = int(params.get('similarity_subsample_size', 0))
    similarity_eps = float(params.get('similarity_eps', 1e-8))
    consistency_aug_scale = float(params.get('consistency_aug_scale', 0.1))
    recon_scale_mode = str(params.get('recon_scale_mode', 'fixed')).lower()
    amp_eps = float(params.get('amp_eps', 1e-8))
    amp_clip_min = params.get('amp_clip_min', None)
    amp_clip_max = params.get('amp_clip_max', None)

    if amp_clip_min is not None:
        amp_clip_min = float(amp_clip_min)
    if amp_clip_max is not None:
        amp_clip_max = float(amp_clip_max)

    if recon_scale_mode not in {'fixed', 'marginalized'}:
        raise ValueError(
            f"Unsupported recon_scale_mode='{recon_scale_mode}'. "
            "Use 'fixed' or 'marginalized'."
        )

    def unpack_training_batch(batch):
        """Normalize variable batch payloads into (spectrum, weights, orig)."""
        spectrum = batch[0]
        if len(batch) >= 4:
            w = batch[2]
            orig = batch[3]
        elif len(batch) == 3:
            # restframe + sig_level_norm=None path provides (spectrum, norms, orig)
            w = None
            orig = batch[2]
        else:
            w = None
            orig = None
        return spectrum, w, orig

    @partial(jax.jit, static_argnames=("is_training",))
    def compute_loss_components(model_params, x, w, rng_key, is_training=True):
        recon_x, z = model.apply({'params': model_params}, x)

        if recon_scale_mode == 'marginalized':
            # Solve per-spectrum least-squares amplitude A* and evaluate residuals on A*recon.
            if w is not None:
                numerator = jnp.sum(w * recon_x * x, axis=-1)
                denominator = jnp.sum(w * recon_x * recon_x, axis=-1) + amp_eps
            else:
                numerator = jnp.sum(recon_x * x, axis=-1)
                denominator = jnp.sum(recon_x * recon_x, axis=-1) + amp_eps

            amp_hat = numerator / denominator
            amp_hat = jnp.where(jnp.isfinite(amp_hat), amp_hat, 1.0)
            if (amp_clip_min is not None) or (amp_clip_max is not None):
                clip_min = -jnp.inf if amp_clip_min is None else amp_clip_min
                clip_max = jnp.inf if amp_clip_max is None else amp_clip_max
                amp_hat = jnp.clip(amp_hat, clip_min, clip_max)

            recon_fid = amp_hat[:, None] * recon_x
            if w is not None:
                loss_fid = jnp.mean(w * (recon_fid - x) ** 2)
            else:
                loss_fid = jnp.mean((recon_fid - x) ** 2)

            amp_stats = jnp.array([
                jnp.mean(amp_hat),
                jnp.std(amp_hat),
                jnp.min(amp_hat),
                jnp.max(amp_hat),
            ])
        else:
            # Backward-compatible baseline behavior.
            if w is not None:
                loss_fid = jnp.mean(w * (recon_x - x) ** 2)
            else:
                loss_fid = jnp.mean((recon_x - x) ** 2)
            amp_stats = jnp.array([1.0, 0.0, 1.0, 1.0])

        n_batch = x.shape[0]
        n_latent = z.shape[1]
        n_wave = recon_x.shape[1]

        if lambda_sim > 0.0:
            # Optional random subsampling for O(N^2) pairwise similarity term.
            if similarity_subsample_size > 0:
                n_keep = min(similarity_subsample_size, n_batch)
            else:
                n_keep = n_batch

            if is_training and n_keep < n_batch:
                idx = jax.random.permutation(rng_key, n_batch)[:n_keep]
                z_sim = z[idx]
                recon_sim = recon_x[idx]
                if w is not None:
                    w_sim = w[idx]
                else:
                    w_sim = None
            else:
                z_sim = z[:n_keep]
                recon_sim = recon_x[:n_keep]
                if w is not None:
                    w_sim = w[:n_keep]
                else:
                    w_sim = None

            z_diff = z_sim[:, None, :] - z_sim[None, :, :]
            latent_term = jnp.sum(z_diff ** 2, axis=-1) / (n_latent + similarity_eps)

            x_diff = recon_sim[:, None, :] - recon_sim[None, :, :]
            if w_sim is not None:
                # Use symmetric pair weights for w' in the paper's similarity term.
                w_pair = 0.5 * (w_sim[:, None, :] + w_sim[None, :, :])
                # Rebalance weights: apply weights to squared residuals (reduces amplification)
                spec_term = jnp.sum(w_pair * (x_diff ** 2), axis=-1) / (n_wave + similarity_eps)
            else:
                spec_term = jnp.sum(x_diff ** 2, axis=-1) / (n_wave + similarity_eps)

            s_ij = latent_term - spec_term

            # Normalize S_ij by a robust scale (std) to avoid sigmoid saturation from large magnitudes
            s_mean = jnp.mean(s_ij)
            s_std = jnp.std(s_ij) + similarity_eps
            s_min = jnp.min(s_ij)
            s_max = jnp.max(s_ij)
            s_ij_scaled = s_ij / s_std

            loss_sim = jnp.mean(jax.nn.sigmoid(sim_k1 * s_ij_scaled - sim_k0)) + \
                       jnp.mean(jax.nn.sigmoid(-sim_k1 * s_ij_scaled - sim_k0))
        else:
            loss_sim = jnp.array(0.0, dtype=loss_fid.dtype)

        if lambda_consistency > 0.0:
            rng_aug, _ = jax.random.split(rng_key)
            if w is not None:
                sigma_x = 1.0 / jnp.sqrt(jnp.maximum(w, similarity_eps))
                x_aug = x + consistency_aug_scale * sigma_x * jax.random.normal(rng_aug, x.shape)
            else:
                x_aug = x + consistency_aug_scale * jax.random.normal(rng_aug, x.shape)

            _, z_aug = model.apply({'params': model_params}, x_aug)
            z_aug_dist = jnp.sum((z - z_aug) ** 2, axis=-1)
            consistency_arg = z_aug_dist / ((sigma_s ** 2 + similarity_eps) * (n_latent + similarity_eps))
            loss_consistency = jnp.mean(jax.nn.sigmoid(consistency_arg) - 0.5)
        else:
            loss_consistency = jnp.array(0.0, dtype=loss_fid.dtype)

        total_loss = loss_fid + lambda_sim * loss_sim + lambda_consistency * loss_consistency
        # Return S_ij stats for monitoring: mean, std, min, max
        s_stats = jnp.array([s_mean, s_std, s_min, s_max]) if lambda_sim > 0.0 else jnp.array([0., 0., 0., 0.])
        return total_loss, loss_fid, loss_sim, loss_consistency, s_stats, amp_stats
    
    @jax.jit
    def train_step(state, x, w, orig, rng_key):
        del orig

        def forward_total_loss(model_params):
            total_loss, loss_fid, loss_sim, loss_consistency, s_stats, amp_stats = compute_loss_components(
                model_params, x, w, rng_key, is_training=True
            )
            return total_loss, (loss_fid, loss_sim, loss_consistency, s_stats, amp_stats)

        (loss, aux_losses), grads = jax.value_and_grad(forward_total_loss, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)

        loss_fid, loss_sim, loss_consistency, s_stats, amp_stats = aux_losses
        return state, loss, loss_fid, loss_sim, loss_consistency, s_stats, amp_stats

    @jax.jit
    def valid_step(model_params, x, w, orig):
        del orig
        fixed_key = jax.random.PRNGKey(0)
        return compute_loss_components(model_params, x, w, fixed_key, is_training=False)

    train_loss_epoch = 0.0
    train_fid_epoch = 0.0
    train_sim_epoch = 0.0
    train_consistency_epoch = 0.0
    train_s_stats_epoch = jnp.zeros((4,), dtype=jnp.float32)
    train_amp_stats_epoch = jnp.zeros((4,), dtype=jnp.float32)
    ctr = 0
    train_rng = jax.random.PRNGKey(0)

    for batch in train_loader:
        spectrum, w, orig = unpack_training_batch(batch)

        spectrum = jax.device_put(spectrum, device=jax.devices('gpu')[0])
        if w is not None:
            w = jax.device_put(w, device=jax.devices('gpu')[0])
        if orig is not None:
            orig = jax.device_put(orig, device=jax.devices('gpu')[0])
        train_rng, step_key = jax.random.split(train_rng)
        
        if ctr==0:
            print('spectrum on device', spectrum.device)
        
        state, loss, loss_fid, loss_sim, loss_consistency, s_stats, amp_stats = train_step(
            state, spectrum, w, orig, step_key
        )
        train_loss_epoch += loss
        train_fid_epoch += loss_fid
        train_sim_epoch += loss_sim
        train_consistency_epoch += loss_consistency
        train_s_stats_epoch += s_stats
        train_amp_stats_epoch += amp_stats
        ctr += 1

    print('counter after train loader is ', ctr)
    train_loss_epoch /= ctr
    train_fid_epoch /= ctr
    train_sim_epoch /= ctr
    train_consistency_epoch /= ctr
    train_s_stats_epoch /= ctr
    train_amp_stats_epoch /= ctr

    valid_loss_epoch = 0.0
    valid_fid_epoch = 0.0
    valid_sim_epoch = 0.0
    valid_consistency_epoch = 0.0
    valid_s_stats_epoch = jnp.zeros((4,), dtype=jnp.float32)
    valid_amp_stats_epoch = jnp.zeros((4,), dtype=jnp.float32)
    ctr = 0
    for valddata in valid_loader:
        
        spectrum, w, orig = unpack_training_batch(valddata)
        spectrum = jax.device_put(spectrum, device=jax.devices('gpu')[0])
        if w is not None:
            w = jax.device_put(w, device=jax.devices('gpu')[0])
        if orig is not None:
            orig = jax.device_put(orig, device=jax.devices('gpu')[0])
        
        if ctr==0:
            print('valid spectrum on device', spectrum.device)
        
        loss, loss_fid, loss_sim, loss_consistency, s_stats, amp_stats = valid_step(state.params, spectrum, w, orig)
        valid_loss_epoch += loss
        valid_fid_epoch += loss_fid
        valid_sim_epoch += loss_sim
        valid_consistency_epoch += loss_consistency
        valid_s_stats_epoch += s_stats
        valid_amp_stats_epoch += amp_stats
        ctr += 1

    print('counter after valid loader is ', ctr)
    valid_loss_epoch /= ctr
    valid_fid_epoch /= ctr
    valid_sim_epoch /= ctr
    valid_consistency_epoch /= ctr
    valid_s_stats_epoch /= ctr
    valid_amp_stats_epoch /= ctr

    return (
        state,
        train_loss_epoch,
        valid_loss_epoch,
        train_fid_epoch,
        train_sim_epoch,
        train_consistency_epoch,
        train_s_stats_epoch,
        train_amp_stats_epoch,
        valid_fid_epoch,
        valid_sim_epoch,
        valid_consistency_epoch,
        valid_s_stats_epoch,
        valid_amp_stats_epoch,
    )

    
def run_ae_sed_fit_jax(dat_obj, run_name=None, property_cat_df=None, save_results=True, \
                params=None, verbose=False, batch_size=128, beta=0.1, variance_penalty_threshold=0.1,\
                        key=None, wav_rest=None, report_callback=None, base_path=None, **kwargs):
    """
    Train a convolutional autoencoder on rest-frame SED data.
    
    Parameters
    ----------
    dat_obj : spec_data_jax
        Data object with training/validation sets
    run_name : str, optional
        Name for this training run (used for saving)
    property_cat_df : DataFrame, optional
        Property catalog (not currently used in training)
    save_results : bool
        Whether to save model checkpoints and figures
    params : dict
        Training parameters (filters, learning rate, epochs, etc.)
    verbose : bool
        Print additional debug information
    batch_size : int
        Batch size for training
    beta : float
        Beta parameter for variational penalty (if used)
    variance_penalty_threshold : float
        Threshold for variance penalty (if used)
    key : jax.random.PRNGKey, optional
        Random key (not currently used)
    wav_rest : array, optional
        Rest-frame wavelength grid for model
    base_path : str, optional
        Base path for saving model runs (default: config.sphx_base_path + 'modl_runs/')
    report_callback : callable, optional
        Callback function for intermediate reporting (e.g., for Optuna pruning).
        Should accept (epoch, metric_dict) and can raise exceptions to stop training.
        
    Returns
    -------
    state : TrainState
        Final training state with optimized parameters
    model : nn.Module
        The autoencoder model
    dat_obj : spec_data_jax
        Data object (with rundir set)
    metric_dict : dict
        Training metrics (train_loss, valid_loss per epoch)
    model_fpath : str
        Path to saved model checkpoint
    """
        
    if params is None:
        print('Should have specified params by now..')
        return None
        
    if save_results:
        if run_name is None:
            print('need run name')
            return None
        # Use custom base_path if provided, otherwise use default
        if base_path is None:
            base_path = config.sphx_base_path + 'modl_runs/'
        dat_obj.rundir = base_path + run_name
        figure_dir = create_result_dir_structure(dat_obj.rundir)

    if wav_rest is None:
        wav_rest = jnp.linspace(0.1, 5, 500)

    input_shape = (batch_size, len(wav_rest))
        
    if params['restframe']: # takes wavelengths from SED data set 
        central_wavelengths = dat_obj.sed_um_wave
    else:
        central_wavelengths = np.sort(np.load(config.sphx_dat_path+'central_wavelengths_sphx102.npz')['central_wavelengths'])

    if verbose:
        print('central wavelengths are ..', central_wavelengths)
        
    print('Instantiating AE model..')
        
    model = instantiate_ae_modl_gen_jax(params, central_wavelengths)
    
    # print(model.tabulate(jax.random.PRNGKey(0), jnp.ones(input_shape)))

    state, scheduler, stopper = create_train_state(model, input_shape, lr=params['lr'])

    metric_dict = dict({
        'train_loss': [],
        'valid_loss': [],
        'train_fid_loss': [],
        'train_sim_loss': [],
        'train_consistency_loss': [],
        'train_s_mean': [],
        'train_s_std': [],
        'train_s_min': [],
        'train_s_max': [],
        'train_amp_mean': [],
        'train_amp_std': [],
        'train_amp_min': [],
        'train_amp_max': [],
        'valid_fid_loss': [],
        'valid_sim_loss': [],
        'valid_consistency_loss': [],
        'valid_s_mean': [],
        'valid_s_std': [],
        'valid_s_min': [],
        'valid_s_max': [],
        'valid_amp_mean': [],
        'valid_amp_std': [],
        'valid_amp_min': [],
        'valid_amp_max': [],
    })
    
    phot_dict_train = dat_obj.phot_dict

    print('initializing train and valid loaders..')
    # t0 = time.time()
    train_loader = batch_generator(dat_obj.data_train, dat_obj.vals_train, batch_size)
    valid_loader = batch_generator(dat_obj.data_valid, dat_obj.vals_valid, batch_size)
    # dt0 = time.time()-t0
    # print('time to initialize batch generator:', dt0)

    # dat_obj.trainloader = jax.device_put(dat_obj.trainloader, device=jax.devices('gpu')[0])
    # dat_obj.validloader = jax.device_put(dat_obj.validloader, device=jax.devices('gpu')[0])

    state = jax.device_put(state, device=jax.devices('gpu')[0])
    
    for epoch in range(1, params['epochs'] + 1):
        
        (
            state,
            train_loss,
            valid_loss,
            train_fid_loss,
            train_sim_loss,
            train_consistency_loss,
            train_s_stats,
            train_amp_stats,
            valid_fid_loss,
            valid_sim_loss,
            valid_consistency_loss,
            valid_s_stats,
            valid_amp_stats,
        ) = train_jax_modl(
            model,
            state,
            train_loader,
            valid_loader,
            params=params,
            beta=beta,
            variance_penalty_threshold=variance_penalty_threshold,
        )

        print('====> Epoch: {} TRAIN Loss: {:.2e} \nVALIDATION Loss: {:.2e}'.format(epoch, train_loss, valid_loss))
        print('      Components: train(fid/sim/c)={:.2e}/{:.2e}/{:.2e}, valid(fid/sim/c)={:.2e}/{:.2e}/{:.2e}'.format(
            train_fid_loss, train_sim_loss, train_consistency_loss,
            valid_fid_loss, valid_sim_loss, valid_consistency_loss))
        train_s_mean, train_s_std, train_s_min, train_s_max = [float(v) for v in train_s_stats]
        valid_s_mean, valid_s_std, valid_s_min, valid_s_max = [float(v) for v in valid_s_stats]
        print('      S_ij stats: train(mean/std/min/max)={:.2e}/{:.2e}/{:.2e}/{:.2e}, valid={:.2e}/{:.2e}/{:.2e}/{:.2e}'.format(
            train_s_mean, train_s_std, train_s_min, train_s_max,
            valid_s_mean, valid_s_std, valid_s_min, valid_s_max))
        train_amp_mean, train_amp_std, train_amp_min, train_amp_max = [float(v) for v in train_amp_stats]
        valid_amp_mean, valid_amp_std, valid_amp_min, valid_amp_max = [float(v) for v in valid_amp_stats]
        print('      Amplitude stats: train(mean/std/min/max)={:.2e}/{:.2e}/{:.2e}/{:.2e}, valid={:.2e}/{:.2e}/{:.2e}/{:.2e}'.format(
            train_amp_mean, train_amp_std, train_amp_min, train_amp_max,
            valid_amp_mean, valid_amp_std, valid_amp_min, valid_amp_max))

        train_loader = batch_generator(dat_obj.data_train, dat_obj.vals_train, batch_size)
        valid_loader = batch_generator(dat_obj.data_valid, dat_obj.vals_valid, batch_size)
    
        if epoch % params['plot_interval'] == 0:
            
            print('====> Epoch: {} TRAIN Loss: {:.2e} \nVALIDATION Loss: {:.2e} '.format(
                  epoch, train_loss, valid_loss))
            
            fig = plot_sed_recon_epoch_jax(
                state,
                model,
                valid_loader,
                wav_rest,
                recon_scale_mode=params.get('recon_scale_mode', 'fixed'),
                amp_eps=float(params.get('amp_eps', 1e-8)),
                amp_clip_min=params.get('amp_clip_min', None),
                amp_clip_max=params.get('amp_clip_max', None),
            )
            
            fig.savefig(figure_dir+'/training/reconstruction_vs_epoch/recon_spec_epoch'+str(epoch)+'.pdf', bbox_inches='tight')
                         

        new_lr = scheduler(epoch)
        state = state.replace(tx=optax.adam(learning_rate=new_lr))

        metric_dict = update_metric_dict(
            metric_dict,
            train_loss=train_loss,
            valid_loss=valid_loss,
            train_fid_loss=train_fid_loss,
            train_sim_loss=train_sim_loss,
            train_consistency_loss=train_consistency_loss,
            train_s_mean=train_s_mean,
            train_s_std=train_s_std,
            train_s_min=train_s_min,
            train_s_max=train_s_max,
            valid_fid_loss=valid_fid_loss,
            valid_sim_loss=valid_sim_loss,
            valid_consistency_loss=valid_consistency_loss,
            valid_s_mean=valid_s_mean,
            valid_s_std=valid_s_std,
            valid_s_min=valid_s_min,
            valid_s_max=valid_s_max,
            train_amp_mean=train_amp_mean,
            train_amp_std=train_amp_std,
            train_amp_min=train_amp_min,
            train_amp_max=train_amp_max,
            valid_amp_mean=valid_amp_mean,
            valid_amp_std=valid_amp_std,
            valid_amp_min=valid_amp_min,
            valid_amp_max=valid_amp_max,
        )
        
        # Call report callback for intermediate reporting (e.g., Optuna pruning)
        if report_callback is not None:
            try:
                report_callback(epoch, metric_dict)
            except Exception as e:
                # Allow callback to stop training (e.g., TrialPruned)
                print(f'Training stopped by callback at epoch {epoch}: {e}')
                # Save partial results before exiting
                if save_results:
                    model_fpath = save_ae_jax(state, dat_obj.rundir)
                    save_train_metrics_jax(metric_dict, dat_obj.rundir)
                raise

        if (not stopper.step(valid_loss)) or (epoch == params['epochs']):
            print('Stopping')
            print('====> Epoch: {} TRAIN Loss: {:.2e} \nVALIDATION Loss: {:.2e}'.format(
                  epoch, train_loss, valid_loss))
                        
            # filename    
            model_fpath = save_ae_jax(state, dat_obj.rundir)
            save_train_metrics_jax(metric_dict, dat_obj.rundir)
        
    #         f = plot_train_validation_logL(len(metric_dict['train_loss']), metric_dict, return_fig=True, logscale=True)
    #         f.savefig(figure_dir+'/training/train_valid_nll_vs_epoch.pdf', bbox_inches='tight')
            
            return state, model, dat_obj, metric_dict, model_fpath


def plot_sed_recon_epoch_jax(state, model, train_loader, sed_lams, figsize=(8, 6), alph=0.4, \
                        bbox_to_anchor=[-1.1, 2.2], legend_fs=12, ncol=2, xlim=[0.5, 5.0], sig_level=None, \
                            xlab = '$\\lambda$ [$\\mu$m]', recon_scale_mode='fixed', amp_eps=1e-8,
                            amp_clip_min=None, amp_clip_max=None):
    
    f = plt.figure(figsize=figsize)

    ctr = 0
    for batch in train_loader:

        if len(batch) >= 4:
            spectrum = batch[0]
            w = batch[2]
        elif len(batch) == 3:
            spectrum = batch[0]
            w = None
        else:
            spectrum = batch[0]
            w = None
        spectrum = jax.device_put(spectrum, device=jax.devices('gpu')[0])
        if w is not None:
            w = jax.device_put(w, device=jax.devices('gpu')[0])
        
        if ctr < 4:
            recon, _ = state.apply_fn({'params':state.params}, spectrum)
            # print('recon is ', recon.shape)
            # print('while spectrum has shape', spectrum.shape)
            recon_plot = recon[0]
            label_recon = 'Reconstructed'
            if str(recon_scale_mode).lower() == 'marginalized':
                if w is not None:
                    num = jnp.sum(w[0] * recon[0] * spectrum[0])
                    den = jnp.sum(w[0] * recon[0] * recon[0]) + amp_eps
                else:
                    num = jnp.sum(recon[0] * spectrum[0])
                    den = jnp.sum(recon[0] * recon[0]) + amp_eps
                amp_hat = num / den
                amp_hat = jnp.where(jnp.isfinite(amp_hat), amp_hat, 1.0)
                if (amp_clip_min is not None) or (amp_clip_max is not None):
                    clip_min = -jnp.inf if amp_clip_min is None else amp_clip_min
                    clip_max = jnp.inf if amp_clip_max is None else amp_clip_max
                    amp_hat = jnp.clip(amp_hat, clip_min, clip_max)
                recon_plot = amp_hat * recon[0]
                label_recon = f'Marginalized recon (A*={float(amp_hat):.3f})'
            xval = sed_lams.copy()
                        
            plt.subplot(2,2,ctr+1)
            plt.plot(xval, recon_plot, label=label_recon, zorder=10, color='C3')

            # plt.plot(xval, np.array(spectrum[0]), label='Original (noiseless)', zorder=10, color='k')
            # plt.plot(xval, np.array(spectrum[0])-recon, label='Original (noiseless) - Reconstruction', linestyle='dashed', color='C3')

            if w is not None:
                flux_unc_norm = 1./np.sqrt(w[0])
                plt.errorbar(xval, np.array(spectrum[0]) ,yerr=flux_unc_norm, color='b', fmt='o', markersize=2, capsize=2, alpha=alph, label='Data')
                # plt.plot(xval, orig[0].detach().numpy(), label='Original (noiseless)', zorder=10, color='k')
                # plt.plot(xval,orig[0].detach().numpy()-normplot*((recon[0].detach().numpy())), label='Original (noiseless) - Reconstruction', linestyle='dashed', color='C3')
                plt.fill_between(xval, -flux_unc_norm, flux_unc_norm, label='1$\\sigma$ flux density uncertainties', alpha=0.5, color='grey')

            plt.tick_params(labelsize=14)
            if ctr==0 or ctr==2:
                plt.ylabel('Flux density [norm.]', fontsize=16)
            if ctr > 1:
                plt.xlabel(xlab, fontsize=16)
            plt.xlim(xlim)
            ctr += 1
        else:
            plt.legend(fontsize=legend_fs, ncol=ncol, loc='lower left', bbox_to_anchor=bbox_to_anchor)
            plt.show()
            break

    return f

def gaussian_prior_kl(mu, log_sigma):
    """
    Compute the KL divergence between the learned Gaussian distribution
    (with mean `mu` and log variance `log_sigma`) and the standard normal prior.
    """
    # Compute the variance from the log-variance
    sigma_squared = jnp.exp(2 * log_sigma)
    
    # KL divergence for each latent dimension
    kl_divergence = 0.5 * jnp.sum(mu**2 + sigma_squared - 1 - 2 * log_sigma)
    
    return kl_divergence
            