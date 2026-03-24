import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
import jax.random as jr
import equinox as eqx
import pickle
import config

# import jaxopt
# import flax.linen as nn
# import optax
# from flax.training import train_state
# import tensorflow_probability.substrates.jax.distributions as tfd

from data_proc.dataloader_jax import *
from data_proc.data_file_utils import *
from .flowjax_modl import *
from .nn_modl_jax import *
from training.train_ae_jax import *
from inference.like_prior import prof_like
from visualization.result_plotting_fns import make_color_corner_plot


def convert_to_bfloat16_recursive(pytree):
    def to_bfloat16(x):
        if isinstance(x, jnp.ndarray) and x.dtype == jnp.float32:
            return x.astype(jnp.bfloat16)
        return x
    return jax.tree_util.tree_map(to_bfloat16, pytree)

class PAE_JAX():
    
    """Probabilistic AutoEncoder

    contains models for the three necessary components:
    encoder: x -> z
    decoder: z -> x'
    flow: z <-> u
    """
    
    def __init__(self, run_name, modl_type=None, central_wavelengths=None, load_flow_decoder=True, params=None, \
                wave_obs=None, params_additional=None, filename_save = 'flow_model_iaf_50k.pkl', filter_set_name='spherex_filters102', filtfiles=None, \
                with_ext_phot=False, inference_dtype=jnp.float32):
        
        super(PAE_JAX, self).__init__()

        self.inference_dtype = inference_dtype
                        
        if central_wavelengths is not None:
            self.central_wavelengths = central_wavelengths

        self.redshift_idx = 1

        self.key = jr.key(42)  # Persistent key for PRNG

        self.wave_obs = None
        if wave_obs is not None:
            self.wave_obs = jax.device_put(jnp.array(wave_obs), jax.devices('gpu')[0])
                        
        self.run_name = run_name
        self.rundir = config.modl_runs_path + run_name + '/'

        # if there are additional keys in provided parameters, add them to self.params
        self.load_update_params(params_additional=params_additional)

        if self.params['filter_integrate']:
            self.load_filters(filter_set_name=filter_set_name, filtfiles=filtfiles, with_ext_phot=with_ext_phot)

        if modl_type is not None:
            self.params['modl_type'] = modl_type

        if load_flow_decoder:
            self.load_flow_decoder(run_name, filename_save=filename_save)


    def load_filters(self, filter_set_name='spherex_filters102', filtfiles=None, with_ext_phot=False, lam_max=5.2, lam_min=None):
        
        print('Loading filters..')
        print('filtfiles:', filtfiles)
        
        # load spherex filters, just fixed 102 channels for now
        central_wavs, bandpass_wavs, bandpass_vals, bandpass_names = load_sphx_filters(filtdir=config.filt_basepath+filter_set_name, filtfiles=filtfiles)

        if with_ext_phot:
            ext_central_wavs, ext_bandpass_wavs,\
                ext_bandpass_vals, ext_bandpass_names = load_ext_filters(base_filt_path=config.filt_basepath+'ext_filters/', wise=True, decam=True)


            central_wavs = np.concatenate((ext_central_wavs, central_wavs))

            bandpass_wavs = ext_bandpass_wavs + bandpass_wavs
            bandpass_vals = ext_bandpass_vals + bandpass_vals
            bandpass_names = np.concatenate((ext_bandpass_names, bandpass_names))


        if lam_min is None:
            if with_ext_phot:
                lam_min = 0.35
            else:
                lam_min = 0.7

        from data_proc.dataloader_jax import convert_filters_to_jax

        self.jax_filters, self.lam_interp = convert_filters_to_jax(central_wavs, bandpass_wavs, bandpass_vals, plot=False, nlam=self.params['nlam_interp'], lam_min=lam_min, lam_max=lam_max)

        self.jax_filters = self.jax_filters.astype(self.inference_dtype) # <--- Convert filters

        print('central wavelengths:', len(central_wavs))
        self.wave_obs = central_wavs



    def load_update_params(self, params_additional=None):
        f = open(self.rundir+'/params.txt', 'rb')
        self.params = pickle.load(f)
        print('self.params:', self.params)
        f.close()

        if params_additional is not None:
            for key in params_additional.keys():
                self.params[key] = params_additional[key]

    def load_flow_decoder(self, run_name, filename_save='flow_model_iaf.pkl'):        
        # load flow

        if 'redshift' in filename_save:
            redshift_in_flow=True
        else:
            redshift_in_flow = False
            
        self.load_flow(filename_save=filename_save, redshift_in_flow=redshift_in_flow)
        # load encoder/decoder
        self.load_ae_modl(run_name)

    def combined_transform(self, x):

        # Apply the normalizing flow's forward pass
        x_flow = self.flow.bijection.transform(x)
        # Apply the affine transformation
        # x_rescaled = self.rescale.transform(x_flow)
        x_rescaled = self.rescale.inverse(x_flow)


        
        return x_rescaled

    def combined_transform_inverse(self, x):


        x_flow = self.rescale.inverse_transform()
        
        return x_flow
    
    def load_flow(self, filename_save='flow_model_iaf.pkl', redshift_in_flow=False):

        latent_fpath = self.rundir+'/latents/latent_loc_std'
        
        if redshift_in_flow:
            latent_fpath += '_with_redshift'
        latent_fpath += '.npz'
        
        print('Loading latent mean std from ', latent_fpath)
        latent_file = np.load(latent_fpath)
        loc, scale = latent_file['loc'], latent_file['scale']
        nlatent = loc.shape[0]
        print('nlatent in flow init is ', nlatent)
        
        """Load the normalizing flow model"""
        key, subkey = jr.split(self.key)

        _, flow = init_flowjax_modl(subkey, nlatent, invert=False)
            
        self.flow = load_model(flow, self.rundir, filename=filename_save)

        # for rescaling to AE latents
        self.rescale = Affine(loc, scale)
        self.combined_transform_jit = eqx.filter_jit(eqx.filter_vmap(self.combined_transform))

        # self.combined_transform_vmap = eqx.filter_vmap(self.combined_transform)
            
    def load_ae_modl(self, run_name):
        """Load the autoencoder model"""
 
        self.model = instantiate_ae_modl_gen_jax(self.params, self.central_wavelengths)
        input_shape = (1, len(self.central_wavelengths))

        state_template, _, _ = create_train_state(self.model, input_shape)
        state = load_jax_state(config.modl_runs_path + run_name, state_template)
        
        self.model.params = state.params

        # self.model.params = convert_to_bfloat16_recursive(state.params) # <--- Convert model params


        self.encoder_params = state.params['encoder']
        self.encoder = SpectrumEncoder_JAX(self.params['nlatent'],\
                                           filter_sizes=self.params['filter_sizes'],\
                                           filters=self.params['filters'])

        
        self.decoder_params = state.params['decoder']
        self.decoder = SpectrumDecoder_JAX(self.central_wavelengths, self.params['nlatent'],\
                                           filter_sizes=reversed(self.params['filter_sizes']),\
                                           filters=reversed(self.params['filters']),\
                                           wave_obs=self.central_wavelengths, \
                                          lam_interp=self.lam_interp)
        
        self.decode_jit = jax.jit(lambda params, z: self.decoder.apply({'params': params}, z))

        # Initialize log-wavelength grid if model was trained on log-spaced wavelengths
        # or set up for optional use with setup_log_wavelength_grid()
        self.use_log_wavelength = self.params.get('use_log_wavelength', False)
        self.log_wav_rest = None
        self.log_wav_obs = None
        self.dloglam = self.params.get('dloglam', None)
        
        if self.use_log_wavelength:
            # Model was trained on log-spaced wavelengths - the central_wavelengths ARE the log grid
            self.log_wav_rest = jnp.array(self.central_wavelengths)
            if self.dloglam is None:
                # Compute dloglam from the grid
                self.dloglam = (jnp.log(self.log_wav_rest[-1]) - jnp.log(self.log_wav_rest[0])) / (len(self.log_wav_rest) - 1)
            print(f"Model trained on log-wavelength grid: dloglam = {self.dloglam:.6f}")

    def setup_log_wavelength_grid(self, n_log_wav=None, log_lam_min=None, log_lam_max=None):
        """
        Set up log-spaced wavelength grid for fast redshift interpolation.
        
        In log-wavelength space, redshifting becomes a simple shift:
        log(λ_obs) = log(λ_rest) + log(1+z)
        
        This allows much faster interpolation using index shifts rather than
        per-wavelength interpolation.
        
        Parameters
        ----------
        n_log_wav : int, optional
            Number of wavelength points in log grid. Default uses same as central_wavelengths.
        log_lam_min : float, optional
            Minimum log10(wavelength/μm). Default from central_wavelengths.
        log_lam_max : float, optional
            Maximum log10(wavelength/μm). Default from central_wavelengths.
        """
        # Use defaults from central wavelengths if not provided
        if n_log_wav is None:
            n_log_wav = len(self.central_wavelengths)
        if log_lam_min is None:
            log_lam_min = np.log10(self.central_wavelengths.min())
        if log_lam_max is None:
            log_lam_max = np.log10(self.central_wavelengths.max())
        
        # Create uniform log10 wavelength grid
        log10_wav = jnp.linspace(log_lam_min, log_lam_max, n_log_wav)
        self.log_wav_rest = 10**log10_wav  # Linear wavelengths on log-spaced grid
        
        # Step size in natural log (for shift calculations)
        # We use natural log because log(1+z) is natural log
        self.dloglam = (jnp.log(self.log_wav_rest[-1]) - jnp.log(self.log_wav_rest[0])) / (n_log_wav - 1)
        
        # Also set up observed wavelength grid for filter convolution
        if hasattr(self, 'lam_interp'):
            log10_wav_obs = jnp.linspace(
                jnp.log10(self.lam_interp.min()),
                jnp.log10(self.lam_interp.max()),
                len(self.lam_interp)
            )
            self.log_wav_obs = 10**log10_wav_obs
            self.dloglam_obs = (jnp.log(self.log_wav_obs[-1]) - jnp.log(self.log_wav_obs[0])) / (len(self.log_wav_obs) - 1)
        
        self.use_log_wavelength = True
        
        print(f"Log-wavelength grid initialized:")
        print(f"  Rest-frame: {n_log_wav} points, λ = [{10**log_lam_min:.3f}, {10**log_lam_max:.3f}] μm")
        print(f"  dloglam (natural log step) = {self.dloglam:.6f}")
    
    def setup_log_filters(self, z_max=3.0, n_z_bins=50):
        """
        Precompute filter curves on the log-wavelength grid for different redshift bins.
        
        This enables fully log-grid forward modeling by avoiding the interpolation
        step back to linear wavelength grid for filter convolution.
        
        Parameters
        ----------
        z_max : float, optional
            Maximum redshift to precompute filters for. Default 3.0.
        n_z_bins : int, optional
            Number of redshift bins to precompute. Default 50.
            
        Notes
        -----
        After calling this, use push_spec_marg_log_full() for fastest forward modeling.
        The filters are linearly interpolated between precomputed redshift bins.
        """
        if not self.use_log_wavelength:
            raise ValueError("Must call setup_log_wavelength_grid() first.")
        
        self.log_z_bins = jnp.linspace(0, z_max, n_z_bins)
        
        # For each redshift bin, compute where log_wav_rest * (1+z) falls relative
        # to the filter wavelength grid, and compute filter response
        def compute_filter_at_z(z):
            # Observed wavelengths for rest-frame log grid at this redshift
            wav_obs = self.log_wav_rest * (1 + z)
            # Interpolate original filters to this observed wavelength grid
            # self.jax_filters shape: (nband, n_lam_interp), evaluated at self.lam_interp
            filters_at_wav_obs = jax.vmap(
                lambda f: jnp.interp(wav_obs, self.lam_interp, f, left=0.0, right=0.0)
            )(self.jax_filters)
            return filters_at_wav_obs  # shape: (nband, n_log_wav)
        
        self.log_filters_by_z = jax.vmap(compute_filter_at_z)(self.log_z_bins)
        # Shape: (n_z_bins, nband, n_log_wav)
        
        self.log_filters_dz = self.log_z_bins[1] - self.log_z_bins[0]
        
        print(f"Log-grid filters precomputed:")
        print(f"  {n_z_bins} redshift bins from z=0 to z={z_max}")
        print(f"  Filter shape: {self.log_filters_by_z.shape}")
        
    def get_filters_at_z_log(self, z):
        """
        Get interpolated filter curves for a given redshift on log grid.
        
        Parameters
        ----------
        z : float or jnp.ndarray
            Redshift value(s)
            
        Returns
        -------
        filters : jnp.ndarray
            Filter curves on log_wav_rest grid, shape (nband, n_log_wav) or (nsamp, nband, n_log_wav)
        """
        # Find the bin index and interpolation weight
        z_idx = z / self.log_filters_dz
        z_idx_floor = jnp.floor(z_idx).astype(jnp.int32)
        z_idx_floor = jnp.clip(z_idx_floor, 0, len(self.log_z_bins) - 2)
        w = z_idx - z_idx_floor
        
        # Linear interpolation between adjacent z bins
        if z.ndim == 0 or (hasattr(z, 'shape') and z.shape == ()):
            # Scalar redshift
            f0 = self.log_filters_by_z[z_idx_floor]
            f1 = self.log_filters_by_z[z_idx_floor + 1]
            return f0 + w * (f1 - f0)
        else:
            # Batched redshifts
            def interp_filters(z_i, w_i):
                f0 = self.log_filters_by_z[z_i]
                f1 = self.log_filters_by_z[z_i + 1]
                return f0 + w_i * (f1 - f0)
            return jax.vmap(interp_filters)(z_idx_floor, w)
        
    def interpolate_to_log_grid(self, spec):
        """
        Interpolate rest-frame SED from central_wavelengths to log-spaced grid.
        
        Parameters
        ----------
        spec : jnp.ndarray
            Rest-frame spectrum on self.central_wavelengths grid, shape (..., n_wav)
            
        Returns
        -------
        spec_log : jnp.ndarray
            Spectrum interpolated to self.log_wav_rest grid
        """
        if self.log_wav_rest is None:
            raise ValueError("Log wavelength grid not initialized. Call setup_log_wavelength_grid() first.")
        
        # Handle batched spectra
        if spec.ndim == 1:
            return jnp.interp(self.log_wav_rest, self.central_wavelengths, spec)
        else:
            return jax.vmap(lambda s: jnp.interp(self.log_wav_rest, self.central_wavelengths, s))(spec)
    
    def shift_on_log_grid(self, spec_log, z):
        """
        Apply redshift as a shift on log-wavelength grid.
        
        This is much faster than interpolation because redshift becomes a simple
        index shift in log-wavelength space:
        λ_obs = λ_rest * (1+z)  =>  log(λ_obs) = log(λ_rest) + log(1+z)
        
        Uses linear interpolation for continuous (non-integer) shifts.
        
        Parameters
        ----------
        spec_log : jnp.ndarray
            Spectrum on log-wavelength grid, shape (n_log_wav,)
        z : float
            Redshift
            
        Returns
        -------
        spec_shifted : jnp.ndarray
            Redshifted spectrum on same log grid (zeros where shifted outside bounds)
        """
        n = spec_log.shape[0]
        
        # Compute continuous shift in grid units
        # shift = log(1+z) / dloglam gives the number of grid points to shift
        shift = jnp.log1p(z) / self.dloglam
        
        # Compute fractional indices for interpolation
        # We're shifting TO observed frame, so we need indices in rest frame
        # that map to each observed-frame position
        i = jnp.arange(n, dtype=jnp.float32) - shift
        
        # Integer part for indexing, fractional part for interpolation
        i0 = jnp.floor(i).astype(jnp.int32)
        w = i - i0  # Fractional weight for linear interpolation
        
        # Clip indices to valid range for safe indexing
        i0c = jnp.clip(i0, 0, n - 2)
        
        # Linear interpolation between adjacent grid points
        y0 = spec_log[i0c]
        y1 = spec_log[i0c + 1]
        y = y0 + w * (y1 - y0)
        
        # Mask out values that fall outside the original grid
        valid = (i >= 0.0) & (i <= (n - 1))
        return jnp.where(valid, y, 0.0)
    
    def shift_on_log_grid_batch(self, spec_log, z):
        """
        Batch version of shift_on_log_grid for multiple spectra and redshifts.
        
        Optimized with JIT compilation and full vectorization (no vmap).
        
        Parameters
        ----------
        spec_log : jnp.ndarray
            Spectra on log-wavelength grid, shape (nsamp, n_log_wav)
        z : jnp.ndarray
            Redshifts, shape (nsamp,) or scalar
            
        Returns
        -------
        spec_shifted : jnp.ndarray
            Redshifted spectra, shape (nsamp, n_log_wav)
        """
        # Use cached JIT-compiled version
        if not hasattr(self, '_shift_batch_jit'):
            self._shift_batch_jit = jax.jit(self._shift_on_log_grid_batch_impl)
        
        return self._shift_batch_jit(spec_log, z, self.dloglam)
    
    @staticmethod
    def _shift_on_log_grid_batch_impl(spec_log, z, dloglam):
        """
        Static implementation for JIT compilation.
        Fully vectorized without vmap for maximum performance.
        """
        nsamp, n = spec_log.shape
        
        # Broadcast z to (nsamp,) if scalar
        z = jnp.atleast_1d(z)
        if z.shape[0] == 1:
            z = jnp.broadcast_to(z, (nsamp,))
        
        # Compute shifts for all samples: shape (nsamp,)
        shifts = jnp.log1p(z) / dloglam
        
        # Create index grid: shape (n,)
        idx_base = jnp.arange(n, dtype=jnp.float32)
        
        # Compute fractional indices for all samples: shape (nsamp, n)
        # Each row is the index array for one sample shifted by its redshift
        i = idx_base[None, :] - shifts[:, None]
        
        # Integer part and fractional part
        i0 = jnp.floor(i).astype(jnp.int32)
        w = i - i0
        
        # Clip indices to valid range
        i0c = jnp.clip(i0, 0, n - 2)
        
        # Gather values using advanced indexing
        # Need to create row indices for gathering
        row_idx = jnp.arange(nsamp)[:, None]
        
        # Gather y0 and y1 for all samples at once
        y0 = spec_log[row_idx, i0c]  # shape: (nsamp, n)
        y1 = spec_log[row_idx, i0c + 1]  # shape: (nsamp, n)
        
        # Linear interpolation
        y = y0 + w * (y1 - y0)
        
        # Mask invalid regions
        valid = (i >= 0.0) & (i <= (n - 1))
        return jnp.where(valid, y, 0.0)

        
    def get_encoded_u(self, data):
        """Encode data into latent space and transform with normalizing flow"""
        latent_z = self.encoder.apply({'params': self.encoder_params}, data)
        latent_u = self.flowmodl.transform_to_noise(latent_z)
        return latent_u

        
    def push_spec(self, latents, redshift, decoder_params=None):
        # latents = latents.astype(self.inference_dtype) # <--- Cast latents
        # Cast redshift if it's used in bfloat16 computations
        redshift = redshift.astype(self.inference_dtype) # <--- Cast redshift
        
        z = self.combined_transform_jit(latents) # NF transformation and affine scaling

        z = z.astype(self.inference_dtype)

        # modify for fine-tuning (8/26/25)
        if decoder_params is None:
            decoder_params = self.model.params['decoder']
        spec = self.decode_jit(decoder_params, z)
        
        # spec = self.decode_jit(self.model.params['decoder'], z)

        nsamp = latents.shape[0]

        if nsamp==1:            
            x_interp = jnp.interp(self.lam_interp, self.central_wavelengths*(1+redshift), spec[0])
        else:
            wave_redshifted = self.central_wavelengths * (1 + redshift[:, None])   
            x_interp = jax.vmap(lambda x, y: jnp.interp(self.lam_interp, x, y))(wave_redshifted, spec)  
            
        spec = jnp.dot(self.jax_filters, x_interp.T).T

        return spec

    def push_spec_marg(self, latents, redshift, observed_flux=None, weight=None, 
                  marginalize_amplitude=True, return_rescaled_flux_and_loglike=False, redshift_in_flow=False, 
                  decoder_params=None, use_jit=True, filter_curves=None, lam_interp=None,
                  return_amplitude=False, log_amplitude=None):
        """
        Forward model: latent -> decoded SED -> redshift -> convolved photometry.
    
        If `marginalize_amplitude=True` and fluxes are provided, computes log-likelihood
        marginalized over amplitude analytically using inverse variance weights.
    
        If `log_amplitude` is provided, uses exp(log_amplitude) as the amplitude instead
        of marginalizing. This allows sampling log-amplitude as a free parameter,
        which enforces positivity and can reduce overfitting at high redshift.
    
        If `return_rescaled_flux_and_loglike=True`, returns (rescaled_flux, loglike, redshift)
        instead of just the log-likelihood.
        
        If `return_amplitude=True`, also returns the best-fit or provided amplitude as the
        last element of the return tuple.

        Returns redshift as well in case where redshift_in_flow and need to grab after NF pass
        
        Parameters
        ----------
        filter_curves : jnp.ndarray, optional
            Per-sample filter curves, shape (nsamp, nband, n_lam).
            If provided, uses these instead of self.jax_filters (homogenized filters).
            For native SPHEREx processing with variable-length measurements.
        lam_interp : jnp.ndarray, optional
            Wavelength grid for interpolation when using filter_curves.
            If None, uses self.lam_interp.
        return_amplitude : bool, optional
            If True, returns the best-fit amplitude (A_hat) as well. Useful for 
            diagnosing negative amplitudes. Default False.
        log_amplitude : jnp.ndarray, optional
            If provided, use exp(log_amplitude) as the amplitude instead of 
            marginalizing. Shape should be (nsamp,) or scalar. This enforces
            positivity and allows amplitude to be sampled as a free parameter.
        """

        if use_jit:
            z = self.combined_transform_jit(latents)
        else:
            z = self.combined_transform_vmap(latents)
            
        if redshift_in_flow:
            z = z[:,:-1] # decoder takes latent variables but not redshift
            redshift = z[:,-1]

        # modify for fine-tuning (8/26/25)
        if decoder_params is None:
            decoder_params = self.model.params['decoder']

        spec = self.decode_jit(decoder_params, z)

        # if use_jit and hasattr(self, 'decode_jit'):
        #     spec = self.decode_jit(decoder_params, z)
        # else:
        #     # direct apply without JIT for training/fine-tuning
        #     spec = self.model.apply({'params':self.model.params}, z, method=self.model.decode_only)

    
        nsamp = latents.shape[0]
        
        # Use custom wavelength grid if provided with native filters
        if lam_interp is None:
            lam_interp_use = self.lam_interp
        else:
            lam_interp_use = lam_interp
    
        if nsamp == 1:
            # x_interp = jnp.interp(lam_interp_use, self.central_wavelengths * (1 + redshift), spec[0])
            x_interp = jnp.interp(lam_interp_use/(1 + redshift), self.central_wavelengths, spec[0])
        else:
            # Broadcast redshift to match number of samples if needed
            if redshift.shape[0] == 1 and nsamp > 1:
                redshift_broadcast = jnp.repeat(redshift, nsamp, axis=0)
            else:
                redshift_broadcast = redshift
            wave_redshifted = self.central_wavelengths * (1 + redshift_broadcast[:, None])
            # x_interp = jax.vmap(lambda x, y: jnp.interp(lam_interp_use, x, y))(wave_redshifted, spec)
            x_interp = jax.vmap(lambda x, y: jnp.interp(lam_interp_use/(1 + redshift_broadcast), x, y))(self.central_wavelengths[None, :].repeat(nsamp, axis=0), spec)        
       
       
        # Use per-source filters if provided, otherwise use homogenized filters
        if filter_curves is not None:
            # filter_curves shape: (nsamp, nband, n_lam)
            # x_interp shape: (nsamp, n_lam) or (n_lam,) if nsamp==1
            if nsamp == 1:
                # For single sample: (nband, n_lam) @ (n_lam,) -> (nband,)
                model_flux = jnp.dot(filter_curves[0], x_interp)
            else:
                # For batch: vmap over samples
                # Each: (nband, n_lam) @ (n_lam,) -> (nband,)
                model_flux = jax.vmap(lambda f, x: jnp.dot(f, x))(filter_curves, x_interp)
        else:
            # Use homogenized filters (original behavior)
            model_flux = jnp.dot(self.jax_filters, x_interp.T).T  # shape [nsamp, nband]
    
        if not marginalize_amplitude and log_amplitude is None:
            return model_flux
    
        assert observed_flux is not None and weight is not None, "Must provide flux and weight for marginalization"
    
        # Case 1: Use log_amplitude as a free parameter (no marginalization)
        if log_amplitude is not None:
            # log_amplitude is provided as a free parameter
            amplitude = jnp.exp(log_amplitude)
            if nsamp > 1:
                f_scaled = amplitude[:, None] * model_flux
                observed_flux = jnp.broadcast_to(observed_flux, model_flux.shape)
                weight = jnp.broadcast_to(weight, model_flux.shape)
            else:
                f_scaled = amplitude * model_flux
            
            chi2 = jnp.sum(((observed_flux - f_scaled) ** 2) * weight, axis=-1)
            loglike = -0.5 * chi2
            
            if return_rescaled_flux_and_loglike:
                if return_amplitude:
                    return f_scaled, loglike, redshift, amplitude
                return f_scaled, loglike, redshift
            else:
                if return_amplitude:
                    return loglike, redshift, amplitude
                return loglike, redshift
    
        # Case 2: Marginalize amplitude analytically (original behavior)
        def marginal_loglike(y, w, f_model):
            A_hat = jnp.sum(f_model * y * w, axis=-1) / jnp.sum(f_model**2 * w, axis=-1)

            # Ensure f_scaled is always 2D: (nsamp, nbands)
            if nsamp > 1:
                f_scaled = A_hat[:, None] * f_model
            else:
                f_scaled = jnp.atleast_2d(A_hat * f_model)
            
            chi2 = jnp.sum(((y - f_scaled) ** 2) * w, axis=-1)
            # logdet = jnp.sum(jnp.log(2 * jnp.pi / (w + 1e-10)))

            loglike = -0.5 * chi2
            return loglike, A_hat, f_scaled
            
        if nsamp > 1:
            observed_flux = jnp.broadcast_to(observed_flux, model_flux.shape)
            weight = jnp.broadcast_to(weight, model_flux.shape)


        loglike, A_hat, rescaled_flux = marginal_loglike(observed_flux, weight, model_flux)
    
        if return_rescaled_flux_and_loglike:
            if return_amplitude:
                return rescaled_flux, loglike, redshift, A_hat
            return rescaled_flux, loglike, redshift
        else:
            if return_amplitude:
                return loglike, redshift, A_hat
            return loglike, redshift

    def push_spec_marg_log(self, latents, redshift, observed_flux=None, weight=None,
                           marginalize_amplitude=True, log_amplitude=None,
                           return_rescaled_flux_and_loglike=False, return_amplitude=False,
                           filter_curves=None, lam_interp=None,
                           decoder_params=None, use_jit=True):
        """
        Forward model using log-wavelength grid for fast redshift shifting.
        
        This is an optimized version of push_spec_marg that uses the log-wavelength
        grid to convert redshift from O(n) interpolation to O(1) index shift with
        linear interpolation. This can provide significant speedups for MCMC sampling.
        
        The key insight is that in log-wavelength space:
            λ_obs = λ_rest * (1+z)  =>  log(λ_obs) = log(λ_rest) + log(1+z)
        
        So redshift becomes a simple shift by log(1+z)/dloglam grid points.
        
        NOTE: You must call setup_log_wavelength_grid() before using this method.
        
        Parameters
        ----------
        latents : jnp.ndarray
            Latent variables, shape (nsamp, nlatent)
        redshift : jnp.ndarray or float
            Redshift value(s), shape (nsamp,) or scalar
        observed_flux : jnp.ndarray, optional
            Observed fluxes for likelihood calculation, shape (nband,)
        weight : jnp.ndarray, optional
            Weights (inverse variance), shape (nband,)
        marginalize_amplitude : bool, optional
            If True, analytically marginalize over amplitude. Default True.
        log_amplitude : jnp.ndarray or float, optional
            If provided, use this fixed log amplitude instead of marginalizing.
        return_rescaled_flux_and_loglike : bool, optional
            If True, return (model_flux, loglike, redshift). Default False.
        return_amplitude : bool, optional
            If True, also return the best-fit or provided amplitude.
        filter_curves : jnp.ndarray, optional
            Custom filter curves, shape (nsamp, nband, n_lam) or (nband, n_lam).
        lam_interp : jnp.ndarray, optional
            Custom wavelength grid for filter convolution. If None, uses self.lam_interp.
        decoder_params : dict, optional
            Custom decoder parameters (for fine-tuning).
        use_jit : bool, optional
            Whether to use JIT-compiled transforms. Default True.
            
        Returns
        -------
        Depends on return_rescaled_flux_and_loglike and return_amplitude flags.
        Default: (loglike, redshift)
        """
        if not self.use_log_wavelength or self.log_wav_rest is None:
            raise ValueError("Log wavelength grid not initialized. Call setup_log_wavelength_grid() first.")
        
        # Decode latents to rest-frame SED
        if use_jit:
            z_transformed = self.combined_transform_jit(latents)
        else:
            z_transformed = self.combined_transform_vmap(latents)
        
        if decoder_params is None:
            decoder_params = self.model.params['decoder']
        
        spec = self.decode_jit(decoder_params, z_transformed)
        nsamp = latents.shape[0]
        
        # Get spectrum on log-wavelength grid
        # If model was trained on log wavelengths, decoder output is already on log grid
        if self.params.get('use_log_wavelength', False):
            spec_log = spec  # Already on log grid from decoder
        else:
            # Need to interpolate from linear to log grid
            spec_log = self.interpolate_to_log_grid(spec)
        
        # Apply redshift via fast log-grid shifting
        redshift = jnp.atleast_1d(redshift)
        if redshift.shape[0] == 1 and nsamp > 1:
            redshift = jnp.repeat(redshift, nsamp)
        
        spec_shifted = self.shift_on_log_grid_batch(spec_log, redshift)
        
        # Interpolate shifted spectrum back to filter wavelength grid for convolution
        if lam_interp is None:
            lam_interp_use = self.lam_interp
        else:
            lam_interp_use = lam_interp
        
        # The shifted spectrum is on log_wav_rest * (1+z) = observed wavelengths
        # We need to interpolate to the filter wavelength grid
        def interp_to_filter_grid(spec_obs, z):
            # Observed wavelength grid for this shifted spectrum
            wav_obs = self.log_wav_rest * (1 + z)
            return jnp.interp(lam_interp_use, wav_obs, spec_obs)
        
        x_interp = jax.vmap(interp_to_filter_grid)(spec_shifted, redshift)
        
        # Filter convolution
        if filter_curves is not None:
            if nsamp == 1:
                model_flux = jnp.dot(filter_curves[0], x_interp[0])
            else:
                model_flux = jax.vmap(lambda f, x: jnp.dot(f, x))(filter_curves, x_interp)
        else:
            model_flux = jnp.dot(self.jax_filters, x_interp.T).T
        
        if not marginalize_amplitude and log_amplitude is None:
            return model_flux
        
        assert observed_flux is not None and weight is not None, "Must provide flux and weight for marginalization"
        
        # Case 1: Use log_amplitude as a free parameter (no marginalization)
        if log_amplitude is not None:
            amplitude = jnp.exp(log_amplitude)
            if nsamp > 1:
                f_scaled = amplitude[:, None] * model_flux
                observed_flux = jnp.broadcast_to(observed_flux, model_flux.shape)
                weight = jnp.broadcast_to(weight, model_flux.shape)
            else:
                f_scaled = amplitude * model_flux
            
            chi2 = jnp.sum(((observed_flux - f_scaled) ** 2) * weight, axis=-1)
            loglike = -0.5 * chi2
            
            if return_rescaled_flux_and_loglike:
                if return_amplitude:
                    return f_scaled, loglike, redshift, amplitude
                return f_scaled, loglike, redshift
            else:
                if return_amplitude:
                    return loglike, redshift, amplitude
                return loglike, redshift
        
        # Case 2: Marginalize amplitude analytically
        def marginal_loglike(y, w, f_model):
            A_hat = jnp.sum(f_model * y * w, axis=-1) / jnp.sum(f_model**2 * w, axis=-1)
            f_scaled = A_hat[:, None] * f_model if nsamp > 1 else A_hat * f_model
            chi2 = jnp.sum(((y - f_scaled) ** 2) * w, axis=-1)
            loglike = -0.5 * chi2
            return loglike, A_hat, f_scaled
        
        if nsamp > 1:
            observed_flux = jnp.broadcast_to(observed_flux, model_flux.shape)
            weight = jnp.broadcast_to(weight, model_flux.shape)
        
        loglike, A_hat, rescaled_flux = marginal_loglike(observed_flux, weight, model_flux)
        
        if return_rescaled_flux_and_loglike:
            if return_amplitude:
                return rescaled_flux, loglike, redshift, A_hat
            return rescaled_flux, loglike, redshift
        else:
            if return_amplitude:
                return loglike, redshift, A_hat
            return loglike, redshift
    
    def push_spec_marg_log_full(self, latents, redshift, observed_flux=None, weight=None,
                                marginalize_amplitude=True, log_amplitude=None,
                                return_rescaled_flux_and_loglike=False, return_amplitude=False,
                                decoder_params=None, use_jit=True):
        """
        Fully log-grid forward model - fastest option for MCMC sampling.
        
        This method operates entirely on log-wavelength grids, avoiding all
        interpolation after the initial rest-frame decode. Requires that both
        setup_log_wavelength_grid() and setup_log_filters() have been called.
        
        The speed advantage comes from:
        1. Redshift is a simple index shift (not per-wavelength interpolation)
        2. Filter convolution uses precomputed filters at the shifted wavelengths
        3. No interpolation back to linear wavelength grid
        
        Parameters
        ----------
        latents : jnp.ndarray
            Latent variables, shape (nsamp, nlatent)
        redshift : jnp.ndarray or float
            Redshift value(s), shape (nsamp,) or scalar
        observed_flux : jnp.ndarray, optional
            Observed fluxes for likelihood calculation, shape (nband,)
        weight : jnp.ndarray, optional
            Weights (inverse variance), shape (nband,)
        marginalize_amplitude : bool, optional
            If True, analytically marginalize over amplitude. Default True.
        log_amplitude : jnp.ndarray or float, optional
            If provided, use this fixed log amplitude instead of marginalizing.
        return_rescaled_flux_and_loglike : bool, optional
            If True, return (model_flux, loglike, redshift). Default False.
        return_amplitude : bool, optional
            If True, also return the best-fit or provided amplitude.
        decoder_params : dict, optional
            Custom decoder parameters (for fine-tuning).
        use_jit : bool, optional
            Whether to use JIT-compiled transforms. Default True.
            
        Returns
        -------
        Depends on return_rescaled_flux_and_loglike and return_amplitude flags.
        Default: (loglike, redshift)
        """
        if not hasattr(self, 'log_filters_by_z') or self.log_filters_by_z is None:
            raise ValueError("Log filters not initialized. Call setup_log_filters() first.")
        
        # Decode latents to rest-frame SED
        if use_jit:
            z_transformed = self.combined_transform_jit(latents)
        else:
            z_transformed = self.combined_transform_vmap(latents)
        
        if decoder_params is None:
            decoder_params = self.model.params['decoder']
        
        spec = self.decode_jit(decoder_params, z_transformed)
        nsamp = latents.shape[0]
        
        # Get spectrum on log-wavelength grid
        # If model was trained on log wavelengths, decoder output is already on log grid
        if self.params.get('use_log_wavelength', False):
            spec_log = spec  # Already on log grid from decoder
        else:
            # Need to interpolate from linear to log grid (one-time cost)
            spec_log = self.interpolate_to_log_grid(spec)
        
        # Get filter curves for these redshifts (interpolated from precomputed grid)
        redshift = jnp.atleast_1d(redshift)
        if redshift.shape[0] == 1 and nsamp > 1:
            redshift = jnp.repeat(redshift, nsamp)
        
        filters_at_z = self.get_filters_at_z_log(redshift)  # shape: (nsamp, nband, n_log_wav)
        
        # Compute model fluxes by convolving rest-frame spectrum with redshift-appropriate filters
        # No need to shift the spectrum - the filters are computed at the right wavelengths!
        model_flux = jax.vmap(lambda f, s: jnp.dot(f, s))(filters_at_z, spec_log)
        # shape: (nsamp, nband)
        
        if not marginalize_amplitude and log_amplitude is None:
            return model_flux
        
        assert observed_flux is not None and weight is not None, "Must provide flux and weight for marginalization"
        
        # Case 1: Use log_amplitude as a free parameter (no marginalization)
        if log_amplitude is not None:
            amplitude = jnp.exp(log_amplitude)
            if nsamp > 1:
                f_scaled = amplitude[:, None] * model_flux
                observed_flux = jnp.broadcast_to(observed_flux, model_flux.shape)
                weight = jnp.broadcast_to(weight, model_flux.shape)
            else:
                f_scaled = amplitude * model_flux
            
            chi2 = jnp.sum(((observed_flux - f_scaled) ** 2) * weight, axis=-1)
            loglike = -0.5 * chi2
            
            if return_rescaled_flux_and_loglike:
                if return_amplitude:
                    return f_scaled, loglike, redshift, amplitude
                return f_scaled, loglike, redshift
            else:
                if return_amplitude:
                    return loglike, redshift, amplitude
                return loglike, redshift
        
        # Case 2: Marginalize amplitude analytically
        def marginal_loglike(y, w, f_model):
            A_hat = jnp.sum(f_model * y * w, axis=-1) / jnp.sum(f_model**2 * w, axis=-1)
            f_scaled = A_hat[:, None] * f_model if nsamp > 1 else A_hat * f_model
            chi2 = jnp.sum(((y - f_scaled) ** 2) * w, axis=-1)
            loglike = -0.5 * chi2
            return loglike, A_hat, f_scaled
        
        if nsamp > 1:
            observed_flux = jnp.broadcast_to(observed_flux, model_flux.shape)
            weight = jnp.broadcast_to(weight, model_flux.shape)
        
        loglike, A_hat, rescaled_flux = marginal_loglike(observed_flux, weight, model_flux)
        
        if return_rescaled_flux_and_loglike:
            if return_amplitude:
                return rescaled_flux, loglike, redshift, A_hat
            return rescaled_flux, loglike, redshift
        else:
            if return_amplitude:
                return loglike, redshift, A_hat
            return loglike, redshift

    def get_restframe_sed(self, latents, redshift_in_flow=False, decoder_params=None, use_jit=True):
        """
        Generate rest-frame SED from latent variables.
        
        This is a separate function from push_spec_marg to avoid JIT compilation issues
        with conditional outputs. Returns only the rest-frame SED before redshifting
        and filter convolution.
        
        Parameters
        ----------
        latents : jnp.ndarray
            Latent variables, shape (nsamp, nlatent) or (nsamp, nlatent+1) if redshift_in_flow
        redshift_in_flow : bool, optional
            If True, redshift is included in latent variables. Default False.
        decoder_params : dict, optional
            Custom decoder parameters (for fine-tuning). If None, uses self.model.params['decoder'].
        use_jit : bool, optional
            Whether to use JIT-compiled transforms. Default True.
            
        Returns
        -------
        spec : jnp.ndarray
            Rest-frame SED, shape (nsamp, n_wavelengths)
        """
        if use_jit:
            z = self.combined_transform_jit(latents)
        else:
            z = self.combined_transform_vmap(latents)
            
        if redshift_in_flow:
            z = z[:,:-1]  # decoder takes latent variables but not redshift

        if decoder_params is None:
            decoder_params = self.model.params['decoder']

        spec = self.decode_jit(decoder_params, z)
        
        return spec

    def find_nearest_neighbors_in_training(self, query_latents, k=5, latents_fpath=None, return_distances=False):
        """
        Find k nearest neighbors in the training set latent space.
        
        This function finds the closest training examples to query latent vectors,
        allowing you to retrieve the original noiseless training SEDs.
        
        Parameters
        ----------
        query_latents : jnp.ndarray or np.ndarray
            Query latent vectors, shape (n_queries, nlatent)
        k : int, optional
            Number of nearest neighbors to return. Default 5.
        latents_fpath : str, optional
            Path to latents.npz file. If None, uses default path from rundir.
        return_distances : bool, optional
            If True, also returns the distances to the nearest neighbors. Default False.
            
        Returns
        -------
        neighbor_indices : np.ndarray
            Indices of the k nearest neighbors in the training set, shape (n_queries, k)
        distances : np.ndarray, optional
            Distances to the k nearest neighbors, shape (n_queries, k)
            Only returned if return_distances=True
            
        Notes
        -----
        The returned indices can be used to retrieve the original training SEDs from
        your training data files (e.g., the data used in dat_obj.build_dataloaders()).
        """
        # Load training latents
        if latents_fpath is None:
            latents_fpath = self.rundir + '/latents/latents.npz'
        
        print(f'Loading training latents from {latents_fpath}')
        latents_file = np.load(latents_fpath)
        training_latents = latents_file['all_z']  # Shape: (n_training, nlatent)
        print(f'Loaded {training_latents.shape[0]} training latents with dimension {training_latents.shape[1]}')
        
        # Convert to numpy for efficient computation
        if isinstance(query_latents, jnp.ndarray):
            query_latents = np.array(query_latents)
        
        # Compute pairwise distances using broadcasting
        # query_latents: (n_queries, nlatent)
        # training_latents: (n_training, nlatent)
        # distances: (n_queries, n_training)
        distances = np.linalg.norm(
            query_latents[:, None, :] - training_latents[None, :, :], 
            axis=2
        )
        
        # Find k nearest neighbors
        neighbor_indices = np.argpartition(distances, k, axis=1)[:, :k]
        
        # Sort the k neighbors by distance
        for i in range(query_latents.shape[0]):
            sorted_idx = np.argsort(distances[i, neighbor_indices[i]])
            neighbor_indices[i] = neighbor_indices[i][sorted_idx]
        
        if return_distances:
            neighbor_distances = np.array([
                distances[i, neighbor_indices[i]] 
                for i in range(query_latents.shape[0])
            ])
            return neighbor_indices, neighbor_distances
        
        return neighbor_indices

    def get_training_seds_from_indices(self, indices, training_data_fpath=None, training_data=None):
        """
        Retrieve original training SEDs given their indices.
        
        Parameters
        ----------
        indices : np.ndarray
            Indices of training examples, shape (n_queries, k) or (n_queries,)
        training_data_fpath : str, optional
            Path to training data file. If None and training_data is None, 
            attempts to load from standard location.
        training_data : np.ndarray, optional
            Pre-loaded training data array, shape (n_training, n_wavelengths).
            If provided, used instead of loading from file.
            
        Returns
        -------
        training_seds : np.ndarray
            Original training SEDs corresponding to the indices.
            Shape matches indices dimensions plus wavelength dimension.
            
        Notes
        -----
        This retrieves the actual noiseless training spectra that were encoded
        into the latent space, not reconstructions from the decoder.
        """
        # Load training data if not provided
        if training_data is None:
            if training_data_fpath is None:
                # Try standard location
                training_data_fpath = self.rundir + '/../../data/training_seds.npz'
                print(f'Attempting to load training data from {training_data_fpath}')
            
            if training_data_fpath.endswith('.npz'):
                data_file = np.load(training_data_fpath)
                # Try common key names
                for key in ['seds', 'spectra', 'data', 'flux', 'all_spec']:
                    if key in data_file:
                        training_data = data_file[key]
                        break
                if training_data is None:
                    raise ValueError(f"Could not find SED data in {training_data_fpath}. "
                                   f"Available keys: {list(data_file.keys())}")
            else:
                # Assume it's a text file or numpy binary
                training_data = np.load(training_data_fpath)
            
            print(f'Loaded training data with shape {training_data.shape}')
        
        # Index into training data
        flat_indices = indices.flatten() if indices.ndim > 1 else indices
        training_seds = training_data[flat_indices]
        
        # Reshape to match input indices shape
        if indices.ndim > 1:
            training_seds = training_seds.reshape(indices.shape + (-1,))
        
        return training_seds

                    
    def sample_spec(self, nsamp=1, key=None, redshift=None):

        if key is None:
            key = self.key  # Store a persistent PRNG key in the class

        # t0 = time.time()
        z = eqx.filter_jit(self.flow.sample)(key, (nsamp,))
        z = jax.vmap(self.rescale.transform)(z)
        # dtnf = time.time()-t0
        # print('dt NF:', dtnf)
        
        spec = self.decode_jit(self.model.params['decoder'], z)
        # dtdec = time.time()-t0-dtnf
        # print('dt decoder:', dtdec)
        
        if redshift is not None:
            nspec = spec.shape[0]
            wave_redshifted = self.central_wavelengths * (1 + redshift[:, None])            
            x_interp = vmap(lambda i: linear_interp_jax(self.lam_interp, wave_redshifted[i], spec[i]))(jnp.arange(nspec)) 

            # lambda_inv = 1.0 / self.lam_interp
            
            spec = jnp.dot(self.jax_filters, x_interp.T).T
        
        return spec

    def push_spec_marg_with_lines(self, latents, redshift, observed_flux, weight, 
                                  emission_line_config, return_rescaled_flux_and_loglike=False, 
                                  redshift_in_flow=False, decoder_params=None, use_jit=True, 
                                  filter_curves=None, lam_interp=None):
        """
        Forward model with emission lines: latent -> decoded SED -> add emission lines -> photometry.
        
        This is a separate method from push_spec_marg to avoid JIT compilation issues with
        conditional branching. Use this method when you want emission line modeling.
        
        Parameters
        ----------
        latents : jnp.ndarray
            Latent variables, shape (nsamp, nlatent)
        redshift : jnp.ndarray
            Redshifts, shape (nsamp,) or (1,)
        observed_flux : jnp.ndarray
            Observed photometry, shape (nsamp, nband) or (nband,)
        weight : jnp.ndarray
            Inverse variance weights, shape (nsamp, nband) or (nband,)
        emission_line_config : EmissionLineConfig
            Configuration for emission line modeling (must have enabled=True)
        return_rescaled_flux_and_loglike : bool
            If True, returns (model_flux, loglike, redshift)
            If False, returns (loglike, redshift)
        redshift_in_flow : bool
            If True, redshift is included in latent variables
        decoder_params : dict, optional
            Custom decoder parameters (for fine-tuning)
        use_jit : bool
            Whether to use JIT-compiled transforms
        filter_curves : jnp.ndarray, optional
            Per-sample filter curves, shape (nsamp, nband, n_lam)
        lam_interp : jnp.ndarray, optional
            Wavelength grid for interpolation
            
        Returns
        -------
        If return_rescaled_flux_and_loglike=True:
            model_flux : jnp.ndarray, shape (nsamp, nband)
            loglike : jnp.ndarray, shape (nsamp,)
            redshift : jnp.ndarray, shape (nsamp,)
        else:
            loglike : jnp.ndarray, shape (nsamp,)
            redshift : jnp.ndarray, shape (nsamp,)
        """
        from .emission_lines import (
            EmissionLineRegistry,
            compute_emission_line_model
        )
        
        # Decode latents to continuum spectrum (same as push_spec_marg)
        if use_jit:
            z = self.combined_transform_jit(latents)
        else:
            z = self.combined_transform_vmap(latents)
            
        if redshift_in_flow:
            z = z[:,:-1]
            redshift = z[:,-1]

        if decoder_params is None:
            decoder_params = self.model.params['decoder']

        spec = self.decode_jit(decoder_params, z)
    
        nsamp = latents.shape[0]
        
        # Use custom wavelength grid if provided
        if lam_interp is None:
            lam_interp_use = self.lam_interp
        else:
            lam_interp_use = lam_interp
    
        # Interpolate continuum to common wavelength grid
        if nsamp == 1:
            x_interp = jnp.interp(lam_interp_use, self.central_wavelengths * (1 + redshift), spec[0])
        else:
            if redshift.shape[0] == 1 and nsamp > 1:
                redshift_broadcast = jnp.repeat(redshift, nsamp, axis=0)
            else:
                redshift_broadcast = redshift
            wave_redshifted = self.central_wavelengths * (1 + redshift_broadcast[:, None])
            x_interp = jax.vmap(lambda x, y: jnp.interp(lam_interp_use, x, y))(wave_redshifted, spec)
        
        # Initialize emission line registry
        registry = EmissionLineRegistry()
        
        # Determine which filters to use
        if filter_curves is not None:
            filters_use = filter_curves
        else:
            # Convert homogenized filters to per-sample format
            filters_use = jnp.broadcast_to(
                self.jax_filters[None, :, :],
                (nsamp, self.jax_filters.shape[0], self.jax_filters.shape[1])
            )
        
        # Broadcast observed flux and weights if needed
        if nsamp > 1:
            observed_flux_use = jnp.broadcast_to(observed_flux, (nsamp, observed_flux.shape[-1]))
            weight_use = jnp.broadcast_to(weight, (nsamp, weight.shape[-1]))
        else:
            observed_flux_use = observed_flux
            weight_use = weight
        
        # Broadcast redshift if needed
        if redshift.shape[0] == 1 and nsamp > 1:
            redshift_use = jnp.repeat(redshift, nsamp, axis=0)
        else:
            redshift_use = redshift
            
        # Call emission line forward model with joint marginalization
        loglike, A_hat, model_flux = compute_emission_line_model(
            wavelength_grid=lam_interp_use,
            filter_curves=filters_use,
            continuum_spectrum=x_interp,
            observed_flux=observed_flux_use,
            weight=weight_use,
            redshift=redshift_use,
            emission_config=emission_line_config,
            registry=registry
        )
        
        if return_rescaled_flux_and_loglike:
            return model_flux, loglike, redshift
        else:
            return loglike, redshift



def load_filter_central_wavelengths(filter_set_name, filtfiles=None):
    """
    Load central wavelengths for a given filter set.
    
    Parameters
    ----------
    filter_set_name : str
        Name of the filter set directory
    filtfiles : list of str, optional
        List of filter file paths. If provided, function will try to extract
        central wavelengths from corresponding file
    
    Returns
    -------
    wave_obs : ndarray
        Central wavelengths of the filters
    filtfiles : list of str
        List of filter file paths (if not provided, will be loaded from directory)
    """
    from pathlib import Path
    
    # Special handling for legacy spherex_filters102 format
    if filter_set_name in ['spherex_filters102', 'spherex_filters102/']:
        # Legacy format: loads from sphx_dat_path instead of scratch_basepath
        legacy_file = Path(config.sphx_dat_path) / 'central_wavelengths_sphx102.npz'
        if legacy_file.exists():
            wave_obs = np.sort(np.load(legacy_file)['central_wavelengths'])
            return wave_obs, filtfiles
    
    filter_dir = Path(config.scratch_basepath) / 'data' / 'filters' / filter_set_name
    
    # Check for central wavelengths file with filter names (new format)
    cent_wave_file = filter_dir / 'fiducial_filters_cent_waves.txt'
    
    if cent_wave_file.exists():
        # Load from two-column text file (filter_name, central_wavelength)
        data = np.loadtxt(cent_wave_file, dtype=str)
        filter_names = data[:, 0]
        wave_obs = data[:, 1].astype(float)
        
        if filtfiles is None:
            # Construct full paths for filter files
            filtfiles = [str(filter_dir / fname) for fname in filter_names]
        
        return wave_obs, filtfiles
    
    # Fallback to old .npz format in filter directory
    cent_wave_npz = filter_dir / f'central_wavelengths_{filter_set_name.replace("/", "")}.npz'
    if not cent_wave_npz.exists():
        # Try without filter_set_name in filename
        cent_wave_npz = filter_dir / 'central_wavelengths.npz'
    
    if cent_wave_npz.exists():
        wave_obs = np.sort(np.load(cent_wave_npz)['central_wavelengths'])
        return wave_obs, filtfiles
    
    # If no central wavelength file found, raise error
    raise FileNotFoundError(f"Could not find central wavelength file for {filter_set_name}. "
                          f"Tried: {cent_wave_file}, {cent_wave_npz}")


def initialize_PAE(run_name, filter_set_name='spherex_filters102/', filtfiles=None, with_ext_phot=False, \
                  filename_flow='flow_model_iaf_50k', redshift_in_flow=False, inference_dtype=jnp.float32, \
                  lam_min_rest=0.1, lam_max_rest=5.0, nlam_rest=500):

    central_wavelengths = np.linspace(lam_min_rest, lam_max_rest, nlam_rest) # rest frame
    
    # Load central wavelengths for the filter set
    wave_obs, filtfiles = load_filter_central_wavelengths(filter_set_name, filtfiles=filtfiles)

    if with_ext_phot:
        wave_obs = np.concatenate((np.array([3.4, 4.6, 0.475, 0.63, 0.9]), wave_obs))
        print('wave obs is now ', wave_obs)
    
    params_additional = dict({'filter_integrate':True, 'nlam_interp':1000, 'redshift_min':0.0001, 'redshift_max':5.0})
    
    PAE_obj = PAE_JAX(run_name=run_name, central_wavelengths=central_wavelengths, modl_type='jax', \
                wave_obs=wave_obs, params_additional=params_additional, load_flow_decoder=False, \
                        filter_set_name=filter_set_name, filtfiles=filtfiles, with_ext_phot=with_ext_phot, \
                     inference_dtype=inference_dtype)

    if redshift_in_flow:
        filename_flow += '_with_redshift'

    filename_flow += '.pkl'
    PAE_obj.load_flow_decoder(run_name, filename_save=filename_flow)

    return PAE_obj


def load_real_spherex_data(parquet_file, filter_set_name,
                           weight_soft=5e-4, abs_norm=True, max_normflux=100,
                           nbands_rf=500):
    """
    Load real SPHEREx photometry from parquet file.
    
    Parameters
    ----------
    parquet_file : str
        Path to parquet file with real SPHEREx observations
    filter_set_name : str
        Name of filter set (e.g., 'SPHEREx_filter_306')
    weight_soft : float
        Soft weighting parameter for normalization
    abs_norm : bool
        Whether to use absolute normalization
    max_normflux : float
        Maximum normalized flux value
    nbands_rf : int
        Number of rest frame bands (for central_wavelengths array)
        
    Returns
    -------
    dat_obs : spec_data_jax
        Observed data object
    property_cat_df_obs : DataFrame
        Property catalog for observed data
    property_cat_df_restframe : None
        Always None for real data (no rest frame equivalents)
    central_wavelengths : ndarray
        Central wavelengths for rest frame (linearly spaced)
    wave_obs : ndarray
        Observed wavelengths
    """
    from pathlib import Path
    
    # Load central wavelengths for the filter set
    wave_obs, _ = load_filter_central_wavelengths(filter_set_name, filtfiles=None)
    nbands_obs = len(wave_obs)
    
    print(f"Loading real data with {nbands_obs} bands from {filter_set_name}")
    
    # Load real photometry from parquet
    dat_obs, property_cat_df_obs = load_real_spherex_parquet(
        parquet_file, filter_set_name, wave_obs,
        weight_soft=weight_soft, abs_norm=abs_norm, max_normflux=max_normflux
    )
    
    # No rest frame data for real observations
    property_cat_df_restframe = None
    central_wavelengths = np.linspace(0.1, 5, nbands_rf)
    
    return dat_obs, property_cat_df_obs, property_cat_df_restframe, central_wavelengths, wave_obs
    

def load_spherex_data(sig_level_norm=0.01, sel_str=None, sed_set='COSMOS', nbands_obs=102, 
                      abs_norm=False, with_ext_phot=False, data_fpath=None, 
                      load_rf_dat=True, load_obs_dat=True, nbands_rf=500, weight_soft=5e-4):
    """
    Load SPHEREx observed and (optionally) rest frame data without initializing PAE model.
    
    This is a leaner version of set_up_pae_wrapper that only handles data loading,
    allowing you to initialize the PAE model separately with initialize_PAE().
    
    Parameters
    ----------
    sig_level_norm : float
        Signal level normalization
    sel_str : str, optional
        Selection string for data subset (e.g., 'zlt22.5')
    sed_set : str
        SED dataset name (default 'COSMOS')
    nbands_obs : int
        Number of observed bands (default 102)
    abs_norm : bool
        Whether to use absolute normalization
    with_ext_phot : bool
        Whether to include external photometry
    data_fpath : str, optional
        Custom data file path
    load_rf_dat : bool
        Whether to load rest frame data
    load_obs_dat : bool
        Whether to load observed frame data
    nbands_rf : int
        Number of rest frame bands (default 500)
    weight_soft : float
        Soft weighting parameter for normalization
    
    Returns
    -------
    dat_obs : spec_data_jax
        Observed frame data object
    property_cat_df_obs : DataFrame
        Observed frame property catalog
    property_cat_df_restframe : DataFrame or None
        Rest frame property catalog (if load_rf_dat=True)
    central_wavelengths : ndarray
        Central wavelengths array
    wave_obs : ndarray
        Observed wavelengths
    """

    # ----------------- rest frame dataloader ------------------
    if load_rf_dat:
        dat_rf = spec_data_jax(nbands_rf) 
        fpath_dict_rf = grab_fpaths_traindat(sed_set, restframe=True)
    
        property_cat_df_restframe = dat_rf.build_dataloaders_new(fpath_dict_rf, train_frac=1.0, load_property_cat=False, property_cat_fpath=None, \
                                         restframe=True, save_property_cat=False)

        central_wavelengths = dat_rf.sed_um_wave
    else:
        central_wavelengths = np.linspace(0.1, 5, nbands_rf)
        property_cat_df_restframe = None
        
    # ------------------ observed frame dataloader ----------------
    if load_obs_dat:
        dat_obs = spec_data_jax(nbands_obs)
        fpath_dict = grab_fpaths_traindat(sed_set, restframe=False, train_mode='full', sel_str=sel_str)
        property_cat_fpath = None
    
        if nbands_obs==102:
            if data_fpath is not None:
                fpath_dict['data_fpath'] = data_fpath
            else:
                fpath_dict['data_fpath'] = config.sphx_dat_path+'phot/catgrid_sphxonly_COSMOS_aug2x_debv_0p02_dustlaw_obs_fullsky_zlt22.5'
            fpath_dict['catgrid_noiseless_fpath'] = config.sphx_dat_path+'phot/catgrid_sphxonly_COSMOS_aug2x_debv_0p02_dustlaw.out'
        elif nbands_obs==408:
            fpath_dict['data_fpath'] = config.sphx_dat_path+'phot/catgrid_sphxonly_'+filter_set_name+'_COSMOS_aug2x_debv_0p02_dustlaw_obs_fullsky_zlt22.5'
            fpath_dict['catgrid_noiseless_fpath'] = config.sphx_dat_path+'phot/catgrid_sphxonly_'+filter_set_name+'_COSMOS_aug2x_debv_0p02_dustlaw.out'
    
        if with_ext_phot:
            fpath_dict['data_fpath'] = fpath_dict['data_fpath'].replace('sphxonly', 'W1W2grz_sphx'+str(nbands_obs))
        
        print('data fpath is ', fpath_dict['data_fpath'])
        
        property_cat_df_obs = dat_obs.build_dataloaders_new(fpath_dict, train_frac=1.0, weight_soft=weight_soft, load_property_cat=False, property_cat_fpath=property_cat_fpath, \
                                         restframe=False, save_property_cat=False, apply_sel=True, abs_norm=abs_norm)
    else:
        property_cat_df_obs, dat_obs = None, None
        
    # Compute observed wavelengths
    wave_obs = np.sort(np.load(config.sphx_dat_path+'central_wavelengths_sphx102.npz')['central_wavelengths'])
    
    if with_ext_phot:
        wave_obs = np.concatenate((np.array([3.4, 4.6, 0.475, 0.63, 0.9]), wave_obs))
        print('wave obs is now ', wave_obs)
    
    return dat_obs, property_cat_df_obs, property_cat_df_restframe, central_wavelengths, wave_obs


def set_up_pae_wrapper(nlatent=8, sig_level_norm=0.01, sel_str=None, sed_set='COSMOS', nbands_obs=102, \
                    filter_set_name = 'spherex_filters102/', filtfiles = None, abs_norm=False, run_name=None, \
                      redshift_in_flow=False, with_ext_phot=False, inference_dtype=jnp.float32, data_fpath=None, 
                      load_rf_dat=True, load_obs_dat=True, nbands_rf=500, filename_flow='flow_model_iaf_092225'):

    filters = [16, 32, 128, 256]
    n_hidden_encoder=[256, 64, 16]
    filter_sizes = [5, 5, 5, 5]

    params = param_dict_gen('jax', filter_sizes=filter_sizes, n_hidden_encoder=n_hidden_encoder, \
                           lr=2e-4, filters=filters, nlatent=nlatent, epochs=100, nbands=nbands_rf, \
                           restframe=True, mean_sub_latents=False, \
                           plot_interval=5, weight_decay=0., nepoch_flow=50)

    if run_name is None:
        run_name='jax_conv1_nlatent='+str(nlatent)+'_siglevelnorm='+str(sig_level_norm)+'_newAllen'
        
    rundir = config.modl_runs_path + run_name + '/'

    # if sel_str is not None:
    #     run_name += '_'+sel_str

    # ----------------- rest frame dataloader ------------------

    if load_rf_dat:
        dat_rf = spec_data_jax(params['nbands']) 
        fpath_dict_rf = grab_fpaths_traindat(sed_set, restframe=True)
    
        property_cat_df_restframe = dat_rf.build_dataloaders_new(fpath_dict_rf, train_frac=1.0, load_property_cat=False, property_cat_fpath=None, \
                                         restframe=True, save_property_cat=False)

        central_wavelengths = dat_rf.sed_um_wave

        srcid_rf = np.array(property_cat_df_restframe['Tractor_ID'])

    else:
        central_wavelengths = np.linspace(0.15, 5, nbands_rf)
        property_cat_df_restframe = None

    # print('central wavelengths rest frame ', central_wavelengths)
    # ------------------ observed frame dataloader ----------------

    if load_obs_dat:
        dat_obs = spec_data_jax(nbands_obs) # 102 bands for observed
        fpath_dict = grab_fpaths_traindat(sed_set, restframe=False, train_mode='full', sel_str=sel_str)
        property_cat_fpath = None
    
        if nbands_obs==102:
            if data_fpath is not None:
                fpath_dict['data_fpath'] = data_fpath
            else:
                fpath_dict['data_fpath'] = config.sphx_dat_path+'phot/catgrid_sphxonly_COSMOS_aug2x_debv_0p02_dustlaw_obs_fullsky_zlt22.5'
            fpath_dict['catgrid_noiseless_fpath'] = config.sphx_dat_path+'phot/catgrid_sphxonly_COSMOS_aug2x_debv_0p02_dustlaw.out'
        elif nbands_obs==408:
            fpath_dict['data_fpath'] = config.sphx_dat_path+'phot/catgrid_sphxonly_'+filter_set_name+'_COSMOS_aug2x_debv_0p02_dustlaw_obs_fullsky_zlt22.5'
            fpath_dict['catgrid_noiseless_fpath'] = config.sphx_dat_path+'phot/catgrid_sphxonly_'+filter_set_name+'_COSMOS_aug2x_debv_0p02_dustlaw.out'
    
        if with_ext_phot:
            fpath_dict['data_fpath'] = fpath_dict['data_fpath'].replace('sphxonly', 'W1W2grz_sphx'+str(nbands_obs))
        
        print('data fpath is ', fpath_dict['data_fpath'])
    
        
        property_cat_df_obs = dat_obs.build_dataloaders_new(fpath_dict, train_frac=1.0, weight_soft=params['weight_soft'], load_property_cat=False, property_cat_fpath=property_cat_fpath, \
                                         restframe=False, save_property_cat=False, apply_sel=True, abs_norm=abs_norm)

    else:
        property_cat_df_obs, dat_obs = None, None
    # ------------- initialize PAE ----------------------------------

    wave_obs = np.sort(np.load(config.sphx_dat_path+'central_wavelengths_sphx102.npz')['central_wavelengths'])

    # def initialize_PAE(run_name, filter_set_name='spherex_filters102/', filtfiles=None, with_ext_phot=False, \
    #                   filename_flow='flow_model_iaf_50k', redshift_in_flow=False, inference_dtype=jnp.float32, \
    #                   lam_min_rest=0.1, lam_max_rest=5.0, nlam_rest=500):

    # PAE_obj = initialize_PAE(run_name, filter_set_name=filter_set_name,
    #                         filtfiles=filtfiles, with_ext_phot=with_ext_phot, \
    #                         filename_flow=filename_flow, redshift_in_flow=redshift_in_flow, 
    #                         inference_dtype=inference_dtype, \
    #                  lam_min_rest=0.15, lam_max_rest=5.0, nlam_rest=params['nbands'])
        
    if with_ext_phot:
        wave_obs = np.concatenate((np.array([3.4, 4.6, 0.475, 0.63, 0.9]), wave_obs))
        print('wave obs is now ', wave_obs)
    
    params_additional = dict({'filter_integrate':True, 'nlam_interp':1000, 'redshift_min':0.0001, 'redshift_max':5.0})
    
    PAE_obj = PAE_JAX(run_name=run_name, central_wavelengths=central_wavelengths, modl_type='jax', \
                wave_obs=wave_obs, params_additional=params_additional, load_flow_decoder=False, \
                        filter_set_name=filter_set_name, filtfiles=filtfiles, with_ext_phot=with_ext_phot, \
                     inference_dtype=inference_dtype)

    # filename_flow = 'flow_model_iaf_50k'
    if redshift_in_flow:
        filename_flow += '_with_redshift'

    filename_flow += '.pkl'
    PAE_obj.load_flow_decoder(run_name, filename_save=filename_flow)

    return PAE_obj, wave_obs, dat_obs, property_cat_df_restframe, property_cat_df_obs, params


def plot_profile_like_input(wave_obs, spec_obs, flux_unc, norm, save_path, run_idx, ztrue):
    """
    Create diagnostic plot showing input photometry for profile likelihood run.
    
    Parameters
    ----------
    wave_obs : array
        Observed wavelengths in microns
    spec_obs : array
        Observed flux (normalized)
    flux_unc : array
        Flux uncertainties (normalized)
    norm : float
        Normalization factor
    save_path : str
        Path to save the figure
    run_idx : int
        Source index
    ztrue : float
        True redshift
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.errorbar(wave_obs, spec_obs*norm, yerr=flux_unc*norm, 
                fmt='o', capsize=2.5, label=f'Source {run_idx}')
    ax.grid(alpha=0.3)
    ax.set_xlabel(r'$\lambda$ [$\mu$m]', fontsize=14)
    ax.set_ylabel('Flux (normalized)', fontsize=14)
    # Dynamic y-limits based on data range
    flux_data = spec_obs*norm
    ymax = np.nanmax(flux_data + flux_unc*norm) * 1.3
    ymin = np.nanmin(flux_data - flux_unc*norm) * 1.3 if np.nanmin(flux_data - flux_unc*norm) < 0 else -ymax * 0.1
    ax.set_ylim(ymin, ymax)
    ax.set_title(f'Source {run_idx}, z_true={ztrue:.3f}', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def prof_like_wrapper(ngal_test, run_idxs, all_spec_obs, weights, 
                      datestr, zpdf_tf, chisq_tf, redshift,
                      run_name, sel_str='zlt22.5',
                      compute_pl=True, save_pl=True,
                      basedir=None,
                      profile_run_subdir=None,
                     Z_MIN=0.0, Z_MAX=3.0, OPTIMIZER_MAXITER=100, 
                     NUM_RESTARTS_PER_Z=3, NZ_GRID=200, plstr='pl_nopost', nf_alpha=1.0, 
                    plot=True, save_bestfit_models=False, save_restframe_seds=False, PAE_obj=None,
                    save_input_plots=False, wave_obs=None, all_flux_unc=None, norms=None,
                    pl_subsample_step=1, srcid_obs=None, tailstring=None,
                    use_warmstart=False, warmstart_coarse_factor=10, 
                    warmstart_num_restarts_coarse=5, warmstart_num_restarts_fine=1,
                    warmstart_adaptive_fallback_restarts=3, warmstart_error_tol=1e-3,
                    mcpl_map_latents_all=None, z_bins_mcpl=None,
                    use_gpu_parallel=False, gpu_batch_size=None):
    """
    Wrapper function for computing profile likelihood for multiple sources.
    
    Parameters:
    -----------
    save_bestfit_models : bool, optional
        If True, compute and save the reconstructed observed-frame SED at each redshift point
        for the profile likelihood. The models will be saved in the output .npz file
        with key 'all_bestfit_models'. Default is False.
    save_restframe_seds : bool, optional
        If True, compute and save the rest-frame SED (before redshifting and filter convolution)
        at each redshift point. The SEDs will be saved in the output .npz file with key
        'all_restframe_seds'. Default is False.
    PAE_obj : PAE_JAX, optional
        Pre-loaded PAE object. If None, will initialize a new PAE from run_name. Default is None.
    save_input_plots : bool, optional
        If True, save diagnostic plots of input photometry for each source. Default is False.
    wave_obs : array, optional
        Observed wavelengths (required if save_input_plots=True)
    all_flux_unc : array, optional
        Flux uncertainties (required if save_input_plots=True)
    norms : array, optional
        Normalization factors (required if save_input_plots=True)
    pl_subsample_step : int, optional
        If >1, save only every Nth best-fit model/restframe SED along the redshift grid
        (e.g., `pl_subsample_step=5` saves indices 0,5,10,...). Default 1 (save all).
    use_gpu_parallel : bool, optional
        If True, process multiple sources in parallel using GPU/TPU devices. Default is False.
    gpu_batch_size : int, optional
        Number of sources to process in parallel. If None and use_gpu_parallel=True, 
        processes sources one at a time (sequential). Default is None.
    """

    if basedir is None:
        basedir = config.profile_like_path

    finez = np.linspace(0.0, 3.002, 1501) # from template fitting

    # Use pre-loaded PAE if provided, otherwise initialize new one
    if PAE_obj is None:
        PAE_COSMOS = initialize_PAE(run_name, filter_set_name='spherex_filters102/', with_ext_phot=False, \
                           inference_dtype=jnp.float32, \
                          lam_min_rest=0.1, lam_max_rest=5.0, nlam_rest=500)
    else:
        PAE_COSMOS = PAE_obj
    
    # Build run-specific output directory under profile_like/<run_name>/<profile_run_subdir>/
    prof_like_dir = os.path.join(basedir, run_name)
    if profile_run_subdir is not None:
        prof_like_dir = os.path.join(prof_like_dir, profile_run_subdir)
    os.makedirs(prof_like_dir, exist_ok=True)

    # Create figures directory for input plots if needed
    if save_input_plots:
        fig_dir = os.path.join(prof_like_dir, 'figures', 'input_phot')
        os.makedirs(fig_dir, exist_ok=True)
    
    # Handle GPU parallelization if requested
    if use_gpu_parallel and gpu_batch_size is not None and gpu_batch_size > 1:
        print(f"\nProcessing {ngal_test} sources with GPU device parallelization (batch_size={gpu_batch_size})...")
        import jax
        import copy
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Get available devices
        devices = jax.devices()
        n_devices = len(devices)
        print(f"  Available devices: {n_devices}")
        for idx, dev in enumerate(devices):
            print(f"    Device {idx}: {dev}")
        
        # Determine effective parallelism
        n_parallel = min(gpu_batch_size, n_devices, ngal_test)
        print(f"  Using {n_parallel} parallel workers")

        # Each worker gets its own model instance to avoid thread-shared Flax/JAX state.
        worker_pae_models = []
        for worker_idx in range(n_parallel):
            try:
                worker_pae_models.append(copy.deepcopy(PAE_COSMOS))
            except Exception as e:
                print(f"  Warning: could not deepcopy PAE model for worker {worker_idx}: {e}")
                if PAE_obj is None:
                    print("  Reinitializing worker model from run_name as fallback")
                    worker_pae_models.append(
                        initialize_PAE(
                            run_name,
                            filter_set_name='spherex_filters102/',
                            with_ext_phot=False,
                            inference_dtype=jnp.float32,
                            lam_min_rest=0.1,
                            lam_max_rest=5.0,
                            nlam_rest=500
                        )
                    )
                else:
                    print("  Falling back to shared model for this worker")
                    worker_pae_models.append(PAE_COSMOS)
        
        def process_source_on_device(device_idx, source_idx):
            """Process a single source on a specific device."""
            run_idx = run_idxs[source_idx]
            device = devices[device_idx % n_devices]
            worker_model = worker_pae_models[device_idx % n_parallel]
            
            # Use source ID if available
            src_identifier = int(srcid_obs[run_idx]) if srcid_obs is not None else run_idx
            
            print(f"  [Device {device_idx % n_devices}] Processing source {source_idx+1}/{ngal_test} (srcid={src_identifier})...")
            
            try:
                # Build filename
                filename_base = f'{plstr}_srcid={src_identifier}_{datestr}'
                if tailstring is not None:
                    filename_base += f'_{tailstring}'
                save_fpath = os.path.join(prof_like_dir, filename_base + '.npz')
                
                # Save diagnostic plot if requested
                if save_input_plots and wave_obs is not None and all_flux_unc is not None and norms is not None:
                    plot_name = f'input_phot_srcid={src_identifier}_{datestr}'
                    if tailstring is not None:
                        plot_name += f'_{tailstring}'
                    plot_path = os.path.join(fig_dir, plot_name + '.png')
                    plot_profile_like_input(wave_obs, all_spec_obs[run_idx], all_flux_unc[run_idx], 
                                           norms[run_idx], plot_path, run_idx, redshift[run_idx])
                
                if compute_pl:
                    # Extract MCLMC MAP latents if available
                    mcpl_map_latents_src = None
                    if mcpl_map_latents_all is not None:
                        mcpl_map_latents_src = mcpl_map_latents_all[source_idx]
                    
                    # Put data on the specific device
                    with jax.default_device(device):
                        x_obs_dev = jax.device_put(all_spec_obs[run_idx], device)
                        weight_dev = jax.device_put(weights[run_idx], device)
                        
                        # Compute profile likelihood on this device
                        z_grid, profile_logL, all_map_thetas, all_bestfit_models, all_restframe_seds = prof_like(
                            x_obs_dev, weight_dev, worker_model,
                            Z_MIN=Z_MIN, Z_MAX=Z_MAX, NZ_GRID=NZ_GRID,
                            OPTIMIZER_MAXITER=OPTIMIZER_MAXITER,
                            NUM_RESTARTS_PER_Z=NUM_RESTARTS_PER_Z,
                            nf_alpha=nf_alpha,
                            save_bestfit_models=save_bestfit_models,
                            save_restframe_seds=save_restframe_seds,
                            use_warmstart=use_warmstart,
                            warmstart_coarse_factor=warmstart_coarse_factor,
                            warmstart_num_restarts_coarse=warmstart_num_restarts_coarse,
                            warmstart_num_restarts_fine=warmstart_num_restarts_fine,
                            warmstart_adaptive_fallback_restarts=warmstart_adaptive_fallback_restarts,
                            warmstart_error_tol=warmstart_error_tol,
                            mcpl_map_latents=mcpl_map_latents_src,
                            z_bins_mcpl=z_bins_mcpl
                        )
                    
                    if save_pl:
                        # Convert to numpy and save
                        save_dict = {
                            'z_grid': np.asarray(z_grid), 
                            'profile_logL': np.asarray(profile_logL),
                            'all_map_thetas': np.asarray(all_map_thetas), 
                            'ztrue': redshift[run_idx]
                        }
                        
                        if zpdf_tf is not None:
                            save_dict['profile_logL_tf'] = zpdf_tf[run_idx,3:]
                        if chisq_tf is not None:
                            save_dict['chisq_tf'] = chisq_tf[run_idx]
                        
                        if save_bestfit_models and all_bestfit_models is not None:
                            models_np = np.asarray(all_bestfit_models)
                            if pl_subsample_step is not None and pl_subsample_step > 1:
                                save_dict['all_bestfit_models'] = models_np[::pl_subsample_step]
                                save_dict['pl_subsample_step'] = pl_subsample_step
                            else:
                                save_dict['all_bestfit_models'] = models_np
                        
                        if save_restframe_seds and all_restframe_seds is not None:
                            seds_np = np.asarray(all_restframe_seds)
                            if pl_subsample_step is not None and pl_subsample_step > 1:
                                save_dict['all_restframe_seds'] = seds_np[::pl_subsample_step]
                            else:
                                save_dict['all_restframe_seds'] = seds_np
                        
                        np.savez(save_fpath, **save_dict)
                        print(f"  [Device {device_idx % n_devices}] ✓ Saved results for source {source_idx+1}/{ngal_test}")
                
                return (source_idx, True, None)
            
            except Exception as e:
                print(f"  [Device {device_idx % n_devices}] ✗ Error processing source {source_idx+1}/{ngal_test}: {e}")
                import traceback
                return (source_idx, False, traceback.format_exc())
        
        # Process sources in parallel using ThreadPoolExecutor
        print(f"\n{'='*70}")
        print(f"Starting parallel processing with {n_parallel} workers")
        print(f"{'='*70}\n")
        
        completed = 0
        failed = []
        
        with ThreadPoolExecutor(max_workers=n_parallel) as executor:
            # Submit bounded batches so active tasks map one-to-one to active devices.
            for batch_start in range(0, ngal_test, n_parallel):
                batch_end = min(batch_start + n_parallel, ngal_test)
                batch_source_indices = list(range(batch_start, batch_end))
                futures = {
                    executor.submit(process_source_on_device, local_dev_idx, source_idx): source_idx
                    for local_dev_idx, source_idx in enumerate(batch_source_indices)
                }

                for future in as_completed(futures):
                    source_idx, success, error = future.result()
                    completed += 1

                    if not success:
                        failed.append((source_idx, error))

                    if completed % 10 == 0 or completed == ngal_test:
                        print(f"\nProgress: {completed}/{ngal_test} sources completed")
        
        print(f"\n{'='*70}")
        print(f"GPU parallel processing complete")
        print(f"  Successful: {completed - len(failed)}/{ngal_test}")
        if failed:
            print(f"  Failed: {len(failed)}/{ngal_test}")
            print(f"  Failed source indices: {[idx for idx, _ in failed[:10]]}" + 
                  ("..." if len(failed) > 10 else ""))
        print(f"{'='*70}\n")
        
        return  # Exit early since we've processed all sources
    
    # Sequential processing (original code path)
    print(f"\nProcessing {ngal_test} sources sequentially...")
    
    for i, run_idx in enumerate(run_idxs[:ngal_test]):
        # Use source ID if available, otherwise fall back to array index
        src_identifier = int(srcid_obs[run_idx]) if srcid_obs is not None else run_idx
        
        # Build filename with optional tailstring
        filename_base = f'{plstr}_srcid={src_identifier}_{datestr}'
        if tailstring is not None:
            filename_base += f'_{tailstring}'
        save_fpath = os.path.join(prof_like_dir, filename_base + '.npz')
        
        # Save diagnostic plot of input photometry
        if save_input_plots and wave_obs is not None and all_flux_unc is not None and norms is not None:
            plot_name = f'input_phot_srcid={src_identifier}_{datestr}'
            if tailstring is not None:
                plot_name += f'_{tailstring}'
            plot_path = os.path.join(fig_dir, plot_name + '.png')
            plot_profile_like_input(wave_obs, all_spec_obs[run_idx], all_flux_unc[run_idx], 
                                   norms[run_idx], plot_path, run_idx, redshift[run_idx])
            print(f'  Saved input photometry plot to {plot_path}')

        if compute_pl:
            # Extract MCLMC MAP latents for this source if available
            mcpl_map_latents_src = None
            if mcpl_map_latents_all is not None:
                mcpl_map_latents_src = mcpl_map_latents_all[i]  # Shape: (n_bins, n_latent)
            
            z_grid, profile_logL, all_map_thetas, all_bestfit_models, all_restframe_seds = prof_like(all_spec_obs[run_idx], weights[run_idx], PAE_COSMOS, \
                                                            Z_MAX=Z_MAX, OPTIMIZER_MAXITER=OPTIMIZER_MAXITER, NUM_RESTARTS_PER_Z=NUM_RESTARTS_PER_Z,
                                                             NZ_GRID=NZ_GRID, nf_alpha=nf_alpha, save_bestfit_models=save_bestfit_models, 
                                                             save_restframe_seds=save_restframe_seds,
                                                             use_warmstart=use_warmstart,
                                                             warmstart_coarse_factor=warmstart_coarse_factor,
                                                             warmstart_num_restarts_coarse=warmstart_num_restarts_coarse,
                                                             warmstart_num_restarts_fine=warmstart_num_restarts_fine,
                                                             warmstart_adaptive_fallback_restarts=warmstart_adaptive_fallback_restarts,
                                                             warmstart_error_tol=warmstart_error_tol,
                                                             mcpl_map_latents=mcpl_map_latents_src, z_bins_mcpl=z_bins_mcpl)
    
            if save_pl:
                print('saving profile likelihood estimate to ', save_fpath)
                save_dict = {
                    'z_grid': z_grid, 
                    'profile_logL': profile_logL,
                    'all_map_thetas': all_map_thetas, 
                    'ztrue': redshift[run_idx]
                }
                # Only save TF photo-z results if they were provided
                if zpdf_tf is not None:
                    save_dict['profile_logL_tf'] = zpdf_tf[run_idx,3:]
                if chisq_tf is not None:
                    save_dict['chisq_tf'] = chisq_tf[run_idx]
                
                if save_bestfit_models and all_bestfit_models is not None:
                    if pl_subsample_step is not None and pl_subsample_step > 1:
                        subsampled = all_bestfit_models[::pl_subsample_step]
                        save_dict['all_bestfit_models'] = subsampled
                        save_dict['pl_subsample_step'] = pl_subsample_step
                        print(f'  Saved subsampled best-fit observed-frame models with shape {subsampled.shape} (step={pl_subsample_step})')
                    else:
                        save_dict['all_bestfit_models'] = all_bestfit_models
                        print(f'  Saved best-fit observed-frame models with shape {all_bestfit_models.shape}')
                    # Also create a diagnostic figure showing a selection of best-fit SEDs vs data
                    try:
                        n_select = 10
                        nz = z_grid.shape[0]
                        if nz >= n_select:
                            idxs = np.linspace(0, nz-1, n_select, dtype=int)
                        else:
                            idxs = np.arange(nz, dtype=int)
                        # ensure max-likelihood z is included
                        max_idx = int(jnp.argmax(profile_logL))
                        if max_idx not in idxs:
                            idxs[len(idxs)//2] = max_idx

                        # Create run-specific subdirectory for best-fit SEDs
                        # Extract descriptor from datestr (format: descriptor_outputsuffix or descriptor_outputsuffix_tailstring)
                        if tailstring is not None:
                            run_subdir = datestr  # Already includes descriptor_suffix_tailstring
                        else:
                            run_subdir = datestr  # Already includes descriptor_suffix
                        
                        fig_dir_best = os.path.join(prof_like_dir, 'figures', 'bestfit_seds', run_subdir)
                        os.makedirs(fig_dir_best, exist_ok=True)
                        
                        # Multipanel plot disabled - only generate overlay plot
                        # fig_name = f'bestfit_seds_srcid={src_identifier}_{datestr}'
                        # if tailstring is not None:
                        #     fig_name += f'_{tailstring}'
                        # fig_path = fig_dir_best + fig_name + '.png'

                        # nplots = len(idxs)
                        # ncols = min(5, nplots)
                        # nrows = int(np.ceil(nplots / ncols))
                        # import matplotlib.pyplot as plt
                        # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows), sharex=True, sharey=False)
                        # axes = np.array(axes).reshape(-1)

                        # # Calculate dynamic y-limits based on observed data and models
                        # if wave_obs is not None and all_flux_unc is not None and norms is not None:
                        #     obs_flux = all_spec_obs[run_idx]*norms[run_idx]
                        #     obs_err = all_flux_unc[run_idx]*norms[run_idx]
                        #     all_model_flux = [np.asarray(all_bestfit_models[sel]).ravel()*norms[run_idx] for sel in idxs]
                        #     ymax = max(np.nanmax(obs_flux + obs_err), np.nanmax([np.nanmax(m) for m in all_model_flux])) * 1.3
                        #     ymin = min(np.nanmin(obs_flux - obs_err), 0) * 1.3 if np.nanmin(obs_flux - obs_err) < 0 else -ymax * 0.1
                        # else:
                        #     ymin, ymax = None, None

                        # for ax_i, sel in enumerate(idxs):
                        #     ax = axes[ax_i]
                        #     # observed data (apply norm)
                        #     if wave_obs is not None and all_flux_unc is not None and norms is not None:
                        #         # Label observed data only on the first subplot to avoid repeated legend entries
                        #         if ax_i == 0:
                        #             ax.errorbar(wave_obs, all_spec_obs[run_idx]*norms[run_idx], yerr=all_flux_unc[run_idx]*norms[run_idx], fmt='o', color='k', markersize=3, capsize=2, label=f'Observed (z_true={redshift[run_idx]:.3f})')
                        #         else:
                        #             ax.errorbar(wave_obs, all_spec_obs[run_idx]*norms[run_idx], yerr=all_flux_unc[run_idx]*norms[run_idx], fmt='o', color='k', markersize=3, capsize=2)
                        #     # best-fit model (at this z)
                        #     bf = np.asarray(all_bestfit_models[sel]).ravel()
                        #     if wave_obs is not None and bf.shape[0] != wave_obs.shape[0]:
                        #         print(f'  Warning: best-fit model length {bf.shape[0]} != wave_obs length {wave_obs.shape[0]}')
                        #     if norms is not None:
                        #         ax.plot(wave_obs, bf*norms[run_idx], color='C1', linewidth=1.5, label=f'z={z_grid[sel]:.3f}')
                        #     else:
                        #         ax.plot(wave_obs, bf, color='C1', linewidth=1.5, label=f'z={z_grid[sel]:.3f}')
                        #     ax.set_xlabel('$\\lambda_{obs}$ [\\mu m]')
                        #     ax.set_ylabel('Flux (norm.)')
                        #     if ymin is not None and ymax is not None:
                        #         ax.set_ylim(ymin, ymax)
                        #     ax.legend(fontsize=8)
                        #     ax.grid(alpha=0.2)

                        # # hide extra axes
                        # for j in range(nplots, len(axes)):
                        #     axes[j].axis('off')

                        # plt.tight_layout()
                        # plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                        # plt.close()
                        # print(f'  Saved best-fit SEDs figure to {fig_path}')

                        # Create overlay version with all models on one plot and residual panel
                        from visualization.result_plotting_fns import plot_bestfit_sed_overlay
                        
                        # Simplified filename (descriptor and suffix in directory name)
                        fig_name_overlay = f'bestfit_seds_overlay_srcid={src_identifier}.png'
                        fig_path_overlay = os.path.join(fig_dir_best, fig_name_overlay)
                        
                        # Prepare data (apply normalization)
                        obs_flux_normed = all_spec_obs[run_idx] * norms[run_idx]
                        obs_err_normed = all_flux_unc[run_idx] * norms[run_idx]
                        models_normed = all_bestfit_models * norms[run_idx]
                        
                        # Create plot using shared function
                        fig_overlay = plot_bestfit_sed_overlay(
                            wave_obs=wave_obs,
                            obs_flux=obs_flux_normed,
                            obs_err=obs_err_normed,
                            z_grid=z_grid,
                            all_bestfit_models=models_normed,
                            true_redshift=redshift[run_idx],
                            n_select=10,
                            ymax=150,
                            ymin=-50,
                            figsize=(10, 5),
                            colormap='YlOrRd'
                        )
                        
                        fig_overlay.savefig(fig_path_overlay, dpi=150, bbox_inches='tight')
                        plt.close(fig_overlay)
                        print(f'  Saved overlay best-fit SEDs figure to {fig_path_overlay}')

                        # Create latent space corner plot showing MAP theta positions
                        fig_name_corner = f'latent_corner_srcid={src_identifier}.png'
                        fig_path_corner = os.path.join(fig_dir_best, fig_name_corner)
                        nlatent = all_map_thetas.shape[1]
                        param_names = [f'$z_{{{i}}}$' for i in range(nlatent)]
                        ticks = [-3, -2, -1, 0, 1, 2, 3]
                        
                        fig_corner = make_color_corner_plot(
                            ncode=nlatent, 
                            latent_z=all_map_thetas, 
                            feature_vals=z_grid, 
                            feature_name='redshift',
                            yticks=ticks, 
                            vmin=0, 
                            vmax=3, 
                            xlim=[-3, 3], 
                            ylim=[-3, 3], 
                            labels=param_names,
                            figsize=(8, 8)
                        )
                        
                        fig_corner.savefig(fig_path_corner, dpi=150, bbox_inches='tight')
                        plt.close(fig_corner)
                        print(f'  Saved latent space corner plot to {fig_path_corner}')
                    except Exception as _e:
                        print('  Warning: failed to save best-fit SEDs figure:', _e)
                if save_restframe_seds and all_restframe_seds is not None:
                    if pl_subsample_step is not None and pl_subsample_step > 1:
                        subsampled_rf = all_restframe_seds[::pl_subsample_step]
                        save_dict['all_restframe_seds'] = subsampled_rf
                        save_dict['pl_subsample_step'] = pl_subsample_step
                        print(f'  Saved subsampled rest-frame SEDs with shape {subsampled_rf.shape} (step={pl_subsample_step})')
                    else:
                        save_dict['all_restframe_seds'] = all_restframe_seds
                        print(f'  Saved rest-frame SEDs with shape {all_restframe_seds.shape}')
                np.savez(save_fpath, **save_dict)
    
        else:
            print('Loading profile likelihood estimate from ', save_fpath)
    
            plfile = np.load(save_fpath)
            z_grid, profile_logL, all_map_thetas = [plfile[key] for key in ['z_grid', 'profile_logL', 'all_map_thetas']]
            all_bestfit_models = plfile.get('all_bestfit_models', None)  # Load if available
            all_restframe_seds = plfile.get('all_restframe_seds', None)  # Load if available
            

        if plot:
            fig =  compare_profile_likes(z_grid, profile_logL, finez, zpdf_tf[run_idx,3:], chisq_tf[run_idx], redshift[run_idx],\
                                         zout_tf=zout_tf[run_idx], \
                                         Z_MIN=Z_MIN, Z_MAX=Z_MAX, sigma_smooth=1, \
                                        ylim=[40, 95], figsize=(6, 4), bbox_to_anchor=[0.05, 3.25], pdf_lw=1.5, ypad=25)
            
    
def train_full_PAE(params, run_name, train_ae=True, train_flow=True, dat_obj=None, cond_features=None, return_PAE=True, make_result_figs_ae=True, make_result_figs_flow=True, \
                  features_plot=None, sed_set='COSMOS', feature_vmin_vmax=None, plot_params=None, dpi=200, fpath_dict=None, \
                  train_mode='deep', property_cat_fpath=None, resave_params=False):

    ''' Not fully updated to JAX yet '''
    
    
    rundir = config.modl_runs_path + run_name
    
    if not os.path.isdir(rundir):
        os.makedirs(rundir)
        
    if fpath_dict is None:
        fpath_dict = grab_fpaths_traindat(sed_set, restframe=params['restframe'], train_mode=train_mode)
        
    if not params['restframe']:
        central_wavelengths = np.sort(np.load(config.sphx_dat_path+'central_wavelengths_sphx102.npz')['central_wavelengths'])
        
    if dat_obj is None:
        if train_ae:
            dat_obj = spec_data(params['nbands'])

            # training
            _ = dat_obj.build_dataloaders(fpath_dict, train_frac=params['train_frac'], weight_soft=params['weight_soft'], load_property_cat=False,\
                                          property_cat_fpath=property_cat_fpath, restframe=params['restframe'])        
        
        # testing on full dataset afterwards
        dat_obj_results = spec_data(params['nbands'])        
        property_cat_df = dat_obj_results.build_dataloaders(fpath_dict, train_frac=1.0, weight_soft=params['weight_soft'], load_property_cat=False, \
                                                            property_cat_fpath=property_cat_fpath, restframe=params['restframe'], save_property_cat=False)
        
        print('property cat df has keys', property_cat_df.keys())
        
    if return_PAE:
        
        if resave_params:
            print('Resaving parameters..')
            save_params(rundir, params)

        # instantiate and load from saved models
        if params['restframe']:
            central_wavelengths = dat_obj_results.sed_um_wave
            
        PAE_obj = PAE(run_name=run_name, load_flow_decoder=False, central_wavelengths=central_wavelengths, params=params)

    if train_ae:
        ae_modl, ae_dat_obj, metric_dict, accelerator = run_ae_sed_fit(dat_obj, train_mode='deep', property_cat_df=None, \
                                                       run_name=run_name, params=params)
    

    # reloads AE from saved file after training   
    ae_modl = PAE_obj.load_ae_modl(run_name, return_ae=True)
    
    # trouble using accelerator for inference tasks (slows things down for some reason?)
    accelerator = None

    print('rundir is ', rundir)
    
    # dat_obj_results.tdata_train = dat_obj_results.tdata_train.to(PAE_obj.device)

    if make_result_figs_ae:
        # grabs latents from full dataset and optionally saves them
        all_mu, ncode = grab_encoded_vars_dataset2(ae_modl, dat_obj_results, property_cat_df, accelerator=accelerator, save=True, rundir=rundir, \
                                                 sed_set=sed_set)
        
        ae_result_fig_wrapper2(ae_modl, dat_obj_results, params, property_cat_df, rundir, features_plot=None, save=True, return_figs=False, \
                             sed_set=sed_set, device=None, all_mu=all_mu)
            
    if train_flow:
        
        NDE_theta, model_file, train_loss_vs_epoch, valid_loss_vs_epoch, latent_means = fit_flow_to_latents(rundir=rundir, accelerator=None, lr=params['lr_flow'], n_epoch=params['nepoch_flow'], \
                                                                                                           mean_sub_latents=params['mean_sub_latents'])
        if make_result_figs_flow:
            latents = np.load(rundir+'/latents/latents.npz')['all_z']                
            nf_result_fig_wrapper(NDE_theta, latents, rundir=rundir, save=True)
            
    PAE_obj.load_flow(return_flow=False) # load flow in place

    if return_PAE:
        return PAE_obj
