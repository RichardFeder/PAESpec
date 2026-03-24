from typing import Callable, Sequence, Union, Optional
import jax.numpy as jnp
from jax import vmap
from flax import linen as nn
from utils.utils_jax import *


class SpectrumEncoder_JAX(nn.Module):
    """
    Spectrum Encoder in JAX with Flax.
    
    Parameters
    ----------
    n_latent : int
        Dimension of the latent space.
    n_hidden_encoder : Sequence[int]
        Dimensions for each hidden layer of the MLP.
    filters : Sequence[int]
        Number of filters for each convolutional layer.
    filter_sizes : Sequence[int]
        Kernel sizes for each convolutional layer.
    act : Sequence[Callable]
        Activation functions for the MLP layers.
    dropout : float
        Dropout probability.
    """

    n_latent: int
    n_hidden_encoder: Sequence[int] = (128, 64, 32)
    filters: Sequence[int] = (32, 64, 128)
    filter_sizes: Sequence[int] = (5, 5, 5, 5)
    act: Sequence[Callable] = None
    dropout: float = 0.0


    def setup(self):
        if self.act is None:
            object.__setattr__(self, 'act', [nn.relu] * len(self.n_hidden_encoder) + [nn.tanh])  # Last activation centered at 0
    @nn.compact
    def __call__(self, y, train: bool = True):

        act = self.act
        # Build convolutional blocks
        x = y[:, None, :]  # Add channel dimension for Conv1D
        for i, (f, s) in enumerate(zip(self.filters, self.filter_sizes)):
            x = nn.Conv(features=f, kernel_size=(s,), padding="SAME")(x)
            x = nn.relu(x)  # Replace PReLU with ReLU for simplicity
            if self.dropout > 0:
                x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
            x = nn.max_pool(x, window_shape=(s,), strides=(1,), padding="SAME")

        # Flatten the output
        x = x.reshape(x.shape[0], -1)  # Flatten all dimensions except batch

        # Small MLP to compress features into the latent space
        for i, h in enumerate(self.n_hidden_encoder):
            x = nn.Dense(h)(x)
            x = act[i](x)
            if self.dropout > 0 and i < len(self.n_hidden_encoder) - 1:
                x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)

        # Output layer
        x = nn.Dense(self.n_latent)(x)
        return x

        # Output two values: `mu` (mean) and `log_sigma` (log variance)
        # mu = nn.Dense(self.n_latent)(x)  # Mean of the latent distribution
        # log_sigma = nn.Dense(self.n_latent)(x)  # Log variance of the latent distribution
        
        # return mu, log_sigma

class SpectrumDecoder_JAX(nn.Module):
    """
    Spectrum Decoder in JAX with Flax.

    Parameters
    ----------
    wave_rest : jnp.ndarray
        Restframe wavelengths.
    n_latent : int
        Dimension of latent space.
    filters : Sequence[int]
        Number of filters for each convolutional layer.
    filter_sizes : Sequence[int]
        Kernel sizes for each convolutional layer.
    dropout : float
        Dropout probability.
    wave_obs : jnp.ndarray, optional
        Observed wavelengths for transformation.
    """

    wave_rest: jnp.ndarray
    n_latent: int = 5
    filters: Sequence[int] = (128, 64, 32)
    filter_sizes: Sequence[int] = (5, 5, 5)
    dropout: float = 0.0
    wave_obs: jnp.ndarray = None
    lam_interp: jnp.ndarray = None

    def setup(self):

        self.convT_layers = [
            nn.ConvTranspose(
                features=f,
                kernel_size=(s,),
                padding="SAME",
            )
            for f, s in zip(self.filters, self.filter_sizes)
        ]
        self.convT_final = nn.ConvTranspose(
            features=1,
            kernel_size=(1,),
            strides=(1,),
            padding="SAME",
        )

        self.lin = nn.Dense(self.wave_rest.shape[0])
        self.shape = (-1, self.wave_rest.shape[0], 1)
        

    def decode(self, s, train: bool = True):
        """
        Decode latents into a restframe spectrum.

        Parameters
        ----------
        s : jnp.ndarray
            Batch of latents.

        Returns
        -------
        x : jnp.ndarray
            Batch of restframe spectra.
        """

        x = self.lin(s) # map latent z to len(wave_rest) features
        x = nn.leaky_relu(x)
        x = x.reshape(self.shape) # then reshape to have (batch_size, nchannel=1, len(wave_rest))
        
        for layer in self.convT_layers: # loop through transpose convolution layers
            x = layer(x)
            x = nn.relu(x)
            if self.dropout > 0:
                x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
                
        x = self.convT_final(x)
        return x.squeeze(-1)  # Remove channel dimension
        
    def __call__(self, s, redshift: jnp.ndarray = None, train: bool = True):
        """
        Forward method for SpectrumDecoder.

        Parameters
        ----------
        s : jnp.ndarray
            Batch of latents.
        redshift : jnp.ndarray, optional
            Redshifts for transformation.

        Returns
        -------
        jnp.ndarray
            Transformed spectra.
        """
        spectrum = self.decode(s, train=train)

        return spectrum


# class SpectrumDecoder_JAX(nn.Module):
#     """
#     Spectrum Decoder in JAX with Flax.

#     Parameters
#     ----------
#     wave_rest : jnp.ndarray
#         Restframe wavelengths.
#     n_latent : int
#         Dimension of latent space.
#     filters : Sequence[int]
#         Number of filters for each convolutional layer.
#     filter_sizes : Sequence[int]
#         Kernel sizes for each convolutional layer.
#     dropout : float
#         Dropout probability.
#     wave_obs : jnp.ndarray, optional
#         Observed wavelengths for transformation.
#     """

#     wave_rest: jnp.ndarray
#     n_latent: int = 5
#     filters: Sequence[int] = (128, 64, 32)
#     filter_sizes: Sequence[int] = (5, 5, 5)
#     dropout: float = 0.0
#     wave_obs: jnp.ndarray = None
#     lam_interp: jnp.ndarray = None

#     def setup(self):

#         self.convT_layers = [
#             nn.ConvTranspose(
#                 features=f,
#                 kernel_size=(s,),
#                 padding="SAME",
#             )
#             for f, s in zip(self.filters, self.filter_sizes)
#         ]
#         self.convT_final = nn.ConvTranspose(
#             features=1,
#             kernel_size=(1,),
#             strides=(1,),
#             padding="SAME",
#         )

#         self.lin = nn.Dense(self.wave_rest.shape[0])
#         self.shape = (-1, self.wave_rest.shape[0], 1)
        

#     def decode(self, s, train: bool = True):
#         """
#         Decode latents into a restframe spectrum.

#         Parameters
#         ----------
#         s : jnp.ndarray
#             Batch of latents.

#         Returns
#         -------
#         x : jnp.ndarray
#             Batch of restframe spectra.
#         """

#         x = self.lin(s) # map latent z to len(wave_rest) features
#         x = nn.leaky_relu(x)
#         x = x.reshape(self.shape) # then reshape to have (batch_size, nchannel=1, len(wave_rest))
        
#         for layer in self.convT_layers: # loop through transpose convolution layers
#             x = layer(x)
#             x = nn.relu(x)
#             if self.dropout > 0:
#                 x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)

#         # print('x has shape', x.shape)
                
#         x = self.convT_final(x)
#         return x.squeeze(-1)  # Remove channel dimension

#     # def transform(self, x, redshift=0, verbose=False):

#     #     nspec = x.shape[0]
#     #     wave_redshifted = self.wave_rest * (1 + redshift[:, None])
    
#     #     if verbose:
#     #         print(f'Wave redshifted shape: {wave_redshifted.shape}, Observed wavelength shape: {self.wave_obs.shape}')

#     #     if self.filters is not None:
#     #         if verbose:
#     #             print('Interpolating onto filter grid')

#     #         x_interp = vmap(lambda i: linear_interp_jax(self.lam_interp, wave_redshifted[i], x[i]))(jnp.arange(nspec)) 

#     #         print('x interp has shape', x_interp)

#     #         if verbose:
#     #             print('Performing matrix multiplication with filter weights')
#     #         spectrum = jnp.dot(self.torch_filters, x_interp.T).T
            
#     #     else:
#     #         if verbose:
#     #             print('Directly interpolating onto observed wavelengths')
#     #         spectrum = jnp.interp(self.wave_obs, wave_redshifted, x)
#     #     return spectrum
        
#     def __call__(self, s, redshift: jnp.ndarray = None, train: bool = True):
#         """
#         Forward method for SpectrumDecoder.

#         Parameters
#         ----------
#         s : jnp.ndarray
#             Batch of latents.
#         redshift : jnp.ndarray, optional
#             Redshifts for transformation.

#         Returns
#         -------
#         jnp.ndarray
#             Transformed spectra.
#         """
#         spectrum = self.decode(s, train=train)
#         if redshift is not None:
#             spectrum = self.transform(spectrum, redshift)
#         return spectrum

        
class SpectrumAutoencoder_JAX(nn.Module):
    """Concrete implementation of spectrum encoder

    Constructs and uses :class:`SpectrumEncoder` as encoder and :class:`SpectrumDecoder`
    as decoder.

    Parameter
    ---------
    instrument: :class:`spender.Instrument`
        Observing instrument
    wave_rest: `torch.tensor`
        Restframe wavelengths
    n_latent: int
        Dimension of latent space
    n_hidden: list of int
        Dimensions for every hidden layer of the decoder :class:`MLP`
    act: list of callables
        Activation functions for the decoder. Needs to have len(n_hidden) + 1
        If `None`, will be set to `LeakyReLU` for every layer.
    """

    wave_rest: jnp.ndarray
    n_latent: int = 5
    n_hidden_encoder: Sequence[int] = (64, 256, 1024)
    filters: Sequence[int] = (32, 64, 128)
    filter_sizes: Sequence[int] = (5, 5, 5)
    act: Sequence[Callable] = None
    dropout: float = 0.0
    conv: bool = False
    wave_obs: jnp.ndarray = None
    

    def setup(self):

        self.encoder = SpectrumEncoder_JAX(self.n_latent, n_hidden_encoder=self.n_hidden_encoder, filter_sizes=self.filter_sizes, filters=self.filters)
        self.decoder = SpectrumDecoder_JAX(self.wave_rest, self.n_latent, filter_sizes=reversed(self.filter_sizes), filters=reversed(self.filters), wave_obs=self.wave_obs)

    def __call__(self, x):
        z = self.encoder(x)
        # mu, log_sigma = self.encoder(x)
        # z = reparameterize(mu, log_sigma)
        
        x_recon = self.decoder(z)
        
        return x_recon, z
        # return x_recon, mu, log_sigma, z

    def encode_only(self, x):
        return self.encoder(x)

    def decode_only(self, z, redshift=None, train=False):
        """
        Decode latent variables into spectra using the trained decoder.
        """
        return self.decoder(z, redshift=redshift, train=train)
        

def instantiate_ae_modl_gen_jax(params, central_wavelengths):
    ''' 
    This is where the desired autoencoder model is specified. 
    Other architectures can either be incorporated into SpectrumAutoencoder or into a separate class.
    
    '''

    wav = jnp.array(central_wavelengths)
    # wav = torch.tensor(central_wavelengths, dtype=torch.float32)
    
    if params['modl_type']=='jax':
        
        print('params[conv_decoder] is ', params['conv_decoder'])
        ae_modl = SpectrumAutoencoder_JAX(wav, n_latent=params['nlatent'], \
                                 filter_sizes=params['filter_sizes'], n_hidden_encoder=params['n_hidden_encoder'], \
                                     filters=params['filters'], conv=params['conv_decoder'])
        
    elif params['modl_type']=='mlp': # deprecated
        ae_modl = vae_modl(params['sizes'], ncode=params['nlatent'], beta=params['beta'], alpha=params['alpha'], lambd=params['lambd'])

    elif params['modl_type'] == 'specformer':
        ae_modl = SpecFormer()

    return ae_modl


class MLP_JAX(nn.Module):
    """
    Multi-Layer Perceptron in JAX with Flax.

    Parameters
    ----------
    n_in : int
        Input dimension
    n_out : int
        Output dimension
    n_hidden : Sequence[int]
        Dimensions for every hidden layer
    act : Sequence[Callable]
        Activation functions after every layer. Needs to have len(n_hidden) + 1.
        If `None`, will be set to `nn.leaky_relu` for every layer.
    dropout : float
        Dropout probability
    """
    n_in: int
    n_out: int
    n_hidden: Sequence[int] = (16, 16, 16)
    act: Sequence[Callable] = None
    dropout: float = 0.0
    train: bool = True

    def setup(self):

        if self.act is None:
            object.__setattr__(self, 'act', [nn.leaky_relu] * len(self.n_hidden) + [lambda x: x])  # Identity for last layer

    
    @nn.compact
    def __call__(self, x):
        # Set default activation functions if not provided
        # act = [nn.leaky_relu] * (len(self.n_hidden) + 1)
        # Ensure activation functions match layer count

        act = self.act
        assert len(act) == len(self.n_hidden) + 1

        # Define layer dimensions
        n_ = [self.n_in, *self.n_hidden, self.n_out]
        
        # Build the network
        for i in range(len(n_) - 1):
            x = nn.Dense(n_[i + 1])(x)  # Linear layer
            x = act[i](x)          # Activation function
            if self.dropout > 0 and i < len(n_) - 2:  # Dropout only on hidden layers
                x = nn.Dropout(rate=self.dropout, deterministic=not self.train)(x)
        return x

