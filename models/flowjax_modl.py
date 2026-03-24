import jax
import jax.numpy as jnp
import jax.random as jr

from flowjax.distributions import Normal
from flowjax.bijections import RationalQuadraticSpline
from flowjax.flows import masked_autoregressive_flow
from flowjax.train import fit_to_data
from flowjax.bijections import Affine, Invert
from flowjax.distributions import Transformed

import paramax
# import equinox as eqx


def init_flowjax_modl(subkey, ncode, invert=False, knots=8, interval=4):
    dist = Normal(jnp.zeros(ncode))
    dist = paramax.non_trainable(dist)
    
    flow = masked_autoregressive_flow(subkey,
        base_dist=dist,
        transformer=RationalQuadraticSpline(knots=knots, interval=interval),
    invert=invert)

    return dist, flow
    

def train_flow_flowjax(latents_train, invert=False, knots=8, interval=4, learning_rate=1e-3, n_epoch_flow=100, 
                       batch_size=128):

    key, subkey = jr.split(jr.key(0))

    ncode = latents_train.shape[1]

    dist, flow = init_flowjax_modl(subkey, ncode, invert=invert, knots=knots, interval=interval)

    # Affine implements: y = (x + loc) * scale
    # For z-score normalization (x - mean) / std, we need:
    loc = -latents_train.mean(axis=0) / latents_train.std(axis=0)
    scale = 1 / latents_train.std(axis=0)

    print('loc, scale:', loc, scale)
    
    preprocess = Affine(loc, scale)

    latents_train_norm = jax.vmap(preprocess.transform)(latents_train)

    print('normalized latents have std:', latents_train_norm.std(axis=0))

    key, subkey = jr.split(key)
    flow, losses = fit_to_data(subkey, flow, latents_train_norm, learning_rate=learning_rate, 
                               max_epochs=n_epoch_flow, batch_size=batch_size)

    flow = Transformed(flow, Invert(preprocess))

    return flow, loc, scale, losses