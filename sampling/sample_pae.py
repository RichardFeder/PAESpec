import blackjax
import jax
import jaxlib
import jax.numpy as jnp
import jax.random as jr
from jax import jit, grad
import torch

import config
import numpy as np

import time
from scipy.stats import uniform, norm, rv_histogram, chi2, multivariate_normal, truncnorm
import pocomc as pmc

from .mclmc import *
from models.flowjax_modl import *
from models.nn_modl_jax import *
from training.train_ae_jax import *
from data_proc.dataloader_jax import *
from data_proc.data_file_utils import *
from utils.utils_jax import *
from models.pae_jax import *
from inference.like_prior import *
from diagnostics.diagnostics_jax import save_redshift_results

import os
os.environ['CONDA_PREFIX'] = '/global/homes/r/rmfeder/.conda/envs/jax-env' # needed for pocoMC sometimes


def grab_map_spec(logP_chain, spec_samples):
    
    nchain = logP_chain.shape[0]
    
    map_spec = torch.zeros((nchain, spec_samples.shape[-1]))
    
    map_logP = torch.zeros((nchain,))
    # print('map_spec.shape:', map_spec.shape)
    
    for x in range(nchain):
        
        whichmax = torch.where((logP_chain[x]==torch.max(logP_chain[x])))[0][0]        
        map_spec[x] = spec_samples[x,whichmax,:]
        map_logP[x] = torch.max(logP_chain[x])
    
    return map_spec, map_logP

def initialize_latents(N, num_samples, key, include_z=True, z_min=0.0, z_max=1.0):
    """
    Initialize a latent variable vector of shape (num_samples, N+1),
    where the first N dimensions follow a standard multivariate Gaussian
    and the last dimension (redshift) follows a uniform distribution.

    Parameters:
    - N (int): Latent dimension of the normalizing flow
    - num_samples (int): Number of samples
    - key (jax.random.PRNGKey): Random number generator key
    - z_min (float): Minimum redshift value
    - z_max (float): Maximum redshift value

    Returns:
    - initial_position (jnp.ndarray): Shape (num_samples, N+1)
    """
    key_gauss, key_uniform = jax.random.split(key)

    # Sample the first N latent variables from a standard normal distribution
    latents = jax.random.normal(key_gauss, shape=(num_samples, N))

    if include_z:
        # Sample the last column (redshift) from a uniform distribution
        # Could compare with empirical prior from N(z)
        redshifts = jax.random.uniform(key_uniform, shape=(num_samples, 1), minval=z_min, maxval=z_max)
    
        # Concatenate along the last axis to get shape (num_samples, N+1)
        initial_position = jnp.concatenate([latents, redshifts], axis=-1)
        
        return initial_position
    else:
        return latents


def pae_spec_sample_pocomc(
    PAE_obj,
    x_obs,
    weight,
    rkey,
    nf_alpha=1.0,
    verbose=False,\
    z_min=0.0, z_max=3.0,\
    n_effective=1024, n_active = 512, n_total=1024, precondition=True, return_sampler=False, z_prior=None, \
    density_thresh_prior=False, frac_cover=0.99, transform='probit', dynamic=False, step_fac=2., \
    **kwargs):
    """
    Run pocoMC to sample the posterior of latent + redshift parameters.

    Args:
        PAE_obj: object with push_spec_marg method for decoding
        x_obs: observed data
        weight: weight matrix for likelihood
        rkey: JAX PRNG key
        nf_alpha: multiplier for flow prior
        verbose: whether to print status messages
        kwargs: passed to sample_param_dict_gen()

    Returns:
        samples: shape (nwalkers, nsamples, ndim)
        log_postmean: posterior components at mean sample
    """
    spd = sample_param_dict_gen(**kwargs)

    if nf_alpha==0.:
        dists = [uniform(loc=-3, scale=6.0) for x in range(PAE_obj.params['nlatent'])]    
        reflective = np.arange(PAE_obj.params['nlatent']+1)
    else:
        dists = [norm(loc=0., scale=1.0) for x in range(PAE_obj.params['nlatent'])]

        reflective = [PAE_obj.params['nlatent']]
    
    if z_prior is None:
        dists.append(uniform(loc=spd['zmin'], scale=3.0))
    else:
        print('Using N(z) redshift prior')
        dists.append(z_prior)
    
    # print('dists has length', len(dists))
    
    pc_prior = pmc.Prior(dists)

    if density_thresh_prior:
        flow_prior = multivariate_normal(mean=np.zeros(PAE_obj.params['nlatent']), cov=np.eye(PAE_obj.params['nlatent']))

        r2_threshold = chi2.ppf(frac_cover, df=PAE_obj.params['nlatent'])

        logp_threshold = -0.5 * r2_threshold - 0.5 * PAE_obj.params['nlatent'] * np.log(2 * np.pi)

        print('logp threshold is ', logp_threshold)

    else:
        flow_prior, logp_threshold = None, None
        
    def loglikelihood_fn(x):
        return logdensity_fn_marg_pmc(x, PAE_obj, x_obs, weight, z_min, z_max, nf_alpha=nf_alpha, flow_prior=flow_prior, logp_threshold=logp_threshold)

    if step_fac is not None:
        n_steps = len(dists)*step_fac

        print('n_steps is ', n_steps)
    else:
        n_steps = len(dists)//2

    
    # Step 3: Instantiate and run sampler
    sampler = pmc.Sampler(
        prior=pc_prior,
        likelihood=loglikelihood_fn,
        n_effective=n_effective, \
        n_active=n_active, vectorize=True, \
        reflective=reflective, \
        precondition=precondition, \
        transform=transform, \
        dynamic=dynamic,       # <-- prevent adaptation
        metric='ess',        # use effective sample size for beta steps
        n_steps=n_steps, 
    )

    if verbose:
        print(f"Running pocoMC for {spd['num_steps']} steps...")

    results = sampler.run(n_total=n_total, n_evidence=0, progress=False)

    if return_sampler:
        return sampler
    else:
        # samples, weights, logl, logp = sampler.posterior()
        samples, logl, logp = sampler.posterior(resample=True)
        return samples, logl, logp

        # return samples, weights, logl, logp


def sample_pocomc_wrapper(PAE_obj, all_spec_obs, weights, redshift, src_idxs, ngal, fix_z=False, nf_alpha=1.0, zmax=3.0, keyidx=102, \
                        save_results=False, save_fpath=None, sample_fpath=None, n_total=2048, n_active=256, n_effective=512,  \
                        return_results=True, redshift_in_flow=False, z_prior=None, precondition=False, density_thresh_prior=False, frac_cover=0.99, \
                         dynamic=True, max_retries=3, step_fac=None):


    all_samples, failed_idx = [], []

    rng_key=jr.key(keyidx)

    runidxs = np.arange(ngal)
    redshifts_true = redshift[src_idxs[runidxs]]

    for idx in runidxs:

        if idx%20==0:
            print('idx ', idx, 'of', ngal)

        attempt = 0
        success = False
        while attempt < max_retries and not success:
            try:
                # Change RNG key per attempt to avoid deterministic failures
                rng_key_i = jr.fold_in(rng_key, idx + 100 * attempt)
                samples, logl, logp = pae_spec_sample_pocomc(
                    PAE_obj, all_spec_obs[src_idxs[idx]], weights[src_idxs[idx]], rng_key_i, nf_alpha=nf_alpha,
                    verbose=False, return_sampler=False, n_active=n_active, n_effective=n_effective, n_total=n_total,
                    precondition=precondition, z_prior=z_prior, density_thresh_prior=density_thresh_prior,
                    frac_cover=frac_cover, transform='logit', dynamic=dynamic, step_fac=step_fac, redshift_in_flow=redshift_in_flow)
                success = True
            except Exception as e:
                attempt += 1
                print(f'Failed on idx={idx}, attempt={attempt}, error: {e}')

        if success:
            all_samples.append(samples)
        else:
            print(f'Giving up on idx={idx} after {max_retries} attempts.')
            failed_idx.append(idx)
            dummy_shape = all_samples[-1].shape if all_samples else (n_total, PAE_COSMOS.params['nlatent'] + 1)
            samples = -99. * np.ones(dummy_shape)
            all_samples.append(samples)
            
    # standardize samples to same length for simplicity
    min_samp = np.min([all_samples[x].shape[0] for x in np.arange(ngal)])
    all_samp_min = np.zeros((ngal, min_samp, PAE_obj.params['nlatent']+1))
    print('Truncated sample array has shape', all_samp_min.shape)

    for x in range(ngal):
        all_samp_min[x] = all_samples[x][:min_samp,:]

    if redshift_in_flow:
        final_redshifts = jax.vmap(lambda nf_latents: PAE_obj.combined_transform_jit(nf_latents)[:,-1])(all_samp_min)
    else:
        final_redshifts = None
    
    if save_results:
        save_redshift_results(save_fpath, all_samp_min, None, None, None, redshifts_true, \
                             sample_fpath=sample_fpath, ae_redshifts=final_redshifts)

    if return_results:
        return all_samp_min, redshifts_true, failed_idx
        

# def pae_spec_sample(PAE_obj, x_obs, weight, spd=None, pretuned_L=None, pretuned_step_size=None, rkey=None, verbose=False, nsamp_init=None,\
#                      reinit_scatter=1e-2, **kwargs):

#     # num_steps=1000, step_size=0.1, z_min=0.0, z_max=4.0, mode='HMC'
#     '''
#     PAE_obj: Probabilistic autoencoder class object
#     x_obs: ~Normalized~ fluxes (divided by mean, revisit)
#     weight: Inverse-variance of ~Normalized~ flux uncertainty
    
#     spd: dictionary, contains sampling configuration parameters
    
#     '''

#     if spd is None:
#         spd = sample_param_dict_gen(**kwargs)

#     if verbose:
#         print("Sampling configuration parameters:")
#         print(spd)

#     if rkey is None:
#         rkey = jr.key(100)

#     def compiled_log_p(x, PAE_obj, x_obs, weight, zmin, zmax):
#         return logdensity_fn(x.reshape(1, -1), PAE_obj, x_obs, weight, zmin, zmax)
    
#     # Specify the log probability density function for the PAE
#     if spd['float_redshift']:
#         log_p_partial = functools.partial(compiled_log_p, PAE_obj=PAE_obj, x_obs=x_obs, weight=weight, zmin=spd['zmin'], zmax=spd['zmax'])
#         log_p = jax.jit(log_p_partial, static_argnames=["x_obs", "weight", "zmin", "zmax"])

#         if spd['find_MAP_first']:
#             nlogP = lambda x: -logdensity_fn(x.reshape(1, -1), PAE_obj, x_obs, weight, spd['zmin'], spd['zmax'])
#     else:
#         if spd['redshift_fix'] is None:
#             print('Need to provide fixed redshift if float_redshift = False, exiting..')
#             return None
#         log_p = lambda x: logdensity_fn_fixz(x,PAE_obj,x_obs,weight, spd['redshift_fix'])

#     initial_position = initialize_latents(PAE_obj.params['nlatent'], spd['nchain'], rkey, z_min=spd['zmin'], z_max=spd['zmax'], \
#                                          include_z=spd['float_redshift'])
#     initial_position = jnp.array(initial_position)

#     if spd['find_MAP_first']:
#         # try ones outside of minimize
#         nlogpval = nlogP(initial_position)
#         opt_result = jso.minimize(nlogP, initial_position, method="BFGS")
#         thetaopt = opt_result.x
        
#         print('nlogp:', nlogpval)
#         print('initial position:', initial_position.shape)
#         print('theta opt:', thetaopt)
#         print('success? ', opt_result.success)
#         print('function value:', opt_result.fun)
#         print('Nfunc calls, Ngradient evals, Niteration:', opt_result.nfev, opt_result.njev, opt_result.nit)

#     # return
#     if verbose:
#         print('initial position has shape:', initial_position.shape)
#         print('initial position is ', initial_position)

#     transform = lambda state, info: state.position

#     if spd['test_mode']:
#         print('xobs, weight shape:', x_obs.shape, weight.shape)
#         # check that gradients are sane
#         grad_logdensity = grad(log_p)
#         print("Gradient at initial position:", grad_logdensity(initial_position))

#     # execute sampling
#     sample_key, rng_key = jax.random.split(rkey)
#     print('Sampling '+str(spd['nchain'])+' chains for '+str(num_steps)+' steps..')

#     if spd['nchain'] > 1:
#         progress_bar = False
#         keys = jax.random.split(sample_key, spd['nchain'])
#     else:
#         progress_bar = True
    
#     if spd['mode']=='MCLMC': # default
#         if spd['nchain'] > 1:
#             if nsamp_init is not None:
#                 all_samples, final_states = run_mclmc_with_reinit_vmap(log_p, num_steps, initial_position, rkey, transform, 
#                                                desired_energy_variance=spd['desired_energy_variance'], pretuned_L=pretuned_L, 
#                                                pretuned_step_size=pretuned_step_size, ninit=nsamp_init, reinit_scatter=reinit_scatter)
                
#             else:
#                 all_samples, all_inits, chain_keys = jax.vmap(lambda k, pos: \
#                                                               run_mclmc(log_p, spd['num_steps'], pos, k, transform,\
#                                                                         pretuned_L=pretuned_L, pretuned_step_size=pretuned_step_size,\
#                                                                         desired_energy_variance=spd['desired_energy_variance'], progress_bar=progress_bar, \
#                                                                         nsamp_init=nsamp_init))(keys, initial_position)
        
#             print('all samples has shape:', all_samples.shape)
#             return all_samples, params
#         else:
#             samples, blackjax_state_after_tuning,\
#                     run_key, pretuned_L, pretuned_step_size = run_mclmc(log_p, spd['num_steps'], initial_position, sample_key, transform, \
#                                                           pretuned_L=pretuned_L, pretuned_step_size=pretuned_step_size, \
#                                                           desired_energy_variance=spd['desired_energy_variance'], progress_bar=progress_bar)

#             print('samples has shape', samples.shape)
#             return samples, params

#     elif spd['mode']=='HMC':
#         samples, state = run_hmc(log_p, initial_position, PAE_obj.key, num_steps=spd['num_steps'], step_size=spd['u_step_size'])
#         return samples, state

#     elif spd['mode']=='SMC':
#         samples = run_smc(log_p, initial_position, num_particles=spd['num_particles'], num_steps=spd['num_steps_smc'])
#         return samples
        

# def run_mclmc_with_reinit_vmap(logdensity_fn, num_steps, initial_positions, key, transform, 
#                                desired_energy_variance=5e-4, pretuned_L=None, 
#                                pretuned_step_size=None, ninit=100, reinit_scatter=1e-1, progress_bar=False):
    
#     num_chains = initial_positions.shape[0]
#     keys = jax.random.split(key, num_chains)  # Generate separate keys for each chain

#     logdensity_fn = jax.jit(logdensity_fn)

#     print('initial pos has shape:', initial_positions.shape)
#     print('keys has shape', keys.shape)
    
#     # Step 1: Run initial phase for `ninit` steps using vmap
#     def run_initial_chain(position, key):
#         return run_mclmc2(logdensity_fn, ninit, position, key, transform, 
#                          desired_energy_variance, 0.5, 0.02, progress_bar=False)

#     initial_samples, initial_states, new_keys,\
#         pretuned_L, pretuned_step_size = jax.vmap(run_initial_chain)(initial_positions, keys)

#     # Step 2: Compute log-densities outside vmap to evaluate initialization quality
#     log_densities = jax.vmap(logdensity_fn)(initial_samples[:, -1])  # Use last sample per chain

#     print('log densities:', log_densities)
#     # Define a threshold: keep the top 50% of chains based on log-density
#     threshold = jnp.max(log_densities)-2  
#     good_chains_mask = (log_densities >= threshold)  

#     print('good chains mask:', good_chains_mask)

#     # Step 3: Reinitialize poor chains using good ones
#     good_chains = initial_samples[:, -1][good_chains_mask]  # Select last samples of good chains
#     if good_chains.shape[0] == 0:  # Edge case: if no good chains, use all chains
#         good_chains = initial_samples[:, -1]

#     if np.sum(good_chains_mask.astype(int)) > 1:
#         print('more than one converged chain..')
#         reinit_scatter=1e-2
    
#     print('replacing bad chains')
#     # Replace bad chains by resampling from good ones with small noise
#     new_initial_positions = jax.random.choice(key, good_chains, shape=(num_chains,))
#     noise = jax.random.normal(key, shape=new_initial_positions.shape) * reinit_scatter
#     reinitialized_positions = new_initial_positions + noise  

#     print('reinitialized positions:', reinitialized_positions)

#     print('new keys has length:', new_keys.shape)
#     print('reinit shape:', reinitialized_positions.shape)
#     print('pretuned_L, pretuned_step_size:', pretuned_L.shape, pretuned_step_size.shape)

#     # pretuned_L_mean = np.median(pretuned_L)
#     # pretuned_step_size_mean = np.median(pretuned_step_size)

    
#     # Step 4: Run remaining steps using vmap with improved initializations
#     def run_final_chain(position, key):
#         return run_mclmc2(logdensity_fn, num_steps - ninit, position, key, transform, 
#                          desired_energy_variance, None, None, progress_bar)

#     final_samples, final_states, _, _, _ = jax.vmap(run_final_chain)(reinitialized_positions, new_keys)

#     return final_samples, final_states


def postproc_samples(PAE_obj, samples, data_spec, weights, redshift_use=None, burn_in=500, thin_fac=1, \
                    plot_post_spec=True, plot_corner=True, plot_trace=True, plot_redshift_post=True, \
                    true_redshift=None):

    samples_proc = samples[burn_in:]
    if thin_fac > 1:
        samples_proc = samples_proc[::int(thin_fac)]

    print('samples_proc has shape:', samples_proc.shape)
    nsamp = samples_proc.shape[0]

    figs = []

    if redshift_use is None:
        log_pz_chain, log_pz_redshift_chain, log_px_given_z_chain = jax.vmap(lambda x: logdensity_fn(x, PAE_obj, data_spec, weights, 0.0, 1.0, True))(samples_proc)
        if samples_proc.shape[1]==1:
            usamp, redshift_samp = samples_proc[:,:,:-1], samples_proc[:,:,-1]
        else:
            usamp, redshift_samp = samples_proc[:,:-1], samples_proc[:,-1]
        pushspec = jax.vmap(lambda x, y: PAE_COSMOS.push_spec(x.reshape(1, -1), y))(usamp, redshift_samp)

    else:
        usamp = samples_proc
        log_pz_redshift_chain = None
        log_pz_chain, log_px_given_z_chain = jax.vmap(lambda x: logdensity_fn_fixz(x, PAE_COSMOS, x_spec_use, weight_use, redshift_use, True))(samples_proc)
        pushspec = jax.vmap(lambda x, y: PAE_COSMOS.push_spec(x.reshape(1, -1), y))(usamp, redshift_use*jnp.ones(samples_proc.shape[0]))
    
    if plot_trace:
        fig_utrace = trace_plot(usamp)
        figs.append(fig_utrace)

        if redshift_use is None:
            fig_redshift_trace = plot_redshift_trace(redshift_samp)
            figs.append(fig_redshift_trace)

        fig_logp_trace = plot_logp_trace_indiv(log_pz_chain, log_px_given_z_chain, log_pz_redshift_chain=log_pz_redshift_chain, figsize=(8, 4))
        figs.append(fig_logp_trace)
    
    if plot_post_spec:

        fig_postspec = plt.figure(figsize=(8, 4))
        for x in range(nsamp):
            plt.plot(PAE_obj.wave_obs, pushspec[x], color='k', alpha=0.005)
        plt.errorbar(PAE_obj.wave_obs, data_spec, yerr=1./np.sqrt(weights), color='r', fmt='o', alpha=0.3)
        plt.ylim(-0.5, 3)
        plt.ylabel('Flux density [norm.]', fontsize=14)
        plt.xlabel('$\\lambda_{obs}$ [$\\mu$m]', fontsize=14) 
        plt.show()

        figs.append(fig_postspec)

    if plot_corner:
        corner_labels=['$u_'+str(n)+'$' for n in range(PAE_obj.params['nlatent'])]

        if samples.shape[1] > PAE_obj.params['nlatent']:
            # add redshift to labels

            corner_labels.append('Redshift')

        fig_corner = corner.corner(np.array(samples[::thin_fac]), labels=corner_labels, show_titles=True, \
                              quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 16})

    if plot_redshift_post:
        fig_zpost = plot_redshift_posterior(redshift_samp, redshift_use=true_redshift)

    if redshift_use is None:
        return figs, usamp, redshift_samp, pushspec
    else:
        return figs, usamp, pushspec
        
        
def compare_chains_logp_indiv(PAE_obj, data_spec, weights, all_samples, burn_in, thin_fac=1, true_redshift=None, plot_trace=True, src_id=None, figsize=(11, 4)):

    nchain = all_samples.shape[0]

    all_samples_proc = all_samples[:,burn_in:]
    if thin_fac > 1:
        all_samples_proc = all_samples_proc[:,::thin_fac]

    print('all_samples_proc has shape', all_samples_proc.shape)

    nsamp = all_samples_proc.shape[1]
    print('Processing ', nchain, 'chains')

    all_log_pz, all_log_px_given_z_chain, all_logp = [np.zeros((nchain, nsamp)) for x in range(3)]
    all_redshift_samples = all_samples_proc[:,:,-1]

    mean_spec, lopct_spec, hipct_spec = [np.zeros((nchain, 102)) for x in range(3)]
    mean_redshifts, upper_unc, lower_unc, chi2_meanspec = [np.zeros(nchain) for x in range(4)]

    for n in range(nchain):

        log_pz_chain, log_pz_redshift_chain, log_px_given_z_chain = jax.vmap(lambda x: logdensity_fn(x, PAE_obj, x_spec_use, weight_use, 0.0, 1.0, True))(all_samples_proc[n])

        usamp, redshift_samp = all_samples_proc[n,:,:-1], all_samples_proc[n,:,-1]
        pushspec = jax.vmap(lambda x, y: PAE_COSMOS.push_spec(x.reshape(1, -1), y))(usamp, redshift_samp)
        
        # log_pz_chain, log_px_given_z_chain = jax.vmap(lambda x: logdensity_fn_fixz(x, PAE_obj, x_spec_use, weight_use, redshift_use, True))(all_samples_proc[n])
        all_log_pz[n] = log_pz_chain
        all_log_px_given_z_chain[n] = log_px_given_z_chain
        all_logp[n] = log_pz_chain+log_px_given_z_chain

        mean_spec[n] = np.mean(pushspec, axis=0)
        lopct_spec[n] = np.percentile(pushspec, 16, axis=0)
        hipct_spec[n] = np.percentile(pushspec, 84, axis=0)

        mean_redshifts[n] = np.mean(redshift_samp)
        upper_unc[n] = np.percentile(redshift_samp, 84)-np.mean(redshift_samp)
        lower_unc[n] = np.mean(redshift_samp) - np.percentile(redshift_samp, 16)
        chi2_meanspec[n] = np.sum(weight_use*(x_spec_use-mean_spec[n])**2)

    print('all_logp has shape', all_logp.shape)
    
    if plot_trace:
        fig_logp_trace = plot_logp_trace_multi(all_log_pz, all_log_px_given_z_chain, all_logp, figsize=figsize, burn_in=burn_in) 

        fig_redshift_trace = plot_redshift_trace_multi(all_redshift_samples, true_redshift=true_redshift, figsize=(7, 4), burn_in=burn_in)

        # fig_redshift_trace = plot_redshift_trace_multi(all_z_samples)

    fig = plt.figure(figsize=(6, 4))

    zrav = all_redshift_samples.ravel()

    med_z, lopct, hipct, lopct2, hipct2 = [np.percentile(zrav, x) for x in [50, 16, 84, 5, 95]]

    lounc = med_z-lopct
    hiunc = hipct-med_z
    
    bins = np.linspace(med_z-5*lounc, med_z+5*hiunc, 50)

    label = '$\\hat{z}='+str(np.round(med_z, 3))+'^{+'+str(np.round(hipct-med_z, 3))+'}_{-'+str(np.round(med_z-lopct, 3))+'}$'
    
    for n in range(nchain):
        plt.hist(all_redshift_samples[n], bins=bins, histtype='step', alpha=0.3, density=True)
    plt.hist(zrav, bins=bins, histtype='step', color='k', density=True, label=label)
    plt.axvline(true_redshift, color='r', label='Truth')
    plt.legend()
    plt.xlabel('$z$', fontsize=12)
    plt.ylabel('$p(z)$', fontsize=12)
    plt.show()
    
    
    plot_spec_posts_multi(PAE_obj.wave_obs, data_spec, weights, mean_spec, lopct_spec, hipct_spec, \
                         true_redshift, mean_redshifts=mean_redshifts, upper_unc=upper_unc, lower_unc=lower_unc, \
                         chi2_meanspec=chi2_meanspec)

    return med_z, lopct, hipct, lopct2, hipct2
        