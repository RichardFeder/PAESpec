import jax
import blackjax
import time

def run_mclmc_simp(logdensity_fn, num_steps, initial_position, key, transform, L=0.5, step_size=0.05, desired_energy_variance= 5e-4, progress_bar=False):
    init_key, tune_key, run_key = jax.random.split(key, 3)

    print('initial position shape:', initial_position.shape)

    t0 = time.time()
    # create an initial state for the sampler
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key)
    print('time for mclmc init:', time.time()-t0)

    t1 = time.time()
    # build the kernel
    kernel = lambda sqrt_diag_cov : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        sqrt_diag_cov=sqrt_diag_cov,
    )
    print('time for kernel init:', time.time()-t1)
    
    t3 = time.time()  
    # use the quick wrapper to build a new kernel with the tuned parameters
    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=L,
        step_size=step_size
    )
    print('time quick wrapper:', time.time()-t3)

    t4 = time.time()
    # run the sampler
    _, samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=initial_state,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=False,
    )
    dts = time.time()-t4
    print('time for sampling:', dts)

    return samples, run_key

def run_mclmc_simp_with_pretune(logdensity_fn, num_steps, initial_position, key, transform, L=0.5, step_size=0.05, desired_energy_variance= 5e-4, progress_bar=False, \
                               frac_tune=0.3, min_L=1e-2, min_step_size=1e-3):
    init_key, tune_key, run_key = jax.random.split(key, 3)

    t0 = time.time()
    # create an initial state for the sampler
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key)
    print('time for mclmc init:', time.time()-t0)

    t1 = time.time()
    # build the kernel
    kernel = lambda sqrt_diag_cov : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        sqrt_diag_cov=sqrt_diag_cov,
    )

    num_steps_tune = int(num_steps*frac_tune)
    # print('time for kernel init:', time.time()-t1)

    # print('Performing tuning over ', num_steps_tune, 'steps..')
    t2 = time.time()
    # find values for L and step_size
    (
        blackjax_state_after_tuning,
        blackjax_mclmc_sampler_params,
    ) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps_tune,
        state=initial_state,
        rng_key=tune_key,
        diagonal_preconditioning=False,
        desired_energy_var=desired_energy_variance
    )

    dtfindL = time.time()-t2
    jax.debug.print('dt find L and step size: '+str(dtfindL))

    tr = time.time()
    pretuned_L = blackjax_mclmc_sampler_params.L
    pretuned_step_size=blackjax_mclmc_sampler_params.step_size

    # Guard against tuner collapse: clamp to minimum values to prevent near-zero
    # L/step_size from causing NaN propagation through subsequent sampling steps.
    import jax.numpy as jnp
    pretuned_L = jnp.maximum(pretuned_L, min_L)
    pretuned_step_size = jnp.maximum(pretuned_step_size, min_step_size)

    t3 = time.time()  
    # use the quick wrapper to build a new kernel with the tuned parameters
    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=pretuned_L,
        step_size=pretuned_step_size
    )
    jax.debug.print('time quick wrapper: '+str(time.time()-t3))

    t4 = time.time()
    # run the sampler
    _, samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=False,
    )

    dts = time.time()-t4
    jax.debug.print('time for sampling: '+str(dts))

    return samples, run_key, pretuned_L, pretuned_step_size
    

def reset_state_if_needed(L, step_size, state, initial_state, default_L=0.5, default_step_size=0.05):
    return jax.lax.cond(
        step_size == 0,  # Condition: check if step_size is zero
        lambda _: (default_L, default_step_size, initial_state),  # If True, reset step_size and state
        lambda _: (L, step_size, state),  # If False, keep current step_size and state
        operand=None
    )


def run_mclmc_with_reinit_vmap(logdensity_fn, num_steps, initial_positions, key, transform, 
                               desired_energy_variance=5e-4, pretuned_L=None, 
                               pretuned_step_size=None, ninit=100, reinit_scatter=1e-1, progress_bar=False):
    
    num_chains = initial_positions.shape[0]
    keys = jax.random.split(key, num_chains)  # Generate separate keys for each chain

    logdensity_fn = jax.jit(logdensity_fn)

    print('initial pos has shape:', initial_positions.shape)
    print('keys has shape', keys.shape)
    
    # Step 1: Run initial phase for `ninit` steps using vmap
    def run_initial_chain(position, key):
        return run_mclmc2(logdensity_fn, ninit, position, key, transform, 
                         desired_energy_variance, 0.5, 0.02, progress_bar=False)

    initial_samples, initial_states, new_keys,\
        pretuned_L, pretuned_step_size = jax.vmap(run_initial_chain)(initial_positions, keys)

    # Step 2: Compute log-densities outside vmap to evaluate initialization quality
    log_densities = jax.vmap(logdensity_fn)(initial_samples[:, -1])  # Use last sample per chain

    print('log densities:', log_densities)
    # Define a threshold: keep the top 50% of chains based on log-density
    threshold = jnp.max(log_densities)-2  
    good_chains_mask = (log_densities >= threshold)  

    print('good chains mask:', good_chains_mask)

    # Step 3: Reinitialize poor chains using good ones
    good_chains = initial_samples[:, -1][good_chains_mask]  # Select last samples of good chains
    if good_chains.shape[0] == 0:  # Edge case: if no good chains, use all chains
        good_chains = initial_samples[:, -1]

    if jnp.sum(good_chains_mask.astype(int)) > 1:
        print('more than one converged chain..')
        reinit_scatter=1e-2
    
    print('replacing bad chains')
    # Replace bad chains by resampling from good ones with small noise
    new_initial_positions = jax.random.choice(key, good_chains, shape=(num_chains,))
    noise = jax.random.normal(key, shape=new_initial_positions.shape) * reinit_scatter
    reinitialized_positions = new_initial_positions + noise  

    print('reinitialized positions:', reinitialized_positions)

    print('new keys has length:', new_keys.shape)
    print('reinit shape:', reinitialized_positions.shape)

    print('pretuned_L, pretuned_step_size:', pretuned_L.shape, pretuned_step_size.shape)

    # pretuned_L_mean = np.median(pretuned_L)
    # pretuned_step_size_mean = np.median(pretuned_step_size)
    
    # Step 4: Run remaining steps using vmap with improved initializations
    def run_final_chain(position, key):
        return run_mclmc(logdensity_fn, num_steps - ninit, position, key, transform, 
                         desired_energy_variance, None, None, progress_bar)

    final_samples, final_states, _, _, _ = jax.vmap(run_final_chain)(reinitialized_positions, new_keys)

    return final_samples, final_states
    
def run_mclmc(logdensity_fn, num_steps, initial_position, key, transform, desired_energy_variance= 5e-4, pretuned_L=None, pretuned_step_size=None, progress_bar=True, \
              nsamp_init=None, default_L=0.5, default_step_size=0.05):

    ''' 
    Wrapper for MCLMC sampling 

    num_steps : total number of steps per chain
    initial_position : initial position of one or multiple chains
    key : random key
    pretuned_L : if using heuristic can provide, otherwise code performs pretuning step first
    pretuned_step_size : same as above
    progress_bar : Used for single chains
    desired_energy_variance : targeted variance for posterior
    nsamp_init : if initializing multiple chains, can run for nsamp_init steps and then re-initialize poor chains
    
    '''
    init_key, tune_key, run_key = jax.random.split(key, 3)

    t0 = time.time()
    # create an initial state for the sampler
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
    )
    print('time for mclmc init:', time.time()-t0)

    # logdensity_fn = jax.jit(logdensity_fn)

    t1 = time.time()
    # build the kernel
    kernel = lambda sqrt_diag_cov : blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        sqrt_diag_cov=sqrt_diag_cov,
    )

    print('time for kernel init:', time.time()-t1)
    t2 = time.time()
    
    if pretuned_L is None and pretuned_step_size is None:
        print('performing tuning..')
        # find values for L and step_size
        (
            blackjax_state_after_tuning,
            blackjax_mclmc_sampler_params,
        ) = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel,
            num_steps=num_steps,
            state=initial_state,
            rng_key=tune_key,
            diagonal_preconditioning=False,
            desired_energy_var=desired_energy_variance
        )

        dtfindL = time.time()-t2
        print('dt find L and step size:', dtfindL)

        tr = time.time()
        pretuned_L = blackjax_mclmc_sampler_params.L
        pretuned_step_size=blackjax_mclmc_sampler_params.step_size
        pretuned_L, pretuned_step_size, blackjax_state_after_tuning = reset_state_if_needed(pretuned_L, pretuned_step_size, blackjax_state_after_tuning, initial_state, \
                                                                                           default_L=default_L, default_step_size=default_step_size)

        print('time for reset state:', time.time()-tr)
    else:
        blackjax_state_after_tuning = initial_state

    print('pre-tuned L, step size:', pretuned_L, pretuned_step_size)
    
    t3 = time.time()  
    # use the quick wrapper to build a new kernel with the tuned parameters
    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=pretuned_L,
        step_size=pretuned_step_size
    )
    print('time quick wrapper:', time.time()-t3)

    t4 = time.time()
    # run the sampler
    _, samples = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=blackjax_state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=progress_bar,
    )
    dts = time.time()-t4
    print('time for sampling:', dts)

    return samples, blackjax_state_after_tuning, run_key, pretuned_L, pretuned_step_size


# def run_mclmc2(logdensity_fn, num_steps, initial_position, key, transform, desired_energy_variance= 5e-4, pretuned_L=None, pretuned_step_size=None, progress_bar=True, \
#               nsamp_init=None, default_L=0.5, default_step_size=0.05):

#     ''' 
#     Wrapper for MCLMC sampling 

#     num_steps : total number of steps per chain
#     initial_position : initial position of one or multiple chains
#     key : random key
#     pretuned_L : if using heuristic can provide, otherwise code performs pretuning step first
#     pretuned_step_size : same as above
#     progress_bar : Used for single chains
#     desired_energy_variance : targeted variance for posterior
#     nsamp_init : if initializing multiple chains, can run for nsamp_init steps and then re-initialize poor chains
    
#     '''
#     init_key, tune_key, run_key = jax.random.split(key, 3)

#     t0 = time.time()
#     # create an initial state for the sampler
#     initial_state = blackjax.mcmc.mclmc.init(
#         position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
#     )
#     print('time for mclmc init:', time.time()-t0)

#     # logdensity_fn = jax.jit(logdensity_fn)

#     t1 = time.time()
#     # build the kernel
#     kernel = lambda sqrt_diag_cov : blackjax.mcmc.mclmc.build_kernel(
#         logdensity_fn=logdensity_fn,
#         integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
#         sqrt_diag_cov=sqrt_diag_cov,
#     )

#     print('time for kernel init:', time.time()-t1)
#     t2 = time.time()
    
#     if pretuned_L is None and pretuned_step_size is None:
#         print('performing tuning..')
#         # find values for L and step_size
#         (
#             blackjax_state_after_tuning,
#             blackjax_mclmc_sampler_params,
#         ) = blackjax.mclmc_find_L_and_step_size(
#             mclmc_kernel=kernel,
#             num_steps=num_steps,
#             state=initial_state,
#             rng_key=tune_key,
#             diagonal_preconditioning=False,
#             desired_energy_var=desired_energy_variance
#         )

#         dtfindL = time.time()-t2
#         print('dt find L and step size:', dtfindL)

#         tr = time.time()
#         pretuned_L = blackjax_mclmc_sampler_params.L
#         pretuned_step_size=blackjax_mclmc_sampler_params.step_size
#         pretuned_L, pretuned_step_size, blackjax_state_after_tuning = reset_state_if_needed(pretuned_L, pretuned_step_size, blackjax_state_after_tuning, initial_state, \
#                                                                                            default_L=default_L, default_step_size=default_step_size)

#         print('time for reset state:', time.time()-tr)
#     else:
#         blackjax_state_after_tuning = initial_state

#     print('pre-tuned L, step size:', pretuned_L, pretuned_step_size)
    
#     t3 = time.time()  
#     # use the quick wrapper to build a new kernel with the tuned parameters
#     sampling_alg = blackjax.mclmc(
#         logdensity_fn,
#         L=pretuned_L,
#         step_size=pretuned_step_size
#     )
#     print('time quick wrapper:', time.time()-t3)

#     t4 = time.time()
#     # run the sampler
#     _, samples = blackjax.util.run_inference_algorithm(
#         rng_key=run_key,
#         initial_state=blackjax_state_after_tuning,
#         inference_algorithm=sampling_alg,
#         num_steps=num_steps,
#         transform=transform,
#         progress_bar=progress_bar,
#     )
#     dts = time.time()-t4
#     print('time for sampling:', dts)

#     return samples, blackjax_state_after_tuning, run_key, pretuned_L, pretuned_step_size