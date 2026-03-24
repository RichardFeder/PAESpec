from models.flowjax_modl import *
from models.nn_modl_jax import *
from training.train_ae_jax import *
from data_proc.dataloader_jax import *
from data_proc.data_file_utils import *
from utils.utils_jax import *
from models.pae_jax import *
import corner
# import torch

def map_and_plot_nf_latent_u_corner(nsamp=10000, nlatent=8, sig_level_norm=0.005, xlim=5):

    filters = [16, 32, 128, 256]
    n_hidden_encoder=[256, 64, 16]
    filter_sizes = [5, 5, 5, 5]

    run_name='jax_conv1_nlatent='+str(nlatent)
    rundir = config.sphx_base_path+'modl_runs/'+run_name+'/'


    params = param_dict_gen('jax', filter_sizes=filter_sizes, n_hidden_encoder=n_hidden_encoder, \
                       lr=2e-4, filters=filters, nlatent=nlatent, epochs=100, nbands=500, \
                       restframe=True, mean_sub_latents=False, \
                       plot_interval=5, weight_decay=0., nepoch_flow=50)

    dat_rf = spec_data_jax(params['nbands']) 


    wave_obs = np.sort(np.load(config.sphx_dat_path+'central_wavelengths_sphx102.npz')['central_wavelengths'])
    params_additional = dict({'filter_integrate':True, 'nlam_interp':1000, 'redshift_min':0.0001, 'redshift_max':4.0})
    
    sed_set = 'COSMOS'
    
    fpath_dict_rf = grab_fpaths_traindat(sed_set, restframe=True)
            
    _ = dat_rf.build_dataloaders(fpath_dict_rf, train_frac=1.0, load_property_cat=False, property_cat_fpath=None, \
                                         restframe=True, save_property_cat=False)
    central_wavelengths = dat_rf.sed_um_wave

    
    print('Initializing PAE model..')
    
    PAE_COSMOS = PAE_JAX(run_name=run_name, params=params, central_wavelengths=central_wavelengths, modl_type='jax', \
                wave_obs=wave_obs, params_additional=params_additional, load_flow_decoder=False)

    PAE_COSMOS.load_flow_decoder(run_name, filename_save='flow_model_iaf_50k.pkl')

    print('Loading latents..')
    latents = np.load(rundir+'/latents/latents.npz')
    latents_train = latents['all_z_train']
    print(latents_train.shape)

    latents_rescaled = jax.vmap(lambda y: PAE_COSMOS.rescale.inverse(y))(latents_train[:nsamp,:])
    latents_u = jax.vmap(lambda z: PAE_COSMOS.flow.bijection.inverse(z))(latents_rescaled)

    # make corner plot figure

    ranges = [(-xlim, xlim) for x in range(latents_u.shape[1])]

    fig_corner = corner.corner(np.array(latents_u[:nsamp]), labels=['$u_'+str(n)+'$' for n in range(8)], show_titles=True, \
                                  quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 16}, range=ranges)

    return fig_corner, latents_u


def compare_redshift_runs(zresult1, zresult2, zmin=0.0, zmax=3.0, cmap='bwr', figsize=(6, 5)):

    logp1 = zresult1['all_log_L']+zresult1['all_log_prior']+zresult1['all_log_redshift']
    logp2 = zresult2['all_log_L']+zresult2['all_log_prior']+zresult2['all_log_redshift']
    
    # compare zest colored by delta chi2
    fig1 = plt.figure(figsize=figsize)
    # plt.scatter(floatz['z_mean'], floatz_102['z_mean'], color='k', s=2)
    plt.scatter(zresult1['z_mean'], zresult2['z_mean'], c=-(zresult1['all_log_L']-zresult2['all_log_L'])*2, s=2, cmap=cmap, vmin=-5, vmax=5)
    cbar = plt.colorbar()
    cbar.set_label('$\\chi^2_1 -\\chi^2_2$', fontsize=14)
    plt.xlabel('$\\hat{z}$ (Run 1)', fontsize=14)
    plt.ylabel('$\\hat{z}$ (Run 2)', fontsize=14)
    plt.plot(np.linspace(zmin, zmax, 100), np.linspace(zmin, zmax, 100), color='r', linestyle='dashed')
    plt.grid(alpha=0.5)
    plt.xlim(zmin, zmax)
    plt.ylim(zmin, zmax)
    plt.show()

    if len(np.unique(zresult1['all_log_prior']))<10 or len(np.unique(zresult2['all_log_prior']))<10:
        fig2 = None
        fig3 = None
    else:
        # same but colored by delta log prior
        fig2 = plt.figure(figsize=figsize)
        # plt.scatter(floatz['z_mean'], floatz_102['z_mean'], color='k', s=2)
        plt.scatter(zresult1['z_mean'], zresult2['z_mean'], c=-(zresult1['all_log_prior']-zresult2['all_log_prior']), s=2, vmin=-5, vmax=5)
        cbar = plt.colorbar()
        cbar.set_label('-$\\log p(u)_1 -\\log p(u)_2$', fontsize=14)
        plt.xlabel('$\\hat{z}$ (Run 1)', fontsize=14)
        plt.ylabel('$\\hat{z}$ (Run 2)', fontsize=14)
        plt.plot(np.linspace(zmin, zmax, 100), np.linspace(zmin, zmax, 100), color='r', linestyle='dashed')
        plt.grid(alpha=0.5)
        plt.xlim(zmin, zmax)
        plt.ylim(zmin, zmax)
        plt.show()

        fig3 = plt.figure(figsize=figsize)
        # plt.scatter(floatz['z_mean'], floatz_102['z_mean'], color='k', s=2)
        plt.scatter(zresult1['z_mean'], zresult2['z_mean'], c=-(logp1-logp2), s=2, vmin=-5, vmax=5)
        cbar = plt.colorbar()
        cbar.set_label('-$\\log p(u)_1 -\\log p(u)_2$', fontsize=14)
        plt.xlabel('$\\hat{z}$ (Run 1)', fontsize=14)
        plt.ylabel('$\\hat{z}$ (Run 2)', fontsize=14)
        plt.plot(np.linspace(zmin, zmax, 100), np.linspace(zmin, zmax, 100), color='r', linestyle='dashed')
        plt.grid(alpha=0.5)
        plt.xlim(zmin, zmax)
        plt.ylim(zmin, zmax)
        plt.show()



def compare_with_fixz_run(float_z_run, fix_z_run, zmin=0.0, zmax=3.0, cmap='bwr', Lmin = -80, Lmax = -30, \
                         nfmin=-20, nfmax=-7.5, sel_sigz_max=0.2, figsize=(5, 4)):
    
    zscore_floatz = (float_z_run['z_mean']-float_z_run['ztrue'])/(0.5*(float_z_run['err_low']+float_z_run['err_high']))

    sigzoneplusz = 0.5*(np.abs(float_z_run['err_low'])+np.abs(float_z_run['err_high']))/(1+float_z_run['z_mean'])

    if sel_sigz_max is not None:

        selmask = np.where((sigzoneplusz < sel_sigz_max))[0]
    else:
        selmask = np.arange(len(sigzoneplusz))

    fig1 = plt.figure(figsize=figsize)
    plt.scatter(float_z_run['all_log_L'][selmask], fix_z_run['all_log_L'][selmask], c=zscore_floatz[selmask], s=3, cmap=cmap, vmin=-3, vmax=3)
    cbar = plt.colorbar()
    cbar.set_label('Redshift z-score', fontsize=14)
    plt.plot(np.linspace(Lmin, Lmax, 100), np.linspace(Lmin, Lmax, 100), linestyle='dashed', color='k') 
    plt.xlim(Lmin, Lmax)
    plt.ylim(Lmin, Lmax)
    plt.xlabel('logL (Float redshift)', fontsize=14)
    plt.ylabel('logL (Fix redshift)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()

    fig2 = plt.figure(figsize=figsize)
    plt.scatter(float_z_run['all_log_prior'][selmask], fix_z_run['all_log_prior'][selmask], c=zscore_floatz[selmask], s=3, cmap='bwr', vmin=-3, vmax=3)
    cbar = plt.colorbar()
    cbar.set_label('Redshift z-score', fontsize=14)

    linsp = np.linspace(nfmin, nfmax, 100)
    plt.plot(linsp, linsp, linestyle='dashed', color='k') 
    plt.xlim(nfmin, nfmax)
    plt.ylim(nfmin, nfmax)
    plt.xlabel('log-NF prior (Float redshift)', fontsize=14)
    plt.ylabel('log-NF prior (Fix redshift)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()
    
    pmin, pmax = Lmin+nfmin, Lmax+nfmax
    fig3 = plt.figure(figsize=figsize)
    plt.scatter(float_z_run['all_log_prior'][selmask]+float_z_run['all_log_L'][selmask], fix_z_run['all_log_prior'][selmask]+fix_z_run['all_log_L'][selmask], c=zscore_floatz[selmask], s=3, cmap='bwr', vmin=-3, vmax=3)
    cbar = plt.colorbar()
    cbar.set_label('Redshift z-score', fontsize=14)
    plt.plot(np.linspace(pmin, pmax, 100), np.linspace(pmin, pmax, 100), linestyle='dashed', color='k') 
    plt.xlim(pmin, pmax)
    plt.ylim(pmin, pmax)
    plt.xlabel('log-p (Float redshift)', fontsize=14)
    plt.ylabel('log-p (Fix redshift)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()

    chisq_float = -2*float_z_run['all_log_L']
    chisq_fix = -2*fix_z_run['all_log_L']
    fig4 = plt.figure(figsize=figsize)
    # plt.scatter(chisq_float-chisq_fix, zscore_floatz, s=3, color='k', alpha=0.5)
    plt.scatter((float_z_run['all_log_L']-fix_z_run['all_log_L'])[selmask], zscore_floatz[selmask], c=np.log10(sigzoneplusz[selmask]), s=3, alpha=0.8, vmin=-3, vmax=-0.5, cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('$\\log \\sigma_z/(1+z)$', fontsize=14)

    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    plt.xlabel('$\\log L_{float-z}-\\log L_{fix-z}$', fontsize=14)
    plt.ylabel('Redshift z-score', fontsize=14)
    plt.grid(alpha=0.3)

    plt.show()

    fig5 = plt.figure(figsize=figsize)
    # plt.scatter(float_z_run['all_log_prior']-fix_z_run['all_log_prior'], zscore_floatz, s=3, color='k', alpha=0.5)
    plt.scatter(float_z_run['all_log_prior'][selmask]-fix_z_run['all_log_prior'][selmask], zscore_floatz[selmask], s=3, c=np.log10(sigzoneplusz[selmask]), alpha=0.8, vmin=-3, vmax=-0.5, cmap='jet')
    cbar = plt.colorbar()
    cbar.set_label('$\\log \\sigma_z/(1+z)$', fontsize=14)
    plt.xlim(-3, 3)
    plt.ylim(-5, 5)
    plt.xlabel('$\\log \\pi_{NF}^{float-z}-\\log \\pi_{NF}^{fix-z}$', fontsize=14)
    plt.ylabel('Redshift z-score', fontsize=14)
    plt.grid(alpha=0.3)
    plt.show()


    figs = [fig1, fig2, fig3, fig4, fig5]

    
    return figs

def compare_mclmc_pocomc_coverage(save_fpaths, sample_fpaths, labels=None, nsrc=100, snrstr='25SNR50', filts_str='sphx102', nf_alpha=1, load_tf=False, src_idxs=None, \
                                 tf_results=None, plot_zscore=True, colors=None):

    # sample_fpath_mclmc = basepath+'PAE_samples_'+str(nsrc_mclmc)+'_srcs_'+filts_str+'_nlatent=5_scatter=0.005_wreinit200_nchain=8_'+snrstr+'_nfalpha='+str(nf_alpha)+'.npz'
    # save_fpath_mclmc = basepath+'PAE_results_'+str(nsrc_pocomc)+'_srcs_'+filts_str+'_nlatent=5_scatter=0.005_wreinit200_nchain=8_'+snrstr+'_nfalpha='+str(nf_alpha)+'.npz'

    if labels is None:
        labels = ['MCLMC', 'pocoMC']
        
    if colors is None:
        colors = ['b', 'C1']

    comb_lists = [[save_fpaths[k], sample_fpaths[k]] for k in range(len(save_fpaths))]
    # comb_lists = [[save_fpath_mclmc, sample_fpath_mclmc], [save_fpath_poco, sample_fpath_poco]]

    all_pit_vals = []
    all_zout, all_ztrue, all_zscore, all_dzout = [[] for x in range(4)]

    which_idxs = src_idxs[:nsrc]

    if load_tf:
        labels = ['Template fitting'] + labels
        colors = ['k']+colors
    
        finez = np.linspace(0.0, 3.002, 1501)

        
        if tf_results is None:
            # zout_tf, dzout_tf, dz_oneplusz_tf, chisq_tf, match_z, zpdf_tf = load_tf_results(load_zpdf=True)
            tf_results = load_tf_results(load_zpdf=True)

        zout_tf, dzout_tf, dz_oneplusz_tf, chisq_tf, match_z, zpdf_tf = tf_results
        
        zpdf_norm = zpdf_tf[which_idxs,3:]
        zpdf_norm /= np.sum(zpdf_norm)
        ztrue_tf = match_z[which_idxs]

        zscore_tf = (zout_tf - match_z)/dzout_tf

        pit_values_tf = compute_pit_values_tf(ztrue_tf, finez, zpdf_norm)
        all_pit_vals.append(pit_values_tf)

        all_zscore.append(zscore_tf[which_idxs])
        all_zout.append(zout_tf[which_idxs])
        all_ztrue.append(ztrue_tf)

    for x in range(len(comb_lists)):

        redshift_res = np.load(comb_lists[x][0])
        sample_res = np.load(comb_lists[x][1])
    
        all_samples = sample_res['all_samples']

        zout, ztrue, zscore = [redshift_res[key] for key in ['z_med', 'ztrue', 'zscore']]
        all_zscore.append(zscore)
        all_zout.append(zout)
        all_ztrue.append(ztrue)
        
        dzout_mclmc = 0.5*(redshift_res['err_low']+redshift_res['err_high'])

        all_dzout.append(dzout_mclmc)

        print('nsrc in file is ', len(ztrue))
    
        pit_values = compute_pit_values_pae(ztrue, all_samples, ae_redshift_samples=None) 

        all_pit_vals.append(pit_values)


    if plot_zscore:
        
        plt.figure(figsize=(5, 4))
        for k in range(len(all_zscore)):
            plt.hist(all_zscore[k], bins=np.linspace(-5, 5, 30), histtype='step', color=colors[k], label=labels[k], linewidth=1.5)
            plt.axvline(np.nanmedian(all_zscore[k]), color=colors[k])
        plt.legend(loc=2, ncol=2, bbox_to_anchor=[-0.1, 1.2])
        plt.xlabel('zscore')
        plt.ylabel('$N_{src}$', fontsize=12)
        plt.grid(alpha=0.3)
        plt.show()

        # dzbins = np.linspace(-0.4, 0.4, 50)

        dzbins = np.linspace(-0.05, 0.05, 30)

        plt.figure(figsize=(5, 4))
        for k in range(len(all_zscore)):
            errz_oneplusz = (all_zout[k]-all_ztrue[k])/(1+all_ztrue[k])
            plt.hist(errz_oneplusz, bins=dzbins, color=colors[k], histtype='step', label=labels[k])
            plt.axvline(np.median(errz_oneplusz), color=colors[k])
        plt.xlabel('$\\Delta z/(1+\\hat{z})$', fontsize=12)
        plt.ylabel('$N_{src}$', fontsize=12)
        plt.show()
                
    fig = plot_qq(all_pit_vals, '$100 < SNR < 300$', colors=colors, labels=labels)

    return fig    