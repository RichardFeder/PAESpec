import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, grad
import torch.optim as optim
import torchinfo
import pickle

import config
from train_ae_modl import *
from pae_modl import *
from data_proc.data_file_utils import *
from spender_modl import *
from utils import *
from visualization.result_plotting_fns import *
from flow import *
from load_phot_data import *
from diagnostics import *

# to be updated    
def train_full_PAE(params, run_name, train_ae=True, train_flow=True, dat_obj=None, cond_features=None, return_PAE=True, make_result_figs_ae=True, make_result_figs_flow=True, \
                  features_plot=None, sed_set='COSMOS', feature_vmin_vmax=None, plot_params=None, dpi=200, fpath_dict=None, \
                  train_mode='deep', property_cat_fpath=None, resave_params=False):
    
    
    rundir = config.sphx_base_path+'modl_runs/'+run_name
    
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

    print('rundir is ', rundir)
    
    if make_result_figs_ae:
        # grabs latents from full dataset and optionally saves them
        all_mu, ncode = grab_encoded_vars_dataset2(ae_modl, dat_obj_results, property_cat_df,  save=True, rundir=rundir, \
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
    