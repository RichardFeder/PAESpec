import os
import config
import pickle
import glob
import numpy as np
import flax.serialization
import equinox as eqx

def create_result_dir_structure(rundir):
    
    if not os.path.isdir(rundir):
        print('making directory path for ', rundir)
        os.makedirs(rundir)
        
    latent_dir = rundir+'/latents'
    if not os.path.isdir(latent_dir):
        print('Making directory for saved latents..')
        os.makedirs(latent_dir)
        
    flow_dir = rundir+'/flows'
    if not os.path.isdir(flow_dir):
        print('Making flow directory..')
        os.makedirs(flow_dir)
        
    post_dir = rundir+'/posterior'
    if not os.path.isdir(post_dir):
        print('Making autoencoder posterior analysis directory..')
        os.makedirs(post_dir)
    
    figure_dir = rundir+'/figures'
    if not os.path.isdir(figure_dir):
        print('Making figure directory and sub-directories..')
        os.makedirs(figure_dir)
        os.makedirs(figure_dir+'/corner_latent_z')
        os.makedirs(figure_dir+'/training')
        os.makedirs(figure_dir+'/training/reconstruction_vs_epoch')
        
    return figure_dir

def check_config_dir(result_dir=None, config_name=None):
    
    if result_dir is None:
        result_dir = config.result_dir

    # make directory for saving results if specified
    if config_name is not None:
        rundir = result_dir + config_name
        print('rundir here is ', rundir)
        if not os.path.isdir(rundir):
            print('No path currently exists for '+str(config_name)+', creating directory in '+str(result_dir)+'..')
            os.mkdir(rundir)
            
    return result_dir, rundir

def save_params(rundir, params):
    print('saving parameter file')

    filename = rundir+'/params.txt'

    print('filename:', filename)
    filehandler = open(filename, 'wb')
    pickle.dump(params, filehandler)
    filehandler.close()


def grab_fpaths_rf(sed_set, tailstr_linecat='aug2x_debv_0p02_dustlaw', hires_fname='hires_sed_COSMOS_0_200000_0p1_8_um_z=0_aug2x_debv_0p02_dustlaw_newAllen.npz'):
    ''' Loading training dataset for rest frame PAE'''

    fpath_dict = dict({})
    fpath_dict['sed_set'] = sed_set
    fpath_dict['srcid_key'] = fpath_dict['srcid_dict'][sed_set]

    fpath_dict['linecat_fpath'] = config.sphx_dat_path+'line_catalogs/'+sed_set+'_'+tailstr_linecat+'.csv'
    phot_fpath = config.sphx_dat_path+'phot/'
    fpath_dict['phot_fpath'] = phot_fpath

    if sed_set=='GAMA':
        fpath_dict['data_fpath'] = phot_fpath+'catgrid_sphx_plus_W1W2grz_'+sed_set+'.out'

    elif sed_set=='COSMOS':
        fpath_dict['data_fpath'] = phot_fpath+hires_fname
        fpath_dict['catgrid_noiseless_fpath'] = None

    return fpath_dict

    

def grab_fpaths_traindat(sed_set, restframe=False, train_mode=None, sel_str='zlt22.5'):

    
    fpath_dict = dict({})
    
    fpath_dict['srcid_dict'] = dict({'COSMOS':'Tractor_ID', 'GAMA':'uberID'})
    fpath_dict['sed_set'] = sed_set
    fpath_dict['srcid_key'] = fpath_dict['srcid_dict'][sed_set]

    
    if restframe:
        fpath_dict['linecat_fpath'] = config.sphx_dat_path+'line_catalogs/'+sed_set+'_aug2x_debv_0p02_dustlaw.csv'
        # fpath_dict['linecat_fpath'] = config.sphx_dat_path+'line_catalogs/'+sed_set+'_aug2x_debv_0p02_dustlaw.csv'
    else:
        fpath_dict['linecat_fpath'] = config.sphx_dat_path+'line_catalogs/'+sed_set+'_catalog_with_lines_feder23.csv'

    phot_fpath = config.sphx_dat_path+'phot/'

    fpath_dict['phot_fpath'] = phot_fpath
    
    if sed_set=='GAMA':
        fpath_dict['deep_phot_fpath'] = phot_fpath+'catgrid_sphx_plus_W1W2grz_1pct_noisefloor_GAMA_full'
        fpath_dict['fullsky_phot_fpath'] = phot_fpath+'catgrid_sphx_plus_W1W2grz_1pct_noisefloor_GAMA_full'
        fpath_dict['catgrid_noiseless_fpath'] = phot_fpath+'catgrid_sphx_plus_W1W2grz_'+sed_set+'.out'

        if train_mode=='deep':
            fpath_dict['data_fpath'] = fpath_dict['deep_phot_fpath']
        elif train_mode in ['full', 'fullsky']:
            fpath_dict['data_fpath'] = fpath_dict['fullsky_phot_fpath']
            
    elif sed_set=='COSMOS':
        if sel_str is not None:
            fpath_dict['ext_info_fpath'] = config.sphx_dat_path+'select_catgrid_info/catgrid_info_COSMOS_'+sel_str
            print('ext info fpath is ', fpath_dict['ext_info_fpath'])
        
        if restframe:
            # fpath_dict['data_fpath'] = phot_fpath+'hires_sed_COSMOS_0_200000_0p1_8_um_z=0_aug2x_debv_0p02_dustlaw.npz'
            fpath_dict['data_fpath'] = phot_fpath+'hires_sed_COSMOS_0_200000_0p1_8_um_z=0_aug2x_debv_0p02_dustlaw_newAllen.npz'

            # fpath_dict['data_fpath'] = phot_fpath+'hires_sed_COSMOS_0_50000_0p1_8_um_z=0_nolines_mean_102824.npz'
            fpath_dict['catgrid_noiseless_fpath'] = None

            # fpath_dict['ext_info_fpath'] = config.sphx_dat_path+'select_catgrid_info/catgrid_info_COSMOS_'+sel_str
            # print('ext info fpath is ', fpath_dict['ext_info_fpath'])
        else:
            fpath_dict['deep_phot_fpath'] = phot_fpath+'catgrid_sphx_plus_W1W2grz_C20_withunc_deep'
            fpath_dict['fullsky_phot_fpath'] = phot_fpath+'catgrid_sphx_plus_W1W2grz_C20_withunc_fullsky'
            # fpath_dict['catgrid_noiseless_fpath'] = phot_fpath+'catgrid_sphx_plus_W1W2grz_C20.out'
            fpath_dict['catgrid_noiseless_fpath'] = phot_fpath+'catgrid_LS_W1W2grz_COSMOS_aug2x_debv_0p02_dustlaw.out'
            
            if train_mode=='deep':
                fpath_dict['data_fpath'] = fpath_dict['deep_phot_fpath']
            elif train_mode in ['full', 'fullsky']:
                fpath_dict['data_fpath'] = fpath_dict['fullsky_phot_fpath']

    return fpath_dict


def load_combined_pae_results(results_pattern, samples_pattern=None, sort_by_index=True):
    """
    Load and combine PAE results from multiple batch jobs.
    
    This function finds all result files matching the pattern, loads them,
    and concatenates them into single arrays. Useful when you've run multiple
    batch jobs with different start indices.
    
    Args:
        results_pattern (str): Glob pattern for result files, e.g.,
            '/path/to/PAE_results_*_start*.npz'
        samples_pattern (str, optional): Glob pattern for sample files. If None,
            attempts to derive from results_pattern by replacing 'results' with 'samples'
        sort_by_index (bool): If True, sorts files by start index before concatenating
    
    Returns:
        tuple: (results_dict, samples_dict) where each dict contains concatenated arrays
            - results_dict: {'ztrue', 'z_mean', 'z_med', 'err_low', 'err_high', 
                            'chi2', 'R_hat', 'zscore', ...}
            - samples_dict: {'all_samples', 'ztrue', ...} or None if samples not found
    
    Example:
        # Load all results from a batch run
        results, samples = load_combined_pae_results(
            '/scratch/data/pae_sample_results/MCLMC/zres/PAE_results_175_srcs_*_121525_start*.npz'
        )
        print(f"Total sources: {len(results['ztrue'])}")
        print(f"Mean R-hat: {np.mean(results['R_hat']):.3f}")
    """
    import re
    
    # Find all matching result files
    result_files = sorted(glob.glob(results_pattern))
    
    if len(result_files) == 0:
        raise FileNotFoundError(f"No files found matching pattern: {results_pattern}")
    
    print(f"Found {len(result_files)} result files")
    
    # Extract start indices for sorting if requested
    if sort_by_index:
        def extract_start_idx(filename):
            match = re.search(r'start(\d+)', filename)
            return int(match.group(1)) if match else 0
        
        result_files = sorted(result_files, key=extract_start_idx)
        print(f"Sorted files by start index")
    
    # Load and concatenate results
    results_list = []
    for i, fpath in enumerate(result_files):
        try:
            data = np.load(fpath, allow_pickle=True)
            results_list.append(data)
            if i == 0:
                print(f"First file: {fpath}")
                print(f"  Keys: {list(data.keys())}")
        except Exception as e:
            print(f"Warning: Failed to load {fpath}: {e}")
    
    if len(results_list) == 0:
        raise RuntimeError("No files could be loaded successfully")
    
    # Concatenate arrays
    results_combined = {}
    for key in results_list[0].keys():
        arrays = []
        for data in results_list:
            if key in data:
                arr = data[key]
                # Handle scalar values (like 'sampling_mode')
                if arr.ndim == 0 or isinstance(arr, str):
                    if key not in results_combined:
                        results_combined[key] = arr
                else:
                    arrays.append(arr)
        
        if arrays:
            results_combined[key] = np.concatenate(arrays, axis=0)
    
    print(f"Combined results: {len(results_combined['ztrue'])} total sources")
    
    # Load samples if pattern provided or can be derived
    samples_combined = None
    if samples_pattern is None:
        # Try to derive samples pattern from results pattern
        samples_pattern = results_pattern.replace('zres/PAE_results', 'samples/PAE_samples')
    
    sample_files = sorted(glob.glob(samples_pattern))
    
    if len(sample_files) > 0:
        print(f"Found {len(sample_files)} sample files")
        
        if sort_by_index:
            sample_files = sorted(sample_files, key=extract_start_idx)
        
        samples_list = []
        for fpath in sample_files:
            try:
                data = np.load(fpath, allow_pickle=True)
                samples_list.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {fpath}: {e}")
        
        if samples_list:
            samples_combined = {}
            for key in samples_list[0].keys():
                arrays = []
                for data in samples_list:
                    if key in data:
                        arr = data[key]
                        if arr.ndim == 0 or isinstance(arr, str):
                            if key not in samples_combined:
                                samples_combined[key] = arr
                        else:
                            arrays.append(arr)
                
                if arrays:
                    samples_combined[key] = np.concatenate(arrays, axis=0)
            
            print(f"Combined samples: {samples_combined['all_samples'].shape}")
    else:
        print("No sample files found")
    
    return results_combined, samples_combined


def load_tf_results(output_photoz=None,
                    output_zpdf=None,
                    load_zpdf=False,
                   select_info=None):
    idx_dict = dict({'truez':4, 'zout':3, 'dzout':4, 'chisq':11})

    if output_photoz is None:
        # output_photoz = config.sphx_dat_path+'tf_results/photoz_result_sphxonly_COSMOS_aug2x_debv_0p02_dustlaw_fullsky_zlt22.5.out'
        output_photoz = config.sphx_dat_path+'jax_tf_results/photoz_result_jax_sphxonly_COSMOS_fineEBV_alltemp_fullsky_zlt22.5_fp32.out'
    
    if select_info is None:
        select_info = config.sphx_dat_path+'select_catgrid_info/catgrid_info_COSMOS_zlt22.5'

    select_cat_info = np.loadtxt(select_info)
    match_z = select_cat_info[:,idx_dict['truez']]

    photoz_out = np.loadtxt(output_photoz)
    zout = photoz_out[:,idx_dict['zout']]
    dzout = photoz_out[:,idx_dict['dzout']]
    chisq_tf = photoz_out[:,idx_dict['chisq']]
    
    dz_oneplusz = dzout/(1+zout)

    if load_zpdf:
        if output_zpdf is None:
            # output_zpdf = config.sphx_dat_path+'tf_results/photoz_pdfs_sphxonly_COSMOS_fineEBV_alltemp_fullsky_zlt22.5.out'
            output_zpdf = config.sphx_dat_path+'jax_tf_results/photoz_pdfs_jax_sphxonly_COSMOS_fineEBV_alltemp_fullsky_zlt22.5_fp32.out'

        print('loading zPDFs from ', output_zpdf)
        zpdf = np.loadtxt(output_zpdf)

    else:
        zpdf = None

    return zout, dzout, dz_oneplusz, chisq_tf, match_z, zpdf

# load SPHEREx filters

def load_sphx_filters(filtdir='data/spherex_filts/', filtfiles=None, to_um=True, sort_by_lam=True):

    ''' 
    Loads files, returns list of central wavelengths and list of wavelengths/filter responses. 
    Converts wavelengths to microns unless otherwise specified. Also sorts filters by increasing wavelength order
    '''

    bandpass_wavs, bandpass_vals, central_wavelengths, bandpass_names = [[] for x in range(4)]
    bband_idxs = np.arange(1, 7)

    if filtfiles is not None:
        print('reading', len(filtfiles), 'filter files..')
        for filtfile in filtfiles:
            # Check if filtfile is already an absolute path
            import os
            if os.path.isabs(filtfile):
                full_path = filtfile
            else:
                full_path = filtdir + '/' + filtfile
            
            bandpass_wav, bandpass_val, cenwav, bandpass_name = load_indiv_filter(full_path)
            
            bandpass_names.append(bandpass_name)
            bandpass_wavs.append(bandpass_wav)
            bandpass_vals.append(bandpass_val)
            central_wavelengths.append(cenwav)
            
    else:
        print('filtdir:', filtdir)
        for bandidx in bband_idxs:
            filtfiles = glob.glob(filtdir+'SPHEREx_band'+str(bandidx)+'*.dat')
            for filtfile in filtfiles:
    
                bandpass_wav, bandpass_val, cenwav, bandpass_name = load_indiv_filter(filtfile)
                
                bandpass_names.append(bandpass_name)
                bandpass_wavs.append(bandpass_wav)
                bandpass_vals.append(bandpass_val)
                central_wavelengths.append(cenwav)
            
    central_wavelengths = np.array(central_wavelengths)
    bandpass_names = np.array(bandpass_names)

    if sort_by_lam:
        sortidx = np.argsort(central_wavelengths)
        
        central_wavelengths = central_wavelengths[sortidx]
        bandpass_wavs = [bandpass_wavs[idx] for idx in sortidx]
        bandpass_vals = [bandpass_vals[idx] for idx in sortidx]
        bandpass_names = bandpass_names[sortidx]

    return central_wavelengths, bandpass_wavs, bandpass_vals, bandpass_names

def grab_ext_phot_filt_files(lsst=False, wise=False, decam=False, uvista=False, subaru=False, base_filt_path=None):
    
    if base_filt_path is None:
        base_filt_path = config.ext_filt_path
    
    lsst_filt_files, wise_filt_files, decam_filt_files, uv_filt_files = [[] for x in range(4)]
    if lsst:
        # LSST bands:
        with open(base_filt_path+'LSST.list', 'r') as f:
            lsst_filt_files = f.read()
            lsst_filt_files = lsst_filt_files.split('\n')[:-2]
    if wise:
        # WISE filters
        wise_filt_files = ['W'+str(ch)+'.res' for ch in [1,2]]
    if decam:
        #DEcam 
        decam_filt_files = [decam_band+'_DEcam.res' for decam_band in ['g', 'r', 'z']]
    if uvista:
        # UltraVista near infrared magnitudes
        uv_filt_files = [uv_band+'_uv.res' for uv_band in ['J', 'H', 'K']]

    all_ext_filt_files = lsst_filt_files + wise_filt_files + decam_filt_files + uv_filt_files
    
    if subaru:
        all_ext_filt_files.append('V_subaru.res')

    return all_ext_filt_files

def load_ext_filters(base_filt_path=None, filter_fpaths=None, wise=False, decam=False, to_um=True):
    if base_filt_path is None:
        base_filt_path = config.ext_filt_path
    
    ext_bandpass_wavs, ext_bandpass_vals, ext_central_wavelengths, ext_bandpass_names = [[] for x in range(4)]

    if filter_fpaths is None:
        filter_fpaths = grab_ext_phot_filt_files(decam=decam, wise=wise)

    for filtfile in filter_fpaths:
        ext_bandpass_wav, ext_bandpass_val, ext_cenwav, ext_bandpass_name = load_indiv_filter(base_filt_path+filtfile)
        ext_bandpass_names.append(ext_bandpass_name)
        ext_bandpass_wavs.append(ext_bandpass_wav)
        ext_bandpass_vals.append(ext_bandpass_val)
        ext_central_wavelengths.append(ext_cenwav)

    return np.array(ext_central_wavelengths), ext_bandpass_wavs, ext_bandpass_vals, np.array(ext_bandpass_names)


def load_indiv_filter(filtfile, norm=True):
    bandpass_name = filtfile.split('.')[0].split('/')[-1]

    x = np.loadtxt(filtfile)
    nonz = (x[:,1] != 0.)
    bandpass_wav = x[nonz,0]*1e-4
    bandpass_val = x[nonz,1]

    if norm:
        bandpass_val /= np.sum(bandpass_val)

    cenwav = np.dot(bandpass_wav, bandpass_val)
    # cenwav = np.dot(x[nonz,0], x[nonz,1])

    return bandpass_wav, bandpass_val, cenwav, bandpass_name


# Function to save model parameters and training state
def save_ae_jax(state, rundir, model_fpath=None):
    
    if model_fpath is None:
        model_fpath = rundir+'/model'
        model_fpath += '.pkl'

    with open(model_fpath, "wb") as f:
        pickle.dump(flax.serialization.to_state_dict(state), f)
    print(f"Model saved to {model_fpath}")

    return model_fpath

def load_jax_state(rundir, state_template, model_fpath=None):
    if model_fpath is None:
        model_fpath = rundir+'/model.pkl'

    with open(model_fpath, "rb") as f:
        state_dict = pickle.load(f)

    state = flax.serialization.from_state_dict(state_template, state_dict)

    return state

def save_train_metrics_jax(metric_dict, rundir, savefpath=None):
    '''
    Saves training/validation losses.
    '''
    
    if savefpath is None:
        savefpath = rundir+'/metrics.npz'

    print('Saving training/validation metrics to ', savefpath)

    # Preserve legacy keys while also storing all available metric series.
    save_dict = {
        'trainloss': np.array(metric_dict['train_loss']),
        'validloss': np.array(metric_dict['valid_loss']),
    }
    for key, values in metric_dict.items():
        save_dict[key] = np.array(values)

    np.savez(savefpath, **save_dict)
    
def save_model(flow_model, rundir, loc=None, scale=None, filename="flow_model.pkl"):

    fname = rundir+'/flows/'+filename
    print('saving flow to ', fname)
    eqx.tree_serialise_leaves(fname, flow_model)

    if loc is not None:
        print('Saving parameters for Affine transformation..')
        np.savez(rundir+'/latents/latent_mean_std.npz', loc=loc, std=scale)

    return fname

def load_model(flow_model_bare, rundir, filename='flow_model.pkl'):

    fname = rundir+'/flows/'+filename
    print('Loading flow from ', fname)
    
    flow_model = eqx.tree_deserialise_leaves(fname, flow_model_bare)

    return flow_model
    