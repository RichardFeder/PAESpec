import pandas as pd
import numpy as np
import config
import matplotlib.pyplot as plt

from visualization.result_plotting_fns import plot_log_phot_weights, plot_norm_phot_fluxes, plot_snr_persource, plot_unnorm_fluxes

def zweight_naninf_vals(weights, fluxes=None):

    weights[weights < 0] = 0.
    weights[np.isinf(weights)] = 0.
    weights[np.isnan(weights)] = 0.

    if fluxes is not None:
        weights[np.isnan(fluxes)] = 0.

    return weights

def make_crossmatch_property_cat(features, catgrid, linecat_df, srcids=None, tid_idx_catgrid=0, which_set='COSMOS'):

    nfeat = len(features)
    nsrc = catgrid.shape[0]
    
    srcid_dict = dict({'COSMOS':'Tractor_ID', 'GAMA':'uberID'})
    
    if srcids is None:
        srcids = catgrid[:,tid_idx_catgrid]
    
    linecat_tid = np.array(linecat_df[srcid_dict[which_set]])
        
    property_cat = np.zeros((nfeat, nsrc))
    features_linecat = [np.array(linecat_df[feature]) for feature in features]

    for x in range(nsrc):
        whichmatch = np.where((linecat_tid==srcids[x]))[0][0]

        for f, feat in enumerate(features):
            property_cat[f, x] = features_linecat[f][whichmatch]

    property_cat_df = pd.DataFrame()

    for f, feat in enumerate(features):
        property_cat_df[feat] = property_cat[f,:]
                
    return property_cat_df

def _load_restframe_data(fpath_dict, apply_sel=False, srcids_ext_sel=None, sig_level=None):
    spec_file = np.load(fpath_dict['data_fpath'])
    sed_um_wave = spec_file['sed_um_wave']
    flux = spec_file['sed_mJy_fluxes']
    srcids = spec_file['srcids']
    flux_unc = None if sig_level is None else np.zeros_like(flux)

    if apply_sel:
        sel = np.isin(srcids, srcids_ext_sel)
        flux = flux[sel]
        srcids = srcids[sel]

    return flux, flux_unc, sed_um_wave, srcids

def _load_observed_data(fpath_dict, tid_idx_catgrid=0, startidx=3, startidx_noiseless=9):
    spec_full = np.loadtxt(fpath_dict['data_fpath'])
    flux = spec_full[:, startidx:-1:2]
    flux_unc = spec_full[:, startidx+1::2]
    srcids = spec_full[:, tid_idx_catgrid]

    sed_um_wave = np.sort(np.load(config.sphx_dat_path + 'central_wavelengths_sphx102.npz')['central_wavelengths'])

    catgrid_noiseless, srcids_noiseless = None, None
    if fpath_dict.get('catgrid_noiseless_fpath'):
        noiseless_cat = np.loadtxt(fpath_dict['catgrid_noiseless_fpath'])
        srcids_noiseless = noiseless_cat[:, 0]
        catgrid_noiseless = noiseless_cat[:, startidx_noiseless:]

    return flux, flux_unc, sed_um_wave, srcids, catgrid_noiseless, srcids_noiseless



def load_in_sphx_dat_new(fpath_dict, restframe=False, features=None, property_cat_fpath=None, save_property_cat=False, load_property_cat=False, plot=False, include_ext_phot=False, tid_idx_catgrid=0, sed_um_wave=None, sig_level=None, \
                    apply_sel=False):
    
    if features is None:
        features = ['Tractor_ID', 'mass_best', 'ebv', 'redshift', 'bfit_tid', 'dustlaw']
        
    catgrid_noiseless = None

    if apply_sel:
        print('ext info fpath is ', fpath_dict['ext_info_fpath'])
        ext_info = np.loadtxt(fpath_dict['ext_info_fpath'])
        srcids_ext_sel = ext_info[:,0]
    else:
        srcids_ext_sel = None
        
    if restframe:
        flux, flux_unc, sed_um_wave, srcids = _load_restframe_data(fpath_dict, apply_sel, srcids_ext_sel, sig_level)
        catgrid_noiseless, srcids_noiseless = None, None
    else:
        flux, flux_unc, sed_um_wave, srcids, catgrid_noiseless, srcids_noiseless = _load_observed_data(fpath_dict, tid_idx_catgrid)

    features = features.copy()
    features[0] = fpath_dict['srcid_key']
    if fpath_dict['sed_set']=='GAMA':
        features.pop()
        
    if property_cat_fpath is not None and load_property_cat:
        property_cat_df = pd.read_csv(property_cat_fpath)
    else:
        print('Making cross match property cat..')
        if apply_sel and restframe:
            print('src ids vs srcids ext sel:', len(srcids), len(srcids_ext_sel))
            sel = np.where(np.isin(srcids, srcids_ext_sel))[0]
            print('len of sel is ', len(sel))            
            srcids = srcids[sel]

        linecat_df = pd.read_csv(fpath_dict['linecat_fpath'])
        property_cat_df = make_crossmatch_property_cat(features, flux, linecat_df, srcids=srcids, which_set=fpath_dict['sed_set'])

        if save_property_cat:
            print('Saving property catalog to ', property_cat_fpath)
            property_cat_df.to_csv(property_cat_fpath, index=False)        

    return flux, flux_unc, sed_um_wave, catgrid_noiseless, property_cat_df, srcids_noiseless


def load_in_sphx_dat(fpath_dict, restframe=False, features=None, property_cat_fpath=None, save_property_cat=False, load_property_cat=False, plot=False, include_ext_phot=False, tid_idx_catgrid=0, sed_um_wave=None, sig_level=None, \
                    apply_sel=False, load_rf_dat=True):
    
    if features is None:
        
        features = ['Tractor_ID', 'mass_best', 'ebv', 'redshift', 'bfit_tid', 'dustlaw']
        
    catgrid_noiseless = None

    if apply_sel:
        print('applying selection')
        print('ext info fpath is ', fpath_dict['ext_info_fpath'])
        ext_info = np.loadtxt(fpath_dict['ext_info_fpath'])
        print('ext info has shape', ext_info.shape)
        srcids_ext_sel = ext_info[:,0]
        

    srcids_noiseless = None
    if restframe:
        print('Were in the rest frame')
        spec_file = np.load(fpath_dict['data_fpath'])
        sed_um_wave = spec_file['sed_um_wave']
        flux = spec_file['sed_mJy_fluxes']
        srcids = spec_file['srcids']

        if sig_level is None:
            flux_unc = None

        # load external info to make selection on srcids
        if apply_sel:
            print(len(srcids), len(srcids_ext_sel))
            sel = np.where(np.isin(srcids, srcids_ext_sel))[0]
            print('len of sel is ', len(sel))
            flux = flux[sel]
            sel_srcids = srcids[sel]
            print('flux now has shape', flux.shape)
        
    else:
        # if include_ext_phot:
        #     startidx, startidx_noiseless = 3, 9
        # else:
        #     startidx, startidx_noiseless = 13, 14
            
        startidx, startidx_noiseless = 3, 9

            
        print('opening photometry from ', fpath_dict['data_fpath'])
        
        spec_full = np.loadtxt(fpath_dict['data_fpath'])
        flux = spec_full[:,startidx:-1:2]
        flux_unc = spec_full[:,startidx+1::2]
        srcids = spec_full[:,tid_idx_catgrid]
        
        sed_um_wave = np.sort(np.load(config.sphx_dat_path+'central_wavelengths_sphx102.npz')['central_wavelengths'])
        
        if fpath_dict['catgrid_noiseless_fpath'] is not None and load_rf_dat:
            print('opening catgrid noiseless fpath:', fpath_dict['catgrid_noiseless_fpath'])

            noiseless_cat = np.loadtxt(fpath_dict['catgrid_noiseless_fpath'])
            srcids_noiseless = noiseless_cat[:,0]
            catgrid_noiseless = noiseless_cat[:,startidx_noiseless:]

            print('srcids noiseless in load sphx dat has shape', srcids_noiseless.shape)
            print('catgrid noiseless:', catgrid_noiseless.shape)

        else:
            catgrid_noiseless, srcids_noiseless = None, None

    nsrc = flux.shape[0]
    
    linecat_df = pd.read_csv(fpath_dict['linecat_fpath'])
    
    features[0] = fpath_dict['srcid_key']
    
    if fpath_dict['sed_set']=='GAMA':
        features.pop()
        
    if property_cat_fpath is not None and load_property_cat:
        property_cat_df = pd.read_csv(property_cat_fpath)
    else:
        if features is None:
            print('need features')
            return None
        
        print('Making cross match property cat..')

        if apply_sel and restframe:
            srcids = sel_srcids
        
        property_cat_df = make_crossmatch_property_cat(features, flux, linecat_df, srcids=srcids, which_set=fpath_dict['sed_set'])

        if save_property_cat:
            print('Saving property catalog to ', property_cat_fpath)
            # print(property_cat_df)
            property_cat_df.to_csv(property_cat_fpath)        

    return flux, flux_unc, sed_um_wave, catgrid_noiseless, property_cat_df, srcids_noiseless

def plot_norms_noiseless_vs_phot_norms(norms_noiseless, phot_norms):
    plt.figure(figsize=(6, 6))
    plt.scatter(norms_noiseless, phot_norms)
    plt.xlabel('norms noiseless')
    plt.ylabel('noisy norms')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.show()

def plot_phot_norms_piv_vs_mean(phot_norms_piv, phot_norms, pivot_wavelength):

    plt.figure()
    plt.hist(phot_norms_piv, bins=50, histtype='step', label='Pivot wavelength norm')
    plt.hist(phot_norms, bins=50, histtype='step', label='Mean norm')
    plt.legend()
    plt.xlabel('pivot norms, $\\lambda=$'+str(pivot_wavelength))
    plt.yscale('log')
    plt.show()

def calc_phot_mean_weights(fluxes, flux_unc=None, max_normflux=1000, weight_soft=5e-4, plot=False, sig_level_norm=None, abs_norm=True, \
                          norms_noiseless=None, pivot_wavelength=None, wav=None, use_weighted_mean=False):
    """
    Calculate photometric normalization and inverse-variance weights.
    
    Parameters
    ----------
    fluxes : array
        Input flux array, shape (n_sources, n_bands)
    flux_unc : array, optional
        Flux uncertainties, shape (n_sources, n_bands)
    max_normflux : float
        Maximum normalized flux (clips extremes)
    weight_soft : float
        Softening parameter for inverse-variance weights
    plot : bool
        Generate diagnostic plots
    sig_level_norm : float, optional
        If provided, add Gaussian noise with this sigma
    abs_norm : bool
        Whether to apply normalization
    norms_noiseless : array, optional
        Pre-computed norms from noiseless spectra
    pivot_wavelength : float, optional
        Wavelength for pivot normalization
    wav : array, optional
        Wavelength array (required if pivot_wavelength used)
    use_weighted_mean : bool
        If True and flux_unc provided, use inverse-variance weighted mean
        for normalization instead of simple mean. More robust for noisy data.
        
    Returns
    -------
    phot_dict : dict
        Dictionary with 'phot_fluxes', 'phot_norms', 'phot_weights', etc.
    """
    
    phot_fluxes = fluxes.copy()

    if plot:
        plot_unnorm_fluxes(phot_fluxes)
    
    if flux_unc is not None:
        phot_flux_unc = flux_unc.copy()

    # Compute normalization factor
    if use_weighted_mean and flux_unc is not None:
        # Weighted mean normalization (more robust for noisy data)
        # Use inverse variance weights: w_i = 1 / (σ_i^2 + weight_soft)
        inv_var = 1.0 / (flux_unc**2 + weight_soft)
        
        # Handle NaN/inf in flux or weights, AND exclude very large uncertainties (bad data)
        # Bad data has flux_unc = 1e10 from preprocessing, giving inv_var ~ 1e-20
        valid_mask = np.isfinite(fluxes) & np.isfinite(inv_var) & (inv_var > 1e-10)
        
        # Weighted mean: Σ(w_i * f_i) / Σ(w_i)
        weighted_sum = np.where(valid_mask, fluxes * inv_var, 0).sum(axis=1)
        weight_sum = np.where(valid_mask, inv_var, 0).sum(axis=1)
        
        # Avoid division by zero
        weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
        phot_norms = (weighted_sum / weight_sum)[:, None]
        
        print(f'Using weighted mean normalization (inverse-variance weights)')
        print(f'  Median norm: {np.median(phot_norms):.3e}')
        print(f'  Mean norm: {np.mean(phot_norms):.3e}')
        print(f'  Std norm: {np.std(phot_norms):.3e}')
    else:
        # Simple mean normalization
        # IMPORTANT: Exclude bad data (flux_unc > 1e9 from preprocessing) from mean
        if flux_unc is not None:
            # Mask out bad data (very large uncertainties set by preprocessing)
            good_mask = flux_unc < 1e9
            flux_masked = np.where(good_mask, fluxes, np.nan)
            phot_norms = np.nanmean(flux_masked, axis=1)[:, None]
            
            # Check for sources with all bad data
            n_bad_sources = np.sum(~np.isfinite(phot_norms))
            if n_bad_sources > 0:
                print(f'WARNING: {n_bad_sources} sources have all bad data, setting norm=1.0')
                phot_norms = np.where(np.isfinite(phot_norms), phot_norms, 1.0)
        else:
            # No uncertainties provided, use simple mean
            phot_norms = np.mean(fluxes, axis=1)[:, None]

    if abs_norm:
        if norms_noiseless is not None:

            plot_norms_noiseless_vs_phot_norms(norms_noiseless, phot_norms)

            assert len(norms_noiseless)==phot_fluxes.shape[0]
            phot_fluxes /= norms_noiseless[:,None]

        elif pivot_wavelength is not None and wav is not None:
            which_piv = np.argmin(np.abs(wav-pivot_wavelength))
            phot_norms_piv = phot_fluxes[:,which_piv]
            plot_phot_norms_piv_vs_mean(phot_norms_piv, phot_norms, pivot_wavelength)
            
            phot_fluxes /= phot_norms_piv[:,None]
            phot_norms = phot_norms_piv.copy()
        else:
            if use_weighted_mean and flux_unc is not None:
                print('Dividing by weighted mean')
            else:
                print('Dividing by mean')
            phot_fluxes /= phot_norms

        phot_fluxes[phot_fluxes > max_normflux] = max_normflux

    if flux_unc is not None: # only if real data, not used if directly injecting scatter to norm fluxes
        if abs_norm:
            print('Dividing flux uncertainties by phot_norms..')
            phot_flux_unc /= np.abs(phot_norms)
        print('norm flux unc ranges from ', np.min(phot_flux_unc), np.max(phot_flux_unc))
        print('norm flux unc ranges from ', np.nanmin(phot_flux_unc), np.nanmax(phot_flux_unc))

    elif sig_level_norm is not None:
        print('adding Gaussian scatter to normalized SEDs with sigma=', sig_level_norm)
        noise_realiz = np.random.normal(0, sig_level_norm, phot_fluxes.shape)
        phot_fluxes += noise_realiz
        phot_flux_unc = sig_level_norm*np.ones_like(phot_fluxes)
        weight_soft = 0.
    else:
        phot_flux_unc, phot_weights = None, None
        
    if phot_flux_unc is not None:
        phot_weights = 1./(phot_flux_unc*phot_flux_unc + weight_soft) # soften largest weights
        phot_weights = zweight_naninf_vals(phot_weights, fluxes=phot_fluxes)            
        phot_fluxes[phot_weights==0] = 0.
        print('min/max phot flux unc:', np.min(phot_flux_unc), np.max(phot_flux_unc))
        
    print('min/max sed fluxes:', np.min(phot_fluxes), np.max(phot_fluxes))
    
    if plot:
        plot_norm_phot_fluxes(phot_fluxes)
        if flux_unc is not None and len(np.unique(phot_weights)) > 1:
            print('min/max weights:', np.min(phot_weights), np.max(phot_weights))
            plot_log_phot_weights(phot_weights)

    phot_dict = dict({'phot_fluxes':phot_fluxes, 'phot_norms':phot_norms})

    if phot_weights is not None:
        phot_dict['phot_weights'] = phot_weights

    if phot_flux_unc is not None:
        phot_dict['phot_flux_unc'] = phot_flux_unc
    
    return phot_dict


def parse_input_sphx_phot(flux, flux_unc, max_normflux=1000, plot=False, weight_soft=5e-4, flux_noiseless=None, \
                         sig_level_norm=None, abs_norm=True, norms_noiseless=None, pivot_wavelength=None, wav=None, 
                         use_weighted_mean=False):
    """
    Parse and normalize SPHEREx photometry.
    
    Parameters
    ----------
    flux : array
        Input flux array, shape (n_sources, n_bands)
    flux_unc : array
        Flux uncertainties
    max_normflux : float
        Maximum normalized flux
    plot : bool
        Generate diagnostic plots
    weight_soft : float
        Softening parameter for inverse-variance weights
    flux_noiseless : array, optional
        Noiseless flux for SNR computation
    sig_level_norm : float, optional
        Add Gaussian noise with this sigma
    abs_norm : bool
        Whether to apply normalization
    norms_noiseless : array, optional
        Pre-computed norms from noiseless spectra
    pivot_wavelength : float, optional
        Wavelength for pivot normalization
    wav : array, optional
        Wavelength array
    use_weighted_mean : bool
        Use inverse-variance weighted mean for normalization
        
    Returns
    -------
    phot_dict : dict
        Dictionary with photometry, weights, SNR, etc.
    """
    
    nsrc, nbands = flux.shape[0], flux.shape[1]
    print('nbands = ', nbands, 'nsrc =', nsrc)

    if flux_unc is not None:
        total_snr = np.zeros((nsrc))

        if flux.shape[1] in [107, 413]:
            print('external photometry, only computing SNR from SPHEREx phot')
            flux_snr_perband = flux[:,5:]/flux_unc[:,5:]
        else:
            flux_snr_perband = flux/flux_unc
            
        total_snr = np.sqrt(np.sum(flux_snr_perband**2, axis=1))
    
        if flux_noiseless is not None:
            flux_snr_perband_true = flux_noiseless/flux_unc
            total_snr_true = np.sqrt(np.sum(flux_snr_perband_true**2, axis=1))

        if plot:
            plot_snr_persource(total_snr)
            
    phot_dict = calc_phot_mean_weights(flux, flux_unc=flux_unc, max_normflux=max_normflux, weight_soft=weight_soft, \
                                      sig_level_norm=sig_level_norm, plot=plot, abs_norm=abs_norm, norms_noiseless=norms_noiseless, \
                                      pivot_wavelength=pivot_wavelength, wav=wav, use_weighted_mean=use_weighted_mean)
    
    if flux_unc is not None:
        phot_dict['phot_snr'] = total_snr
        
        if flux_noiseless is not None:
            phot_dict['phot_snr_true'] = total_snr_true

    return phot_dict
    