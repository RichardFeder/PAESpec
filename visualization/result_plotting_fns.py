import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors as colors
from diagnostics.diagnostics_jax import *
from scipy.stats import gaussian_kde
import scipy
import corner
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import LogNorm # Import LogNorm for log-scaled colormap

from utils.utils_jax import format_estimates, make_plotstr, make_plotstr_count, compute_pit_values_pae, compute_pit_values_tf
from scipy.stats import pearsonr

# from scipy.stats import norm

import scipy.ndimage


def compare_sbi_pae_configs(
    config_data,
    row_configs=None,
    figsize_scatter=(10, 7),
    figsize_hist=(4, 7),
    zmin=0.0,
    zmax=3.0,
    ngal=None,
    compute_stats=True,
    text_fontsize=13,
    title_fontsize=14,
    axis_fontsize=14,
    text_pos=(0.05, 0.95),
    stats_pos=(0.05, 0.05),
    alpha=0.2,
    point_size=2,
    save_path=None,
    show_plot=True,
    colors=['b', 'C1', 'k', 'C1', 'k'],
    hist_bins=30,
    hist_range=(-0.3, 0.3),
    hist_alpha=0.6,
    use_hexbin=True,
    hexbin_gridsize=60,
):
    """
    Create two separate comparison figures: one for scatter plots and one for histograms.
    
    Parameters
    ----------
    config_data : dict or list
        If dict: Dictionary with keys corresponding to each configuration.
        If list: List where each element is a list of config names for that row.
        Each config dict should contain:
        - 'z_true' or 'truez': array of true redshifts
        - 'z_est', 'med_sbi_est', or 'zest': array of estimated redshifts  
        - 'title': string title for the subplot
        - 'method': 'SBI' or 'PAE' for appropriate key mapping
        - 'sigma_z' (optional): array of redshift uncertainties
    row_configs : list of lists, optional
        Specifies which configs go in which row
    figsize_scatter : tuple, optional
        Figure size for scatter plot figure (width, height)
    figsize_hist : tuple, optional
        Figure size for histogram figure (width, height)
        
    Returns
    -------
    fig_scatter : matplotlib.figure.Figure
        The scatter plot figure object
    fig_hist : matplotlib.figure.Figure
        The histogram figure object
    """
    
    # Determine row configuration
    if row_configs is None:
        config_keys = list(config_data.keys())
        # Default: first 2 in row 0, rest in row 1 (assuming 2 total configs)
        row_configs = [config_keys[:2], []]
    
    # Create first figure with 2x2 subplots for scatter plots
    fig_scatter, axes_scatter = plt.subplots(2, 2, figsize=figsize_scatter, sharex=True, sharey=True)
    
    # Create second figure with 2x1 subplots for histograms
    fig_hist, axes_hist = plt.subplots(2, 1, figsize=figsize_hist)
    
    # Create line for perfect agreement
    linsp = np.linspace(zmin, zmax, 100)
    
    # Storage for histogram data
    row_errors = [[], []]  # Errors for each row
    row_labels = [[], []]  # Labels for each row
    row_colors = [[], []]  # Colors for each row
    
    # Storage for histogram data
    row_errors = [[], []]  # Errors for each row
    row_labels = [[], []]  # Labels for each row
    row_colors = [[], []]  # Colors for each row
    
    # Process each row
    for row_idx, row_config_names in enumerate(row_configs):
        for col_idx, config_name in enumerate(row_config_names):
            config = config_data[config_name]
            
            # Get the appropriate axis for scatter plot
            ax = axes_scatter[row_idx, col_idx]
            
            z_true = config['truez']

            if config['method'] == 'SBI':
                z_est = config['med_sbi_est']
                sigma_z = config['sigma_z']
                    
            elif config['method'] == 'PAE':
                z_est = config['zest']
                sigma_z = config['sigma_z']
     
            else:
                raise ValueError(f"Unknown method: {config['method']}. Must be 'SBI' or 'PAE'")
            
            # Limit number of galaxies if specified
            if ngal is not None:
                n_use = min(ngal, len(z_true))
                z_true = z_true[:n_use]
                z_est = z_est[:n_use]
                if sigma_z is not None:
                    sigma_z = sigma_z[:n_use]
            else:
                n_use = len(z_true)

            sigz_oneplusz = sigma_z / (1 + z_est)

            # Create subplot
            if config['method'] == 'PAE':
                textypos = 0.8
            else:
                textypos = text_pos[1]
                
            ax.text(text_pos[0], textypos, config['title'], fontsize=title_fontsize, 
                    transform=ax.transAxes)
            
            # Compute errors for histogram
            mask = (~np.isnan(z_est))
            dz = (z_est[mask] - z_true[mask]) / (1 + z_true[mask])
            row_errors[row_idx].append(dz)
            row_labels[row_idx].append(config['title'].replace('\n', ' '))
            
            # Determine color for this config
            config_idx = list(config_data.keys()).index(config_name)
            row_colors[row_idx].append(colors[config_idx] if config_idx < len(colors) else 'k')
            
            # Compute and display statistics if requested
            if compute_stats:
                try:
                    # Use your existing functions
                    arg_bias, arg_std, bias, NMAD, cond_outl, \
                    outl_rate, outl_rate_15pct = compute_redshift_stats(
                        z_est[mask], z_true[mask], sigma_z_select=sigz_oneplusz[mask]
                    )
                    
                    # Compute median sigma_z/(1+z) if uncertainty data is available
                    if sigma_z is not None:
                        sigz_oneplusz = sigma_z / (1 + z_est)
                        med_sigz = np.nanmedian(sigz_oneplusz)
                    else:
                        med_sigz = np.nan
                        
                    # Use your existing plot string function
                    plotstr = make_plotstr_count(n_use, NMAD, med_sigz, bias, outl_rate * 100)
                    
                    ax.text(stats_pos[0], stats_pos[1], plotstr, fontsize=text_fontsize, 
                            color='k', transform=ax.transAxes, bbox=dict(
                            boxstyle='round,pad=0.4',
                            facecolor='white',
                            alpha=0.8,
                            edgecolor='gray',
                            linewidth=0.5
                        ))
                                        
                except NameError:
                    print(f"Warning: compute_redshift_stats or make_plotstr_count not found. Skipping statistics for {config_name}")
                    pass
            
            # Plot scatter or hexbin
            if use_hexbin:
                hb = ax.hexbin(z_true, z_est, bins='log', cmap='plasma',
                               mincnt=1, gridsize=hexbin_gridsize,
                               extent=[zmin, zmax, zmin, zmax])
                plt.colorbar(hb, ax=ax, label='log10(count)', pad=0.01)
            else:
                ax.scatter(z_true, z_est, s=point_size, color=row_colors[row_idx][col_idx], alpha=alpha)

            # Set labels
            if row_idx == 1 or col_idx==0:  # Bottom row
                ax.set_xlabel('True redshift', fontsize=axis_fontsize)

            if col_idx == 0:  # First column
                ax.set_ylabel('Estimated redshift', fontsize=axis_fontsize)
                
            ax.grid(alpha=0.3)
            ax.plot(linsp, linsp, color='grey', linestyle='dashed')
            ax.set_xlim(zmin, zmax)
            ax.set_ylim(zmin, zmax)

            if col_idx == 0:
                ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 0.5, 1.0, 1.5, 2.0])
            else:
                ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0], ['' for _ in range(5)])
        
        # All axes in 2x2 layout are used, no need to hide any
        
        # Plot histogram for this row

        
        hist_ax = axes_hist[row_idx]
        
        for i, (errors, label, color) in enumerate(zip(row_errors[row_idx], row_labels[row_idx], row_colors[row_idx])):
            if row_idx == 0:
                hist_range_row = [-0.06, 0.06]
            else:
                hist_range_row = [-0.5, 0.5]
            hist_ax.hist(errors, bins=np.linspace(hist_range_row[0], hist_range_row[1], 40),
                        label=label, color=color, histtype='step')
        
        hist_ax.axvline(0, color='grey', linestyle='dashed', linewidth=1.5)
        hist_ax.set_xlabel('$\\Delta z / (1 + z_{\\rm true})$', fontsize=axis_fontsize)
        hist_ax.set_ylabel('Counts', fontsize=axis_fontsize)
        hist_ax.set_yticks([], [])
        if row_idx==0:
            hist_ax.legend(fontsize=text_fontsize, loc=2, bbox_to_anchor=[0.0, 1.3])
    
    # Adjust spacing for scatter plot figure
    fig_scatter.subplots_adjust(hspace=0.05, wspace=0.05)
    
    # Adjust spacing for histogram figure
    fig_hist.subplots_adjust(hspace=0.3)
    
    if show_plot:
        plt.show()
    
    return fig_scatter, fig_hist

def compare_multiple_profile_likes(z_grid, profile_logL_paelist, labels, 
                                   finez, zpdf_tf, chisq_tf, ztrue,
                                   zout_tf=None, sigma_smooth=1, Z_MIN=None, Z_MAX=None,
                                   Z_MIN_BOTTOM=0.0, Z_MAX_BOTTOM=3.0,
                                   ylim=[35, 75], figsize=(8, 7), all_samples_pae=None, 
                                   max_logL_samples=None, bbox_to_anchor=[0.1, 2.7], 
                                   ypad=40, pdf_lw=3.0, legend_fs=11, colors=None,
                                   n_bins_samples=200, mcpl=None, sample_color='forestgreen', 
                                   wspace=0.1, hspace=0.15,
                                   zpdf_tf_coarse=None, chisq_tf_coarse=None,
                                   bpz_prior_pdf=None, textxpos=2.0):

    """
    Compare multiple PAE profile likelihoods against template fitting, 
    with three rows:
      1. NLL curves
      2. Smoothed PDFs
      3. Raw posterior samples histograms

    Parameters
    ----------
    z_grid : array
        Redshift grid for all PAE profile likelihoods (same grid for all runs).
    profile_logL_paelist : list of arrays
        Log-likelihood arrays for each PAE profile.
    labels : list of str
        Labels for each PAE curve.
    finez : array
        Fine redshift grid for template fitting.
    zpdf_tf : array
        Template fitting PDF.
    chisq_tf : float
        Chi-squared from template fitting.
    ztrue : float
        True redshift value.
    all_samples_pae : list of arrays, optional
        Posterior samples for each PAE run.
    max_logL_samples : list of floats, optional
        Max log-likelihood during sampling for each PAE run.
    colors : list of colors, optional
        Colors for each PAE curve.
    n_bins_samples : int
        Number of bins for the samples histogram in row 3.
    """

    # Determine Z range for top panel
    if Z_MIN is None:
        Z_MIN = np.min(z_grid)
    if Z_MAX is None:
        Z_MAX = np.max(z_grid)
    
    # Determine Z range for bottom panel (defaults to top panel range)
    if Z_MIN_BOTTOM is None:
        Z_MIN_BOTTOM = Z_MIN
    if Z_MAX_BOTTOM is None:
        Z_MAX_BOTTOM = Z_MAX

    # Normalize fine-grid TF PDF with guards against zeros/non-finite values.
    zpdf_arr = np.asarray(zpdf_tf, dtype=float)
    valid_pdf = np.isfinite(zpdf_arr) & (zpdf_arr > 0)
    if not np.any(valid_pdf):
        raise ValueError("Template-fitting PDF has no finite positive values.")

    zpdf_norm = np.where(valid_pdf, zpdf_arr, 0.0)
    zpdf_sum = np.sum(zpdf_norm)
    if not np.isfinite(zpdf_sum) or zpdf_sum <= 0:
        raise ValueError("Template-fitting PDF normalization failed (non-positive sum).")
    zpdf_norm = zpdf_norm / zpdf_sum
    zpdf_norm = np.clip(zpdf_norm, 1e-300, None)
    zpdf_norm = zpdf_norm / np.sum(zpdf_norm)

    nll_temp = -np.log(zpdf_norm)
    dnll = np.min(nll_temp) - 0.5 * chisq_tf
    nll_temp -= dnll

    # Coarse TF comparison intentionally disabled in this view.
    zpdf_norm_coarse = None
    nll_temp_coarse = None
    finez_coarse = None

    pl_linewidth = 2.5

    # Create figure with two rows (NLL and combined PDFs)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize,
                             gridspec_kw={'height_ratios':[1, 1], 'hspace':0.15})
    ax1, ax2 = axes

    # Assign colors if not provided
    if colors is None:
        colors = plt.cm.tab10.colors

    # ---------------- Row 1: Profile likelihoods (NLL) ----------------
    miny = np.inf
    for i, (profile_logL_pae, label) in enumerate(zip(profile_logL_paelist, labels)):
        conv_logL = scipy.ndimage.gaussian_filter1d(profile_logL_pae, sigma=sigma_smooth)
        
        # Make second and later curves dashed
        linestyle = 'solid' if i == 0 else 'dotted'
        
        # ax1.plot(z_grid, -profile_logL_pae, color=colors[i % len(colors)], marker='.', markersize=4, alpha=0.2, zorder=5)
        ax1.plot(z_grid, -conv_logL, color=colors[i % len(colors)], alpha=1.0, linewidth=pdf_lw, label=label, linestyle=linestyle)

        miny = min(miny, np.min(-conv_logL))
        
        # if max_logL_samples is not None and max_logL_samples[i] is not None:
            # ax1.axhline(-max_logL_samples[i], color=colors[i % len(colors)], linestyle='dashed', alpha=0.7)




    # MCPL overlay intentionally disabled: keep only profile likelihood + TF on top,
    # and MCLMC posterior samples on the lower panel.

    # Template fitting curves
    ax1.plot(finez, nll_temp, color='k', linewidth=pdf_lw, label='Template fitting', alpha=0.8)
    ax1.axvline(ztrue, color='grey', linestyle='dashed', linewidth=2,
                label=f'True $z={np.round(ztrue, 3)}$')

    ax1.axvspan(Z_MIN, Z_MIN_BOTTOM, color='lightgrey', alpha=0.2)
    ax1.axvspan(Z_MAX_BOTTOM, Z_MAX, color='lightgrey', alpha=0.2)

    # ax1.axhline(0.5 * chisq_tf, color='k', linestyle='dashed')

    ax1.set_ylabel('NLL', fontsize=14)
    ax1.set_xlim(Z_MIN, Z_MAX)
    # ax1.set_xlim(0, 1)
    # ax1.set_ylim(miny - 2, miny + ypad)
    ax1.set_ylim(ylim)
    ax1.tick_params(labelbottom=True)  # Enable x-axis tick labels on top panel

    # ax1.axvline(1.0, color='lightgrey', linestyle='solid', linewidth=1)
    ax1.grid(alpha=0.2)
    # ax1.legend(fontsize=legend_fs, ncols=2, loc=1, facecolor='white')

    # ---------------- Row 2: Combined PDFs (profile likelihoods + MCLMC samples) ----------------
    # Plot smoothed PDFs from profile likelihoods
    for i, profile_logL_pae in enumerate(profile_logL_paelist):
        profile_like = np.exp(profile_logL_pae)
        if sigma_smooth != 0:
            conv_prof_like = scipy.ndimage.gaussian_filter1d(profile_like, sigma=sigma_smooth)
            ax2.plot(z_grid, conv_prof_like / np.max(conv_prof_like), color=colors[i % len(colors)],
                     label=f"{labels[i]}", linewidth=pdf_lw, alpha=0.8)

    # Template fitting PDFs
    ax2.plot(finez, zpdf_norm / np.max(zpdf_norm), color='k', alpha=0.8,
             linewidth=pdf_lw, label='Template fitting')

    # Optional BPZ prior overlay (normalized to max=1 to match the panel scale).
    if bpz_prior_pdf is not None:
        bpz_arr = np.asarray(bpz_prior_pdf, dtype=float)
        if bpz_arr.shape == zpdf_norm.shape:
            bpz_max = np.nanmax(bpz_arr)
            if np.isfinite(bpz_max) and bpz_max > 0:
                ax2.plot(
                    finez,
                    bpz_arr / bpz_max,
                    color='firebrick',
                    alpha=0.7,
                    linewidth=1.5,
                    linestyle='--',
                    label='$p_{\\rm BPZ}(z)$',
                )
    
    # MCLMC samples histogram (overlaid with proper normalization)
    if all_samples_pae is not None and len(all_samples_pae) > 0:
        bin_edges = np.linspace(Z_MIN_BOTTOM, Z_MAX_BOTTOM, n_bins_samples + 1)
        
        for i, samples in enumerate(all_samples_pae):
            if samples is None:
                continue
            if i==0:
                label_samp = 'Posterior (PAE)'
            else:
                label_samp = None
            
            # Compute histogram counts and normalize to max=1 (same as PDFs above)
            counts, _ = np.histogram(samples, bins=bin_edges)
            counts_norm = counts / np.max(counts) if np.max(counts) > 0 else counts
            
            # Use matching color for each run's samples
            sample_color_i = colors[i % len(colors)]
            
            # Plot as histogram using step style
            ax2.hist(bin_edges[:-1], bins=bin_edges, weights=counts_norm, 
                    alpha=0.8, linewidth=1.5, color='C1', 
                    label=label_samp, histtype='step')
    
    ax2.axvline(ztrue, color='grey', linestyle='dashed', linewidth=2,
                label=f"$z_{{true}}$={np.round(ztrue, 2)}")
    
    # Add redshift estimate annotations
    text_x = float(textxpos)
    text_fontsize = 13
    
    text_lines = []
    
    # Helper function to compute 68% credible interval from PDF
    def compute_68_ci(z_arr, pdf_arr):
        """Compute 68% credible interval from a PDF."""
        pdf_norm = pdf_arr / np.sum(pdf_arr)
        cdf = np.cumsum(pdf_norm)
        # Find 16th and 84th percentiles
        z_16 = np.interp(0.16, cdf, z_arr)
        z_84 = np.interp(0.84, cdf, z_arr)
        return z_16, z_84
    
    # 1. Template fitting redshift estimates with 68% CI
    pdf_norm = zpdf_norm / np.sum(zpdf_norm)
    z_tf_est = np.sum(finez * pdf_norm)
    z_tf_16, z_tf_84 = compute_68_ci(finez, zpdf_norm)
    z_tf_upper = z_tf_84 - z_tf_est
    z_tf_lower = z_tf_est - z_tf_16
    text_lines.append((f"$z={z_tf_est:.3f}^{{+{z_tf_upper:.3f}}}_{{-{z_tf_lower:.3f}}}$", 'k'))
    
    # 2. PAE profile likelihood redshift estimates with 68% CI
    for i, (profile_logL_pae, label) in enumerate(zip(profile_logL_paelist, labels)):
        # Convert log-likelihood to posterior (unnormalized)
        posterior = np.exp(profile_logL_pae - np.max(profile_logL_pae))
        # Use PDF weighted mean instead of peak
        posterior_norm = posterior / np.sum(posterior)
        z_pae_est = np.sum(z_grid * posterior_norm)
        z_pae_16, z_pae_84 = compute_68_ci(z_grid, posterior)
        z_pae_upper = z_pae_84 - z_pae_est
        z_pae_lower = z_pae_est - z_pae_16
        color_i = colors[i % len(colors)]
        text_lines.append((f"$z={z_pae_est:.3f}^{{+{z_pae_upper:.3f}}}_{{-{z_pae_lower:.3f}}}$", color_i))
    
    # 3. PAE posterior sample redshift estimates with 68% CI (16th-84th percentiles)
    if all_samples_pae is not None and len(all_samples_pae) > 0:
        for i, samples in enumerate(all_samples_pae):
            if samples is None or len(samples) == 0:
                continue
            z_samples_est = np.median(samples)
            z_samples_16 = np.percentile(samples, 16)
            z_samples_84 = np.percentile(samples, 84)
            z_samples_upper = z_samples_84 - z_samples_est
            z_samples_lower = z_samples_est - z_samples_16
            color_i = 'C1' if i == 0 else colors[i % len(colors)]
            text_lines.append((f"$z={z_samples_est:.3f}^{{+{z_samples_upper:.3f}}}_{{-{z_samples_lower:.3f}}}$", color_i))
    
    # Add all text lines to the plot
    y_top_data = 1.18
    y_step_data = 0.23
    for i, (text, color) in enumerate(text_lines):
        ax2.text(text_x, y_top_data - i * y_step_data, text,
                fontsize=text_fontsize,
                color=color,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax2.set_xlim(Z_MIN_BOTTOM, Z_MAX_BOTTOM)
    ax2.set_ylim(0, 1.3)
    ax2.set_xlabel('redshift', fontsize=14)
    ax2.set_ylabel('p(z) (norm.)', fontsize=14)
    ax2.legend(fontsize=legend_fs, ncols=2, loc=2, bbox_to_anchor=bbox_to_anchor)
    ax2.grid(alpha=0.2)

    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    plt.show()

    return fig


def plot_umap_latent_tracks(umap_embedding_train, all_map_thetas_list, z_grid, 
                            ztrue, umap_model, labels=None, colors_list=None,
                            figsize=(7, 6), cmap_track='coolwarm', 
                            train_alpha=0.05, train_s=1, track_lw=2.0,
                            track_s=15, marker_s=80, lab_fs=14, legend_fs=10,
                            title=None, show_colorbar=True):
    """
    Plot profile likelihood latent points in 2D UMAP space.
    
    For each source/configuration, the best-fit latent vector at each redshift 
    in the profile likelihood grid is projected into the 2D UMAP space trained
    on the full training set. Points are color-coded by redshift.
    
    Parameters
    ----------
    umap_embedding_train : array, shape (N_train, 2)
        2D UMAP embedding of the training set latents.
    all_map_thetas_list : list of arrays, each shape (Nz, n_latent)
        MAP latent vectors at each redshift grid point, one per configuration.
    z_grid : array, shape (Nz,)
        Redshift grid used in the profile likelihood.
    ztrue : float
        True redshift of the source.
    umap_model : umap.UMAP
        Fitted UMAP model to transform new points.
    labels : list of str, optional
        Labels for each configuration track.
    colors_list : list of colors, optional
        Colors for each track (used for start/end markers). If None, uses tab10.
    figsize : tuple
        Figure size.
    cmap_track : str
        Colormap for color-coding the track by redshift.
    train_alpha : float
        Alpha for training set background points.
    train_s : float
        Marker size for training set background points.
    track_lw : float
        Unused (kept for backward compatibility).
    track_s : float
        Marker size for track points.
    marker_s : float
        Unused (kept for backward compatibility).
    lab_fs : int
        Font size for axis labels.
    legend_fs : int
        Font size for legend.
    title : str, optional
        Plot title.
    show_colorbar : bool
        Whether to show the redshift colorbar.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training set as grey background
    ax.scatter(umap_embedding_train[:, 0], umap_embedding_train[:, 1],
               c='lightgrey', s=train_s, alpha=train_alpha, rasterized=True)
    
    if colors_list is None:
        colors_list = plt.cm.tab10.colors
    
    if labels is None:
        labels = [f'Config {i}' for i in range(len(all_map_thetas_list))]
    
    # Color normalization for redshift
    z_norm = plt.Normalize(vmin=z_grid.min(), vmax=z_grid.max())
    cmap = plt.cm.get_cmap(cmap_track)
    
    for k, all_map_thetas in enumerate(all_map_thetas_list):
        # Project MAP thetas into UMAP space and render marker-only points.
        track_2d = umap_model.transform(all_map_thetas)
        ax.scatter(
            track_2d[:, 0],
            track_2d[:, 1],
            c=z_grid,
            cmap=cmap,
            norm=z_norm,
            marker='o',
            s=track_s,
            alpha=0.9,
            edgecolors='none',
            label=labels[k],
        )
    
    # Colorbar for redshift
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=z_norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Redshift $z$', fontsize=lab_fs)
    
    ax.set_xlabel('UMAP 1', fontsize=lab_fs)
    ax.set_ylabel('UMAP 2', fontsize=lab_fs)
    if title is not None:
        ax.set_title(title, fontsize=lab_fs)
    ax.legend(fontsize=legend_fs, loc='best')
    ax.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_bestfit_sed_overlay(wave_obs, obs_flux, obs_err, z_grid, all_bestfit_models,
                              true_redshift, n_select=10, ymax=125, ymin=-50,
                              figsize=(9, 6), colormap='RdBu_r'):
    """
    Create overlay plot of best-fit SEDs at selected redshifts with residual panel.
    
    Parameters
    ----------
    wave_obs : array
        Observed wavelengths
    obs_flux : array
        Observed fluxes (already normalized)
    obs_err : array
        Flux uncertainties (already normalized)
    z_grid : array
        Redshift grid
    all_bestfit_models : array
        Best-fit models at each redshift (shape: Nz x nbands), already normalized
    true_redshift : float
        True redshift of the source
    n_select : int
        Number of redshift points to plot
    ymax, ymin : float
        Y-axis limits for main panel
    figsize : tuple
        Figure size
    colormap : str
        Matplotlib colormap name for model curves
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    # Find index closest to true redshift
    true_z_idx = np.argmin(np.abs(z_grid - true_redshift))
    
    # Select evenly spaced redshift indices
    nz = len(z_grid)
    idxs = np.linspace(0, nz-1, min(n_select, nz), dtype=int)
    
    # Create figure with two panels
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1], sharex=ax_main)
    
    # Plot observed data (more transparent, will appear last in legend)
    ax_main.errorbar(wave_obs, obs_flux, yerr=obs_err, 
                     fmt='o', color='k', markersize=3, capsize=2, 
                     label=f'Observed ($z_{{\\rm{{true}}}}={true_redshift:.2f}$)', alpha=0.3)
    
    # Calculate and display SNR statistics
    snr_per_channel = obs_flux / obs_err
    mean_snr = np.mean(snr_per_channel)
    total_snr = np.sqrt(np.sum(snr_per_channel**2))  # Combined SNR
    
    # Add SNR text in top left corner
    textxpos = 0.02
    textypos = 0.97
    ax_main.text(textxpos, textypos, f'Total SNR: {total_snr:.1f}', 
                transform=ax_main.transAxes, fontsize=14, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85))
    ax_main.text(textxpos, textypos - 0.15, f'Mean SNR/channel: {mean_snr:.1f}', 
                transform=ax_main.transAxes, fontsize=14, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85))
    
    # Plot best-fit model at true redshift first (if not in selected indices)
    # This will appear in legend before "Observed"
    if true_z_idx not in idxs:
        bf_true = np.asarray(all_bestfit_models[true_z_idx]).ravel()
        ax_main.plot(wave_obs, bf_true, color='k', 
                     linewidth=2.0, label=f'Best fit at z={z_grid[true_z_idx]:.2f}', 
                     alpha=0.9, zorder=5, linestyle='dashed')
        # Compute chi for true-z model
        chi_true = (obs_flux - bf_true) / obs_err
        ax_resid.plot(wave_obs, chi_true, color='k', linewidth=2.0, 
                      alpha=0.9, zorder=5, linestyle='dashed')
    
    # Overlay best-fit models at selected redshifts
    colors_map = plt.cm.get_cmap(colormap)(np.linspace(0.15, 0.85, len(idxs)))
    for i, sel in enumerate(idxs):
        bf = np.asarray(all_bestfit_models[sel]).ravel()
        
        # Compute chi for this model
        chi = (obs_flux - bf) / obs_err
        
        # Highlight if this is closest to true redshift
        if sel == true_z_idx:
            ax_main.plot(wave_obs, bf, color='k', 
                         linewidth=2.0, label=f'Best fit at z={z_grid[sel]:.2f}', 
                         alpha=0.9, zorder=5, linestyle='dashed')
            ax_resid.plot(wave_obs, chi, color='k', linewidth=2.0, 
                          alpha=0.9, zorder=5, linestyle='dashed')
        else:
            ax_main.plot(wave_obs, bf, color=colors_map[i], 
                         linewidth=2.0, label=f'z={z_grid[sel]:.2f}', alpha=0.8)
            ax_resid.plot(wave_obs, chi, color=colors_map[i], linewidth=1.5, alpha=0.8)
    
    # Format main panel
    ax_main.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax_main.set_ylabel('Flux ($\\mu$Jy)', fontsize=16)
    ax_main.set_ylim(ymin, ymax)
    ax_main.legend(fontsize=11, ncol=4, loc=2, bbox_to_anchor=(-0.1, 1.30))
    ax_main.grid(alpha=0.3)
    plt.setp(ax_main.get_xticklabels(), visible=False)
    
    # Format residual panel
    ax_resid.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax_resid.set_ylabel('$\\chi$', fontsize=16)
    ax_resid.set_xlabel('$\\lambda_{\\rm obs}$ [$\\mu$m]', fontsize=16)
    ax_resid.grid(alpha=0.3)
    ax_resid.set_ylim(-4, 4)
    
    # plt.tight_layout()
    
    return fig


def plot_spec_with_incomplete_coverage(wave_obs, all_datclass, src_idx,
                                       all_spec_recon_med=None,
                                       all_spec_pct5=None,
                                       all_spec_pct16=None,
                                       all_spec_pct84=None,
                                       all_spec_pct95=None,
                                       true_redshifts=None,
                                       all_z_samples=None,
                                       figsize=(6, 8), 
                                       lab_fs=14, text_fs=12,
                                       post_color=None,
                                       labels=None, 
                                       alpha=0.3,
                                       second_col_width_ratio=0.5,
                                      dz=0.05):

    # fig, ax = plt.subplots(figsize=figsize, ncols=1, nrows=len(all_datclass), sharex=True)

    fig, ax = plt.subplots(
        figsize=figsize,
        ncols=2,
        nrows=len(all_datclass),
        gridspec_kw={'width_ratios': [1, second_col_width_ratio]}
    )
    
    for x in range(len(all_datclass)):

        srcid = all_datclass[x].srcid_obs[src_idx]
        print('srcid = ', srcid)
        spec_obs = all_datclass[x].all_spec_obs[src_idx].copy()
        flux_unc = all_datclass[x].all_flux_unc[src_idx].copy()

        spec_obs[flux_unc==np.inf] = np.nan

        norm = all_datclass[x].norms[src_idx]

        # spec_obs *= norm
        # flux_unc *= norm

        if x==0:
            ylim = [0, norm*np.nanmax(spec_obs)*1.4]

        if labels is not None:
            ax[x,0].text(0.85, ylim[1]*0.85, labels[x], fontsize=text_fs, color='C'+str(x))
        
        # ax[x].errorbar(wave_obs, spec_obs, yerr=flux_unc, fmt='o', color='C'+str(x), markersize=3, capsize=2.5)
        ax[x,0].errorbar(wave_obs, norm*spec_obs, yerr=norm*flux_unc, fmt='o', color='k', markersize=2, capsize=2.5, alpha=alpha)


        if all_spec_recon_med is not None:
            ax[x,0].plot(wave_obs, all_spec_recon_med[x][src_idx], color='C'+str(x), linestyle='solid', label='Posterior mean')

        if all_spec_pct16 is not None and all_spec_pct84 is not None:
            ax[x,0].fill_between(wave_obs, all_spec_pct16[x][src_idx], all_spec_pct84[x][src_idx], alpha=0.4, color='C'+str(x), label='68% C.I.')

        if all_spec_pct5 is not None and all_spec_pct95 is not None:
            ax[x,0].fill_between(wave_obs, all_spec_pct5[x][src_idx], all_spec_pct95[x][src_idx], alpha=0.2, color='C'+str(x), label='95% C.I.')

        ax[x,0].set_xlim(0.7, 5)
        ax[x,0].set_ylim(ylim)

        if x==len(all_datclass)-1:
            ax[x,0].set_xlabel('$\\lambda_{obs}$ [$\\mu$m]', fontsize=lab_fs)
            ax[x,0].set_xticks([1, 2, 3, 4])
            ax[x,1].set_xlabel('z', fontsize=lab_fs)

        else:
            ax[x,0].set_xticks([1, 2, 3, 4], ['' for _ in range(4)])
            ax[x,1].set_xticks([], [])
        ax[x,0].set_ylabel('Flux [$\\mu$Jy]', fontsize=lab_fs)


        if true_redshifts is not None:
            print(true_redshifts[src_idx])
            ax[x,1].axvline(true_redshifts[src_idx], color='k', linestyle='solid')

            xlim_z = [true_redshifts[src_idx]-dz, true_redshifts[src_idx]+dz]

            if all_z_samples is not None:

                med_z_est = np.median(all_z_samples[x][src_idx])

                pct84, pct16 = np.percentile(all_z_samples[x][src_idx], 84), np.percentile(all_z_samples[x][src_idx], 16)
                
                ax[x,1].hist(all_z_samples[x][src_idx].ravel(), bins=np.linspace(xlim_z[0], xlim_z[1], 40), histtype='step', color='C'+str(x))
                ax[x,1].axvline(med_z_est, color='C'+str(x), linestyle='dashed')

                textstr = '$\\hat{z}='+str(np.round(med_z_est, 3))+'^{+'+str(np.round(pct84-med_z_est, 3))+'}_{-'+str(np.round(med_z_est-pct16, 3))+'}$'

                ax[x,0].text(3.2, ylim[1]*0.65, textstr, color='C'+str(x), fontsize=text_fs)
                
                textstr_true = '$z_{true}='+str(np.round(true_redshifts[src_idx], 3))+'$'

                ax[x,0].text(3.2, ylim[1]*0.85, textstr_true, color='k', fontsize=text_fs)

            
            ax[x,1].set_xlim(true_redshifts[src_idx]-dz, true_redshifts[src_idx]+dz)
        
        ax[x,1].set_yticks([], [])
        

    plt.subplots_adjust(hspace=0.02, wspace=0.05)
    plt.show()

    return fig

def plot_fine_tune_progress(loss_hist, pct_vals=None, pct_loss_hists=None):
    # Plot training progress
    fig = plt.figure(figsize=(8, 5))
    plt.plot(loss_hist, label='Median')

    if pct_loss_hists is not None and pct_vals is not None:

        for p in range(len(pct_vals)):
            plt.plot(pct_loss_hists[p], label='Percentile '+str(pct_vals[p]))

    plt.legend()
            
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Fine-tuning Progress')
    plt.grid(True)
    plt.show()

    return fig


def plot_gr_statistic_chains(sample_fpath, burn_in=500, figsize=(8,6), bins=None):

    if bins is None:
        bins = np.linspace(0.8, 5, 40)


    floatzsamp = np.load(sample_fpath)
    all_samples = floatzsamp['all_samples']
    nparam = all_samples.shape[-1]

    all_rhat = calc_all_gr(all_samples, burn_in=burn_in)
    
    fig = plt.figure(figsize=figsize)
    
    for y in range(nparam):
        ax = plt.subplot(3, 4, y+1)  # main axis
        
        color = 'C'+str(y)
    
        if y < nparam-1:
            parlab = '$u_'+str(y+1)+'$'
        else:
            parlab = 'redshift $z$'
    
        # Histogram (main y-axis)
        counts, bins, patches = ax.hist(
            all_rhat[y],
            bins=bins,
            color=color,
            histtype='stepfilled',
            alpha=0.5
        )
    
        ax.set_title(parlab, fontsize=14)
        ax.set_xlabel('$\\hat{R}$', fontsize=14)
        ax.axvline(np.nanmedian(all_rhat[y]), color='k', linestyle='dashed')

        if y < nparam-1:
            ax.set_xticks([1, 2, 3, 4, 5])
    
            # if y < 5:
            ax.set_xlim(np.min(bins), np.max(bins))
        else:
            ax.set_xticks([1, 2, 3])
            ax.set_xlim(0.8, 3)
        # else:
            # ax.set_xlim(0.8, 3)
        # Twin axis for CDF
        ax_cdf = ax.twinx()

        if y < nparam - 1:
            ax_cdf.set_xticks([1, 2, 3, 4, 5])
    
        sorted_rhat = np.sort(all_rhat[y])
        cdf = np.arange(1, len(sorted_rhat) + 1) / len(sorted_rhat)
        ax_cdf.plot(sorted_rhat, cdf, color='k', lw=1.5)
        ax_cdf.set_ylim(0, 1)
        ax_cdf.set_ylabel('CDF', fontsize=10)
        ax_cdf.grid(alpha=0.5)
    
    
    plt.tight_layout()
    # plt.savefig('../figures/gr_statistic_uvars_and_redshift_'+tailstr+'.pdf', bbox_inches='tight')
    plt.show()

    return fig, all_rhat
    
def compare_PDFs_TF_PAE(zgrid_tf, zpdf_tf_use, all_zsamp_pae,
                        ztrue, figsize = (8, 7), nrows=2, 
                        ncols=5, lab_fs=12, legend_fs=14, 
                        return_fig=True, bbox_to_anchor=[0.0, 1.2],
                        color_pae='C0', alpha_pae=0.3,
                        zout_tf=None, dzout_tf=None,
                        zout_pae=None, dzout_pae=None, text_fs=9):

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    ax = ax.ravel()
    
    for x in range(len(ax)):


        truestr = '$z_{true}='+str(np.round(ztrue[x], 3))+'$'
        if zout_tf is not None and dzout_tf is not None:
            tfstr = '$\\hat{z}^{TF}$ $='+str(np.round(zout_tf[x], 3))+' \\pm '+str(np.round(dzout_tf[x], 3))+'$'            
        else:
            tfstr = None

        if zout_pae is not None and dzout_pae is not None:
            paestr = '$\\hat{z}^{PAE}='+str(np.round(zout_pae[x], 3))+' \\pm '+str(np.round(dzout_pae[x], 3))+'$'
        else:
            paestr = None

        zsamp_pae = all_zsamp_pae[x,:,1000:,-1].ravel()
        counts, bins, _ = ax[x].hist(zsamp_pae, color=color_pae, label='PAE', histtype='stepfilled', alpha=alpha_pae, bins=25, density=True)

        pdf_scaled = zpdf_tf_use[x]/np.sum(zpdf_tf_use[x])
        pdf_scaled *= counts.max() / pdf_scaled.max()
        
        ax[x].plot(zgrid_tf, pdf_scaled, color='k', label='Template fitting')

        ylim = [0, np.max(pdf_scaled)*2.0]
        ax[x].set_ylim(ylim)

        pae_min, pae_max = np.min(zsamp_pae), np.max(zsamp_pae)
        
        # Data range from template fitting where PDF is significant
        tf_mask = zpdf_tf_use[x] > 1e-4 * zpdf_tf_use[x].max()  # only where PDF is non-negligible
        tf_min, tf_max = zgrid_tf[tf_mask][[0, -1]]
        
        # Combined range
        z_min = min(pae_min, tf_min)
        z_max = max(pae_max, tf_max)

        # Add margin
        pad = 0.05 * (z_max - z_min)

        xlim = [max(z_min - pad, 0), z_max + pad]
        dx = xlim[1]-xlim[0]
        ax[x].set_xlim(xlim)

        bbox_use = None
        ax[x].text(xlim[0]+0.03*dx, ylim[1]*0.9, truestr, color='r', fontsize=text_fs, bbox=bbox_use)
        ax[x].text(xlim[0]+0.03*dx, ylim[1]*0.6, paestr, color='C0', fontsize=text_fs, bbox=bbox_use)
        ax[x].text(xlim[0]+0.03*dx, ylim[1]*0.75, tfstr, color='k', fontsize=text_fs, bbox=bbox_use)        
        ax[x].axvline(ztrue[x], linestyle='dashed', color='r', label='Truth', linewidth=1., alpha=0.5)
        
        if x==0:
            ax[x].legend(fontsize=legend_fs, loc=2, bbox_to_anchor=bbox_to_anchor, ncol=3)

        ax[x].set_yticks([], [])

        if x%ncols==0:
            ax[x].set_ylabel('$p(z)$', fontsize=lab_fs)

        ax[x].set_xlabel('$z$', fontsize=lab_fs)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    
    plt.show()

    if return_fig:
        return fig

def _plot_binned_data(ax, correlations, z_scores, n_bins, error_type, color, label):
    """
    Helper function to calculate and plot binned statistics.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    bin_indices = np.digitize(correlations, bins)
    
    central_values = np.zeros(n_bins)
    error_low = np.zeros(n_bins)
    error_high = np.zeros(n_bins)
    valid_bins = np.zeros(n_bins, dtype=bool)

    for j in range(1, n_bins + 1):
        mask = (bin_indices == j)
        if np.sum(mask) > 1: # Ensure at least 2 points for std/sem
            valid_bins[j-1] = True
            if error_type == 'sem':
                central_values[j-1] = np.mean(z_scores[mask])
                sem = np.std(z_scores[mask]) / np.sqrt(np.sum(mask))
                error_low[j-1] = sem
                error_high[j-1] = sem
            else: # Default to 'std' (using percentiles)
                central_values[j-1] = np.median(z_scores[mask])
                p16 = np.percentile(z_scores[mask], 16)
                p84 = np.percentile(z_scores[mask], 84)
                error_low[j-1] = central_values[j-1] - p16
                error_high[j-1] = p84 - central_values[j-1]

    # Plot the binned central values and error bars
    ax.errorbar(bin_centers[valid_bins], central_values[valid_bins],
                yerr=[error_low[valid_bins], error_high[valid_bins]],
                fmt='o-', capsize=3, color=color, linewidth=1.5, label=label)

    return ax

def plot_zscore_analysis_combined(correlations, z_scores, n_bins=10,
                                 figsize=(4, 10), ylim=[-4, 4], alpha=0.2,
                                 nbin_hist=30, error_type='std',
                                 z_scores_tf=None, z_scores_tf_color='red',
                                 use_hexbin=True, hexbin_gridsize=40):
    """
    Creates a single-column plot combining scatter/hexbin points with a binned z-score analysis.

    Args:
        correlations (np.ndarray): Array of shape (ngal, n_latent) with per-galaxy correlations.
        z_scores (np.ndarray): Array of shape (ngal,) with per-galaxy z-scores.
        n_bins (int): Number of bins to use for the binned analysis.
        error_type (str): Type of error to plot. 'std' for standard deviation (68% CI)
                          or 'sem' for standard error of the mean.
        z_scores_tf (np.ndarray, optional): A second array of z-scores to plot. Defaults to None.
        z_scores_tf_color (str): Color for the binned points of the second array.
    """
    n_latent = correlations.shape[1]

    # Create a 6x1 grid of subplots with shared x and y axes
    fig, axes = plt.subplots(nrows=n_latent+1, ncols=1, figsize=figsize,
                             sharex=True, sharey=False)

    # Plotting loop for each latent vector
    ax = axes[0]
    for i in range(n_latent):
        ax.hist(np.abs(correlations[:,i]), bins=np.linspace(0, 1, nbin_hist), histtype='step', color='C'+str(i))
        ax.grid(alpha=0.5)
        ax.set_yticks([], [])

    for i in range(n_latent):
        ax = axes[i+1]
        abs_corr = np.abs(correlations[:, i])

        # Plot the scatter/hexbin points
        # ax.scatter(abs_corr, z_scores, s=1, alpha=alpha, color='C'+str(i))
        if use_hexbin:
            hb = ax.hexbin(abs_corr, z_scores, bins='log', cmap='viridis',
                           mincnt=1, gridsize=hexbin_gridsize,
                           extent=[0, 1, ylim[0], ylim[1]])
            plt.colorbar(hb, ax=ax, label='log10(count)', pad=0.01)
        else:
            ax.scatter(abs_corr, z_scores, s=1, alpha=alpha, color='k')

        # Use the helper function to plot the binned data for the primary z-scores
        _plot_binned_data(ax, abs_corr, z_scores, n_bins, error_type,
                          color='k', label='Primary Data (binned)')

        # Use the helper function to plot the binned data for the second z-score array
        if z_scores_tf is not None:
            _plot_binned_data(ax, abs_corr, z_scores_tf, n_bins, error_type,
                              color=z_scores_tf_color, label='Second Data (binned)')

        # Plot a horizontal line at Z=0 for reference
        ax.axhline(0, color='k', linestyle='--', linewidth=1, zorder=0)

        # Set titles and limits
        # ax.text(0.75, -3., '$|\\rho(z, u_'+str(i+1)+')|$', fontsize=14, color='C'+str(i))
        ax.text(0.75, -3., '$|\\rho(z, u_'+str(i+1)+')|$', fontsize=14, color='k')

        ax.set_ylim(ylim)
        ax.grid(alpha=0.5)
        ax.set_xlim(0, 1)
        # ax.legend(loc='lower left', fontsize=8) # Add a legend for the binned lines

        # Set the x-axis label only for the last plot
        if i == n_latent - 1:
            ax.set_xlabel('$|\\rho(z, u)|$', fontsize=16)
            ax.xaxis.set_tick_params(labelbottom=True)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

    # Set the common y-axis label
    fig.text(-0.02, 0.45, r'Redshift Z-score $= (\hat{z} - z_{true})/\hat{\sigma}_z$',
              va='center', rotation='vertical', fontsize=18)

    plt.subplots_adjust(hspace=0.1)
    plt.show()

    return fig


def compare_chi2_tf_pae(chi2_tf, chi2_pae, nbands=102, nlatent=5, figsize=(5, 4), chi2lim=[0.5, 2.0]):
    fig = plt.figure(figsize=(5, 4))
    plt.scatter(chi2_tf/(nbands-nlatent), chi2_pae, color='k', s=2)
    plt.xlabel('$\\chi^2_{red}$ (TF)', fontsize=14)
    plt.ylabel('$\\chi^2_{red}$ (PAE)', fontsize=14)
    plt.xlim(chi2lim)
    plt.ylim(chi2lim)
    plt.grid(alpha=0.5)
    linsp = np.linspace(chi2lim[0], chi2lim[1], 50)
    plt.plot(linsp, linsp, color='r', linestyle='dashed')
    plt.show()

    return fig

def create_and_evaluate_custom_colormap(colormap_name, data_min, data_max):
    """
    Creates a custom colormap from a specified built-in colormap,
    scaled to a custom data range, and returns a callable function
    to get colors for a set of numbers.

    Args:
        colormap_name (str): The name of the built-in colormap (e.g., 'magma').
        data_min (float): The minimum value of the data range.
        data_max (float): The maximum value of the data range.

    Returns:
        callable: A function that takes a number or a list of numbers
                  and returns their corresponding RGBA colors from the
                  scaled colormap.
    """
    # 1. Get the original colormap object
    original_cmap = cm.get_cmap(colormap_name)

    # 2. Create a normalization object that maps the data range to [0, 1]
    # This is the key step. The Normalize object will take a value in
    # the range [data_min, data_max] and return a value in the range [0, 1].
    normalize = colors.Normalize(vmin=data_min, vmax=data_max)
    
    # 3. Create a function that combines the normalization and the colormap
    # The returned function will take a number, normalize it, and then
    # use that normalized value to get a color from the original colormap.
    def get_color_from_value(value):
        # Normalize the value
        normed_value = normalize(value)
        # Get the color from the colormap
        return original_cmap(normed_value)

    return get_color_from_value

def compare_pae_tf_redshifts(all_med_z, all_err_low, all_err_high, redshifts_true,
                             z_min=0, z_max=3.0, sig_z_oneplusz_max=0.2, sig_z_oneplusz_min=None, color_val=None, val_label='$\\log_{10}(\\sigma_{z/1+z})$',
                             figsize=(7, 4), alpha=0.1, cmap='magma', vmin=None, vmax=None, lab_fs=16,
                             redshift_stats=True, textxpos=0.1, textypos=1.7, text_fs=12, persqdeg_fac=1./1.27,
                             ylabels=['$\\hat{z}_{TF}$', '$\\hat{z}_{PAE}$'], xlabel='$z_{true}$', logmin=-2.5, logmax=-0.5,
                             use_hexbin=True, gridsize=50, show_15pct_outlier=False):
    
    linsp = np.linspace(z_min, z_max, 100)
    
    norm = colors.Normalize(vmin=logmin, vmax=logmax)
    original_cmap = cm.get_cmap(cmap)
    magma_scaled = create_and_evaluate_custom_colormap('magma', logmin, logmax)
    
    # Create two main subplots of equal sizef
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharey=True)

    # Calculate colorbar position based on figure size
    # This is a robust way to ensure equal subplot sizes and
    # place a colorbar on the side
    cbar_width = 0.03 # Fraction of the figure width
    cbar_pad = 0.02   # Padding between subplots and colorbar
    
    # Get the position of the rightmost subplot
    pos = ax[1].get_position()
    
    # Define the position for the new colorbar axes
    cax_pos = [pos.x1 + cbar_pad, pos.y0, cbar_width, pos.height]
    cax = fig.add_axes(cax_pos)

    # Loop through the panels
    for x in range(2):
        # Handle case where redshifts_true might be a list of arrays (one per method)
        if isinstance(redshifts_true, list):
            truez_for_method = redshifts_true[x]
        else:
            truez_for_method = redshifts_true
            
        sigz_oneplusz = 0.5 * (all_err_high[x] + all_err_low[x]) / (1 + all_med_z[x])
        sigz_mask = (sigz_oneplusz < sig_z_oneplusz_max)

        if sig_z_oneplusz_min is not None:
            sigz_mask *= (sigz_oneplusz > sig_z_oneplusz_min)
            
        med_z_use, err_low_use, err_high_use, truez_use, sigz_oneplusz_use = \
            all_med_z[x][sigz_mask], all_err_low[x][sigz_mask], all_err_high[x][sigz_mask], truez_for_method[sigz_mask], sigz_oneplusz[sigz_mask]
        
        color_val_use = magma_scaled(np.log10(sigz_oneplusz_use))
        
        if redshift_stats and len(med_z_use) > 0:
            arg_bias, arg_std, bias, NMAD, cond_outl, outl_rate, outl_rate_15pct = compute_redshift_stats(
                med_z_use, truez_use, sigma_z_select=sigz_oneplusz_use, nsig_outlier=3)
            n_sources = len(med_z_use)
            med_sigz = np.median(sigz_oneplusz_use)
            # Use 3-sigma outlier rate by default
            plotstr = make_plotstr_count(n_sources, NMAD, med_sigz, bias, outl_rate * 100)
            
            # Optionally add 15% outlier fraction line below
            if show_15pct_outlier:
                plotstr += f'\n$f_{{outlier}}^{{15\\%}}={outl_rate_15pct*100:.2f}\\%$'
        elif redshift_stats:
            plotstr = "No sources in bin"
        else:
            plotstr = ""
            
        ax[x].text(textxpos, textypos, plotstr, fontsize=text_fs, color='k',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5))
        ax[x].plot(linsp, linsp, color='k', linestyle='dashed', zorder=10, alpha=0.5)
        
        if use_hexbin:
            # Use hexbin with log color scale showing counts
            hb = ax[x].hexbin(truez_use, med_z_use, gridsize=gridsize, cmap='magma', 
                             mincnt=1, bins='log', extent=[z_min, z_max, z_min, z_max])
            sc = hb  # For consistency with scatter version
        else:
            sc = ax[x].scatter(truez_use, med_z_use, c=color_val_use, s=8, edgecolor='None', alpha=0.8)

        ax[x].set_xlim(z_min, z_max)
        ax[x].set_ylim(z_min, z_max)
        ax[x].set_xlabel(xlabel, fontsize=lab_fs)
        ax[x].set_ylabel(ylabels[x], fontsize=lab_fs)
        ax[x].grid(alpha=0.5)
        
        # Add title showing selection criteria
        method_label = 'TF' if x == 0 else 'PAE'
        if sig_z_oneplusz_min is not None:
            title = f"${sig_z_oneplusz_min:.2g}<\\sigma_{{z/(1+z)}}^{{{method_label}}}<{sig_z_oneplusz_max:.2g}$"
        else:
            title = f"$\\sigma_{{z/(1+z)}}^{{{method_label}}}<{sig_z_oneplusz_max:.2g}$"
        ax[x].set_title(title, fontsize=lab_fs)
    
    # Hide the y-axis label for the right plot to avoid repetition
    plt.setp(ax[1].get_yticklabels(), visible=False)

    # 3. Create the colorbar in the new cax
    if use_hexbin:
        # For hexbin, use the hexbin object's colorbar (log counts)
        cbar = fig.colorbar(sc, cax=cax)
        # cbar.set_label('Counts (log scale)', fontsize=16)
    else:
        # For scatter, use the custom colormap
        mappable = cm.ScalarMappable(norm=norm, cmap=original_cmap)
        mappable.set_array([]) 
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(val_label, fontsize=16)

    plt.show()
    return fig


def compare_pae_tf_redshifts_multisel(
    all_med_z, all_err_low, all_err_high, redshifts_true,
    sig_z_thresholds=(0.01, 0.03, 0.1, 0.2),
    z_min=0.0, z_max=2.5,
    figsize=(12, 5),
    gridsize=50,
    lab_fs=13, text_fs=9,
    textxpos=0.05, textypos=0.93,
    row_labels=None,
    show_15pct_outlier=False,
):
    """
    2 × 4 hexbin comparison figure.

    Row 0 : Template-fitting (TF) results
    Row 1 : PAE results

    Columns correspond to σ_z/(1+z) < threshold for thresholds in
    sig_z_thresholds (default [0.01, 0.03, 0.1, 0.2]).

    All panels share the same z range [z_min, z_max] (default 0–2.5) and the
    same hexbin gridsize, so cell sizes are consistent across columns.

    Each panel title shows the selection as
    σ_{z/(1+z)}^{TF} < threshold  (top row)
    σ_{z/(1+z)}^{PAE} < threshold  (bottom row).

    row_labels : list of method name strings for y-axis / title labels,
                 default ['TF', 'PAE'].

    Parameters
    ----------
    all_med_z : list of two arrays [med_z_tf, med_z_pae]
    all_err_low, all_err_high : same structure
    redshifts_true : array or list of two arrays
    """
    n_cols = len(sig_z_thresholds)

    if row_labels is None:
        row_labels = ['TF', 'PAE']

    n_rows = len(row_labels)

    linsp = np.linspace(z_min, z_max, 100)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize)
    fig.subplots_adjust(hspace=0.08, wspace=0.10, top=0.97, bottom=0.10,
                        left=0.07, right=0.88)

    hb_last = None
    for col, threshold in enumerate(sig_z_thresholds):
        for row in range(n_rows):
            ax = axes[row, col]

            # True redshifts for this method
            if isinstance(redshifts_true, list):
                zt = redshifts_true[row]
            else:
                zt = redshifts_true

            sigz  = 0.5 * (all_err_high[row] + all_err_low[row]) / (1.0 + all_med_z[row])
            m = (sigz < threshold) & np.isfinite(zt) & np.isfinite(all_med_z[row]) & np.isfinite(sigz)

            z_hat = all_med_z[row][m]
            zt_m  = zt[m]
            sz_m  = sigz[m]

            ax.plot(linsp, linsp, 'k--', lw=1, alpha=0.5, zorder=10)
            ax.set_xlim(z_min, z_max)
            ax.set_ylim(z_min, z_max)
            ax.grid(alpha=0.3)

            if len(z_hat) > 0:
                hb = ax.hexbin(zt_m, z_hat, gridsize=gridsize, cmap='magma',
                               mincnt=1, bins='log',
                               extent=[z_min, z_max, z_min, z_max])
                hb_last = hb

                _, _, bias, NMAD, _, outl_rate, outl_rate_15pct = \
                    compute_redshift_stats(z_hat, zt_m, sigma_z_select=sz_m, nsig_outlier=3)
                med_sigz = np.median(sz_m)
                n_src = int(m.sum())
                stat_str = make_plotstr_count(n_src, NMAD, med_sigz, bias, outl_rate * 100)
                if show_15pct_outlier:
                    stat_str += f'\n$f_{{outlier}}^{{15\%}}={outl_rate_15pct*100:.2f}\%$'
                ax.text(textxpos, textypos, stat_str, fontsize=9,
                        transform=ax.transAxes, va='top',
                        bbox=dict(boxstyle='round', facecolor='white',
                                  alpha=0.85, edgecolor='gray', lw=0.5))
            else:
                ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                        transform=ax.transAxes, fontsize=text_fs)

            # Per-panel annotation in bottom-right corner (beige background)
            method = row_labels[row]
            sel_label = r'$\sigma_{z/(1+z)}^{\rm ' + method + r'}<' + f'{threshold}$'
            ax.text(0.97, 0.04, sel_label, fontsize=text_fs,
                    transform=ax.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='cornsilk',
                              alpha=0.95, edgecolor='gray', lw=0.5))

            # y-axis label on leftmost column only
            if col == 0:
                ax.set_ylabel(f'$\\hat{{z}}_{{\\rm {method}}}$', fontsize=lab_fs)
            else:
                ax.tick_params(labelleft=False)

            # x-axis label on bottom row only
            if row == n_rows - 1:
                ax.set_xlabel(r'$z_{\rm true}$', fontsize=lab_fs)
            else:
                ax.tick_params(labelbottom=False)

    # Single shared colorbar at right of figure
    if hb_last is not None:
        cbar_ax = fig.add_axes([0.90, 0.10, 0.015, 0.87])
        fig.colorbar(hb_last, cax=cbar_ax, label='log$_{10}$(counts)')

    return fig


def plot_cross_selection_zin_zout(
    z_true,
    z_out_pae, err_low_pae, err_high_pae,
    z_out_tf, err_tf,
    sigma_threshold=0.2,
    z_min=0.0, z_max=2.5,
    figsize=(8, 8),
    use_hexbin=False,
    gridsize=40,
    lab_fs=16, title_fs=18, row_label_fs=13, text_fs=12,
    textxpos=0.05, textypos=0.92,
    logmin=-2.5, logmax=-0.5,
    xlabel='$z_{\\rm true}$',
    show_15pct_outlier=False,
):
    """
    Two-row, two-column figure for cross-selected zin/zout comparisons.

    Row 0 — sources where TF is precise but PAE is not
        (σ^TF/(1+z) < threshold  AND  σ^PAE/(1+z) > threshold)
    Row 1 — sources where PAE is precise but TF is not
        (σ^PAE/(1+z) < threshold  AND  σ^TF/(1+z) > threshold)

    Columns:  [TF z_out vs z_true,  PAE z_out vs z_true]

    When use_hexbin=True  : cells are coloured by log10(counts) with a shared
                            magma colorbar on the right.
    When use_hexbin=False : points are coloured by log10(σ_z/(1+z)) of the
                            plotted estimator, with a shared magma colorbar on
                            the right labelled accordingly.

    Parameters
    ----------
    z_true : array (N,)
    z_out_pae, err_low_pae, err_high_pae : PAE median redshifts and asymmetric 1-σ uncertainties.
    z_out_tf, err_tf : TF median redshifts and symmetric 1-σ uncertainty.
    sigma_threshold : float
        σ/(1+z) boundary used for both selections (default 0.2).
    """
    # Fractional uncertainties
    sigz_pae = 0.5 * (err_low_pae + err_high_pae) / (1.0 + z_out_pae)
    sigz_tf  = err_tf / (1.0 + z_out_tf)

    # Validity mask: both estimators must have finite z and uncertainty
    valid = (np.isfinite(z_true) & np.isfinite(z_out_pae) &
             np.isfinite(z_out_tf) & np.isfinite(sigz_pae) & np.isfinite(sigz_tf))

    # Cross-selection masks
    mask_tf_good_pae_poor = valid & (sigz_tf  < sigma_threshold) & (sigz_pae >= sigma_threshold)
    mask_pae_good_tf_poor = valid & (sigz_pae < sigma_threshold) & (sigz_tf  >= sigma_threshold)

    row_labels = [
        r'$\sigma_{z/(1+z)}^{\rm TF}<' + f'{sigma_threshold}' + r',\;\sigma_{z/(1+z)}^{\rm PAE}\geq' + f'{sigma_threshold}$',
        r'$\sigma_{z/(1+z)}^{\rm PAE}<' + f'{sigma_threshold}' + r',\;\sigma_{z/(1+z)}^{\rm TF}\geq' + f'{sigma_threshold}$',
    ]
    masks      = [mask_tf_good_pae_poor, mask_pae_good_tf_poor]
    col_zout   = [[z_out_tf,  z_out_pae], [z_out_tf,  z_out_pae]]
    col_sigz   = [[sigz_tf,   sigz_pae],  [sigz_tf,   sigz_pae]]
    col_ylabels = [[r'$\hat{z}_{\rm TF}$', r'$\hat{z}_{\rm PAE}$'],
                   [r'$\hat{z}_{\rm TF}$', r'$\hat{z}_{\rm PAE}$']]
    col_names  = [['TF', 'PAE'], ['TF', 'PAE']]

    cmap_obj = cm.get_cmap('magma')
    norm_obj = colors.Normalize(vmin=logmin, vmax=logmax)
    linsp    = np.linspace(z_min, z_max, 100)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize,
                             sharex=True, sharey=True)
    fig.subplots_adjust(top=0.86, bottom=0.08, left=0.10, right=0.97,
                        hspace=0.18, wspace=0.30)

    # Shared horizontal colorbar at the top
    cax = fig.add_axes([0.15, 0.91, 0.70, 0.025])

    last_mappable = None  # track the last drawable for colorbar

    for row in range(2):
        m = masks[row]
        for col in range(2):
            ax = axes[row, col]
            z_hat  = col_zout[row][col][m]
            sz     = col_sigz[row][col][m]
            zt     = z_true[m]
            ylabel = col_ylabels[row][col]
            method = col_names[row][col]

            ax.plot(linsp, linsp, 'k--', lw=1, alpha=0.5, zorder=10)
            ax.set_xlim(z_min, z_max)
            ax.set_ylim(z_min, z_max)
            ax.grid(alpha=0.4)

            n_src = len(z_hat)
            if n_src == 0:
                ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                        transform=ax.transAxes, fontsize=text_fs)
            else:
                if use_hexbin:
                    hb = ax.hexbin(zt, z_hat, gridsize=gridsize, cmap='magma',
                                   mincnt=1, bins='log',
                                   extent=[z_min, z_max, z_min, z_max])
                    last_mappable = hb
                else:
                    log_sz = np.log10(np.clip(sz, 10**logmin, 10**logmax))
                    sc = ax.scatter(zt, z_hat,
                                    c=log_sz, cmap='magma',
                                    vmin=logmin, vmax=logmax,
                                    s=6, edgecolor='none', alpha=0.7)
                    last_mappable = sc

                # Statistics
                arg_bias, arg_std, bias, NMAD, cond_outl, outl_rate, outl_rate_15pct = \
                    compute_redshift_stats(z_hat, zt, sigma_z_select=sz, nsig_outlier=3)
                med_sigz = np.median(sz)
                stat_str = make_plotstr_count(n_src, NMAD, med_sigz, bias, outl_rate * 100)
                if show_15pct_outlier:
                    stat_str += f'\n$f_{{outlier}}^{{15\\%}}={outl_rate_15pct*100:.2f}\\%$'
                ax.text(textxpos, textypos, stat_str, fontsize=text_fs,
                        transform=ax.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white',
                                  alpha=0.85, edgecolor='gray', lw=0.5),
                        va='top')

            ax.set_ylabel(ylabel, fontsize=lab_fs)
            if row == 0:
                ax.set_title(method, fontsize=title_fs)
            if row == 1:
                ax.set_xlabel(xlabel, fontsize=lab_fs)

        # Row label on the right (outside the subplots, before the colorbar)
        axes[row, 1].annotate(
            row_labels[row],
            xy=(1.18, 0.5), xycoords='axes fraction',
            rotation=270, va='center', ha='left', fontsize=row_label_fs,
        )

    # Shared horizontal colorbar at the top
    if last_mappable is not None:
        cbar = fig.colorbar(last_mappable, cax=cax, orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        if use_hexbin:
            cbar.set_label('log$_{10}$(counts)', fontsize=lab_fs - 2, labelpad=8)
        else:
            cbar.set_label(r'$\log_{10}(\sigma_{z/(1+z)})$', fontsize=lab_fs - 2, labelpad=8)

    return fig


def plot_sigz_binned_zin_zout(
    z_true,
    z_out_pae, sigz_pae,
    z_out_tf, sigz_tf,
    sigz_bin_edges=None,
    z_min=0.0, z_max=2.5,
    figsize=None,
    use_hexbin=False,
    gridsize=30,
    lab_fs=13, title_fs=12, text_fs=9,
    textxpos=0.04, textypos=0.97,
    logmin=-2.5, logmax=-0.5,
    xlabel='$z_{\\rm true}$',
    show_15pct_outlier=False,
):
    """
    Two-row (TF top, PAE bottom), N-column (one per sigma bin) zin/zout grid.

    sigz_bin_edges : sequence of N+1 edges; default [0, 0.003, 0.01, 0.03, 0.1, 0.2].
    Sources outside the last edge are excluded from all panels.
    """
    if sigz_bin_edges is None:
        sigz_bin_edges = [0, 0.003, 0.01, 0.03, 0.1, 0.2]
    sigz_bin_edges = np.asarray(sigz_bin_edges)
    n_bins = len(sigz_bin_edges) - 1

    if figsize is None:
        figsize = (3.5 * n_bins, 7)

    cmap_obj = cm.get_cmap('magma')
    norm_obj = colors.Normalize(vmin=logmin, vmax=logmax)
    linsp    = np.linspace(z_min, z_max, 100)

    fig, axes = plt.subplots(nrows=2, ncols=n_bins, figsize=figsize,
                             sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.10, wspace=0.12)

    row_data = [
        (z_out_tf,  sigz_tf,  r'$\hat{z}_{\rm TF}$',  'TF'),
        (z_out_pae, sigz_pae, r'$\hat{z}_{\rm PAE}$', 'PAE'),
    ]

    for row, (z_out, sigz, ylabel, method) in enumerate(row_data):
        valid = (np.isfinite(z_true) & np.isfinite(z_out) & np.isfinite(sigz))
        for col in range(n_bins):
            lo, hi = sigz_bin_edges[col], sigz_bin_edges[col + 1]
            m = valid & (sigz >= lo) & (sigz < hi)

            ax = axes[row, col]
            ax.plot(linsp, linsp, 'k--', lw=1, alpha=0.5, zorder=10)
            ax.set_xlim(z_min, z_max)
            ax.set_ylim(z_min, z_max)
            ax.grid(alpha=0.3)

            n_src = int(m.sum())
            z_hat = z_out[m]
            sz    = sigz[m]
            zt    = z_true[m]

            if n_src == 0:
                ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                        transform=ax.transAxes, fontsize=text_fs)
            else:
                if use_hexbin:
                    ax.hexbin(zt, z_hat, gridsize=gridsize, cmap='magma',
                              mincnt=1, bins='log',
                              extent=[z_min, z_max, z_min, z_max])
                else:
                    log_sz = np.log10(np.clip(sz, 10**logmin, 10**logmax))
                    ax.scatter(zt, z_hat, c=log_sz, cmap='magma',
                               vmin=logmin, vmax=logmax,
                               s=4, edgecolor='none', alpha=0.6)

                arg_bias, arg_std, bias, NMAD, cond_outl, outl_rate, outl_rate_15pct = \
                    compute_redshift_stats(z_hat, zt, sigma_z_select=sz, nsig_outlier=3)
                med_sigz = np.median(sz)
                stat_str = make_plotstr_count(n_src, NMAD, med_sigz, bias, outl_rate * 100)
                ax.text(textxpos, textypos, stat_str, fontsize=text_fs,
                        transform=ax.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white',
                                  alpha=0.85, edgecolor='gray', lw=0.5),
                        va='top')

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=lab_fs)
            if row == 1:
                ax.set_xlabel(xlabel, fontsize=lab_fs - 1)

            # Bin label as title on row 0 only
            if row == 0:
                if lo == 0:
                    bin_str = r'$\sigma_{z/(1+z)}<' + f'{hi}$'
                else:
                    bin_str = f'${lo}' + r'\leq\sigma_{z/(1+z)}<' + f'{hi}$'
                ax.set_title(bin_str, fontsize=title_fs)

        # Method label on the left
        axes[row, 0].annotate(
            method, xy=(-0.28, 0.5), xycoords='axes fraction',
            rotation=90, va='center', ha='center', fontsize=lab_fs,
            fontweight='bold',
        )

    fig.tight_layout(rect=[0, 0, 1, 1])
    return fig


def plot_cross_selection_by_sigz_bin(
    z_true,
    z_out_pae, sigz_pae,
    z_out_tf,  sigz_tf,
    sigz_bin_edges=None,
    z_min=0.0, z_max=2.5,
    figsize=None,
    use_hexbin=False,
    gridsize=30,
    lab_fs=15, title_fs=13, text_fs=9,
    textxpos=0.04, textypos=0.97,
    logmin=-3.0, logmax=-0.5,
    sigma_ratio_threshold=None,
    sigma_fixed_threshold=None,
):
    """
    N-row × 2-column cross-selection grid, one row per TF sigma_z bin.

    Left column  ("Full sample"):   all TF sources in that sigma_z bin
                                    → sigz_tf in [lo, hi]
    Right column ("With PAE selection"): TF sources in that bin that PAE
                 would NOT have placed there.
                 Default (both thresholds=None):
                     sigz_tf in [lo, hi]  AND  sigz_pae >= hi
                 Ratio mode (sigma_ratio_threshold=<float>, e.g. 1.5):
                     sigz_tf in [lo, hi]  AND  sigz_pae > sigma_ratio_threshold * sigz_tf
                 Fixed mode (sigma_fixed_threshold=<float>, e.g. 0.2):
                     sigz_tf in [lo, hi]  AND  sigz_pae > sigma_fixed_threshold
                     (takes precedence over ratio mode if both are set)

    Column headers: "Full sample" / "With PAE selection".
    Lower-right annotation in each panel shows the exact selection criteria.

    sigz_bin_edges : array-like, default [0, 0.03, 0.1, 0.2]
        N+1 edges → N rows.
    sigma_ratio_threshold : float or None
        When set, use ratio-based cross-selection for the right column:
        sigz_pae > sigma_ratio_threshold * sigz_tf.
    sigma_fixed_threshold : float or None
        When set, use fixed-threshold cross-selection for the right column:
        sigz_pae > sigma_fixed_threshold.
        Takes precedence over sigma_ratio_threshold if both are set.
    """
    if sigz_bin_edges is None:
        sigz_bin_edges = [0, 0.03, 0.1, 0.2]
    sigz_bin_edges = np.asarray(sigz_bin_edges)
    n_bins = len(sigz_bin_edges) - 1

    if figsize is None:
        figsize = (6, 2.5 * n_bins)

    linsp = np.linspace(z_min, z_max, 100)
    valid = (np.isfinite(z_true) & np.isfinite(z_out_pae) &
             np.isfinite(z_out_tf) & np.isfinite(sigz_pae) & np.isfinite(sigz_tf))

    fig, axes = plt.subplots(nrows=n_bins, ncols=2, figsize=figsize,
                             sharex=True, sharey=True)
    if n_bins == 1:
        axes = axes[np.newaxis, :]
    fig.subplots_adjust(hspace=0.06, wspace=0.05, top=0.93, bottom=0.09)

    # Column headers on the top row only
    axes[0, 0].set_title('Full sample', fontsize=title_fs)
    axes[0, 1].set_title('With PAE selection', fontsize=title_fs)

    for row in range(n_bins):
        lo, hi = sigz_bin_edges[row], sigz_bin_edges[row + 1]

        # Left column: all TF sources in bin (no PAE condition)
        m_all = valid & (sigz_tf >= lo) & (sigz_tf < hi)
        # Right column: TF in bin, PAE worse — boundary, ratio, or fixed mode
        if sigma_fixed_threshold is not None:
            m_cross = m_all & (sigz_pae > sigma_fixed_threshold)
        elif sigma_ratio_threshold is not None:
            m_cross = m_all & (sigz_pae > sigma_ratio_threshold * sigz_tf)
        else:
            m_cross = m_all & (sigz_pae >= hi)

        for col, m in enumerate((m_all, m_cross)):
            ax = axes[row, col]
            ax.plot(linsp, linsp, 'k--', lw=1, alpha=0.5, zorder=10)
            ax.set_xlim(z_min, z_max)
            ax.set_ylim(z_min, z_max)
            ax.grid(alpha=0.3)

            n_src = int(m.sum())
            if n_src == 0:
                ax.text(0.5, 0.5, 'No sources', ha='center', va='center',
                        transform=ax.transAxes, fontsize=text_fs)
            else:
                z_hat = z_out_tf[m]
                sz    = sigz_tf[m]
                zt    = z_true[m]

                if use_hexbin:
                    ax.hexbin(zt, z_hat, gridsize=gridsize, cmap='magma',
                              mincnt=1, bins='log',
                              extent=[z_min, z_max, z_min, z_max])
                else:
                    log_sz = np.log10(np.clip(sz, 10**logmin, 10**logmax))
                    ax.scatter(zt, z_hat, c=log_sz, cmap='magma',
                               vmin=logmin, vmax=logmax,
                               s=4, edgecolor='none', alpha=0.7)

                _, _, bias, NMAD, _, outl_rate, _ = \
                    compute_redshift_stats(z_hat, zt, sigma_z_select=sz, nsig_outlier=3)
                med_sigz = np.median(sz)
                stat_str = make_plotstr_count(n_src, NMAD, med_sigz, bias, outl_rate * 100)
                # Stats in upper left
                ax.text(textxpos, textypos, stat_str, fontsize=text_fs,
                        transform=ax.transAxes,
                        bbox=dict(boxstyle='round', facecolor='white',
                                  alpha=0.85, edgecolor='gray', lw=0.5),
                        va='top')

            # Selection label in lower right — always includes the bin range
            if lo == 0:
                bin_str = r'$\sigma_{\rm TF}<' + f'{hi}$'
            else:
                bin_str = f'${lo}' + r'\leq\sigma_{\rm TF}<' + f'{hi}$'

            if col == 0:
                sel_str = bin_str
            else:
                if sigma_fixed_threshold is not None:
                    sel_str = (bin_str + '\n'
                               + r'AND $\sigma_{z}^{\rm PAE} > '
                               + f'{sigma_fixed_threshold}$')
                elif sigma_ratio_threshold is not None:
                    sel_str = (bin_str + '\n'
                               + r'AND $\sigma_{z}^{\rm PAE} > '
                               + f'{sigma_ratio_threshold}'
                               + r'\,\sigma_{z}^{\rm TF}$')
                else:
                    sel_str = bin_str + '\n' + r'AND $\sigma_{\rm PAE}\geq ' + f'{hi}$'

            ax.text(0.97, 0.04, sel_str, fontsize=11,
                    transform=ax.transAxes, ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white',
                              alpha=0.80, edgecolor='gray', lw=0.4))

            if col == 0:
                ax.set_ylabel(r'$\hat{z}_{\rm TF}$', fontsize=lab_fs)
            if row == n_bins - 1:
                ax.set_xlabel(r'$z_{\rm true}$', fontsize=lab_fs)

    return fig


def plot_bias_vs_ztrue_by_sigz_bins(
    z_true_pae, z_out_pae, sigz_pae,
    z_true_tf,  z_out_tf,  sigz_tf,
    sigz_bin_edges=None,
    z_min=0.0, z_max=2.5,
    n_ztrue_bins=12,
    figsize=(11, 4.5),
    lab_fs=14, legend_fs=10, text_fs=11,
    colors_list=None,
    linestyles=None,
    show_nmad=True,
):
    """
    Two-panel figure: left = TF bias vs z_true per sigma bin,
                      right = PAE bias vs z_true per sigma bin.

    Bias defined as median[ (z_out - z_true) / (1 + z_true) ] in z_true bins.
    One curve per sigma bin; shaded band = 16th-84th percentile of the bias
    distribution in each z_true bin.

    sigz_bin_edges : sequence of N+1 edges; default [0, 0.003, 0.01, 0.03, 0.1, 0.2].
    """
    if sigz_bin_edges is None:
        sigz_bin_edges = [0, 0.003, 0.01, 0.03, 0.1, 0.2]
    sigz_bin_edges = np.asarray(sigz_bin_edges)
    n_sigma_bins   = len(sigz_bin_edges) - 1

    if colors_list is None:
        colors_list = plt.cm.viridis(np.linspace(0.1, 0.9, n_sigma_bins))
    if linestyles is None:
        linestyles = ['-'] * n_sigma_bins

    ztrue_edges   = np.linspace(z_min, z_max, n_ztrue_bins + 1)
    ztrue_centers = 0.5 * (ztrue_edges[:-1] + ztrue_edges[1:])

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=figsize, sharey=True)
    method_data = [
        (z_true_tf,  z_out_tf,  sigz_tf,  'TF'),
        (z_true_pae, z_out_pae, sigz_pae, 'PAE'),
    ]

    for ax, (z_true, z_out, sigz, method) in zip(axes, method_data):
        valid = np.isfinite(z_true) & np.isfinite(z_out) & np.isfinite(sigz)
        bias_all = (z_out - z_true) / (1.0 + z_true)

        for k in range(n_sigma_bins):
            lo, hi = sigz_bin_edges[k], sigz_bin_edges[k + 1]
            m = valid & (sigz >= lo) & (sigz < hi)

            med_bias = np.full(n_ztrue_bins, np.nan)
            pct16    = np.full(n_ztrue_bins, np.nan)
            pct84    = np.full(n_ztrue_bins, np.nan)
            n_in_bin = np.zeros(n_ztrue_bins, dtype=int)

            for j in range(n_ztrue_bins):
                zlo, zhi = ztrue_edges[j], ztrue_edges[j + 1]
                in_zbin = m & (z_true >= zlo) & (z_true < zhi)
                n_in_bin[j] = in_zbin.sum()
                if n_in_bin[j] >= 5:
                    b = bias_all[in_zbin]
                    med_bias[j] = np.median(b)
                    pct16[j]    = np.percentile(b, 16)
                    pct84[j]    = np.percentile(b, 84)

            # Build label
            if lo == 0:
                lbl = r'$\sigma<' + f'{hi}$  (N={m.sum()})'
            else:
                lbl = f'${lo}' + r'\leq\sigma<' + f'{hi}$  (N={m.sum()})'

            good = np.isfinite(med_bias)
            if not good.any():
                continue

            ax.plot(ztrue_centers[good], med_bias[good],
                    color=colors_list[k], ls=linestyles[k], lw=1.8, label=lbl)
            ax.fill_between(ztrue_centers[good], pct16[good], pct84[good],
                            color=colors_list[k], alpha=0.15)

        ax.axhline(0, color='k', lw=1, ls='--', alpha=0.6)
        ax.set_xlim(z_min, z_max)
        ax.set_xlabel('$z_{\\rm true}$', fontsize=lab_fs)
        ax.set_title(method, fontsize=lab_fs + 1)
        ax.grid(alpha=0.35)
        ax.legend(fontsize=legend_fs, loc='lower right', framealpha=0.85)

    axes[0].set_ylabel(r'Median bias $(\hat{z}-z_{\rm true})/(1+z_{\rm true})$',
                       fontsize=lab_fs)
    fig.tight_layout()
    return fig


def plot_raw_chi2_histogram(
    chi2,
    chi2_min=50,
    chi2_max=200,
    nbins=50,
    figsize=(5, 3),
    color='steelblue',
    alpha=0.75,
    log_y=True,
    lab_fs=12,
    text_fs=9,
    xlabel=r'$\chi^2$',
):
    """
    Single-panel histogram of raw (un-normalized) chi-squared values.

    Parameters
    ----------
    chi2 : array-like
        Raw chi-squared values (not reduced).
    chi2_max : float or None
        Upper x-axis limit.  If None, uses the 99th percentile.
    nbins : int
        Number of histogram bins.
    log_y : bool
        If True (default), use a log-scale y-axis.
    """
    chi2 = np.asarray(chi2, dtype=float)
    finite = np.isfinite(chi2)
    chi2_use = chi2[finite]

    if chi2_max is None:
        chi2_max = float(np.percentile(chi2_use, 99))

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.hist(chi2_use, bins=nbins, range=(chi2_min, chi2_max),
            color=color, alpha=alpha, edgecolor='none')

    med   = float(np.median(chi2_use))
    mn    = float(np.mean(chi2_use))
    p95   = float(np.percentile(chi2_use, 95))

    stat_str = (f'Median $= {med:.1f}$\n'
                f'Mean $= {mn:.1f}$')
    ax.text(0.97, 0.97, stat_str, fontsize=text_fs,
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.85, edgecolor='gray', lw=0.5))

    ax.axvline(med, color='k', lw=1.2, linestyle='--',
               label=f'median = {med:.1f}')
    ax.axvline(p95, color='firebrick', lw=1.2, linestyle=':',
               label=f'95th pct = {p95:.1f}')

    ax.set_xlim(chi2_min, chi2_max)
    ax.set_xlabel(xlabel, fontsize=lab_fs)
    ax.set_ylabel('Count', fontsize=lab_fs)
    if log_y:
        ax.set_yscale('log')

    # Twin axis: CDF over the same x range
    ax_cdf = ax.twinx()
    chi2_sorted = np.sort(chi2_use)
    cdf = np.arange(1, len(chi2_sorted) + 1) / len(chi2_sorted)
    ax_cdf.plot(chi2_sorted, cdf, color='dimgray', lw=1.5, alpha=0.8)
    ax_cdf.set_xlim(chi2_min, chi2_max)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.set_ylabel('CDF', fontsize=lab_fs)
    ax_cdf.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))

    return fig


def plot_chi2_histograms_by_uncertainty(z_out_pae, sigmaz_pae, chi2_pae,
                                         sigz_bins, 
                                         figsize=(12, 4), 
                                         label_fs=14, 
                                         title_fs=14,
                                         nbins=30,
                                         chi2_max=10.0,
                                         color='steelblue'):
    """
    Generates chi-squared histograms binned by PAE's fractional redshift uncertainty.
    
    Args:
        z_out_pae (np.ndarray): PAE median redshifts.
        sigmaz_pae (np.ndarray): PAE reported redshift uncertainties.
        chi2_pae (np.ndarray): Reduced chi-squared values for each source.
        sigz_bins (np.ndarray): Edges of the fractional uncertainty bins.
        figsize (tuple): Size of the overall figure.
        label_fs (int): Font size for axis labels.
        title_fs (int): Font size for subplot titles.
        nbins (int): Number of histogram bins.
        chi2_max (float): Maximum chi2 value to display on x-axis.
        color (str): Color for the histograms.
        
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    
    # Calculate fractional uncertainties
    fractional_sigma_pae = sigmaz_pae / (1 + z_out_pae)
    
    # Setup the figure and subplots
    ncols = len(sigz_bins) - 1
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, 
                             gridspec_kw={'wspace': 0.3})
    
    # Make axes iterable even for single column
    if ncols == 1:
        axes = [axes]
    
    # Loop over fractional uncertainty bins
    for i in range(ncols):
        ax = axes[i]
        low_sigz, high_sigz = sigz_bins[i], sigz_bins[i+1]
        
        # Create mask for this uncertainty bin
        mask = (fractional_sigma_pae >= low_sigz) & (fractional_sigma_pae < high_sigz)
        mask &= np.isfinite(chi2_pae)  # Exclude NaN/inf chi2 values
        
        chi2_bin = chi2_pae[mask]
        n_sources = len(chi2_bin)
        
        # Set title
        if i == 0:
            title = r'$\sigma_{z/1+z}<' + f'{high_sigz}$'
        else:
            title = f'${low_sigz}' + r'<\sigma_{z/1+z}<' + f'{high_sigz}$'
        ax.set_title(title + f'\n(n={n_sources})', fontsize=title_fs)
        
        if n_sources < 2:
            ax.text(0.5, 0.5, 'Insufficient data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, chi2_max)
            ax.set_xlabel('$\\chi^2_\\mathrm{red}$', fontsize=label_fs)
            continue
        
        # Create histogram
        bins = np.linspace(0, chi2_max, nbins)
        ax.hist(chi2_bin, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add vertical line at chi2=1 (expected for good fit)
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label=r'$\chi^2_\mathrm{red}=1$')
        
        # Add median line
        median_chi2 = np.median(chi2_bin)
        ax.axvline(median_chi2, color='darkblue', linestyle=':', linewidth=2, 
                  label=f'Median={median_chi2:.2f}')
        
        # Set labels and formatting
        ax.set_xlabel(r'$\chi^2_\mathrm{red}$', fontsize=label_fs)
        ax.set_xlim(0, chi2_max)
        
        if i == 0:
            ax.set_ylabel('Count', fontsize=label_fs)
            ax.legend(loc='upper right', fontsize=10)
        else:
            ax.set_ylabel('')
        
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_coverage_comparison_grid(z_true, z_out_pae, sigmaz_pae, pae_samples,
                                   z_out_tf, dzout_tf, tf_zpdfs, tf_zpdf_fine_z,
                                   sigz_bins, labels=['PAE', 'TF'], colors=['b', 'k'],
                                   figsize=(9, 8), z_score_xlim=[-5, 5], dz_norm_xlim=[-0.5, 0.5], \
                                 fracz_widths=None, label_fs=14, nbins=30, height_ratios=[1, 1, 2, 2], \
                                 hspace=0.3, wspace=0.05, alpha=0.6, s=2, title_fs=14,
                                 bias_correct_pit=False, sample_log_amplitude=False):
    """
    Generates a four-row figure comparing PAE and Template Fitting results
    binned by PAE's fractional redshift uncertainty.
    
    Row 1: Fractional redshift error histograms
    Row 2: Z-score histograms
    Row 3: PIT without bias correction
    Row 4: PIT with bias correction

    Args:
        z_true (np.ndarray): The true redshifts for all sources.
        z_out_pae (np.ndarray): PAE median redshifts.
        sigmaz_pae (np.ndarray): PAE reported redshift uncertainties.
        pae_samples (np.ndarray): PAE posterior redshift samples (n_src x n_samples).
        z_out_tf (np.ndarray): TF median redshifts.
        dzout_tf (np.ndarray): TF redshift uncertainties (e.g., from NMAD).
        tf_zpdfs (np.ndarray): TF redshift PDFs (n_src x n_z_bins).
        tf_zpdf_fine_z (np.ndarray): The fine redshift grid for the TF PDFs.
        sigz_bins (np.ndarray): Edges of the fractional uncertainty bins for columns.
        labels (list): Labels for the two methods.
        colors (list): Colors for the two methods.
        figsize (tuple): Size of the overall figure.
        z_score_xlim (list): x-axis limits for the z-score plots.
        dz_norm_xlim (list): x-axis limits for the fractional error plots.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    
    # Calculate fractional uncertainties and errors
    fractional_sigma_pae = sigmaz_pae / (1 + z_out_pae)
    dz_norm_pae = (z_out_pae - z_true) / (1 + z_true)
    z_score_pae = (z_out_pae - z_true) / sigmaz_pae

    fractional_sigma_tf = dzout_tf / (1 + z_out_tf)
    dz_norm_tf = (z_out_tf - z_true) / (1 + z_true)
    z_score_tf = (z_out_tf - z_true) / dzout_tf

    # Setup the figure and subplots grid
    ncols = len(sigz_bins) - 1
    fig, axes = plt.subplots(nrows=4, ncols=ncols, figsize=figsize, 
                             gridspec_kw={'hspace': hspace, 'wspace': wspace}, \
                            height_ratios=height_ratios)

    # Main loop over the fractional uncertainty bins (columns)
    for i in range(ncols):
        # 1. Create masks for the current bin
        low_sigz, high_sigz = sigz_bins[i], sigz_bins[i+1]
        mask = (fractional_sigma_pae >= low_sigz) & (fractional_sigma_pae < high_sigz)
        
        # Apply masks to all data
        z_true_bin = z_true[mask]
        
        dz_norm_pae_bin = dz_norm_pae[mask]
        z_score_pae_bin = z_score_pae[mask]
        pae_samples_bin = pae_samples[mask]

        tf_mask = (fractional_sigma_tf >= low_sigz) & (fractional_sigma_tf < high_sigz)

        dz_norm_tf_bin = dz_norm_tf[mask]
        z_score_tf_bin = z_score_tf[mask]
        tf_zpdfs_bin = tf_zpdfs[mask]

        # Handle columns with insufficient data
        if np.sum(mask) < 2:
            for row in range(4):
                axes[row, i].set_title(f'Bin {i+1}\n(n=0)')
                axes[row, i].axis('off')
            continue

        # --- Row 1: Fractional Redshift Error Histograms ---
        ax = axes[0, i]

        if i==0:
            title = '$\\sigma_{z/1+z}<'+str(high_sigz)+'$'
        else:
            title = str(low_sigz)+'$<\\sigma_{z/1+z}<'+str(high_sigz)+'$'
        ax.set_title(title, fontsize=title_fs)

        if fracz_widths is not None:
            ax.set_xlim(-fracz_widths[i], fracz_widths[i])

            dzbins = np.linspace(-fracz_widths[i], fracz_widths[i], nbins)
        else:
            dzbins = nbins
            
        ax.hist(dz_norm_pae_bin, bins=dzbins, histtype='step', color=colors[0], label=labels[0], density=True)
        ax.hist(dz_norm_tf_bin, bins=dzbins, histtype='step', color=colors[1], label=labels[1], density=True)
        ax.axvline(np.median(dz_norm_pae_bin), color=colors[0], linestyle='--')
        ax.axvline(np.median(dz_norm_tf_bin), color=colors[1], linestyle='--')

        ax.set_yticks([], [])
        ax.set_xlabel('$\\Delta z/1+z$', fontsize=label_fs)

        # --- Row 2: Z-score Histograms ---
        ax = axes[1, i]

        zscore_bins = np.linspace(-5, 5, nbins)
        ax.hist(z_score_pae_bin, bins=zscore_bins, histtype='step', color=colors[0], label=labels[0], density=True)
        ax.hist(z_score_tf_bin, bins=zscore_bins, histtype='step', color=colors[1], label=labels[1], density=True)
        # ax.axvline(0, color='k', linestyle=':')
        ax.axvline(np.median(z_score_pae_bin), color=colors[0], linestyle='--')
        ax.axvline(np.median(z_score_tf_bin), color=colors[1], linestyle='--')
        ax.set_xlim(z_score_xlim)
        ax.set_xlabel('Redshift z-score', fontsize=label_fs)

        ax.set_yticks([], [])

        # --- Row 3: PIT without bias correction ---
        ax = axes[2, i]
        
        # Add text label in first panel
        if i == 0:
            ax.text(0.05, 0.9, 'No bias correction', 
                   transform=ax.transAxes, fontsize=12, verticalalignment='top')
        
        # Compute standard PIT (no bias correction)
        pit_pae_uncorrected = compute_pit_values_pae(z_true_bin, pae_samples_bin, sample_log_amplitude=sample_log_amplitude)
        pit_tf_uncorrected = compute_pit_values_tf(z_true_bin, tf_zpdf_fine_z, tf_zpdfs_bin)

        # Plot QQ for PAE
        pit_pae_sorted = np.sort(pit_pae_uncorrected)
        uniform_quantiles = np.linspace(1/(2*len(z_true_bin)), 1-1/(2*len(z_true_bin)), len(z_true_bin))
        ax.plot(uniform_quantiles, pit_pae_sorted, 'o', color=colors[0], label=labels[0], markersize=s, alpha=alpha)
        
        # Plot QQ for TF
        pit_tf_sorted = np.sort(pit_tf_uncorrected)
        ax.plot(uniform_quantiles, pit_tf_sorted, 'o', color=colors[1], label=labels[1], markersize=s, alpha=alpha)

        ax.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Theoretical Quantiles', fontsize=label_fs)

        yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        if i==0:
            ax.set_ylabel('Empirical Quantiles', fontsize=label_fs)
            ax.set_yticks(yticks)
        else:
            ax.set_yticks(yticks, ['' for x in range(len(yticks))])
        
        # --- Row 4: PIT with bias correction ---
        ax = axes[3, i]
        
        # Add text label in first panel
        if i == 0:
            ax.text(0.05, 0.9, 'With bias correction', 
                   transform=ax.transAxes, fontsize=12, verticalalignment='top')
        
        print('z_true_bin shape:', z_true_bin.shape)
        print('pae_samples_bin shape:', pae_samples_bin.shape)
        print('tf_zpdfs_bin shape:', tf_zpdfs_bin.shape)
        print('tf_zpdf_fine_z shape:', tf_zpdf_fine_z.shape)

        # Compute bias-corrected PIT values
        from utils.utils_jax import compute_pit_values_pae_bias_corrected, compute_pit_values_tf_bias_corrected
        
        z_out_pae_bin = z_out_pae[mask]
        z_out_tf_bin = z_out_tf[mask]
        
        pit_pae_corrected, bias_pae = compute_pit_values_pae_bias_corrected(
            z_true_bin, pae_samples_bin, z_out_pae_bin, sample_log_amplitude=sample_log_amplitude)
        pit_tf_corrected, bias_tf = compute_pit_values_tf_bias_corrected(
            z_true_bin, tf_zpdf_fine_z, tf_zpdfs_bin, z_out_tf_bin)
        
        print(f'  Bin {i+1}: PAE mean frac bias: {bias_pae:.4f}, TF mean frac bias: {bias_tf:.4f}')

        # Plot QQ for PAE
        pit_pae_sorted = np.sort(pit_pae_corrected)
        uniform_quantiles = np.linspace(1/(2*len(z_true_bin)), 1-1/(2*len(z_true_bin)), len(z_true_bin))
        ax.plot(uniform_quantiles, pit_pae_sorted, 'o', color=colors[0], label=labels[0], markersize=s, alpha=alpha)
        
        # Plot QQ for TF
        pit_tf_sorted = np.sort(pit_tf_corrected)
        ax.plot(uniform_quantiles, pit_tf_sorted, 'o', color=colors[1], label=labels[1], markersize=s, alpha=alpha)

        ax.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Theoretical Quantiles', fontsize=label_fs)

        yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        if i==0:
            ax.set_ylabel('Empirical Quantiles', fontsize=label_fs)
            ax.set_yticks(yticks)
        else:
            ax.set_yticks(yticks, ['' for x in range(len(yticks))])

        # --- Final formatting for the column ---
        ax.grid(alpha=0.5)

    # Shared Y-labels and legends
    axes[2, 0].set_ylabel('Empirical Quantiles')
    # Place a single legend in the first column's first panel
    axes[0, 0].legend(loc='upper right')

    plt.tight_layout()
    return fig


def prepare_data_for_plotting(pae_save_fpath, pae_sample_fpath,
                              tf_results, tf_zpdf_fine_z,
                              nsrc=100, src_idxs=None, burnin=1000):
    """
    Prepares and loads the necessary data arrays for the plotting function.

    Args:
        pae_save_fpath (str): Path to the PAE redshift results file (e.g., z_med, ztrue).
        pae_sample_fpath (str): Path to the PAE posterior samples file.
        tf_results (tuple): A tuple containing template fitting results,
                            in the format (zout_tf, dzout_tf, ..., match_z, zpdf_tf).
        tf_zpdf_fine_z (np.ndarray): The fine redshift grid for the TF PDFs.
        nsrc (int): The number of sources to select.
        src_idxs (np.ndarray): The indices of the sources to use.
        burnin (int): The number of burn-in samples to discard from PAE samples.
    Returns:
        tuple: A tuple containing the processed data arrays, ready to be passed
               to the plotting function:
               (z_true, z_out_pae, sigmaz_pae, pae_samples, z_out_tf,
                dzout_tf, tf_zpdfs, tf_zpdf_fine_z)
    """

    # --- Load PAE data from file paths ---
    try:
        pae_redshift_res = np.load(pae_save_fpath)
        pae_sample_res = np.load(pae_sample_fpath)
    except FileNotFoundError:
        print("Error: PAE data files not found. Please check paths.")
        return None

    # --- Extract and process PAE data ---
    z_true = pae_redshift_res['ztrue']
    z_out_pae = pae_redshift_res['z_med']
    err_low = pae_redshift_res['err_low']
    err_high = pae_redshift_res['err_high']
    sigmaz_pae = 0.5 * (err_low + err_high)

    print('all samples has shape:', pae_sample_res['all_samples'].shape)
    pae_samples = pae_sample_res['all_samples'][:, :, burnin:]  # Discard burn-in samples

    redshift_Rhat = pae_redshift_res['R_hat']

    # --- Extract and process TF data from the provided tuple ---
    # Assuming tf_results is in the format (zout, dzout, dz_oneplusz, chisq, ztrue, zpdf)
        
    z_out_tf, dzout_tf, _, _, z_true_tf, tf_zpdfs = tf_results

    # --- Apply source indexing to all arrays ---
    if src_idxs is None:
        src_idxs = np.arange(len(z_true))
    
    which_idxs = src_idxs[:nsrc]

    z_true_sel = z_true[which_idxs]
    z_out_pae_sel = z_out_pae[which_idxs]
    sigmaz_pae_sel = sigmaz_pae[which_idxs]
    pae_samples_sel = pae_samples[which_idxs]

    z_out_tf_sel = z_out_tf[which_idxs]

    dzout_tf_sel = dzout_tf[which_idxs]
    tf_zpdfs_sel = tf_zpdfs[which_idxs,3:]
    redshift_Rhat_sel = redshift_Rhat[which_idxs]

    tf_zpdfs_sel /= np.sum(tf_zpdfs_sel, axis=1)[:, np.newaxis]


    # Return the processed data
    return (z_true_sel, z_out_pae_sel, sigmaz_pae_sel, pae_samples_sel,
            z_out_tf_sel, dzout_tf_sel, tf_zpdfs_sel, tf_zpdf_fine_z, redshift_Rhat_sel)

# def plot_zscore_analysis_combined(correlations, z_scores, n_bins=10, \
#                                  figsize=(4, 10), ylim=[-4, 4], alpha=0.2, \
#                                  nbin_hist=30):
#     """
#     Creates a single-column plot combining scatter points with a binned z-score analysis.

#     Each row corresponds to a latent vector u_i and shows a scatter plot of z-scores
#     vs. absolute correlation, with binned median and 68% CIs as errorbars.

#     Args:
#         correlations (np.ndarray): Array of shape (ngal, n_latent) with per-galaxy correlations.
#         z_scores (np.ndarray): Array of shape (ngal,) with per-galaxy z-scores.
#         labels (list): List of strings for the latent vector labels.
#         n_bins (int): Number of bins to use for the binned analysis.
#     """
#     n_latent = correlations.shape[1]
    
#     # Create a 6x1 grid of subplots with shared x and y axes
#     fig, axes = plt.subplots(nrows=n_latent+1, ncols=1, figsize=figsize, 
#                              sharex=True, sharey=False)

#     # Calculate the binned statistics once for each latent vector
#     bin_results = []
#     bins = np.linspace(0, 1, n_bins + 1)
#     bin_centers = (bins[:-1] + bins[1:]) / 2

#     for i in range(n_latent):
#         abs_corr = np.abs(correlations[:, i])
        
#         bin_indices = np.digitize(abs_corr, bins)
        
#         median_zscores = np.zeros(n_bins)
#         p16_zscores = np.zeros(n_bins)
#         p84_zscores = np.zeros(n_bins)
#         valid_bins = np.zeros(n_bins, dtype=bool)

#         for j in range(1, n_bins + 1):
#             mask = (bin_indices == j)
#             if np.sum(mask) > 0:
#                 median_zscores[j-1] = np.median(z_scores[mask])
#                 p16_zscores[j-1] = np.percentile(z_scores[mask], 16)
#                 p84_zscores[j-1] = np.percentile(z_scores[mask], 84)
#                 valid_bins[j-1] = True
        
#         bin_results.append({
#             'bin_centers': bin_centers[valid_bins],
#             'median': median_zscores[valid_bins],
#             'p16': p16_zscores[valid_bins],
#             'p84': p84_zscores[valid_bins]
#         })

#     # Plotting loop for each latent vector

#     ax = axes[0]
#     for i in range(n_latent):
#         ax.hist(np.abs(correlations[:,i]), bins=np.linspace(0, 1, nbin_hist), histtype='step', color='C'+str(i))
#         ax.grid(alpha=0.5)
#         ax.set_yticks([], [])
    
    
#     for i in range(n_latent):
#         ax = axes[i+1]
#         abs_corr = np.abs(correlations[:, i])
        
#         # Plot the scatter points
#         ax.scatter(abs_corr, z_scores, s=1, alpha=alpha, color='C'+str(i))
        
#         # Overplot the binned median line and error bars
#         error_low = bin_results[i]['median'] - bin_results[i]['p16']
#         error_high = bin_results[i]['p84'] - bin_results[i]['median']
#         ax.errorbar(bin_results[i]['bin_centers'], bin_results[i]['median'],
#                     yerr=[error_low, error_high], fmt='o-', capsize=3,
#                     color='k', linewidth=1.5)

#         # Plot a horizontal line at Z=0 for reference
#         ax.axhline(0, color='k', linestyle='--', linewidth=1, zorder=0)

#         # Set titles and limits
#         ax.text(0.75, -3., '$|\\rho(z, u_'+str(i+1)+')|$', fontsize=14, color='C'+str(i))
#         ax.set_ylim(ylim)
#         ax.grid(alpha=0.5)
#         ax.set_xlim(0, 1)

#         # Set the x-axis label only for the last plot
#         if i == n_latent - 1:
#             ax.set_xlabel('$|\\rho(z, u)|$', fontsize=16)
#             ax.xaxis.set_tick_params(labelbottom=True)
#         else:
#             plt.setp(ax.get_xticklabels(), visible=False)

#     # Set the common y-axis label
#     fig.text(-0.02, 0.45, r'Redshift Z-score $= (\hat{z} - z_{true})/\hat{\sigma}_z$', 
#              va='center', rotation='vertical', fontsize=18)

#     plt.subplots_adjust(hspace=0.1)
#     plt.show()

#     return fig

def plot_zerr_tf_pae_with_binned_corr(zout_tf, ztrue_tf, zout_pae, ztrue, sigmaz_pae, 
                                     figsize=(5, 5),
                                     logmin=-2.6, logmax=-0.7, xlim=[-0.5, 0.5], ylim=[-0.5, 0.5], s=8, 
                                     cmap='Blues_r', sigzoneplusz_list=[0.01, 0.03, 0.1, 0.2],
                                     color_by='pae_sigma', tf_sigmaz_oneplusz=None):
    """
    Creates a two-panel plot comparing TF and PAE redshift errors.
    
    The top panel shows the Pearson correlation coefficient between the errors,
    binned by PAE's fractional redshift uncertainty. The bottom panel shows the 
    scatter of errors, colored by PAE's reported fractional redshift uncertainty.

    Args:
        zout_tf (np.ndarray): The estimated redshifts from the template fitting method.
        ztrue_tf (np.ndarray): The true redshifts for the TF data.
        zout_pae (np.ndarray): The estimated redshifts from the PAE method.
        ztrue (np.ndarray): The true redshifts for the PAE data.
        sigmaz_pae (np.ndarray): The reported redshift uncertainties from the PAE.
        sigz_bins (np.ndarray): An array defining the uncertainty bins for the correlation panel.
        figsize (tuple): Figure size (width, height).
        logmin (float): Min value for the log10 of the colorbar.
        logmax (float): Max value for the log10 of the colorbar.
        xlim (list): x-axis limits for the scatter plot.
        ylim (list): y-axis limits for the scatter plot.
        s (int): Marker size for the scatter plot.
    """
    # 1. Calculate errors for both methods and the fractional uncertainty
    dz_tf = zout_tf - ztrue_tf
    dz_pae = zout_pae - ztrue
    fractional_sigma = sigmaz_pae / (1 + zout_pae)
    # fractional_sigma = sigmaz_pae / (1 + zout_tf)

    if color_by == 'pae_sigma':
        color_metric = fractional_sigma
        colorbar_label = "$\\sigma_{z/1+z}$ (PAE)"
        tick_vals = np.array([1e-1, 3e-2, 1e-2, 3e-3, 1e-3], dtype=float)
        tick_lbls = [
            "$10^{-1}$",
            "$3 \\times 10^{-2}$",
            "$10^{-2}$",
            "$3 \\times 10^{-3}$",
            "$10^{-3}$",
        ]
    elif color_by == 'sigma_ratio':
        if tf_sigmaz_oneplusz is None:
            raise ValueError("tf_sigmaz_oneplusz must be provided when color_by='sigma_ratio'.")
        tf_sigmaz_oneplusz = np.asarray(tf_sigmaz_oneplusz, dtype=float)
        tf_safe = np.clip(tf_sigmaz_oneplusz, 1e-12, None)
        color_metric = fractional_sigma / tf_safe
        colorbar_label = "$\\sigma_{z/1+z}^{PAE}/\\sigma_{z/1+z}^{TF}$"
        tick_vals = np.array([1e-1, 3e-1, 1.0, 3.0, 1e1], dtype=float)
        tick_lbls = [
            "$10^{-1}$",
            "$3 \\times 10^{-1}$",
            "$1$",
            "$3$",
            "$10$",
        ]
    else:
        raise ValueError(f"Unknown color_by mode: {color_by}")

    # 2. Setup the figure and subplots (sharex is now False)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=figsize, 
                                   sharex=False, gridspec_kw={'height_ratios': [1, 4]})

    # --- Top Panel: Binned Correlation ---
    # bin_centers = (sigz_bins[:-1] + sigz_bins[1:]) / 2
    binned_corr = np.zeros(len(sigzoneplusz_list))
    
    # Digitize the fractional uncertainties into the specified bins
    # sigz_bin_indices = np.digitize(fractional_sigma, sigz_bins)

    all_labls = []
    
    for i in range(len(sigzoneplusz_list)):

        if i==0:
            mask = (fractional_sigma < sigzoneplusz_list[i])
            labl = '$\\sigma_{z/1+z} < '+str(sigzoneplusz_list[i])+'$'
        else:
            mask = (fractional_sigma < sigzoneplusz_list[i])*(fractional_sigma > sigzoneplusz_list[i-1])
            labl = '$\\sigma_{z/1+z}\\in ['+str(sigzoneplusz_list[i-1])+','+str(sigzoneplusz_list[i])+']$'
            # labl = '('+str(sigzoneplusz_list[i-1])+'$<\\sigma_{z/1+z} < $'+str(sigzoneplusz_list[i])+')'
        all_labls.append(labl)

        if np.sum(mask) > 1: # Need at least 2 points to calculate a correlation
            corr, _ = pearsonr(dz_tf[mask], dz_pae[mask])
            binned_corr[i] = corr
        else:
            binned_corr[i] = np.nan # Set to NaN if not enough points

    ax1.plot(np.arange(len(sigzoneplusz_list)), binned_corr, marker='x', color='k', markersize=8, linestyle='dashed')
    ax1.set_xticks(np.arange(len(sigzoneplusz_list)), all_labls) 
    ax1.set_ylabel('$\\rho(\\Delta z_{PAE}, \\Delta z_{TF})$', fontsize=14)
    ax1.grid(alpha=0.5)
    ax1.set_ylim(0, 1)
    ax1.xaxis.tick_top()
    ax1.tick_params(axis='x', rotation=15)

    # --- Bottom Panel: Scatter Plot ---
    # Draw larger-uncertainty points first so high-precision points are visible on top.
    finite_mask = np.isfinite(dz_tf) & np.isfinite(dz_pae) & np.isfinite(fractional_sigma) & (fractional_sigma > 0)
    finite_mask &= np.isfinite(color_metric) & (color_metric > 0)
    dz_tf_plot = dz_tf[finite_mask]
    dz_pae_plot = dz_pae[finite_mask]
    frac_sigma_plot = fractional_sigma[finite_mask]
    color_metric_plot = color_metric[finite_mask]

    order = np.argsort(frac_sigma_plot)[::-1]  # high sigma first, low sigma last
    dz_tf_plot = dz_tf_plot[order]
    dz_pae_plot = dz_pae_plot[order]
    frac_sigma_plot = frac_sigma_plot[order]
    color_metric_plot = color_metric_plot[order]
    log_sigma_plot = np.log10(np.clip(color_metric_plot, 10**logmin, 10**logmax))

    scatter = ax2.scatter(
        dz_tf_plot,
        dz_pae_plot,
        s=s,
        alpha=0.6,
        c=log_sigma_plot,
        cmap=cmap, vmin=logmin, vmax=logmax,
        edgecolors='none',
        zorder=2,
    )

    # Emphasize high-precision points with a slightly larger marker and thin white edge.
    hi_precision_mask = frac_sigma_plot <= 2e-2
    if np.any(hi_precision_mask):
        ax2.scatter(
            dz_tf_plot[hi_precision_mask],
            dz_pae_plot[hi_precision_mask],
            s=s,
            alpha=0.95,
            c=log_sigma_plot[hi_precision_mask],
            cmap=cmap,
            vmin=logmin,
            vmax=logmax,
            zorder=3,
        )
    ax2.set_xlabel('$\\Delta z$ (TF)', fontsize=14)
    ax2.set_ylabel('$\\Delta z$ (PAE)', fontsize=14)
    ax2.plot(xlim, ylim, color='k', linestyle='dashed', linewidth=1) # 1:1 line
    ax2.grid(alpha=0.5)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    # --- Add the colorbar with sigma values shown on the log-scaled mapping ---
    cbar = fig.colorbar(scatter, ax=ax2, pad=0.01)
    tick_pos = np.log10(tick_vals)
    in_range = (tick_pos >= logmin) & (tick_pos <= logmax)
    cbar.set_ticks(tick_pos[in_range])
    cbar.set_ticklabels([lbl for lbl, keep in zip(tick_lbls, in_range) if keep])
    cbar.set_label(colorbar_label, fontsize=14)

    plt.subplots_adjust(hspace=0.05)
    # plt.tight_layout()
    plt.show()
    return fig

# def plot_zerr_tf_pae_with_sigz_color(zout_tf, ztrue_tf, zout_pae, ztrue, sigmaz_pae, figsize=(5, 4), \
#                                     logmin=-3, logmax=-0.5, xlim=[-0.5, 0.5], ylim=[-0.5, 0.5], s=10):

#     fig = plt.figure(figsize=figsize)
#     scatter = plt.scatter(
#         zout_tf-ztrue_tf,
#         zout_pae-ztrue,
#         s=s,        # Marker size (adjust as needed)
#         alpha=0.5,   # Transparency
#         c=np.log10(sigmaz_pae),
#         cmap='jet', vmin=logmin, vmax=logmax, # Colormap for the 'true_redshifts'
#         edgecolors='none' # No edge color for markers
#     )
#     cbar = plt.colorbar(scatter)
#     cbar.set_label("$\\log(\\sigma_z)$", fontsize=12)
#     plt.xlabel('$\\Delta z$ (TF)', fontsize=14)
#     plt.ylabel('$\\Delta z$ (PAE)', fontsize=14)
#     plt.legend()
#     plt.xlim(xlim)
#     plt.ylim(ylim)
#     plt.grid(alpha=0.5)
#     plt.show()

#     return fig

# def plot_zerr_correlation_tf_pae(all_corr_zerrs, all_labls,\
#                                  figsize=(6, 3), lab_fs=12, ylim=[0, 1], s=50):

#     fig_corr = plt.figure(figsize=(7, 3))
#     plt.scatter(np.arange(len(all_corr_zerrs)), all_corr_zerrs, marker='x', color='r', s=s)
#     plt.plot(np.arange(len(all_corr_zerrs)), all_corr_zerrs, color='k')
#     plt.ylabel('Correlation of redshift errors', fontsize=lab_fs)
#     plt.xlabel('Redshift precision bin', fontsize=lab_fs)
#     plt.xticks(np.arange(len(all_corr_zerrs)), all_labls)
#     plt.grid(alpha=0.5)
#     plt.ylim(ylim)
#     plt.show()

#     return fig_corr

def compare_zscore_dists_tf_pae(zscore_pae, zscore_tf, figsize=(8, 4),\
                                    hist_lim=[-5, 5], nbin_1d=40, zscore_lim=[-10, 10], nbin_2d=50):

    bin_edges_2d = np.linspace(zscore_lim[0], zscore_lim[1], nbin_2d) # 50 bins from -10 to 10 for both axes
    bin_edges_1d = np.linspace(hist_lim[0], hist_lim[1], nbin_1d) # 50 bins from -10 to 10 for both axes

    fig_zscore_compare = plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.hist(zscore_pae, bins=bin_edges_1d, histtype='step', color='C3', label='PAE')
    plt.hist(zscore_tf, bins=bin_edges_1d, histtype='step', color='k', label='Template fitting')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlabel('redshift z-score', fontsize=14)
    
    plt.subplot(1,2,2)
    h = plt.hist2d(zscore_tf, zscore_pae, bins=bin_edges_2d, cmap='viridis', norm=LogNorm(), cmin=1)
    plt.colorbar(label='Galaxy Density (log scale)')
    plt.xlim(zscore_lim)
    plt.ylim(zscore_lim)
    plt.axvline(0, color='k', alpha=0.5)
    plt.axhline(0, color='k', alpha=0.5)
    plt.xlabel('Template Fitting z-score', fontsize=12)
    plt.ylabel('PAE z-score', fontsize=12)
    plt.grid(alpha=0.2)
    plt.gca().set_aspect('equal', adjustable='box') # Ensure square aspect ratio for symmetry
    plt.tight_layout()
    plt.show()

    return fig_zscore_compare

def compare_sigmaz_hdpi_secondmom(sigz_hdpi, sigz_secondmom, figsize=(5, 4)):

    fig = plt.figure(figsize=figsize)
    plt.scatter(sigz_hdpi, sigma_z_secondmom, color='k', s=2, alpha=0.5)
    plt.xlabel('68% HDPI')
    plt.ylabel('Second moment')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(alpha=0.3)
    plt.show()

    return fig

def plot_reduced_chi2(chi2, nphot=102, nparam=5, bins=50, figsize=(4, 3)):
        
    fig = plt.figure()
    plt.hist(chisq/(102-5), bins=bins)
    plt.yscale('log')
    plt.show()
    return fig
    
def plot_phot_snr_vs_redshift_error(phot_snr, zout, ztrue, figsize=(5, 4), xlim=[10, 300], \
                                   lab_fs=16, s=3, alpha=0.3):

    fig = plt.figure(figsize=figsize)
    plt.scatter(phot_snr, zout-ztrue, color='k', s=s, alpha=alpha)
    plt.xscale('log')
    plt.ylim(-2, 2)    
    plt.grid(alpha=0.5)
    plt.xlabel('Observed SNR', fontsize=lab_fs)
    plt.ylabel('$\\hat{z}-z_{in}$', fontsize=lab_fs)
    plt.title('PAE', fontsize=lab_fs)
    plt.xlim(xlim)
    plt.show()

    return fig
    
def plot_chi2_vs_redshift_error(chi2, zout, ztrue, figsize=(6, 6), fontsize=14, s=2, color='k'):
    fig = plt.figure(figsize=figsize)
    plt.scatter(chi2, zout-ztrue, color=color, s=s)
    plt.xlabel('$\\chi^2$', fontsize=fontsize)
    plt.ylabel('PAE redshift error', fontsize=fontsize)
    plt.show()

def compare_profile_likes(z_grid, profile_logL_pae, finez, zpdf_tf, chisq_tf, ztrue,
                         zout_tf=None, sigma_smooth=2, Z_MIN=None, Z_MAX=None,
                         ylim=[40, 75], figsize=(8, 6), samples_pae=None, max_logL_sample=None, bbox_to_anchor=[0.0, 2.5], \
                          ypad=40, pdf_lw=3.0, legend_fs=11):

    if Z_MIN is None:
        Z_MIN, Z_MAX = np.min(z_grid), np.max(z_grid)
    
    # Normalize TF PDF to unit area
    zpdf_norm = zpdf_tf / np.sum(zpdf_tf)
    nll_temp = -np.log(zpdf_norm)

    dnll = np.min(nll_temp) - 0.5 * chisq_tf
    nll_temp -= dnll
    
    # Smoothed profile likelihood (in log space)
    conv_logL = scipy.ndimage.gaussian_filter1d(profile_logL_pae, sigma=sigma_smooth)

    # Create figure with two rows, first row taller
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize,
                             gridspec_kw={'height_ratios':[1.5, 1], 'hspace':0.15}, sharex=True)

    ax1, ax2 = axes

    # ---------------- Row 1: Profile likelihoods ----------------
    ax1.plot(z_grid, -profile_logL_pae, color='b', marker='.', markersize=4, alpha=0.2,
             label='Profile likelihood (PAE)', zorder=5)
    ax1.plot(z_grid, -conv_logL, color='b', alpha=1.0, linewidth=pdf_lw)
    
    ax1.plot(finez, nll_temp, color='k', linewidth=pdf_lw, label='Template fitting')
    ax1.axvline(ztrue, color='r', linestyle='solid', linewidth=2,
                label=f'Input redshift $z={np.round(ztrue, 2)}$')
    ax1.axhline(0.5 * chisq_tf, color='k', linestyle='dashed')
    ax1.axhline(np.min(-conv_logL), color='b', linestyle='dashed')

    if max_logL_sample is not None:
        ax1.axhline(-max_logL_sample, color='g', linestyle='dashed', label='ML during sampling')

    # if zout_tf is not None:
        # ax1.axvline(zout_tf, color='k', alpha=0.8)


    miny = min(np.min(-conv_logL), np.min(nll_temp))
        
    ax1.set_ylabel('NLL', fontsize=14)
    # ax1.set_xlabel('redshift', fontsize=14)
    ax1.set_xlim(Z_MIN, Z_MAX)
    ax1.set_ylim(miny-2, miny+ypad)
    ax1.grid(alpha=0.2)
    # ax1.legend(fontsize=12, loc='upper right')

    # ---------------- Row 2: Redshift PDFs ----------------
    profile_like = np.exp(profile_logL_pae)

    if sigma_smooth != 0:
        conv_prof_like = scipy.ndimage.gaussian_filter1d(profile_like, sigma=sigma_smooth)
        ax2.plot(z_grid, conv_prof_like / np.max(conv_prof_like), color='b',
                 label='Profile likelihood (PAE)', linewidth=pdf_lw)

    if samples_pae is not None:

        n, bins, patches = ax2.hist(samples_pae, bins=30,
                                    density=True, alpha=0.3, color='g', label='PAE posterior')
        
        max_density = n.max()
        for patch in patches:
            patch.set_height(patch.get_height() / max_density)

    ax2.plot(finez, zpdf_norm / np.max(zpdf_norm), color='k', alpha=0.8,
             linewidth=pdf_lw, label='Template fitting')
    ax2.axvline(ztrue, color='r', linestyle='solid', linewidth=2,
                label='$z_{true}$='+str(np.round(ztrue, 2)))
    ax2.set_xlim(Z_MIN, Z_MAX)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel('redshift', fontsize=14)
    ax2.set_ylabel('p(z) (norm.)', fontsize=14)
    ax2.legend(fontsize=legend_fs, ncols=2, loc=2, bbox_to_anchor=bbox_to_anchor)
    ax2.set_yticks([], [])

    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    plt.show()

    return fig

def plot_pit_histogram(pit_values, title):
    """
    Plots a histogram of PIT values.

    Args:
        pit_values (np.array): Array of PIT values.
        title (str): Title for the plot.
    """
    fig = plt.figure(figsize=(6, 4))
    plt.hist(pit_values, bins=20, range=(0, 1), density=True, color='skyblue', edgecolor='black')
    plt.plot([0, 1], [1, 1], 'r--', label='Uniform distribution')
    plt.title(f'PIT Histogram for {title}')
    plt.xlabel('PIT Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

    return fig

def plot_qq(all_pit_values, title, colors=['k', 'b', 'C1'], labels = ['Template fitting', 'PAE (MCLMC)', 'PAE (pocoMC)'], \
           figsize=(5, 4), lab_fs=12):
    """
    Generates a Q-Q plot for PIT values against a uniform distribution.

    Args:
        pit_values (np.array): Array of PIT values.
        title (str): Title for the plot.
    """

    fig = plt.figure(figsize=figsize)
    for p, pit_vals in enumerate(all_pit_values):
        pit_values_sorted = np.sort(pit_vals)
        # Generate quantiles from a uniform distribution
        n = len(pit_values_sorted)
        uniform_quantiles = np.linspace(1/(2*n), (2*n-1)/(2*n), n) # Using (i-0.5)/n for quantiles
        plt.plot(uniform_quantiles, pit_values_sorted, 'o', color=colors[p], label=labels[p], markersize=2, alpha=0.7)
    
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
    plt.title(f'Q-Q Plot for {title}')
    plt.xlabel('Theoretical Quantiles (Uniform)', fontsize=lab_fs)
    plt.ylabel('Empirical Quantiles (PIT)', fontsize=lab_fs)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

    return fig
    
def plot_per_source_proc(PAE_obj, spherex_dat, all_samples, which_run_idx, fix_z=False, \
                        ylim_fix=None, bbox_to_anchor=[0.15, 1.3], figsize=(8, 4.5), verbose=True, 
                        plot_corner=True, umin=-3, umax=3, zpost_figsize=(4, 3), corner_figsize=(6, 6), \
                        dz_pad=0.01, zpost_bbox_to_anchor=[1.0, 1.2], zpost_color='b', zpost_fillcolor='lightblue', \
                        ztrue_color='r'):

    src_idxs = spherex_dat.src_idxs
    src_idxs_sel = src_idxs[which_run_idx]

    if verbose:
        print('src idxs sel:', src_idxs_sel)

    norms_array = np.array(spherex_dat.norms)[:,0]
    norms_sel = norms_array[src_idxs_sel]
    
    spec_obs_sel, weights_sel, ztrue_sel, flux_unc_sel, srcids_obs_sel = [
        getattr(spherex_dat, key)[src_idxs_sel]
            for key in ['all_spec_obs', 'weights', 'redshift', 'all_flux_unc', 'srcid_obs']]

    srcids_noiseless = spherex_dat.srcids_noiseless
    all_noiseless_spec = spherex_dat.all_noiseless_spec

    if verbose:
        print('spec obs sel has shape', spec_obs_sel.shape)
        print('srcids noiselss has length', srcids_noiseless.shape)
        print('all noiseless spec has shape', all_noiseless_spec.shape)

    
    for i, idx in enumerate(which_run_idx):

        if fix_z:
            redshift_fix = ztrue_sel[i]
        else:
            redshift_fix = None
            
        recon_x, logL, redshift_post = proc_spec_post(PAE_COSMOS, all_samples[idx], spec_obs_sel[i], weights_sel[i], combine_chains=True, burn_in=1000, thin_fac=1, redshift_fix=redshift_fix)

        recon_x *= norms_sel[i]
        whichmax = np.argmax(logL)
        
        map_recon = recon_x[whichmax]
        idx_noiseless = np.where((srcids_noiseless==srcids_obs_sel[i]))[0][0]

        spec_true = all_noiseless_spec[idx_noiseless]
        spec_lopct, spec_hipct, spec_med = np.percentile(recon_x, 5, axis=0), np.percentile(recon_x, 95, axis=0), np.median(recon_x, axis=0)
        spec_pcts = [np.percentile(recon_x, pct, axis=0) for pct in [5, 16, 5, 68, 95]]
            
        spec_68pct_range = [spec_pcts[1], spec_pcts[3]]
        spec_95pct_range = [spec_pcts[0], spec_pcts[4]]
    
        spec_med = np.mean(recon_x, axis=0)
        if ylim_fix is None:
            ylim = [-30, max(100, 1.5*np.max(spec_true))]
        else:
            ylim = ylim_fix
        print('ylim:', ylim)
        fig_recon = plot_post_spec_recon(PAE_obj.wave_obs, norms_sel[i]*spec_obs_sel[i], MAP_spec=None, spec_truth=spec_true, flux_unc=norms_sel[i]*flux_unc_sel[i], \
              spec_recon_med=spec_med, spec_68pct_interval=spec_68pct_range, spec_95pct_interval=spec_95pct_range, \
                     redshift_post=redshift_post, redshift_true=ztrue_sel[i], recon_indiv=None, ylim=ylim, \
                    bbox_to_anchor=bbox_to_anchor, figsize=figsize, legend_fs=11, alph=0.2/np.sqrt(len(PAE_obj.wave_obs)/102), \
                                         post_color=zpost_fillcolor, color='k', ztrue_color=ztrue_color)


        if plot_corner:
            param_names = ['$u_'+str(i+1)+'$' for i in np.arange(5)]
            if not fix_z:
                param_names.append('$z_{gal}$')
                ztrue_plot = ztrue_sel[i]
            else:
                ztrue_plot = None
            all_samp_plot = all_samples[idx].reshape(-1, len(param_names))
            fig_corner = make_corner_plot(all_samp_plot, param_names=param_names, figsize=corner_figsize, redshift_true=ztrue_plot, smooth=0.5, nbin=20, \
                                   title_fontsize=9, fix_z=fix_z, umin=umin, umax=umax, levels=[0.68, 0.95], dz=0.03, title_fmt='.2f')  
        else:
            fig_corner = None
        
        if not fix_z:

            fig_z = plot_redshift_posterior(redshift_post, redshift_use=ztrue_sel[i], include_pcts=True, \
                                    include_mean=True, include_median=False, figsize=zpost_figsize, nbins=40, bbox_to_anchor=zpost_bbox_to_anchor, \
                                            dz_pad=dz_pad, color=zpost_color, fillcolor=zpost_fillcolor, ztrue_color=ztrue_color)
            
        else:
            fig_z = None

        figs = [fig_recon, fig_corner, fig_z]

    return figs
    
def make_corner_plot(samples, param_names=None, figsize=(8, 8), filename=None,
                     label_fontsize=10, tick_fontsize=8, title_fontsize=10,
                     redshift_true=None, samples2=None, color2='C3',
                     smooth=1.0, nbin=40, dz=0.05, fix_z=False, umin=-5, umax=5, **kwargs):
    """
    Generate a corner plot from MCMC samples with customized titles and colors.
    """
    truth_color='k'

    if redshift_true is not None:
        truths = [None] * (samples.shape[1] - 1) + [redshift_true]
    else:
        truths = None

    ndim = samples.shape[1]

    if fix_z:
        ranges = [(umin, umax) for _ in range(ndim)]
    else:
        ranges = [(umin, umax) for _ in range(ndim - 1)]
        if dz is None:
            ranges += [(0, 3.0)]
        else:
            ranges += [(redshift_true - dz, redshift_true + dz)]

    fig, axes = plt.subplots(ndim, ndim, figsize=figsize)  # Increase figsize from default
    # corner.corner(..., fig=fig)
    fig = corner.corner(
        samples,
        labels=param_names,
        truths=truths,
        truth_color=truth_color,
        show_titles=False,
        bins=nbin,
        range=ranges,
        label_kwargs={"fontsize": label_fontsize},
        plot_datapoints=False,
        smooth=smooth,
        color="C0",
        figsize=figsize,\
        fig=fig,
        **kwargs
    )

    if samples2 is not None:
        corner.corner(
            samples2,
            color=color2,
            bins=nbin,
            range=ranges,
            plot_datapoints=False,
            smooth=smooth,
            show_titles=False,
            fig=fig
        )

    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=tick_fontsize)

    med_a, err_lo_a, err_hi_a = format_estimates(samples)
    if samples2 is not None:
        med_b, err_lo_b, err_hi_b = format_estimates(samples2)

    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        ax = axes[i, i]
        param_label = param_names[i] if param_names else f"param {i}"

        # # Parameter name title

        if i==ndim-1 and not fix_z:
            decimal_places = 3
            paramstr = f'$z_{{true}}={redshift_true:.3f}$'
            
            ax.text(0.5, 1.4, paramstr, transform=ax.transAxes, ha="center", va="top",
                    fontsize=title_fontsize, color="black")
        else:
            decimal_places = 2

        format_str = f'.{decimal_places}f'

        # Format with superscript/subscript
        txt_a = rf"{param_label} $= {med_a[i]:{format_str}}^{{+{err_hi_a[i]:{format_str}}}}_{{-{err_lo_a[i]:{format_str}}}}$"
        ax.text(0.5, 1.2, txt_a, transform=ax.transAxes, ha="center", va="top",
                fontsize=title_fontsize, color="C0")
        
        if samples2 is not None:
            txt_b = rf"{param_label} $= {med_b[i]:{format_str}}^{{+{err_hi_b[i]:.{format_str}}}}_{{-{err_lo_b[i]:{format_str}}}}$"
            ax.text(0.5, 1.4, txt_b, transform=ax.transAxes, ha="center", va="top",
                    fontsize=title_fontsize, color=color2)

    # fig.tight_layout()

    if filename:
        fig.savefig(filename, bbox_inches="tight")
        print(f"Corner plot saved to {filename}")
    else:
        fig.set_size_inches(figsize)
        return fig


def plot_post_spec_recon(central_wavelengths, spec_test, spec_recon_init=None, MAP_spec=None, flux_unc=None, figsize=(8, 4), bbox_to_anchor=[0.9, 1.35], \
                   spec_recon_med=None, spec_68pct_interval=None, spec_95pct_interval=None, spec_truth=None, redshift_true=None, redshift_post=None, ylim=None, \
                        recon_indiv=None, legend_fs=10, post_color='C3', alph=0.2, color='C3', ztrue_color='k'):

    fig = plt.figure(figsize=figsize)
    
    if flux_unc is not None:
        unit = '[$\\mu$Jy]'
        plt.errorbar(central_wavelengths, spec_test, yerr=flux_unc, fmt='o', color='k', label='Observed', markersize=3, alpha=alph)
        plt.fill_between(central_wavelengths, -flux_unc, flux_unc, alpha=0.1, color='k')
    
    if spec_truth is not None:
        plt.plot(central_wavelengths, spec_truth, color=ztrue_color, label='Truth')
        
        if spec_recon_med is not None:
            plt.plot(central_wavelengths, spec_recon_med-spec_truth, color='b', linestyle='dashed', linewidth=1, label='Residual')

    if recon_indiv is not None:
        alpha = 10./recon_indiv.shape[0]

        for x in range(recon_indiv.shape[0]):
            plt.plot(central_wavelengths, recon_indiv[x], color=color, alpha=alpha)

    if spec_recon_med is not None:
        plt.plot(central_wavelengths, spec_recon_med, color=post_color, linestyle='solid', label='Posterior mean')

        chi2_med = ((spec_test - spec_recon_med)/flux_unc)**2
        sum_chi2_med = np.sum(chi2_med)

    if spec_68pct_interval is not None:
        plt.fill_between(central_wavelengths, spec_68pct_interval[0], spec_68pct_interval[1], alpha=0.4, color=post_color, label='68% C.I.')
    
    if spec_95pct_interval is not None:
        plt.fill_between(central_wavelengths, spec_95pct_interval[0], spec_95pct_interval[1], alpha=0.2, color=post_color, label='95% C.I.')

        
    if MAP_spec is not None:  
        plt.plot(central_wavelengths, MAP_spec, color='b', label='MAP')

    if redshift_post is not None:
        
        z_mean, zlo, zhi = np.mean(redshift_post), np.percentile(redshift_post, 16), np.percentile(redshift_post, 84)
        err_low = z_mean - zlo
        err_high = zhi - z_mean
        
        result_text = (
            f"$\\chi^2_{{recon}} = {sum_chi2_med:.1f} (/ {len(central_wavelengths)}) \quad "
            f"z_{{true}} = {redshift_true:.3f} \quad "
            f"\\hat{{z}} = {z_mean:.3f}^{{+{err_high:.3f}}}_{{-{err_low:.3f}}}$"
        )

        plt.text(0.8, 0.85*ylim[1], result_text, fontsize=14, bbox=dict({'facecolor':'white', 'alpha':0.5, 'edgecolor':'None'}))
        
    plt.ylim(ylim)
    plt.xlim(0.7, 5.05)
    plt.legend(loc=2, ncol=3, fontsize=legend_fs, bbox_to_anchor=bbox_to_anchor)
    plt.xlabel('$\\lambda$ [$\\mu$m]', fontsize=14)
    plt.ylabel('Flux density '+unit, fontsize=14)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
    
    return fig



def plot_noisy_vs_noiseless_phot(wave_obs, spec_obs, flux_unc, spec_noiseless=None, figsize=(8, 3), \
                                xlim=[0.75, 5.0], title=None, title_fs=14, norm=1.):

    fig = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title, fontsize=title_fs)
    plt.errorbar(wave_obs, spec_obs, yerr=flux_unc, fmt='o', color='k', alpha=0.5)

    print('mean obs:', np.mean(spec_obs))
    if spec_noiseless is not None:
        
        plt.plot(wave_obs, spec_noiseless/np.mean(spec_noiseless), color='r')
        # plt.plot(wave_obs, spec_noiseless, color='r')

    print('noiseless mean is ', np.mean(spec_noiseless))

    plt.xlim(xlim)
    plt.xlabel('$\\lambda_{obs}$ [$\\mu$m]', fontsize=14)
    plt.ylabel('Flux', fontsize=14)
    plt.legend()
    # plt.ylim(-10, 100)
    plt.show()

    return fig

def plot_input_recovered_redshifts(med_z, err_low, err_high, redshifts_true, \
                                   z_min=0, z_max=2.5, sig_z_oneplusz_max=None, color_val=None, val_label=None, \
                                  figsize=(7, 5), alpha=0.1, cmap='jet', vmin=None, vmax=None, lab_fs=18, \
                                  redshift_stats=True, textxpos=0.1, textypos=1.0, text_fs=14, persqdeg_fac=1./1.27, \
                                  ylabel='$\\hat{z}_{PAE}$', xlabel='$z_{true}$'):


    linsp = np.linspace(z_min, z_max, 100)
    sigz_oneplusz = 0.5*(err_high+err_low)/(1+med_z)

    if sig_z_oneplusz_max is not None:
        sigz_mask = (sigz_oneplusz < sig_z_oneplusz_max)
        med_z_use, err_low_use, err_high_use, truez_use, sigz_oneplusz_use = med_z[sigz_mask], err_low[sigz_mask], err_high[sigz_mask], redshifts_true[sigz_mask], sigz_oneplusz[sigz_mask]
        if color_val is not None:
            color_val_use = color_val[sigz_mask]
    else:
        med_z_use, err_low_use, err_high_use, truez_use, sigz_oneplusz_use = med_z, err_low, err_high, redshifts_true, sigz_oneplusz
        color_val_use = color_val

    if redshift_stats and len(med_z_use) > 0:
        arg_bias, arg_std, bias, NMAD, cond_outl, outl_rate, outl_rate_15pct = compute_redshift_stats(med_z_use, truez_use, sigma_z_select=sigz_oneplusz_use, nsig_outlier=3)
        nbar = len(med_z_use)*persqdeg_fac
        plotstr = make_plotstr(nbar, NMAD, np.median(sigz_oneplusz_use), bias, outl_rate*100)
    elif redshift_stats:
        plotstr = "No sources available"
    else:
        plotstr = ""
          

    fig = plt.figure(figsize=figsize)
    
    plt.text(textxpos, textypos, plotstr, fontsize=text_fs, color='k')
    plt.plot(linsp, linsp, color='k', linestyle='dashed', zorder=10, alpha=0.5)

    if sig_z_oneplusz_max is not None:
        plt.title('$\\hat{\\sigma}_{z}/(1+z)<$'+str(sig_z_oneplusz_max), fontsize=18)
        plt.plot(linsp, linsp + sig_z_oneplusz_max*(1+linsp), color='C3', linestyle='solid')
        plt.plot(linsp, linsp - sig_z_oneplusz_max*(1+linsp), color='C3', linestyle='solid')

    plt.errorbar(truez_use, med_z_use, yerr=[err_high_use, err_low_use], fmt='o', c='k', capsize=3, markersize=3.5, alpha=alpha)

    if color_val is not None:
        sc = plt.scatter(truez_use, med_z_use, c=color_val_use, s=10, cmap=cmap, edgecolor='None', zorder=10, vmin=vmin, vmax=vmax, alpha=0.5)
        cbar = plt.colorbar()
        cbar.set_label(val_label, fontsize=lab_fs)
        
    plt.xlim(z_min, z_max)
    plt.ylim(z_min, z_max)
    plt.xlabel(xlabel, fontsize=lab_fs)
    plt.ylabel(ylabel, fontsize=lab_fs)
    plt.grid(alpha=0.5)
    plt.show()

    return fig

def plot_spec_posts_multi(wave_obs, data_spec, weights, mean_spec, lopct_spec, hipct_spec,\
                          true_redshift, mean_redshifts=None, upper_unc=None, lower_unc=None, \
                          chi2_meanspec=None, nrows=4, ncols=2, lab_fs=14, lopct=16, hipct=84, figsize=(10, 7), color='C3', \
                         linewidth=1.5, ylim=[-0.5, 4], textxpos=-0.4, textypos=3):

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    ax = ax.ravel()
    nchain = mean_spec.shape[0]

    for n in range(8):

        if mean_redshifts is not None:

            ztext= '$\\hat{z}='+str(np.round(mean_redshifts[n], 3))+'^{+'+str(np.round(upper_unc[n], 3))+'}_{-'+str(np.round(lower_unc[n], 3))+'}$'
            
            ax[n].text(1.1*np.min(wave_obs), textypos, ztext, color=color, fontsize=10)
            ax[n].text(1.1*np.min(wave_obs), 0.8*textypos, '$z_{in}='+str(np.round(true_redshift, 3))+'$', color='k', fontsize=10)

            
            ax[n].text(2, textypos, '$\\chi^2='+str(np.round(chi2_meanspec[n], 1))+' ( / '+str(len(wave_obs))+')$', color=color, fontsize=10)
        
        ax[n].errorbar(wave_obs, data_spec, yerr=1./np.sqrt(weights), color='k', fmt='o', alpha=0.4, markersize=3, capsize=2)

        if n+1>nchain:
            ax[n].set_xlabel('$\\lambda$ [$\\mu$m]', fontsize=lab_fs)

        if n%ncols==0:
            ax[n].set_ylabel('Flux [norm.]', fontsize=lab_fs)

        ax[n].plot(wave_obs, mean_spec[n], color='C'+str(n), linewidth=linewidth, label='Posterior mean', zorder=10)
        ax[n].fill_between(wave_obs, lopct_spec[n], hipct_spec[n], color='C'+str(n), alpha=0.3, linewidth=linewidth, label='Posterior uncertainty (68\\%)', zorder=10)

        ax[n].set_ylim(ylim)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    return fig


def plot_redshift_trace_multi(all_z_samples, true_redshift=None, figsize=(7, 4), burn_in=1000, lab_fs=14, alpha=0.8):


    isamp = np.arange(burn_in, burn_in+all_z_samples.shape[1])

    fig = plt.figure(figsize=figsize)
    
    for n in range(nchain):
        plt.plot(isamp, all_z_samples[n], alpha=alpha)

    if true_redshift is not None:
        plt.axhline(true_redshift, color='b', linewidth=2, label='Input redshift')
    
    plt.legend()
    plt.xlabel('Sample index', fontsize=lab_fs)
    plt.ylabel('Redshift', fontsize=lab_fs)
    plt.show()

    return fig

def plot_logp_trace_multi(log_pz_chains, logL_chains, logp_chains, burn_in=1000, log_pz_redshift_chains=None, figsize=(7,3), lab_fs=14, \
                         alpha=0.3):

    nchain = logp_chains.shape[0]
    isamp = np.arange(burn_in, burn_in+logp_chains.shape[1])

    fig_logp_trace, ax = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)
    pvec = [log_pz_chains, logL_chains, logp_chains]

    for n in range(nchain):
        for v, val in enumerate([log_pz_chains[n], logL_chains[n], logp_chains[n]]):
            ax[v].plot(isamp, val, alpha=alpha)
            if n==0:
                print(np.nanmin(val), np.nanmax(val))
                ax[v].set_ylim(np.nanmin(pvec[v]), np.nanmax(pvec[v]))
                
    labels = ['log $p(u)$', 'log $p(flux|u, z)$', 'Total log$p$']

    for k in range(len(labels)):
        ax[k].set_ylabel(labels[k], fontsize=lab_fs)
        if k==2:
            ax[k].set_xlabel('Sample index', fontsize=lab_fs)
    plt.show()

    return fig_logp_trace


def plot_density_contours(zscore_tf, zscore, levels=10, scatter_alpha=0.2, scatter_s=2):
    """
    Plot density contours of (zscore_tf, zscore) with optional scatter overlay.

    Parameters:
    - zscore_tf: array-like, x-axis data
    - zscore: array-like, y-axis data
    - levels: int, number of contour levels
    - scatter_alpha: float, alpha transparency for scatter points
    - scatter_s: float, scatter point size
    """
    xy = np.vstack([zscore_tf, zscore])
    kde = gaussian_kde(xy)

    # Create grid to evaluate kde
    xgrid = np.linspace(-5, 5, 100)
    ygrid = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(xgrid, ygrid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)

    plt.figure(figsize=(6, 5))
    plt.contour(X, Y, Z, levels=levels, colors='k')
    plt.scatter(zscore_tf, zscore, color='k', alpha=scatter_alpha, s=scatter_s)
    plt.xlabel('z-score (TF)')
    plt.ylabel('z-score (PAE)')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def plot_redshift_posterior(redshift_samp, redshift_use=None, include_pcts=True, which_pcts=[16, 84],\
                            include_mean=True, include_median=False, \
                            figsize=(6, 3), bins=None, nbins=30, legend_fs=11, color='grey', bbox_to_anchor=[1.1, 1.2], \
                            dz_pad=0.02, fillcolor='lightblue', ztrue_color='r'):

    zmean = np.mean(redshift_samp)
    
    if bins is None:
        bins = np.linspace(np.percentile(redshift_samp, 1)-dz_pad, np.percentile(redshift_samp, 99)+dz_pad, nbins)
        
    fig = plt.figure(figsize=figsize)
    
    plt.hist(redshift_samp, bins=bins, histtype='stepfilled', color=fillcolor, alpha=0.2,
             range=(np.percentile(redshift_samp, 16), np.percentile(redshift_samp, 84)))

    
    if redshift_use is not None:
        # plt.axvline(redshift_use, color='r', linestyle='dashed', label='$z_{true}$='+str(np.round(redshift_use, 3)), linewidth=3)
        plt.axvline(redshift_use, color=ztrue_color, linestyle='solid', label='Truth', linewidth=2.5)

    if include_mean:
        plt.axvline(np.mean(redshift_samp), color=color, linestyle='dashed', linewidth=2.5, label='Posterior mean')
    if include_median:
        plt.axvline(np.median(redshift_samp), linestyle='solid', linewidth=2.5, color='b', label='Posterior median')
        
    if include_pcts:
        for pct in which_pcts:
            plt.axvline(np.percentile(redshift_samp, pct), color=color, linestyle='dashdot')

        plt.axvline(np.percentile(redshift_samp, 5), color=color, linestyle='solid', linewidth=1.0)
        plt.axvline(np.percentile(redshift_samp, 95), color=color, linestyle='solid', linewidth=1.0)
        

    plt.xlabel('redshift', fontsize=14)
    plt.ylabel('$p(z)$', fontsize=14)
    plt.yticks([], [])
    plt.legend(fontsize=legend_fs, ncol=2, bbox_to_anchor=bbox_to_anchor)
    plt.show()

    return fig

def plot_history(history_lbfgs, figsize=(5, 4)):
    fig = plt.figure(figsize=figsize)
    plt.scatter(np.arange(len(history_lbfgs)), history_lbfgs)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()
    return fig

def plot_bandpass_and_interp(lam_interp, filter_interp, bandpass_wav, bandpass_val, central_wavelength, buffer_um=0.2, figsize=(8, 3), \
                            return_fig=False):
    
    ''' For checking accuracy of filter interpolation '''
    fig = plt.figure(figsize=(8, 3))
    plt.plot(lam_interp, filter_interp/np.max(filter_interp), color='r')
    plt.plot(bandpass_wav, bandpass_val/np.max(bandpass_val))
    plt.xlim(central_wavelength-buffer_um, central_wavelength+buffer_um)
    plt.show()
    
    if return_fig:
        return fig

def plot_norm_phot_fluxes(phot_fluxes, bins=30):  
    plt.figure(figsize=(5, 4))
    plt.title('Normalized photometric fluxes')
    plt.hist(phot_fluxes.ravel(), bins=50)
    plt.yscale('log')
    plt.show()

def plot_log_phot_weights(phot_weights, bins=30):
    plt.figure(figsize=(5, 4))
    plt.hist(np.log10(phot_weights.ravel()), bins=50)
    plt.yscale('log')
    plt.xlabel('log10(Photometry weights)')
    plt.show()

def plot_unnorm_fluxes(phot_fluxes):
    plt.figure(figsize=(5, 4))
    plt.title('Unnormalized photometric fluxes')
    plt.hist(phot_fluxes.ravel(), bins=50)
    plt.yscale('log')
    plt.show()

def plot_snr_persource(snr_persource, snr_min=0, snr_max=500, nbin=50):

    snr_bins = np.linspace(snr_min, snr_max, nbin)
    
    plt.figure(figsize=(5, 4))
    plt.hist(snr_persource, bins=snr_bins, histtype='step')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Per-source SNR')
    plt.show()

def plot_lbfgs_result_spec(wav_plot, result_dict_lbfgs, figsize=(7, 4), lab_fs=14):
    
    n_init = result_dict_lbfgs['u_map_all'].shape[0]

    map_spec_all = result_dict_lbfgs['map_spec_all'].cpu().detach().numpy()
    map_spec_glob = result_dict_lbfgs['map_spec_glob'].cpu().detach().numpy()
    spec_noiseless = result_dict_lbfgs['spec_noiseless']
    

    fig = plt.figure(figsize=figsize)

    for initidx in range(n_init):
        plt.plot(wav_plot, map_spec_all[initidx], color='grey')
        
    plt.plot(wav_plot, map_spec_glob, color='r', label='Global MAP')
    
    plt.plot(wav_plot, spec_noiseless, color='k', linewidth=2, label='Truth (Noiseless)')
    plt.legend()
    plt.ylabel('Flux [normalized]', fontsize=lab_fs)
    plt.xlabel('$\\lambda$ [$\\mu$m]', fontsize=lab_fs)
    
    plt.show()
    
    return fig

def plot_map_recon(central_wavelengths, spec_test, spec_recon_init=None, MAP_spec=None, flux_unc=None, figsize=(8, 4), bbox_to_anchor=[0.9, 1.35], \
                   spec_recon_med=None, spec_recon_lopct=None, spec_recon_hipct=None, spec_truth=None, ylim=None):
    
    fig = plt.figure(figsize=figsize)
    
    if flux_unc is not None:
        unit = '[uJy]'
        plt.errorbar(central_wavelengths, spec_test, yerr=flux_unc, fmt='o', color='k', label='Observed', markersize=3, alpha=0.2)
        plt.fill_between(central_wavelengths, -flux_unc, flux_unc, label='1$\\sigma$ flux density uncertainties', alpha=0.1, color='k')
        chi2 = ((spec_test - MAP_spec)/flux_unc)**2
        sum_chi2 = np.sum(chi2)
        print('sum chi2 MAP spec is ', sum_chi2)
        
        if spec_recon_init is not None:
            chi2_init = ((spec_test - spec_recon_init)/flux_unc)**2
            sum_chi2_init = np.sum(chi2_init)
            print('sum chi2 from initialized value is ', sum_chi2_init)
            
    else:
        unit = '[norm.]'
    
    if spec_truth is not None:
        plt.plot(central_wavelengths, spec_truth, color='limegreen', label='Truth', linewidth=3)
        
        if spec_recon_med is not None:
            plt.plot(central_wavelengths, spec_recon_med-spec_truth, color='limegreen', linestyle='dashed', linewidth=2, label='Posterior median - truth')
            
    else:
        if spec_recon_med is not None:
            plt.plot(central_wavelengths, spec_test-spec_recon_med, color='k', label='Observed-median')

    if spec_recon_med is not None:
        chi2_med = ((spec_test - spec_recon_med)/flux_unc)**2
        sum_chi2_med = np.sum(chi2_med)
        print('sum chi2 from posterior median is ', sum_chi2_med)

        plt.fill_between(central_wavelengths, spec_recon_lopct, spec_recon_hipct, alpha=0.5, color='k')
        plt.fill_between(central_wavelengths, spec_recon_lopct-spec_recon_med, spec_recon_hipct-spec_recon_med, alpha=0.5, color='k', label='Reconstruction uncertainty')

        plt.plot(central_wavelengths, spec_recon_med, color='k', label='Median reconstruction')
        
    elif MAP_spec is not None:  
        plt.plot(central_wavelengths, MAP_spec, color='C3', label='MAP')
        plt.plot(central_wavelengths, MAP_spec-spec_truth, color='b', label='Truth (noiseless) - MAP')

    if spec_recon_init is not None:
        plt.plot(central_wavelengths, spec_recon_init, color='b', label='Initial point', linestyle='dashed')
    plt.ylim(ylim)
    plt.legend(loc=2, ncol=2, fontsize=9)
    plt.xlabel('$\\lambda_{rest}$ [$\\mu$m]', fontsize=14)
    plt.ylabel('Flux density '+unit, fontsize=14)
    plt.show()
    
    return fig


def plot_train_validation_logL(nstep, metric_dict, show=True, return_fig=True, figsize=(5, 4), logscale=False):
    
    f = plt.figure(figsize=figsize)
    
    plt.plot(np.arange(nstep), -1.*np.array(metric_dict['train_logL']), label='Train')
    plt.plot(np.arange(nstep), -1.*np.array(metric_dict['valid_logL']), label='Validation')
    
    plt.ylabel('-logL', fontsize=14)
    plt.xlabel('Epoch', fontsize=14) 
    plt.legend(fontsize=14)
    plt.tick_params(labelsize=14)
    
    if logscale:
        plt.yscale('log')

    plt.tight_layout()
    if show:
        plt.show()
    if return_fig:
        return f
    

def plot_train_validation_results_simp(nstep, metric_dict, show=True, return_fig=True, figsize=(8, 6)):

    metric_list = ['loss', 'logL', 'KLD', 'MMD']
    f = plt.figure(figsize=figsize)
    for m, metric_name in enumerate(metric_list):
        
        if metric_name=='logL':
            metric_name_plot = '-logL'
            fac = -1
        else:
            metric_name_plot = metric_name
            fac = 1.
            
        plt.subplot(2,2,1+m)
        plt.plot(np.arange(nstep), fac*np.array(metric_dict['train_'+metric_name]), label='Train')
        plt.plot(np.arange(nstep), fac*np.array(metric_dict['valid_'+metric_name]), label='Validation')
        plt.xlabel('Epoch', fontsize=16)

        if metric_name in ['logL', 'loss']:
            plt.yscale('log')
        plt.ylabel(metric_name_plot, fontsize=16)
        plt.legend(fontsize=14)
        plt.tick_params(labelsize=14)

    plt.tight_layout()
    if show:
        plt.show()
    if return_fig:
        return f


def plot_train_validation_loss_jax(metric_dict=None, metrics_file=None, show=True, return_fig=True, figsize=(7, 5), logscale=True):
    """
    Plot training and validation loss curves for JAX autoencoder.
    
    Parameters
    ----------
    metric_dict : dict, optional
        Dictionary with 'train_loss' and 'valid_loss' keys
    metrics_file : str, optional
        Path to metrics.npz file (alternative to metric_dict)
    show : bool
        Whether to display the plot
    return_fig : bool
        Whether to return the figure object
    figsize : tuple
        Figure size
    logscale : bool
        Whether to use log scale for y-axis
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object (if return_fig=True)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load metrics from file if provided
    if metrics_file is not None:
        metrics_data = np.load(metrics_file)
        train_loss = metrics_data['trainloss']
        valid_loss = metrics_data['validloss']
    elif metric_dict is not None:
        train_loss = np.array(metric_dict['train_loss'])
        valid_loss = np.array(metric_dict['valid_loss'])
    else:
        raise ValueError("Must provide either metric_dict or metrics_file")
    
    nstep = len(train_loss)
    epochs = np.arange(1, nstep + 1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(epochs, train_loss, 'o-', label='Train', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(epochs, valid_loss, 's-', label='Validation', linewidth=2, markersize=4, alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss (MSE)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    if logscale:
        ax.set_yscale('log')
        ax.set_ylabel('Loss (MSE, log scale)', fontsize=14)
    
    # Add final loss values to legend
    ax.text(0.98, 0.98, 
            f'Final Train: {train_loss[-1]:.2e}\\nFinal Valid: {valid_loss[-1]:.2e}',
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            fontsize=10)

    plt.tight_layout()

    if show:
        plt.show()

    if return_fig:
        return fig


def plot_sed_recon_epoch(ae_modl, trainloader, flux_unc, train_mode, sed_lams, figsize=(8, 6), alph=0.4, \
                        bbox_to_anchor=[-1.1, 2.2], legend_fs=12, ncol=2, xlim=[0.5, 5.0]):
    f = plt.figure(figsize=figsize)

    for batch_idx, data in enumerate(trainloader):

        if batch_idx < 4:

            if flux_unc is None:
                norms, orig = data[1].cpu(), data[2].cpu()
            else:
                weig, norms, orig = data[1].cpu(), data[2].cpu(), data[3].cpu()
            normplot = norms[0].detach().numpy()

            recon = ae_modl.forward(data[0]).cpu()

            xval = sed_lams.copy()
            xlab = '$\\lambda$ [$\\mu$m]'

            data_flux_uJy = normplot*(data[0][0].cpu().detach().numpy())
            modl_recon_flux_uJy = normplot*((recon[0].cpu().detach().numpy()))

            plt.subplot(2,2,batch_idx+1)

            if flux_unc is not None:
                plt.errorbar(xval, normplot*(data[0][0].cpu().detach().numpy()),yerr=flux_unc, color='b', fmt='o', markersize=2, capsize=2, alpha=alph, label='Data')

            if flux_unc is not None:

                plt.plot(xval, normplot*(recon[0].detach().numpy()), label='Reconstructed', zorder=10, color='C3')
                plt.plot(xval, orig[0].detach().numpy(), label='Original (noiseless)', zorder=10, color='k')
                plt.plot(xval,orig[0].detach().numpy()-normplot*((recon[0].detach().numpy())), label='Original (noiseless) - Reconstruction', linestyle='dashed', color='C3')
                plt.fill_between(xval, -flux_unc, flux_unc, label='1$\\sigma$ flux density uncertainties', alpha=0.5, color='grey')

            else:
                plt.plot(xval, recon[0].detach().numpy(), label='Reconstructed', zorder=10, color='C3')

                plt.plot(xval, data[0][0].cpu().detach().numpy(), label='Original (noiseless)', zorder=10, color='k')
                plt.plot(xval,data[0][0].cpu().detach().numpy()-((recon[0].detach().numpy())), label='Original (noiseless) - Reconstruction', linestyle='dashed', color='C3')

            plt.tick_params(labelsize=14)
            if batch_idx==0 or batch_idx==2:
                if flux_unc is not None:
                    plt.ylabel('Flux density [$\\mu$Jy]', fontsize=16)
                else:
                    plt.ylabel('Flux density [norm.]', fontsize=16)

            if batch_idx > 1:
                plt.xlabel(xlab, fontsize=16)

            if flux_unc is not None:
                chi2 = np.sum((data_flux_uJy-modl_recon_flux_uJy)**2/flux_unc**2)
                textstr = '$\\chi^2 = $'+str(np.round(chi2, 1))+'/'+str(len(flux_unc))
                ymin = min(-4, 1.5*np.min((orig[0]-recon[0]).detach().numpy()))
                ymax = (4+1.5*np.max(orig[0].detach().numpy()))
                plt.text(0.6, ymax*0.8, textstr, fontsize=13)

                plt.ylim(ymin, ymax)

            plt.xlim(xlim)

        else:
            plt.legend(fontsize=legend_fs, ncol=ncol, loc='lower left', bbox_to_anchor=bbox_to_anchor)
            plt.show()
            break

    return f


def plot_sphx_photometry(property_cat_df, dat_obj, nsrc=3, figsize=(7, 3), include_ext_phot=False, ext_phot_flux=None, ext_phot_flux_unc=None, \
                        bbox_to_anchor=[-0.05, 1.3], ylim=[5e-1, 300], xlim=[0.3, 5.1], which_set='COSMOS', return_figs=False):

    tids = []
    figs = []
    
    central_wavelengths = dat_obj.sed_um_wave
    
    for x in range(nsrc):
        f = plt.figure(figsize=figsize)

        # whichmatch = np.where((linecat_tid==catgrid_deep_withunc[x+50,0]))[0][0]
        if which_set=='COSMOS':
            tractor_ID = property_cat_df['Tractor_ID'][x]
            tids.append(tractor_ID)
            
        zplot, logMplot = property_cat_df['redshift'][x], property_cat_df['mass_best'][x]

        plt.scatter(central_wavelengths, dat_obj.catgrid_flux_noiseless[x], color='grey', \
                    alpha=0.8, s=8, label='SPHEREx (noiseless)')

        plt.errorbar(central_wavelengths, dat_obj.flux[x], yerr=dat_obj.flux_unc[x], fmt='o', color='C3', \
            markersize=4, zorder=10, label='SPHEREx observed', alpha=0.5)
        
        plt.plot(central_wavelengths, dat_obj.flux_unc[x], color='C3', linestyle='dashed', label='1$\\sigma$ noise uncertainty')

        if include_ext_phot: # not currently using
            plt.errorbar([3.4, 4.5, 0.48, 0.65, 0.9], ext_phot_flux[x], yerr=ext_phot_flux_unc[x], fmt='o', color='k', \
                markersize=6,  label='External photometry (LS, WISE)', zorder=11, marker='o', capthick=2)

        plt.yscale('log')
        plt.legend(loc=2, ncol=3, bbox_to_anchor=bbox_to_anchor)
        plt.xlabel('$\\lambda$ [$\\mu$m]', fontsize=14)
        plt.ylabel('$S_{\\lambda}$ [uJy]', fontsize=14)
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.tick_params(labelsize=14)
        plt.text(0.5, 150, 'z='+str(np.round(zplot, 3))+', logM$_*$='+str(np.round(logMplot, 1)), fontsize=14)
        plt.show()
        
        figs.append(f)
        
    if return_figs:
        return figs, tids
    else:
        return tids

    
def plot_mse_stats(mse_perobj, mse_vs_lam, wav, labels=['Test galaxies'], ylim_lam=[1e-4, 1e-2], figsize=(5, 5), \
                  logmin=-4.5, logmax=-1, nbins=30, lab_fs=14, title=None, title_fs=14, bbox_to_anchor=[-0.05, 1.35]):


    if len(mse_perobj)==1:
        mse_perobj = [mse_perobj]
    
    fig1, ax = plt.subplots(figsize=(5, 5), nrows=2, ncols=1) 

    plt.subplot(2,1,1)
    
    if title is not None:
        plt.title(title, fontsize=title_fs)
    for x in range(len(mse_perobj)):
        plt.hist(mse_perobj[x], bins=np.logspace(logmin, logmax, nbins), histtype='step', label=labels[x])
    plt.xlabel('MSE per object [normalized flux units]', fontsize=lab_fs)
    plt.ylabel('$N_{gal}$', fontsize=lab_fs)
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=2, ncol=2)

    plt.xscale('log')

    if len(mse_vs_lam)==1:
        mse_vs_lam = [mse_vs_lam]

    plt.subplot(2,1,2)

    for x in range(len(mse_vs_lam)):
        plt.plot(wav, mse_vs_lam[x], label=labels[x])
    plt.xlabel('Rest frame wavelength [$\\mu$m]', fontsize=lab_fs)
    plt.ylabel('MSE', fontsize=lab_fs)
    plt.yscale('log')
    plt.ylim(ylim_lam)
    plt.grid(alpha=0.3)        
    plt.tight_layout()
    plt.show()
    
    return fig1

    
def plot_chi2_stats(chi2, phot_snr, ncode, log_snr_nbin=20, show=True, snr_min=15.0, snr_max=1e3, central_wavelengths=None):
    
    chi2_perobj = np.sum(chi2, axis=1)
    print('chi2 per obj has shape', chi2_perobj.shape)
    nbands = chi2.shape[1]

    fig1 = plt.figure(figsize=(4, 3))
    plt.text(1e3, 1e3, '102 bands\n$n_{latent}=$'+str(ncode), fontsize=14)
    plt.hist(chi2_perobj, bins=np.logspace(np.log10(30), 4, 50))
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('$N_{gal}$', fontsize=12)
    plt.xlabel('Reconstruction $\\chi^2$', fontsize=12)
    if show:
        plt.show()
    
    fig2 = plt.figure(figsize=(4, 3))
    plt.scatter(phot_snr, chi2_perobj, alpha=0.02, s=2, color='k')    
    plt.axhline(102-ncode, linestyle='dashed', color='r', label='$N_{chan}-dim(z)$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Source SNR', fontsize=12)
    plt.ylabel('Reconstruction $\\chi^2$', fontsize=12)
    plt.legend()
    plt.ylim(50, 1e3)
    plt.xlim(1e1, 3e3)
    if show:
        plt.show()
        
    chi2_lam_snr = np.zeros((log_snr_nbin, nbands))
    log_snr_bins = np.logspace(np.log10(snr_min), np.log10(snr_max), log_snr_nbin+1)
    snr_bin_idxs = np.digitize(phot_snr, log_snr_bins)
    
    for x in range(log_snr_nbin):
        
        which_inbin = np.where((phot_snr > log_snr_bins[x])*(phot_snr <= log_snr_bins[x+1]))[0]
        chi2_inbin = chi2[which_inbin, :]
        # print('chi2 in bin is ', chi2_inbin)
        
        chi2_lam_snr[x] = np.mean(chi2_inbin, axis=0)
    
    # print(chi2_lam_snr)
    fig3 = plt.figure(figsize=(8, 6))
    
    plt.imshow(chi2_lam_snr, cmap='jet', vmax=5.0, vmin=0.0)
    
    ytick_idx = 0, 5, 10, 15, 19
    mean_snr = [np.sqrt(log_snr_bins[x]*log_snr_bins[x+1]) for x in range(len(log_snr_bins)-1)]
    plt.yticks(ytick_idx, [np.round(mean_snr[idx], 1) for idx in ytick_idx])
    
    if central_wavelengths is not None:
        xtick_idx = [0]+[x*17 for x in np.arange(1, 6)] + [len(central_wavelengths)-1]
        plt.xticks(xtick_idx, [np.round(central_wavelengths[idx], 2) for idx in xtick_idx])
    cbar = plt.colorbar(orientation='horizontal')
    plt.title('Mean $\\chi^2$ vs. $\\lambda$ vs. SNR', fontsize=16)
    plt.xlabel('$\\lambda_{obs}$ [$\\mu$m]', fontsize=14)
    plt.ylabel('Observed SNR', fontsize=13)
    
    if show:
        plt.show()
    
    
    return fig1, fig2, fig3

def make_color_corner_plot(ncode, latent_z, feature_vals, feature_name, feature_logscale=False, \
                          vmin=None, vmax=None, figsize=(7, 6), hist_bins=None, color='k', xlim=None, ylim=None, \
                          yticks=None, alph=0.5, s=2, use_contour=False, labels=None, legend_fs=16, bbox_to_anchor=[2, 2]):
        
    if type(latent_z)!=list:
        latent_z = [latent_z]
    
    if type(color) != list:
        color = [color]
        
    if labels is None:
        labels = [None for n in range(len(color))]
            
    fig, ax = plt.subplots(figsize=figsize, nrows=ncode, ncols=ncode, sharex=True)
    
    for ix in range(ncode):
        
        for iy in range(ncode):
            
            if iy > 0 or ix==iy:
                ax[ix,iy].set_yticks([], [])
            else:
                ax[ix,iy].set_yticks(yticks)
            
            if iy < ix+1:
                ax[ix, iy].set_xticks(yticks)
                ax[ix, iy].set_yticks(yticks)

                if iy==0 and ix>0:
                    ax[ix, iy].set_ylabel('$z_'+str(ix)+'$')
                if ix==ncode-1:
                    ax[ix, iy].set_xlabel('$z_'+str(iy)+'$')
                
                for n in range(len(latent_z)):

                    if ix==iy:
                        if ix==0:
                            label = labels[n]
                        else:
                            label = None
                        ax[ix,iy].hist(latent_z[n][:,ix], histtype='step', bins=hist_bins, color=color[n], density=True, label=label)
                    else: 
                        if feature_name is not None:

                            if feature_logscale:
                                sub = ax[ix,iy].scatter(latent_z[n][:,iy], latent_z[n][:,ix], c=feature_vals, alpha=alph, s=s, cmap='jet', norm=colors.LogNorm(vmin=vmin, vmax=vmax))
                            else:
                                sub = ax[ix,iy].scatter(latent_z[n][:,iy], latent_z[n][:,ix], c=feature_vals, alpha=alph, s=s, vmin=vmin, vmax=vmax, cmap='jet')
                        else:
                            sub = ax[ix,iy].scatter(latent_z[n][:,iy], latent_z[n][:,ix], color=color[n], alpha=alph, s=s)

                        ax[ix,iy].set_xlim(xlim)
                        ax[ix,iy].set_ylim(xlim)
                ax[ix,iy].grid(alpha=0.3)


            else:
                fig.delaxes(ax[ix,iy])
    
    if feature_name is not None:
        cbar_ax = fig.add_axes([0.75, 0.35, 0.05, 0.5])
        cbar = fig.colorbar(sub, cax=cbar_ax)
        cbar.set_label(feature_name, size=14)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()
    
    return fig
               
def plot_udist_train_validation(nf_latents_train, nf_latents_valid=None, figsize=(7, 5)):

    fig = plt.figure(figsize=figsize)

    return fig


def make_latent_corner_plots(latents_rescaled, latents_u, nlatent, nsamp=10000, 
                             xlim=5, figsize=(6, 6)):
    """
    Generate corner plots for autoencoder and normalizing flow latents.
    
    This function creates two corner plots: one for autoencoder latents (zeta space)
    and one for normalizing flow latents (u space). Useful for visualizing the
    distribution of latent variables after training.
    
    Parameters
    ----------
    latents_rescaled : array_like, shape (n_samples, nlatent)
        Rescaled autoencoder latents (ζ space)
    latents_u : array_like, shape (n_samples, nlatent)
        Normalizing flow latents (u space)
    nlatent : int
        Number of latent dimensions
    nsamp : int, optional
        Number of samples to use for plotting (default: 10000)
    xlim : float, optional
        Axis limits for corner plots, symmetric around 0 (default: 3)
    figsize : tuple, optional
        Figure size in inches (default: (6, 6))
        
    Returns
    -------
    fig_corner_zeta : matplotlib.figure.Figure
        Autoencoder latent corner plot (ζ space)
    fig_corner_u : matplotlib.figure.Figure
        Normalizing flow latent corner plot (u space)
        
    """
    import corner
    
    # Set up ranges
    ranges = [(-xlim, xlim) for _ in range(nlatent)]
    
    # Subsample if needed
    n_available = min(latents_rescaled.shape[0], latents_u.shape[0])
    if n_available > nsamp:
        rng = np.random.RandomState(42)
        idx = rng.choice(n_available, nsamp, replace=False)
        latents_rescaled_subset = latents_rescaled[idx]
        latents_u_subset = latents_u[idx]
    else:
        latents_rescaled_subset = latents_rescaled[:n_available]
        latents_u_subset = latents_u[:n_available]
    
    # Generate autoencoder latent corner plot
    fig_zeta, axes_zeta = plt.subplots(figsize=figsize, nrows=nlatent, ncols=nlatent)
    
    fig_corner_zeta = corner.corner(
        np.array(latents_rescaled_subset), 
        fig=fig_zeta, 
        axes=axes_zeta,
        labels=[f'$\\zeta_{{{n}}}$' for n in range(nlatent)], 
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84], 
        plot_contours=False, 
        plot_density=False, 
        title_kwargs={"fontsize": 10}, 
        range=ranges,
        data_kwargs={'alpha': 0.02}
    )
    
    # Adjust tick labels
    for ax in fig_corner_zeta.get_axes():
        ax.tick_params(labelsize=8)
    
    # Add label to autoencoder plot
    if nlatent >= 2:
        ax_top_right = fig_corner_zeta.axes[nlatent - 2]
        ax_top_right.text(
            1.1, 0.9, 'Autoencoder latents',
            fontsize=18,
            ha='right',
            va='top',
            transform=ax_top_right.transAxes,
            color='r'
        )
    
    # Generate normalizing flow latent corner plot
    fig_u, axes_u = plt.subplots(figsize=figsize, nrows=nlatent, ncols=nlatent)
    
    fig_corner_u = corner.corner(
        np.array(latents_u_subset),
        fig=fig_u,
        axes=axes_u,
        labels=[f'$u_{{{n}}}$' for n in range(nlatent)], 
        show_titles=True,
        quantiles=[0.16, 0.5, 0.84], 
        title_kwargs={"fontsize": 10}, 
        range=ranges, 
        smooth=0.5,
        levels=[0.68, 0.95],
        data_kwargs={'alpha': 0.002}, 
        plot_contours=False, 
        plot_density=False
    )
    
    # Adjust tick labels
    for ax in fig_corner_u.get_axes():
        ax.tick_params(labelsize=8)
    
    # Add label to normalizing flow plot
    if nlatent >= 2:
        ax_top_right_u = fig_corner_u.axes[nlatent - 2]
        ax_top_right_u.text(
            1.15, 0.85, 'Normalizing flow latents',
            fontsize=18,
            ha='right',
            va='top',
            transform=ax_top_right_u.transAxes,
            color='b'
        )
    
    return fig_corner_zeta, fig_corner_u


def plot_latent_corner_plots(rundir, nlatent, nsamp=10000, xlim=5, figsize=(6, 6), 
                             save_ae=True, save_nf=True, dpi=300, flow_name='flow_model_iaf'):
    """
    Generate corner plots for autoencoder and normalizing flow latents.
    
    This function loads the saved latents and creates corner plots showing the
    distribution of latent variables in both autoencoder space (zeta) and 
    normalizing flow space (u).
    
    Parameters
    ----------
    rundir : str
        Path to the model run directory containing latents
    nlatent : int
        Number of latent dimensions
    nsamp : int, optional
        Number of samples to use for plotting (default: 10000)
    xlim : float, optional
        Axis limits for corner plots (default: 5)
    figsize : tuple, optional
        Figure size in inches (default: (6, 6))
    save_ae : bool, optional
        Whether to save autoencoder latent corner plot (default: True)
    save_nf : bool, optional
        Whether to save normalizing flow latent corner plot (default: True)
    dpi : int, optional
        DPI for saved figures (default: 300)
    flow_name : str, optional
        Name of the flow model file (without .pkl extension) to include in output filenames
        (default: 'flow_model_iaf'). Use this to distinguish between different flow configs.
        
    Returns
    -------
    fig_ae : matplotlib.figure.Figure or None
        Autoencoder corner plot figure (if save_ae=True)
    fig_nf : matplotlib.figure.Figure or None
        Normalizing flow corner plot figure (if save_nf=True)
        

    """
    import corner
    import jax
    from pathlib import Path
    
    print("\n" + "="*70)
    print("GENERATING LATENT CORNER PLOTS")
    print("="*70)
    
    # Only proceed if latents_with_u_space.npz exists (with both z and u latents)
    latent_file = Path(rundir) / 'latents' / 'latents_with_u_space.npz'
    if not latent_file.exists():
        print(f"✗ Required file not found: {latent_file}")
        print("  This function requires latents_with_u_space.npz from flow training")
        print("  Run flow training first to generate this file")
        return None, None
    
    print(f"Loading latents from: {latent_file.name}")
    latents = np.load(latent_file)
    
    print(f"Available keys in latent file: {list(latents.keys())}")
    
    # Check that all required keys are present
    required_keys = ['latents_z_train', 'latents_u_train', 'loc', 'scale']
    missing_keys = [k for k in required_keys if k not in latents]
    if missing_keys:
        print(f"✗ Missing required keys: {missing_keys}")
        print(f"  File must contain: {required_keys}")
        return None, None
    
    # Load all required data
    latents_train = latents['latents_z_train']
    latents_u_train = latents['latents_u_train']
    loc = latents['loc']
    scale = latents['scale']
    
    print(f"Loaded autoencoder latents (z-space): {latents_train.shape}")
    print(f"Loaded normalizing flow latents (u-space): {latents_u_train.shape}")
    
    # Apply affine transformation to get rescaled latents for plotting
    # This gives unit Gaussian normalization: (latents * scale) + loc
    print("Applying affine rescale transformation to autoencoder latents...")
    latents_rescaled = (latents_train * scale) + loc
    latents_rescaled = np.array(latents_rescaled)
    
    print(f"Rescaled autoencoder latents: {latents_rescaled.shape}")
    print("\nAutoencoder latent statistics (rescaled, unit Gaussian):")
    for dim in range(nlatent):
        std_val = np.std(latents_rescaled[:, dim])
        mean_val = np.mean(latents_rescaled[:, dim])
        print(f"  Dimension {dim}: mean={mean_val:+.4f}, std={std_val:.4f}")
    
    # Randomly select samples if we have more than nsamp
    n_available = latents_train.shape[0]
    if n_available > nsamp:
        rng = np.random.RandomState(42)  # Fixed seed for reproducibility
        idx = rng.choice(n_available, nsamp, replace=False)
        latents_subset = latents_rescaled[idx]
        latents_u_subset = latents_u_train[idx]
    else:
        latents_subset = latents_rescaled[:nsamp]
        latents_u_subset = latents_u_train[:nsamp]
        nsamp = n_available
    
    print(f"\nUsing {nsamp} samples for corner plots")
    
    # Set up ranges
    ranges = [(-xlim, xlim) for _ in range(nlatent)]
    
    # Create output directory
    fig_dir = Path(rundir) / 'figures' / 'latent_corner_plots'
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    fig_ae, fig_nf = None, None
    
    # Generate autoencoder latent corner plot
    if save_ae:
        print("\nGenerating autoencoder latent corner plot...")
        fig_ae, axes_ae = plt.subplots(figsize=figsize, nrows=nlatent, ncols=nlatent)
        
        fig_ae = corner.corner(
            latents_subset, 
            fig=fig_ae, 
            axes=axes_ae,
            labels=[f'$\\zeta_{{{n}}}$' for n in range(nlatent)], 
            show_titles=True,
            quantiles=[0.16, 0.5, 0.84], 
            plot_contours=False, 
            plot_density=False, 
            title_kwargs={"fontsize": 10}, 
            range=ranges,
            data_kwargs={'alpha': 0.02}
        )
        
        # Adjust tick labels
        for ax in fig_ae.get_axes():
            ax.tick_params(labelsize=8)
        
        # Add label
        if nlatent >= 2:
            ax_top_right = fig_ae.axes[nlatent - 2]
            ax_top_right.text(
                1.1, 0.9, 'Autoencoder latents',
                fontsize=18,
                ha='right',
                va='top',
                transform=ax_top_right.transAxes,
                color='r'
            )
        
        # Save
        save_path_ae = fig_dir / f'corner_plot_ae_latents_{flow_name}.png'
        fig_ae.savefig(save_path_ae, bbox_inches='tight', dpi=dpi)
        print(f"✓ Saved autoencoder corner plot: {save_path_ae}")
    
    # Generate normalizing flow latent corner plot
    if save_nf:
        print("\nGenerating normalizing flow latent corner plot...")
        
        print(f"Using normalizing flow latents: {latents_u_train.shape}")
        print("\nNormalizing flow latent statistics (u-space):")
        for dim in range(nlatent):
            std_val = np.std(latents_u_train[:, dim])
            mean_val = np.mean(latents_u_train[:, dim])
            print(f"  Dimension {dim}: mean={mean_val:+.4f}, std={std_val:.4f}")
        
        fig_nf, axes_nf = plt.subplots(figsize=figsize, nrows=nlatent, ncols=nlatent)
        
        fig_nf = corner.corner(
            latents_u_subset,
            fig=fig_nf,
            axes=axes_nf,
            labels=[f'$u_{{{n}}}$' for n in range(nlatent)], 
            show_titles=True,
            quantiles=[0.16, 0.5, 0.84], 
            title_kwargs={"fontsize": 10}, 
            range=ranges, 
            smooth=0.5,
            levels=[0.68, 0.95],
            data_kwargs={'alpha': 0.02}, 
            plot_contours=False, 
            plot_density=False
        )
        
        # Adjust tick labels
        for ax in fig_nf.get_axes():
            ax.tick_params(labelsize=8)
        
        # Add label
        if nlatent >= 2:
            ax_top_right_u = fig_nf.axes[nlatent - 2]
            ax_top_right_u.text(
                1.15, 0.85, 'Normalizing flow latents',
                fontsize=18,
                ha='right',
                va='top',
                transform=ax_top_right_u.transAxes,
                color='b'
            )
        
        # Save
        save_path_nf = fig_dir / f'corner_plot_nf_latents_{flow_name}.png'
        fig_nf.savefig(save_path_nf, bbox_inches='tight', dpi=dpi)
        print(f"✓ Saved normalizing flow corner plot: {save_path_nf}")
    
    print("\n" + "="*70)
    print("CORNER PLOT GENERATION COMPLETE")
    print("="*70)
    
    return fig_ae, fig_nf


def regenerate_bestfit_overlay_plots(prof_like_files, wave_obs, all_spec_obs, all_flux_unc, 
                                     norms, redshift, output_dir, n_select=10, 
                                     y_scale_factor=1.5, highlight_true_z=True):
    """
    Regenerate best-fit SED overlay plots from saved profile likelihood files.
    
    Parameters
    ----------
    prof_like_files : list of str
        Paths to profile likelihood .npz files
    wave_obs : array
        Observed wavelengths
    all_spec_obs : array
        Observed spectra (Nsrc, Nbands)
    all_flux_unc : array
        Flux uncertainties (Nsrc, Nbands)
    norms : array
        Normalization factors per source
    redshift : array
        True redshifts per source
    output_dir : str
        Directory to save regenerated plots
    n_select : int, optional
        Number of redshift points to display. Default 10.
    y_scale_factor : float, optional
        Scale factor for y-axis upper limit (multiplied by max data value). Default 1.5.
    highlight_true_z : bool, optional
        If True, plot the best-fit closest to true redshift with enhanced linewidth. Default True.
    
    Returns
    -------
    fig_paths : list of str
        Paths to saved figures
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig_paths = []
    
    for pl_file in prof_like_files:
        try:
            # Load profile likelihood results
            pldata = np.load(pl_file)
            z_grid = pldata['z_grid']
            all_bestfit_models = pldata.get('all_bestfit_models', None)
            
            if all_bestfit_models is None:
                print(f"  Warning: No best-fit models in {pl_file}, skipping")
                continue
            
            # Extract source index/ID from filename
            filename = Path(pl_file).stem
            # Parse srcid from filename (e.g., pl_nopost_srcid=12345_...)
            if 'srcid=' in filename:
                src_identifier = filename.split('srcid=')[1].split('_')[0]
            elif 'srcidx=' in filename:
                src_identifier = filename.split('srcidx=')[1].split('_')[0]
            else:
                src_identifier = 'unknown'
            
            # Find source in arrays (assumes order matches or we need better matching)
            # For now, assume it's passed in the right order
            src_idx = 0  # This needs to be properly matched - user should pass correct subset
            
            # Select evenly spaced redshift indices
            nz = len(z_grid)
            idxs = np.linspace(0, nz-1, min(n_select, nz), dtype=int)
            
            # Find index closest to true redshift if highlighting
            true_z_idx = None
            if highlight_true_z and redshift is not None:
                true_z_idx = np.argmin(np.abs(z_grid - redshift[src_idx]))
            
            # Create overlay plot with two panels (SED + chi-squared)
            import matplotlib.gridspec as gridspec
            fig = plt.figure(figsize=(10, 5))
            gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.05)
            ax_main = fig.add_subplot(gs[0])
            ax_resid = fig.add_subplot(gs[1], sharex=ax_main)
            
            # Plot observed data
            obs_flux = all_spec_obs[src_idx] * norms[src_idx]
            obs_err = all_flux_unc[src_idx] * norms[src_idx]
            ax_main.errorbar(wave_obs, obs_flux, yerr=obs_err, 
                       fmt='o', color='k', markersize=3, capsize=2, 
                       label=f'Observed (z_true={redshift[src_idx]:.3f})', alpha=0.7)
            
            # Fixed y-limits for consistency
            ymax = 120
            ymin = -50
            
            # Plot best-fit model at true redshift first (if not in selected indices)
            if true_z_idx is not None and true_z_idx not in idxs:
                bf_true = np.asarray(all_bestfit_models[true_z_idx]).ravel()
                bf_true_z = bf_true * norms[src_idx]
                ax_main.plot(wave_obs, bf_true_z, color='k', 
                       linewidth=3.0, label=f'z={z_grid[true_z_idx]:.3f} (true)', 
                       alpha=0.9, zorder=5, linestyle='dashed')
                # Compute chi for true-z model
                chi_true = ((obs_flux - bf_true_z) / obs_err)
                ax_resid.plot(wave_obs, chi_true, color='k', linewidth=2.5, 
                            alpha=0.9, zorder=5, linestyle='dashed')
            
            # Overlay best-fit models at selected redshifts
            colors_map = plt.cm.jet(np.linspace(0, 1, len(idxs)))
            for i, sel in enumerate(idxs):
                bf = np.asarray(all_bestfit_models[sel]).ravel()
                bf_flux = bf * norms[src_idx]
                
                # Compute chi for this model
                chi = ((obs_flux - bf_flux) / obs_err)
                
                # Check if this is the true-z point
                if highlight_true_z and sel == true_z_idx:
                    ax_main.plot(wave_obs, bf_flux, color='k', 
                           linewidth=3.0, label=f'z={z_grid[sel]:.3f} (true)', 
                           alpha=0.9, zorder=5, linestyle='dashed')
                    ax_resid.plot(wave_obs, chi, color='k', linewidth=3.0, 
                                alpha=0.9, zorder=5, linestyle='dashed')
                else:
                    ax_main.plot(wave_obs, bf_flux, color=colors_map[i], 
                           linewidth=1.5, label=f'z={z_grid[sel]:.3f}', alpha=0.8)
                    ax_resid.plot(wave_obs, chi, color=colors_map[i], linewidth=1.5, alpha=0.8)
            
            # Format chi panel
            ax_resid.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax_resid.set_ylabel('$\\chi$ per channel', fontsize=12)
            ax_resid.set_xlabel('$\\lambda_{obs}$ [μm]', fontsize=14)
            ax_resid.grid(alpha=0.3)
            ax_resid.set_ylim(-5, 5)
            
            ax_main.set_ylabel('Flux ($\\mu$Jy)', fontsize=14)
            ax_main.set_ylim(ymin, ymax)
            ax_main.legend(fontsize=10, ncol=4, bbox_to_anchor=(0.0, 1.3), loc=2)
            ax_main.grid(alpha=0.3)
            plt.setp(ax_main.get_xticklabels(), visible=False)
            
            # plt.tight_layout()
            
            # Save figure
            fig_path = output_dir / f'bestfit_overlay_srcid={src_identifier}_regenerated.png'
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  Saved: {fig_path}")
            fig_paths.append(str(fig_path))
            
        except Exception as e:
            print(f"  Error processing {pl_file}: {e}")
            continue
    
    return fig_paths


# =============================================================================
# NEW PAPER PLOTTING FUNCTIONS
# =============================================================================

def plot_nz_comparison(z_true, z_out_pae, z_out_tf, sigmaz_pae=None, z_out_sbi=None,
                       sigmaz_tf=None, z_bins=None, figsize=(10, 6), 
                       sigz_cut=None, z_range=(0, 3.0), legend_fs=12,
                       labels=['Truth', 'PAE', 'TF', 'SBI'], colors=['k', 'C0', 'C1', 'C2'],
                       alpha=0.7, linewidth=2, title=None):
    """
    Plot N(z) distribution comparison between true and estimated redshifts.
    
    Parameters
    ----------
    z_true : array
        True redshifts
    z_out_pae : array
        PAE estimated redshifts
    z_out_tf : array
        Template fitting estimated redshifts
    sigmaz_pae : array, optional
        PAE redshift uncertainties (for quality cuts)
    z_out_sbi : array, optional
        SBI estimated redshifts (if available)
    sigmaz_tf : array, optional
        TF uncertainties (for quality cuts)
    z_bins : array, optional
        Custom redshift bins (default: 30 bins from z_range)
    figsize : tuple
        Figure size
    sigz_cut : float, optional
        Fractional uncertainty cut: sigma_z/(1+z) < sigz_cut
    z_range : tuple
        Redshift range for histogram
    labels : list
        Labels for [truth, PAE, TF, SBI]
    colors : list
        Colors for each curve
    alpha : float
        Transparency for histograms
    linewidth : float
        Line width for histograms
    title : str, optional
        Plot title
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if z_bins is None:
        z_bins = np.linspace(z_range[0], z_range[1], 31)
    
    # Apply quality cut if provided
    if sigz_cut is not None and sigmaz_pae is not None:
        mask_pae = (sigmaz_pae / (1 + z_out_pae)) < sigz_cut
        if sigmaz_tf is not None:
            mask_tf = (sigmaz_tf / (1 + z_out_tf)) < sigz_cut
        else:
            mask_tf = np.ones_like(z_out_tf, dtype=bool)
    else:
        mask_pae = np.ones_like(z_out_pae, dtype=bool)
        mask_tf = np.ones_like(z_out_tf, dtype=bool)
    
    # Plot true N(z)
    n_true, _, _ = ax.hist(z_true, bins=z_bins, histtype='step', 
                           color=colors[0], linewidth=linewidth, alpha=alpha,
                           label=f'{labels[0]} (N={len(z_true)})', density=False)
    
    # Plot PAE N(z)
    n_pae = np.sum(mask_pae)
    ax.hist(z_out_pae[mask_pae], bins=z_bins, histtype='step',
            color=colors[1], linewidth=linewidth, alpha=alpha,
            label=f'{labels[1]} (N={n_pae})', density=False)
    
    # Plot TF N(z)
    n_tf = np.sum(mask_tf)
    ax.hist(z_out_tf[mask_tf], bins=z_bins, histtype='step',
            color=colors[2], linewidth=linewidth, alpha=alpha,
            label=f'{labels[2]} (N={n_tf})', density=False)
    
    # Plot SBI N(z) if available
    if z_out_sbi is not None:
        ax.hist(z_out_sbi, bins=z_bins, histtype='step',
                color=colors[3], linewidth=linewidth, alpha=alpha,
                label=f'{labels[3]} (N={len(z_out_sbi)})', density=False)
    
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel('N(z)', fontsize=14)
    
    if title is None and sigz_cut is not None:
        title = r'$N(z)$ Comparison: $\sigma_z/(1+z) < $' + f'{sigz_cut:.3f}'
    elif title is None:
        title = r'$N(z)$ Comparison'
    
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=legend_fs, loc='best')
    ax.grid(alpha=0.3)
    ax.set_xlim(z_range)
    
    plt.tight_layout()
    return fig


def plot_goodness_of_fit_distribution(chi2, sigmaz=None, z_out=None, z_true=None,
                                       nphot=102, nparam=5, bins=50, figsize=(12, 5),
                                       chi2_max=3.0, sigz_bins=None):
    """
    Plot goodness of fit (reduced chi-squared) distributions, optionally in uncertainty bins.
    
    Parameters
    ----------
    chi2 : array
        Reduced chi-squared values
    sigmaz : array, optional
        Redshift uncertainties for binning
    z_out : array, optional
        Estimated redshifts
    z_true : array, optional
        True redshifts
    nphot : int
        Number of photometric bands
    nparam : int
        Number of parameters
    bins : int
        Number of histogram bins
    figsize : tuple
        Figure size
    chi2_max : float
        Maximum chi2 for x-axis
    sigz_bins : list, optional
        Uncertainty bin edges (e.g., [0.0, 0.01, 0.05, 0.2])
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if sigz_bins is None:
        # Single panel
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        axes = [ax]
    else:
        # Multiple panels
        n_bins = len(sigz_bins) - 1
        ncols = min(3, n_bins)
        nrows = int(np.ceil(n_bins / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if n_bins > 1 else [axes]
    
    valid = np.isfinite(chi2)
    
    # Compute expected distribution for comparison
    dof = nphot - nparam
    expected_mean = 1.0
    expected_std = np.sqrt(2.0 / dof)
    
    if sigz_bins is None:
        # Single histogram
        ax = axes[0]
        ax.hist(chi2[valid], bins=np.linspace(0, chi2_max, bins), 
                edgecolor='k', alpha=0.7, density=True)
        
        # Add expected distribution
        x_theory = np.linspace(0, chi2_max, 200)
        y_theory = scipy.stats.norm.pdf(x_theory, loc=expected_mean, scale=expected_std)
        ax.plot(x_theory, y_theory, 'r--', linewidth=2, 
                label=f'Expected (μ={expected_mean:.1f}, σ={expected_std:.3f})')
        
        ax.axvline(1.0, color='green', linestyle='--', alpha=0.5, label=r'$\chi^2_{\nu}=1$')
        ax.set_xlabel(r'Reduced $\chi^2$', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_title(f'Goodness of Fit Distribution (N={np.sum(valid)})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        # Add statistics
        med_chi2 = np.median(chi2[valid])
        mean_chi2 = np.mean(chi2[valid])
        stats_text = f'Median: {med_chi2:.3f}\nMean: {mean_chi2:.3f}'
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
    else:
        # Multiple histograms by uncertainty bin
        sigz_frac = sigmaz / (1 + z_out) if sigmaz is not None and z_out is not None else None
        
        for i, (ax, sigz_low, sigz_high) in enumerate(zip(axes, sigz_bins[:-1], sigz_bins[1:])):
            if sigz_frac is not None:
                mask = valid & (sigz_frac >= sigz_low) & (sigz_frac < sigz_high)
            else:
                mask = valid
            
            n_in_bin = np.sum(mask)
            
            if n_in_bin > 0:
                ax.hist(chi2[mask], bins=np.linspace(0, chi2_max, bins),
                        edgecolor='k', alpha=0.7, density=True)
                
                # Add expected distribution
                x_theory = np.linspace(0, chi2_max, 200)
                y_theory = scipy.stats.norm.pdf(x_theory, loc=expected_mean, scale=expected_std)
                ax.plot(x_theory, y_theory, 'r--', linewidth=1.5, alpha=0.7)
                
                ax.axvline(1.0, color='green', linestyle='--', alpha=0.5)
                
                # Add title with bin range
                title_str = f'{sigz_low:.3f}' + r' $\leq \sigma_z/(1+z) < $' + f'{sigz_high:.3f}\n(N={n_in_bin})'
                ax.set_title(title_str, fontsize=11)
                
                # Add statistics
                med_chi2 = np.median(chi2[mask])
                mean_chi2 = np.mean(chi2[mask])
                stats_text = f'Med: {med_chi2:.2f}\nMean: {mean_chi2:.2f}'
                ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                        fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
            
            ax.set_xlabel(r'Reduced $\chi^2$', fontsize=12)
            if i % ncols == 0:
                ax.set_ylabel('Density', fontsize=12)
            ax.grid(alpha=0.3)
        
        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        fig.suptitle('Goodness of Fit by Uncertainty Bin', fontsize=16, y=1.00)
    
    plt.tight_layout()
    return fig


def plot_rhat_distribution(R_hat, sigmaz=None, z_out=None, figsize=(12, 5),
                           rhat_max=3.0, bins=50, sigz_bins=None):
    """
    Plot R-hat (Gelman-Rubin) convergence diagnostic distribution.
    
    Parameters
    ----------
    R_hat : array
        Gelman-Rubin R-hat values
    sigmaz : array, optional
        Redshift uncertainties for binning
    z_out : array, optional
        Estimated redshifts
    figsize : tuple
        Figure size
    rhat_max : float
        Maximum R-hat for x-axis
    bins : int
        Number of histogram bins
    sigz_bins : list, optional
        Uncertainty bin edges
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if sigz_bins is None:
        # Single panel
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        axes = [ax]
    else:
        # Multiple panels
        n_bins = len(sigz_bins) - 1
        ncols = min(3, n_bins)
        nrows = int(np.ceil(n_bins / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if n_bins > 1 else [axes]
    
    valid = np.isfinite(R_hat)
    
    if sigz_bins is None:
        # Single histogram
        ax = axes[0]
        ax.hist(R_hat[valid], bins=np.linspace(1.0, rhat_max, bins),
                edgecolor='k', alpha=0.7)
        
        ax.axvline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.7,
                   label=r'$\hat{R}=1$ (perfect)')
        ax.axvline(1.1, color='orange', linestyle='--', linewidth=2, alpha=0.7,
                   label=r'$\hat{R}=1.1$ (threshold)')
        
        ax.set_xlabel(r'Gelman-Rubin $\hat{R}$', fontsize=14)
        ax.set_ylabel('Number of Sources', fontsize=14)
        ax.set_title(f'Convergence Diagnostic Distribution (N={np.sum(valid)})', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        
        # Add statistics
        med_rhat = np.median(R_hat[valid])
        mean_rhat = np.mean(R_hat[valid])
        frac_converged = np.sum(R_hat[valid] < 1.1) / np.sum(valid)
        stats_text = (f'Median: {med_rhat:.3f}\n'
                     f'Mean: {mean_rhat:.3f}\n'
                     f'Frac < 1.1: {frac_converged:.3f}')
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
    else:
        # Multiple histograms by uncertainty bin
        sigz_frac = sigmaz / (1 + z_out) if sigmaz is not None and z_out is not None else None
        
        for i, (ax, sigz_low, sigz_high) in enumerate(zip(axes, sigz_bins[:-1], sigz_bins[1:])):
            if sigz_frac is not None:
                mask = valid & (sigz_frac >= sigz_low) & (sigz_frac < sigz_high)
            else:
                mask = valid
            
            n_in_bin = np.sum(mask)
            
            if n_in_bin > 0:
                ax.hist(R_hat[mask], bins=np.linspace(1.0, rhat_max, bins),
                        edgecolor='k', alpha=0.7)
                
                ax.axvline(1.0, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
                ax.axvline(1.1, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
                
                # Add title with bin range
                title_str = f'{sigz_low:.3f}' + r' $\leq \sigma_z/(1+z) < $' + f'{sigz_high:.3f}\n(N={n_in_bin})'
                ax.set_title(title_str, fontsize=11)
                
                # Add statistics
                med_rhat = np.median(R_hat[mask])
                frac_converged = np.sum(R_hat[mask] < 1.1) / n_in_bin
                stats_text = f'Med: {med_rhat:.2f}\n<1.1: {frac_converged:.2f}'
                ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                        fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
            
            ax.set_xlabel(r'$\hat{R}$', fontsize=12)
            if i % ncols == 0:
                ax.set_ylabel('Count', fontsize=12)
            ax.grid(alpha=0.3)
        
        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        fig.suptitle(r'$\hat{R}$ Distribution by Uncertainty Bin', fontsize=16, y=1.00)
    
    plt.tight_layout()
    return fig


def plot_individual_pz_comparisons(pae_samples, tf_zpdfs, tf_zpdf_z_grid, z_true,
                                    src_indices, sbi_samples=None, 
                                    figsize=(15, 10), n_examples=None,
                                    z_range=(0, 3.0), burn_in=0):
    """
    Plot p(z) comparisons for individual objects between PAE, TF, and optionally SBI.
    
    Parameters
    ----------
    pae_samples : array
        PAE posterior samples, shape (n_sources, n_chains, n_steps, n_dim)
        or (n_sources, n_steps, n_dim). Redshift assumed to be last dimension.
    tf_zpdfs : array
        Template fitting p(z), shape (n_sources, n_z_bins)
    tf_zpdf_z_grid : array
        Redshift grid for TF p(z)
    z_true : array
        True redshifts
    src_indices : array-like
        Indices of sources to plot
    sbi_samples : array, optional
        SBI posterior samples (if available)
    figsize : tuple
        Figure size
    n_examples : int, optional
        Maximum number of examples to plot (overrides src_indices length)
    z_range : tuple
        Redshift range for plotting
    burn_in : int
        Number of burn-in samples to discard from PAE chains
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if n_examples is not None:
        src_indices = src_indices[:n_examples]
    
    n_examples = len(src_indices)
    ncols = min(3, n_examples)
    nrows = int(np.ceil(n_examples / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_examples == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (ax, src_idx) in enumerate(zip(axes, src_indices)):
        # Extract PAE samples for this source
        if len(pae_samples.shape) == 4:
            # Shape: (n_sources, n_chains, n_steps, n_dim)
            z_samples = pae_samples[src_idx, :, burn_in:, -1].ravel()
        else:
            # Shape: (n_sources, n_steps, n_dim)
            z_samples = pae_samples[src_idx, burn_in:, -1]
        
        # Plot PAE p(z) as histogram
        z_bins = np.linspace(z_range[0], z_range[1], 50)
        counts, edges = np.histogram(z_samples, bins=z_bins, density=True)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(bin_centers, counts, color='C0', linewidth=2, label='PAE', alpha=0.8)
        ax.fill_between(bin_centers, counts, alpha=0.3, color='C0')
        
        # Plot TF p(z)
        if tf_zpdfs is not None and src_idx < len(tf_zpdfs):
            tf_pdf = tf_zpdfs[src_idx]
            # Normalize TF PDF to match histogram density
            tf_pdf_norm = tf_pdf / np.trapz(tf_pdf, tf_zpdf_z_grid)
            ax.plot(tf_zpdf_z_grid, tf_pdf_norm, color='C1', linewidth=2,
                    label='TF', alpha=0.8)
        
        # Plot SBI p(z) if available
        if sbi_samples is not None:
            if len(sbi_samples.shape) == 3:
                sbi_z = sbi_samples[src_idx, :, -1]
            else:
                sbi_z = sbi_samples[src_idx, -1]
            
            counts_sbi, _ = np.histogram(sbi_z, bins=z_bins, density=True)
            ax.plot(bin_centers, counts_sbi, color='C2', linewidth=2,
                    label='SBI', alpha=0.8, linestyle='--')
        
        # Plot true redshift
        ax.axvline(z_true[src_idx], color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Truth: z={z_true[src_idx]:.3f}')
        
        # Formatting
        ax.set_xlabel('Redshift', fontsize=11)
        ax.set_ylabel('p(z)', fontsize=11)
        ax.set_title(f'Source {src_idx}', fontsize=12)
        ax.set_xlim(z_range)
        ax.grid(alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=9, loc='best')
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('Individual p(z) Comparisons', fontsize=16, y=0.995)
    plt.tight_layout()
    return fig


def plot_redshift_error_vs_magnitude(z_out, z_true, magnitudes, sigmaz=None,
                                      mag_label='z-band magnitude', figsize=(10, 6),
                                      mag_bins=None, plot_type='scatter',
                                      hexbin_gridsize=30, vmin=1, vmax=None):
    """
    Plot redshift error (or scatter) as a function of magnitude or SNR.
    
    Parameters
    ----------
    z_out : array
        Estimated redshifts
    z_true : array
        True redshifts
    magnitudes : array
        Magnitudes or SNR values
    sigmaz : array, optional
        Redshift uncertainties (for coloring points)
    mag_label : str
        Label for magnitude/SNR axis
    figsize : tuple
        Figure size
    mag_bins : array, optional
        Magnitude bins for computing binned statistics
    plot_type : str
        'scatter', 'hexbin', or 'binned'
    hexbin_gridsize : int
        Grid size for hexbin plot
    vmin, vmax : float
        Color scale limits for hexbin
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Compute redshift error
    dz = (z_out - z_true) / (1 + z_true)
    abs_dz = np.abs(dz)
    
    valid = np.isfinite(magnitudes) & np.isfinite(dz)
    
    if plot_type == 'binned':
        # Binned statistics plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                       gridspec_kw={'height_ratios': [2, 1]})
        
        if mag_bins is None:
            mag_bins = np.linspace(np.percentile(magnitudes[valid], 5),
                                  np.percentile(magnitudes[valid], 95), 15)
        
        mag_centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])
        
        # Compute statistics in each bin
        median_dz = []
        nmad_dz = []
        n_per_bin = []
        
        for i in range(len(mag_bins) - 1):
            mask = valid & (magnitudes >= mag_bins[i]) & (magnitudes < mag_bins[i + 1])
            n_in_bin = np.sum(mask)
            n_per_bin.append(n_in_bin)
            
            if n_in_bin > 5:
                median_dz.append(np.median(dz[mask]))
                nmad = 1.4826 * np.median(abs_dz[mask])
                nmad_dz.append(nmad)
            else:
                median_dz.append(np.nan)
                nmad_dz.append(np.nan)
        
        # Plot bias
        ax1.plot(mag_centers, median_dz, 'o-', color='C0', linewidth=2, markersize=8,
                label='Median bias')
        ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax1.set_ylabel(r'$\Delta z / (1+z_{\rm true})$', fontsize=14)
        ax1.set_title('Redshift Bias vs ' + mag_label, fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
        
        # Plot scatter
        ax2.plot(mag_centers, nmad_dz, 's-', color='C1', linewidth=2, markersize=8,
                label='NMAD')
        ax2.set_xlabel(mag_label, fontsize=14)
        ax2.set_ylabel(r'NMAD($\Delta z$)', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)
        
        # Add second y-axis showing counts
        ax3 = ax2.twinx()
        ax3.bar(mag_centers, n_per_bin, width=mag_bins[1]-mag_bins[0],
                alpha=0.2, color='gray', label='N')
        ax3.set_ylabel('N per bin', fontsize=12, color='gray')
        ax3.tick_params(axis='y', labelcolor='gray')
        
    elif plot_type == 'hexbin':
        # Hexbin density plot
        fig, ax = plt.subplots(figsize=figsize)
        
        hb = ax.hexbin(magnitudes[valid], abs_dz[valid], gridsize=hexbin_gridsize,
                      cmap='YlOrRd', mincnt=1, vmin=vmin, vmax=vmax,
                      norm=LogNorm() if vmax is None else None)
        
        cb = plt.colorbar(hb, ax=ax, label='Number of sources')
        ax.set_xlabel(mag_label, fontsize=14)
        ax.set_ylabel(r'$|\Delta z| / (1+z_{\rm true})$', fontsize=14)
        ax.set_title('Redshift Error vs ' + mag_label, fontsize=14)
        ax.grid(alpha=0.3)
        
    else:
        # Scatter plot
        fig, ax = plt.subplots(figsize=figsize)
        
        if sigmaz is not None:
            # Color by uncertainty
            sigz_frac = sigmaz / (1 + z_out)
            scatter = ax.scatter(magnitudes[valid], abs_dz[valid], c=sigz_frac[valid],
                               s=10, alpha=0.5, cmap='viridis', vmin=0, vmax=0.2)
            cb = plt.colorbar(scatter, ax=ax, label=r'$\sigma_z/(1+z)$')
        else:
            ax.scatter(magnitudes[valid], abs_dz[valid], s=10, alpha=0.5, color='C0')
        
        ax.set_xlabel(mag_label, fontsize=14)
        ax.set_ylabel(r'$|\Delta z| / (1+z_{\rm true})$', fontsize=14)
        ax.set_title('Redshift Error vs ' + mag_label, fontsize=14)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, np.percentile(abs_dz[valid], 99))
    
    plt.tight_layout()
    return fig
