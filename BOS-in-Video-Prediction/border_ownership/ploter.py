# functions for plotting different figures
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import colors
import scipy.stats as sci_st
from mpl_toolkits.axes_grid1 import make_axes_locatable
from border_ownership.rf_finder import RF_Finder_Local_Sparse_Noise_Small_Memory_Center
from scipy.ndimage import gaussian_filter
import matplotlib.gridspec as gridspec
import numpy as np
from kitti_settings import *

def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both',which='both',labelbottom=False,bottom=False,left=False, labelleft=False)
    #ax.get_xaxis().tick_bottom()
    #ax.get_yaxis().tick_left()

def plot_seq_prediction(stimuli, prediction):
    '''
    plot prediction of one sequence
    input:
        stimuli (n_image, *imshape, 3): rgb color
        prediction (n_image, *imshape, 3): the output of Agent() while the output_mode is prediction. The value should be 0 to 255 int
    output:
        fig, ax
    '''

    n_image = stimuli.shape[0]
    fig = plt.figure(figsize = (n_image, 2))
    gs = gridspec.GridSpec(2, n_image)
    gs.update(wspace=0., hspace=0.)
    
    for t, sq_s, sq_p in zip(range(n_image), stimuli, prediction):
        plt.subplot(gs[t])

        ## the image can be ploted without explicit normalization anyway
        #sq_s_norm = cv2.normalize(sq_s, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        #sq_p_norm = cv2.normalize(sq_p, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        
        #sq_s_norm = sq_s_norm.astype(np.uint8)
        #sq_p_norm = sq_p_norm.astype(np.uint8)

        sq_s_norm = sq_s
        sq_p_norm = sq_p

        plt.imshow(sq_s_norm)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

        plt.subplot(gs[t + n_image])
        plt.imshow(sq_p_norm)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    return fig, gs

def plot_example_neuron_response_and_input_single_wrapper(stim_images, respons, rf_mask, figsize=(10, 4.8)):
    pass

def plot_example_neuron_response_and_input(output_module_X, output_module_neuron, angle_idx, neuron_idx, rf_mask=None, figsize=(10, 4.8)):
    '''
    input:
      the output_module_neuron should be in the rf_parametereization (alpha, beta, gamma, t, chs), where chs also means neurons
      the output_module_X are the input images, should be in the rf_parametereization (alpha, beta, gamma, t, width, height, chs)
      angle_idx, neuron_idx: int number
      rf_mask: binary matrix indicating the position of rf
    output:
      figures
    '''
    # show example neural responses
    n_time = output_module_neuron.shape[-2]
    idx = (angle_idx, neuron_idx)

    beta_gamma_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
    color_list = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']

    bg_to_ax_id = {(0, 0): 241, (0, 1): 242, (1, 0): 245, (1, 1): 246}
    fig = plt.figure(figsize=figsize)
    ax_res = fig.add_subplot(122)
    stim_ax_list = []
    for i, bg in enumerate(beta_gamma_list):
        ci = color_list[i]
        stimulus = output_module_X[idx[0], bg[0], bg[1], 0]
        if rf_mask is not None: stimulus = stimulus + rf_mask[..., np.newaxis]
        stimulus = np.clip(stimulus, a_min=0, a_max=1)

        stim_ax_list.append(fig.add_subplot(bg_to_ax_id[(bg[0], bg[1])]))
        ax = stim_ax_list[-1]
        ax.imshow(stimulus)
        simpleaxis(ax)
        ax.set_title(ci, fontsize=5)

        ax_res.plot(np.arange(n_time), output_module_neuron[idx[0], bg[0], bg[1], :, idx[1]], label='{} {} {}'.format(idx, bg[0], bg[1]), color=ci)
    ax_res.legend()
    ax_res.set_xlabel('time step')
    ax_res.set_ylabel('response')
    fig.tight_layout()

    return fig, stim_ax_list, ax_res

def single_imshow_colorbar(im, vcenter=1.0, vmin=None, vmax=None, fig=None, ax=None, cmap='seismic', cbar_label='', sci_format=False):
    '''
    im ([width, height, chs]): a single image
    '''
    if fig is None:
        fig, ax = plt.subplots(1)
    if vmin is None:
        heatmap = ax.imshow(im, cmap=cmap)
    else:
        if vcenter is None:
            vcenter = (vmax + vmin) / 2
        divnorm=colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        heatmap = ax.imshow(im, cmap=cmap, norm=divnorm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(heatmap, cax=cax, orientation='vertical')
    cbar.set_label(cbar_label)
    if sci_format:
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
    return fig, ax, cax

def plot_white_black_heatmap(im, vcenter=1, fig=None, ax=None, cmap='seismic', cbar_label='', sci_format=False):
    '''
    im ([2, im_width, im_height]): two heatmaps
    '''
    if fig is None:
        fig, ax = plt.subplots(2, 1, figsize=(2, 4))
    vmin = np.min(im)
    vmax = np.max(im)
    if vcenter is None:
        vcenter = (vmax + vmin) / 2
    else:
        dmin, dmax = vmin - vcenter, vmax-vcenter
        dis = np.maximum(np.abs(dmin), np.abs(dmax))
        vmin, vmax = vcenter - dis, vcenter + dis
        if dis < ERROR_TOL:
            vmin, vcenter, vmax = None, None, None # if this neuron doesn't respond at all

    _, _, cax0 = single_imshow_colorbar(im[0], vcenter, vmin, vmax, fig, ax[0], cmap=cmap, cbar_label=cbar_label, sci_format=sci_format)
    _, _, cax1 = single_imshow_colorbar(im[1], vcenter, vmin, vmax, fig, ax[1], cmap=cmap, cbar_label=cbar_label, sci_format=sci_format)
    simpleaxis(ax[0])
    simpleaxis(ax[1])

    return fig, ax, [cax0, cax1]

def plot_heatmap_with_contour(zmap, im, zthresh=1, vcenter=1, fig=None, ax=None, contour_color='black', linestyles='solid'):
    '''
    zmap ([2, im_width, im_height]): two zmaps corresponding to 2 im
    '''
    fig, ax = plot_white_black_heatmap(im, vcenter, fig, ax)

    ax[0].contour(zmap[0], levels=[-zthresh, zthresh], colors=contour_color, linestyles=linestyles)
    ax[1].contour(zmap[1], levels=[-zthresh, zthresh], colors=contour_color, linestyles=linestyles)
    return fig, ax

def add_ax_contour(ax, neural_id, center_heatmap_dir, contour_smooth_sigma=2, z_thresh=1, linestyle=[(0, (1, 1))]):
    # add contour
    rff = RF_Finder_Local_Sparse_Noise_Small_Memory_Center() # add rf contour
    rff.load_center_heatmap(center_heatmap_dir)
    zmap = rff.query_zmap(neural_id, smooth_sigma=contour_smooth_sigma, merge_black_white=True)
    ax.contour(zmap, levels=[z_thresh], colors='black', linestyles=linestyle)
    return ax

from matplotlib.ticker import ScalarFormatter

def plot_polar_boi(boi, angles, ax=None, color='black', fillstyle='full'):
    # Adjust angles for negative boi values
    angles_adj = np.where(boi < 0, angles + 180, angles)
    
    # Take absolute values of boi
    boi_abs = np.abs(boi)

    # Create polar plot
    if ax is None:
        ax = plt.subplot(111, polar=True)
    for angle, radius in zip(np.deg2rad(angles_adj), boi_abs):
        ax.plot([angle, angle], [0, radius], color=color, linestyle='-', linewidth=2, fillstyle=fillstyle, marker='o', markersize=10)
    ax.set_xticklabels([])
    # ax.tick_params(axis='y', colors='grey')

    y_formatter = ScalarFormatter(useOffset=False, useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((-3,4))  # Adjust power limits for scientific notation
    ax.yaxis.set_major_formatter(y_formatter)

    return ax

def plot_layer_violin_helper(score_exps, layer_order, ax=None, fig=None, color_palette=None, sns_voi_kwgs={}):
    '''
    If there are a lot of data for each class, you can use violin plot instead of boxplot
    score_exps (dict): {layer_name1: [list_of_score], layer_name2: [list_of_score], ...}
    layer_order (dict): {layer_name1: 0, layer_name2: 0, ...}. A dict uses layer name as key and the order as value
    '''
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))

    df = pd.DataFrame([(k, v) for k, values in score_exps.items() for v in values],
                      columns=['name', 'value'])
    df['order'] = df['name'].map(layer_order)
    df.sort_values('order', inplace=True)

    ax = sns.violinplot(x='name', y='value', data=df, fill=False, inner_kws=dict(whis_width=2), ax=ax, palette=color_palette, order=sorted(df['name'].unique(), key=lambda n: layer_order[n]), **sns_voi_kwgs)

    return fig, ax

def plot_layer_boxplot_helper(score_exps, layer_order, color="#747473", ax=None, fig=None, patch_artist=False, jitter=0.04, jitter_s=50, jitter_alpha=0.4, jitter_color='#1f77b4', show_outlier=True, **box_kwarg):
    '''
    score_exps (dict): {layer_name1: [list_of_score], layer_name2: [list_of_score], ...}
    layer_order (dict): {layer_name1: 0, layer_name2: 1, ...}. A dict uses layer name as key and the order as value
    jitter (float or None): if None, no jitter, if float
    '''
    boxprops = dict(
        linewidth=2,
        color=color,
    )
    if patch_artist:
        boxprops['facecolor'] = color

    whisprops = dict(
        linewidth=2,
        color=color,
    )

    medianprops = dict(
        linewidth=4,
        color=color,
        solid_capstyle="butt"
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    lo_order_id = []
    score_list = []
    jitter_color_list = []
    for reg, value in score_exps.items():
        lo_order_id.append(layer_order[reg])
        score_list.append(value)

        if type(jitter_color) == dict:
            jitter_color_list.append(jitter_color[reg])
        else:
            jitter_color_list.append(jitter_color)

    box_kwarg_default = {'positions': lo_order_id, 'showfliers': False, 'showcaps': False, 'medianprops': medianprops, 'whiskerprops': whisprops, 'boxprops': boxprops, 'patch_artist': patch_artist}
    box_kwarg.update(box_kwarg_default)

    ax.boxplot(score_list, **box_kwarg)

    if not show_outlier:
        score_list = [removeOutliers(np.array(v)) for v in score_list]

    if jitter is not None:
        x_data = [np.array([lo_order_id[i]] * len(d)) for i, d in enumerate(score_list)]
        x_jit = [x + sci_st.t(df=6, scale=jitter).rvs(len(x)) for x in x_data]

        # Add jittered dots ----------------------------------------------
        #for x, y, color in zip(x_jit, score_list, COLOR_SCALE):
        for x, y, c in zip(x_jit, score_list, jitter_color_list):
            ax.scatter(x, y, s=jitter_s, alpha=jitter_alpha, c=c)

    pos, label = [], []
    for key in layer_order:
        pos.append(layer_order[key])
        label.append(str(key))
    ax.set_xticks(pos)
    ax.set_xticklabels(label)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig, ax

def bootstrap_confidence_intervals(y, num_bootstrap_samples=1000, ci=95, score='mean'):
    '''
    input:
        y ([len_x, len_repeat]): y values
        num_bootstrap_samples (int): number of bootstrap samples
        ci (int): confidence interval
    output:
        [len_x, 3]: mean, ci_lower, ci_upper
    '''
    if score == 'mean':
        means = [np.mean(sublist) for sublist in y]

        bootstrapped = [
            [np.mean(np.random.choice(sublist, replace=True, size=len(sublist))) for _ in range(num_bootstrap_samples)]
            for sublist in y
        ]
    elif score == 'median':
        means = [np.median(sublist) for sublist in y]

        bootstrapped = [
            [np.median(np.random.choice(sublist, replace=True, size=len(sublist))) for _ in range(num_bootstrap_samples)]
            for sublist in y
        ]

    ci_lower = [np.percentile(sublist, (100 - ci) / 2) for sublist in bootstrapped]
    ci_upper = [np.percentile(sublist, ci + (100 - ci) / 2) for sublist in bootstrapped]

    return np.column_stack((means, ci_lower, ci_upper))

def removeOutliers(a, outlierConstant=1.5):
    '''
    remove outliers from a 1d array
    a (np.array): 1d array
    '''
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant # 1.5
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    return a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]

def compute_y_and_ybound(x, y, error_mode='se', mean_mode='mean', remove_outlier=False):
    '''
    compute the center y (can be mean or quantile) and error bar (can be se or std)
    input:
        x (list [len_x]): x values
        y ([len_x, len_repeat]): y values
        error_mode (str): 'se', 'std', 'quantile', 'ci_mean', 'ci_median'
        mean_mode (str): 'mean', 'median'
        remove_outlier (bool): whether to remove outliers before calculating the error bar and the mean (median). This is incompatible with error_mode='ci_mean'
    '''
    if remove_outlier:
        y = [removeOutliers(v) for v in y]

    if mean_mode == 'mean':
        mean_y = np.array([np.mean(v) for v in y])
    else:
        mean_y = np.array([np.median(v) for v in y])

    if error_mode == 'se':
        se_y = np.array([np.std(v) / np.sqrt(len(v)) for v in y])
        lower_y = mean_y - se_y; upper_y = mean_y + se_y
    elif error_mode == 'std':
        std_y = np.array([np.std(v) for v in y])
        lower_y = mean_y - std_y; upper_y = mean_y + std_y
    elif error_mode == 'quantile':
        y_25 = np.array([np.percentile(v, 25) for v in y])
        y_75 = np.array([np.percentile(v, 75) for v in y])
        lower_y = y_25; upper_y = y_75
    elif error_mode == 'ci_mean':
        mean_y, ci_lower, ci_upper = bootstrap_confidence_intervals(y).T
        lower_y = ci_lower; upper_y = ci_upper
    elif error_mode == 'ci_median':
        median_y, ci_lower, ci_upper = bootstrap_confidence_intervals(y, score='median').T
        mean_y = median_y
        lower_y = ci_lower; upper_y = ci_upper

    return x, mean_y, lower_y, upper_y

def error_bar_plot(x, y, fig=None, ax=None, color='tab:blue', label='', error_mode='se', mean_mode='mean', remove_outlier=False, error_band=False, line_style='-', with_line=True, with_scatter=False):
    '''
    draw the error bars
    input:
        x (list [len_x]): x values
        y ([len_x, len_repeat]): y values
        error_mode (str): 'se', 'std', 'quantile', 'ci_mean'
        mean_mode (str): 'mean', 'median'
        remove_outlier (bool): whether to remove outliers before calculating the error bar and the mean (median). This is incompatible with error_mode='ci_mean'
    '''
    if fig is None: fig, ax = plt.subplots()
    x, mean_y, lower_y, upper_y = compute_y_and_ybound(x, y, error_mode=error_mode, mean_mode=mean_mode, remove_outlier=remove_outlier)
    if error_band:
        ax.fill_between(x, lower_y, upper_y, color=color, alpha=0.2)
    else:
        ax.errorbar(x, mean_y, yerr=[mean_y - lower_y, upper_y - mean_y], fmt='o', color=color)
    if with_line:
        ax.plot(x, mean_y, color=color, label=label, linestyle=line_style)
    if with_scatter:
        ax.scatter(x, mean_y, color=color)
    return fig, ax

def show_center_region(image, region_size=None):
    """
    Display the center region of a PIL Image using matplotlib's imshow.

    :param image: PIL Image object
    :param region_size: tuple (width, height) of the center region size; defaults to image's quarter size
    """
    plt.figure(figsize=(5, 5))
    width, height = image.size
    if region_size is None:
        region_size = (width // 4, height // 4)

    center_x, center_y = width // 2, height // 2
    half_width, half_height = region_size[0] // 2, region_size[1] // 2

    center_region = image.crop((
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height
    ))

    plt.imshow(center_region)
    plt.axis('off')  # Hide axes
