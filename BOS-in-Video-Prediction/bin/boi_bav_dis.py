# draw example neural responseav
import border_ownership.neuron_info_processor as nip
from statsmodels.stats.proportion import proportion_confint
import numpy as np
import matplotlib.pyplot as plt
from border_ownership.ploter import plot_layer_boxplot_helper
import os
import hickle as hkl
from kitti_settings import *


def draw_n_cadidate(bo_info, module_list, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    n_candidate = [bo_info[module].shape[0] for module in module_list]

    ax.bar(module_list, n_candidate, color='tab:blue')  # Choose a pleasant color
    ax.set_xlabel('Module Name')
    ax.set_ylabel('Number of Candidate Units')
    ax.set_xticklabels(module_list, rotation=45)  # Rotate labels if they overlap
    fig.tight_layout()  # Adjust layout to prevent clipping of labels
    return fig, ax

def draw_n_bo(bo_info, module_list, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    n_candidate = np.array([bo_info[module].shape[0] for module in module_list])
    n_bo = np.array([(bo_info[module]['bav_pvalue'] < p_thre).sum() for module in module_list])
    percentage = np.array([n_bo[i] / n_candidate[i] * 100 for i in range(len(n_bo))])

    ci_low, ci_upp = proportion_confint(n_bo, n_candidate, alpha=0.05, method='wilson')
    print('percentage of BOS units: ', percentage)
    print('CI low: ', ci_low)
    print('CI upp: ', ci_upp)

    yerr = np.array([percentage - ci_low * 100, ci_upp * 100 - percentage])

    ax.bar(module_list, n_bo, yerr=yerr / 100.0 * n_candidate[None], color='tab:green')  # Choose a pleasant color
    ax.set_xlabel('Module Name')
    ax.set_ylabel('Number of BOS Units')
    ax.set_xticklabels(module_list, rotation=45)  # Rotate labels if they overlap
    ax.spines['left'].set_color('tab:green')  # Hide the right spine
    ax.tick_params(axis='y', colors='tab:green')
    ax.yaxis.label.set_color('tab:green')

    ax_p = ax.twinx()
    ax_p.scatter(x=np.arange(len(module_list)), y=percentage, color='tab:red')
    ax_p.plot(np.arange(len(module_list)), percentage, color='tab:red')
    ax_p.errorbar(np.arange(len(module_list)), percentage, yerr=yerr, fmt='o', color='tab:red')
    ax_p.hlines(5, -0.5, len(module_list) - 0.5, color='tab:red', linestyle='--')
    ax_p.set_ylim(0, 100)  # Set limits to make percentage visible
    ax_p.set_ylabel('Percentage (%) of BOS units \n among the candidate units')
    ax_p.yaxis.tick_right()
    ax_p.spines['right'].set_visible(True) 
    ax_p.spines['left'].set_visible(False) 
    ax_p.spines['right'].set_color('tab:red')
    ax_p.tick_params(axis='y', colors='tab:red')
    ax_p.yaxis.label.set_color('tab:red')

    fig.tight_layout()  # Adjust layout to prevent clipping of labels
    return fig, ax

def draw_bav_dis(metric, module_list, ylabel=r'Absolute $B_{av}$', **kwargs):
    layer_order = {module: i for i, module in enumerate(module_list)}

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    # fig, ax = plot_layer_boxplot_helper(metric, layer_order, jitter_s=10, jitter_alpha=0.4, fig=fig, ax=ax, color='#565656', box_lw=1.5, **kwargs)
    fig, ax = plot_layer_boxplot_helper(metric, layer_order, jitter_s=10, jitter_alpha=0.4, fig=fig, ax=ax, color='#565656', **kwargs)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, ax

def obtain_bav(bo_info, module_list, bo_unit=False):
    if bo_unit:
        bo_flag = {module: bo_info[module]['bav_pvalue'] < p_thre for module in module_list}
        bav = {module: bo_info[module]['bav'][bo_flag[module]] for module in module_list}
    else:
        bav = {module: bo_info[module]['bav'] for module in module_list}
    return bav

#################### Hyperparameters ####################
# module_list = ['R0', 'R1', 'R2', 'R3', 'E0', 'E1', 'E2', 'E3']
rmodule_list = ['R0', 'R1', 'R2', 'R3']
emodule_list = ['E0', 'E1', 'E2', 'E3']
center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')
p_thre = 0.05

#################### Main ####################
data = hkl.load(center_info_path)
bo_info, res_info, stim_info, unique_orientation = data['bo_info'], data['res_info'], data['stim_info'], data['unique_orientation']

fig, ax = draw_n_cadidate(bo_info, rmodule_list)
fig.savefig(os.path.join(FIGURE_DIR, 'n_candidate_r.svg'), format='svg')
fig, ax = draw_n_cadidate(bo_info, emodule_list)
fig.savefig(os.path.join(FIGURE_DIR, 'n_candidate_e.svg'), format='svg')

bav = obtain_bav(bo_info, emodule_list)
fig, ax = draw_bav_dis(bav, emodule_list)
ax.set_title('Candidate Unit')
fig.savefig(os.path.join(FIGURE_DIR, 'bav_dis_e_candidate.svg'), format='svg')
bav = obtain_bav(bo_info, rmodule_list)
fig, ax = draw_bav_dis(bav, rmodule_list)
ax.set_title('Candidate Unit')
fig.savefig(os.path.join(FIGURE_DIR, 'bav_dis_r_candidate.svg'), format='svg')

fig, ax = draw_n_bo(bo_info, emodule_list)
fig.savefig(os.path.join(FIGURE_DIR, 'n_bo_e.svg'), format='svg')
fig, ax = draw_n_bo(bo_info, rmodule_list)
fig.savefig(os.path.join(FIGURE_DIR, 'n_bo_r.svg'), format='svg')

bav = obtain_bav(bo_info, emodule_list, bo_unit=True)
fig, ax = draw_bav_dis(bav, emodule_list, jitter_color='tab:green')
ax.set_title('BOS Unit')
fig.savefig(os.path.join(FIGURE_DIR, 'bav_dis_e_bo.svg'), format='svg')
bav = obtain_bav(bo_info, rmodule_list, bo_unit=True)
fig, ax = draw_bav_dis(bav, rmodule_list, jitter_color='tab:green')
ax.set_title('BOS Unit')
fig.savefig(os.path.join(FIGURE_DIR, 'bav_dis_r_bo.svg'), format='svg')

plt.show()
