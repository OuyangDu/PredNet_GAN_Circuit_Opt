import border_ownership.neuron_info_processor as nip
from statsmodels.stats.proportion import proportion_confint
import border_ownership.ploter as ploter
from border_ownership.border_response_analysis import compute_BO, permute_beta_axis
from border_ownership.util import df_to_arr
from border_ownership.para_robust import get_res_para_ori
import matplotlib.pyplot as plt
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
import hickle as hkl
from kitti_settings import *

import cProfile, pstats

def compute_boi(arr):
    arr = arr.swapaxes(-2, -1) # (alpha, beta, gamma, time, size)
    boi = compute_BO(arr) # (alpha, time, size)
    boi = np.mean(boi, axis=(1, 2)) # (alpha)
    return boi

def permutation_test(boi, arr, n_shuffle=1000):
    boi_shuffle = np.empty((n_shuffle, len(boi)), dtype=np.float32)

    for i in range(n_shuffle):
        arr_shuffle = permute_beta_axis(arr, for_each_axis=[0, 2, 4])
        boi_shuffle[i] = compute_boi(arr_shuffle)

    # two-tailed test for each orientation
    p_value = np.empty(len(boi), dtype=np.float32)
    for i in range(len(boi)):
        more_extreme = np.sum(boi[i] < boi_shuffle[:, i]) / n_shuffle
        p_value[i] = 2 * min(more_extreme, 1 - more_extreme) # (alpha)
    return p_value

def convert_dire(boi, orientation):
    boi_bool = boi > 0

    dire = []
    for i, ori in enumerate(orientation):
        if boi_bool[i]:
            dire.append(ori)
        else:
            dire.append(ori + 180)
    return np.array(dire)

def circular_substraction(a, b):
    return (a - b + 180) % 360 - 180

def compute_span(significant_dire, method='min_max'):
    if len(significant_dire) == 0:
        return 0 # no significant direction
    elif method=='min_max':
        min_dire, max_dire = min(significant_dire), max(significant_dire)
        span = max_dire - min_dire if max_dire - min_dire < 180 else 360 - (max_dire - min_dire)
    # alternatively, using brutal force
    elif method=='brutal_force':
        span = 0
        for i in range(len(significant_dire)):
            for j in range(i + 1, len(significant_dire)):
                span = max(span, circular_substraction(significant_dire[i], significant_dire[j]))
    return span

def process_one_unit(neuron_bo_info, neuron_res_info):
    arr, unique_values = df_to_arr(neuron_res_info, value_name='response', condition_names=['orientation', 'beta', 'gamma', 'size'])
    boi = compute_boi(arr)
    p_value = permutation_test(boi, arr)
    dire = convert_dire(boi, unique_values[0]) # (alpha)
    significant_dire = dire[p_value < 0.05]
    span = compute_span(significant_dire)
    return span

if __name__ == '__main__':
    #################### Hyperparameters ####################
    module_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
    max_workers = 15
    center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')

    # #################### Main ####################
    data = hkl.load(center_info_path)
    bo_info, res_info, stim_info, unique_orientation = data['bo_info'], data['res_info'], data['stim_info'], data['unique_orientation']
    res_size_info, stim_size_info = data['res_size_info'], data['stim_size_info'] # used for testing statistical significance of BOI along a orientation

    #################### generate data ####################
    # nipor = nip.Neuron_Info_Processor()
    # span_dict = {}
    # for module in module_list:
    #     span_dict[module] = []
    #     nipor.load_data(bo_info, res_size_info, stim_size_info, module, unique_orientation)
    #     n_bo = nipor.bo_infom['bo_only_rank'].max()

    #     pool_data = [nipor.get_target_neuron_info(neural_rank, rank_method='bo_only') for neural_rank in range(1, int(n_bo) + 1)]
    #     with Pool(max_workers) as p:
    #         result = p.starmap(process_one_unit, pool_data)
    #     span_dict[module] = result

    # hkl.dump(span_dict, os.path.join(DATA_DIR_HOME, 'span_dict.hkl'))
    ##################################################

    span_dict = hkl.load(os.path.join(DATA_DIR_HOME, 'span_dict.hkl'))
    thresh_values = np.linspace(0, 180, 10)
    thresh_values = thresh_values[1:] # remove 0

    fig, axes = plt.subplots(2, 4, figsize=(11, 4))
    axes = axes.flatten()
    for i, module in enumerate(module_list):
        span_module = span_dict[module]
        counts = np.array([(span_module >= t).sum() for t in thresh_values])
        percent = counts / len(span_module) * 100

        ci_low, ci_high = proportion_confint(counts, len(span_module), method='wilson', alpha=0.05)

        axes[i].plot(thresh_values, counts, marker='o', color='k')
        axes[i].errorbar(thresh_values, counts, yerr=[counts - ci_low * len(span_module), ci_high * len(span_module) - counts], fmt='o', color='k')
        if i in [4, 5, 6, 7]:
            axes[i].set_xlabel('Span of BOI \n (degree)')
        if i in [0, 4]:
            axes[i].set_ylabel('Number of \n BO units')
        axes[i].set_title(f'{module}: {len(span_module)}', fontsize=12)

        # Secondary y-axis for percentages
        ax2i = axes[i].twinx()
        ax2i.errorbar(thresh_values, percent, yerr=[percent - ci_low * 100, ci_high * 100 - percent], fmt='o', color='k')
        ax2i.spines['right'].set_visible(True)
        if i in [3, 7]:
            ax2i.set_ylabel('Percentage (%)')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'span_BOI.svg'))

    module = 'E2'
    span_module = span_dict[module]
    counts = np.array([(span_module >= t).sum() for t in thresh_values])
    percent = counts / len(span_module) * 100
    ci_low, ci_high = proportion_confint(counts, len(span_module), method='wilson', alpha=0.05)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(thresh_values, counts, marker='o', color='k')
    ax.errorbar(thresh_values, counts, yerr=[counts - ci_low * len(span_module), ci_high * len(span_module) - counts], fmt='o', color='k')
    ax.set_xlabel('Span of BOI \n (degree)')
    ax.set_ylabel('Number of \n BO units')
    ax.set_title(f'{module}', fontsize=14)
    ax2 = ax.twinx()
    ax2.plot(thresh_values, percent, color='k')  # Using different style for percent
    ax2.errorbar(thresh_values, percent, yerr=[percent - ci_low * 100, ci_high * 100 - percent], fmt='o', color='k')
    ax2.spines['right'].set_visible(True)
    ax2.set_ylabel('Percentage in \n the whole population (%)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'span_BOI_E2.svg'))

    plt.show()
