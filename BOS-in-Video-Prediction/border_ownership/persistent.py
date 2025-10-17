# functions related to reproducing persistent activity
import border_ownership.neuron_info_processor as nip
import copy
import numpy as np
from typing import Optional, Tuple
from kitti_settings import *

class Res_Diff_Calculator():
    def __init__(self, bo_info, res_info, stim_info, unique_orientation):
        self.bo_info = bo_info
        self.unique_orientation = unique_orientation
        self.res_info = res_info
        self.stim_info = stim_info

        for key in self.res_info.keys(): # all modules have the same time and switch time
            self.tot_time = self.res_info[key]['response'].iloc[0].shape[0]
            try:
                self.switch_time = self.res_info[key]['t_len0'].iloc[0]
                self.switch_time = int(self.switch_time)
            except KeyError:
                self.switch_time = int(0) # no switch, then set to 0
            break

    def set_module(self, module):
        self.module = module
        self.nipor = nip.Neuron_Info_Processor()
        self.nipor.load_data(self.bo_info, self.res_info, self.stim_info, self.module, self.unique_orientation)

    def get_res_diff(self, neural_rank, pref_beta=None):
        res_diff, pref_beta = get_pref_non_pref_res(self.nipor, neural_rank, pref_beta=pref_beta, output_mode='response_diff', normalize=True, switch_time=self.switch_time)
        return res_diff, pref_beta

    def get_res_diff_module(self) -> np.ndarray:
        max_rank = int(self.bo_info[self.module]['bo_only_rank'].max())
        res_diff_module = []
        for neural_rank in range(1, max_rank + 1):
            res_diff, _ = self.get_res_diff(neural_rank)
            res_diff_module.append(res_diff)
        return np.array(res_diff_module)

def get_response_along_pref_ori(nipor, neural_rank):
    neuron_bo_info, neuron_res_info = nipor.get_target_neuron_info(neural_rank, rank_method='bo_only')

    # get the preferred orientation
    pref_ori = nip.get_preferred_orientation(neuron_bo_info)
    neuron_res_info_pref_ori = neuron_res_info[neuron_res_info['orientation'] == pref_ori]

    # average on the contrast polarity
    res_bf_gf = neuron_res_info_pref_ori[(neuron_res_info_pref_ori['gamma'] == False) & (neuron_res_info_pref_ori['beta'] == False)]['response'].iloc[0]
    res_bf_gt = neuron_res_info_pref_ori[(neuron_res_info_pref_ori['gamma'] == True) & (neuron_res_info_pref_ori['beta'] == False)]['response'].iloc[0]
    res_bt_gf = neuron_res_info_pref_ori[(neuron_res_info_pref_ori['gamma'] == False) & (neuron_res_info_pref_ori['beta'] == True)]['response'].iloc[0]
    res_bt_gt = neuron_res_info_pref_ori[(neuron_res_info_pref_ori['gamma'] == True) & (neuron_res_info_pref_ori['beta'] == True)]['response'].iloc[0]
    return [res_bf_gf, res_bf_gt, res_bt_gf, res_bt_gt]

def get_pref_non_pref_res(nipor_s, neural_rank, pref_beta=None, avg_res=None, output_mode='corrected_activation', normalize=True, switch_time=True):
    res_bf_gf_s, res_bf_gt_s, res_bt_gf_s, res_bt_gt_s = get_response_along_pref_ori(nipor_s, neural_rank)
    # correct the response
    avg_res_bf_s = (res_bf_gf_s + res_bf_gt_s) / 2
    avg_res_bt_s = (res_bt_gf_s + res_bt_gt_s) / 2

    if pref_beta is None:
        avg_t_bf_s = np.mean(avg_res_bf_s[:switch_time + 1]); avg_t_bt_s = np.mean(avg_res_bt_s[:switch_time + 1])
        if avg_t_bf_s > avg_t_bt_s:
            pref_beta = False
        else:
            pref_beta = True

    if pref_beta: # assign pref and non-pref
        pref_res_s = avg_res_bt_s
        non_pref_res_s = avg_res_bf_s
    else:
        pref_res_s = avg_res_bf_s
        non_pref_res_s = avg_res_bt_s

    if output_mode == 'corrected_activation':
        if avg_res is None: # shift the mean
            avg_res = (avg_res_bf_s + avg_res_bt_s) / 2
        pref_res_s = pref_res_s - avg_res
        non_pref_res_s = non_pref_res_s - avg_res
        if normalize: # divided by the mean
            normalize_factor = np.mean((avg_res_bf_s + avg_res_bt_s) / 2) # unlike the avg_res for substraction, this normalize_factor is the average but cannot be set externally
            pref_res_s = pref_res_s / normalize_factor; non_pref_res_s = non_pref_res_s / normalize_factor
        return pref_res_s, non_pref_res_s, pref_beta
    elif output_mode == 'response_diff':
        if normalize:
            res_diff = (pref_res_s - non_pref_res_s) / (pref_res_s + non_pref_res_s)
        else: res_diff = pref_res_s - non_pref_res_s
        return res_diff, pref_beta
    else:
        raise ValueError('Invalid output_mode. Please choose from corrected_activation and response_diff.')

def configure_calculators(data: dict, bo_info: dict, unique_orientation: list, configs: dict) -> dict:
    """
    Configures various Res_Diff_Calculator instances based on data configurations.

    Parameters:
    - configs (dict): Dictionary containing various configurations. e.g. 
      configs = {
        's': ('res_switch_info', 'stim_switch_info'),
        'sags': ('res_switch_ambiguous_grey_info', 'stim_switch_ambiguous_grey_info', True)}.
      The first element is the calculator's name, first value is the res_info name, second value is the stim_info name, and the third value is whether to swap gamma and beta values.
    - data (dict): Dictionary containing various datasets.
    - bo_info (dict): Border ownership information.
    - unique_orientation (list): List of unique orientations.

    Returns:
    - dict: A dictionary of Res_Diff_Calculator instances keyed by their configuration names.
    """

    calculators = {}
    for key, config in configs.items():
        res_info, stim_info = data[config[0]], data[config[1]]
        if len(config) > 2 and config[2]:
            res_info = swap_gamma_beta(copy.deepcopy(res_info))
        calculators[key] = Res_Diff_Calculator(bo_info, res_info, stim_info, unique_orientation)

    return calculators

def swap_gamma_beta(data_info: dict) -> dict:
    """
    Swaps 'gamma' and 'beta' values in the provided data information.

    Parameters:
    - data_info (dict): A dictionary containing 'gamma' and 'beta' keys for each module.

    Returns:
    - dict: The modified dictionary with 'gamma' and 'beta' swapped.
    """
    for module in data_info:
        data_info[module][['gamma', 'beta']] = data_info[module][['beta', 'gamma']]
    return data_info
