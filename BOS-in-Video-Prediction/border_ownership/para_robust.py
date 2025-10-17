# study the response and boi robustness to different square conditions
import border_ownership.neuron_info_processor as nip
import pandas as pd
import numpy as np

def get_res_para_ori(ori, neuron_bo_info, neuron_res_info, res_avg_start=0, mode='shift'):
    '''
    Get the response parameters for an orientation
    :param ori: orientation
    :param neuron_bo_info: border ownership information
    '''
    neuron_res = neuron_res_info[neuron_res_info['orientation'] == ori]

    neuron_res['mean_response'] = neuron_res['response'].apply(lambda x: np.mean(x[res_avg_start:], axis=0))

    res_beta_false = neuron_res[neuron_res['beta'] == False]
    res_beta_false_avg_gamma = res_beta_false.groupby(mode)['mean_response'].mean().reset_index()
    mean_response_beta_false = res_beta_false_avg_gamma['mean_response'].to_numpy()

    res_beta_true = neuron_res[neuron_res['beta'] == True]
    res_beta_true_avg_gamma = res_beta_true.groupby(mode)['mean_response'].mean().reset_index() # average two gammas
    mean_response_beta_true = res_beta_true_avg_gamma['mean_response'].to_numpy()

    para_beta = res_beta_true_avg_gamma[mode].to_numpy() # should be the same as para_beta_false
    return mean_response_beta_true, mean_response_beta_false, para_beta

def get_res_para(neuron_bo_info, neuron_res_info, res_avg_start=0, mode='shift'):
    pref_ori = nip.get_preferred_orientation(neuron_bo_info)
    mean_response_beta_true, mean_response_beta_false, para_beta = get_res_para_ori(pref_ori, neuron_bo_info, neuron_res_info, res_avg_start=res_avg_start, mode=mode)

    # compute the preferred beta
    pref_dire = neuron_bo_info['boi_pref_dire'].to_numpy()[0]
    if pref_dire < 180:
        mean_response_beta_pref = mean_response_beta_true
        mean_response_beta_npref = mean_response_beta_false
    else:
        mean_response_beta_pref = mean_response_beta_false
        mean_response_beta_npref = mean_response_beta_true

    return mean_response_beta_pref, mean_response_beta_npref, para_beta

