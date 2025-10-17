import border_ownership.neuron_info_processor as nip
import border_ownership.ploter as ploter
import matplotlib.pyplot as plt
import numpy as np
import os
import hickle as hkl
from kitti_settings import *

def plot_para_response(neural_res_pref, neural_res_npref, para, mode='shift'):
    c_pref='tab:blue'
    c_npref='tab:green'
    fig = plt.figure(figsize=(3, 2.5))
    ax_para = fig.add_subplot()
    ax_para.scatter(para, neural_res_pref, color=c_pref)
    ax_para.plot(para, neural_res_pref, color=c_pref)
    ax_para.scatter(para, neural_res_npref, color=c_npref)
    ax_para.plot(para, neural_res_npref, color=c_npref)
    ax_para.set_xlabel('{} (pixel)'.format(mode))
    ax_para.set_ylabel('Unit Activation')

    fig.tight_layout()
    return fig, ax_para

def get_res_para(neuron_bo_info, neuron_res_info, res_avg_start=0, mode='shift'):
    pref_ori = nip.get_preferred_orientation(neuron_bo_info)
    neuron_res = neuron_res_info[neuron_res_info['orientation'] == pref_ori]
    neuron_res['mean_response'] = neuron_res['response'].apply(lambda x: np.mean(x[res_avg_start:], axis=0))

    res_beta_false = neuron_res[neuron_res['beta'] == False]
    res_beta_false_avg_gamma = res_beta_false.groupby(mode)['mean_response'].mean().reset_index()
    mean_response_beta_false = res_beta_false_avg_gamma['mean_response'].to_numpy()
    para_beta_false = res_beta_false_avg_gamma[mode].to_numpy() # should be the same as para_beta_false

    res_beta_true = neuron_res[neuron_res['beta'] == True]
    res_beta_true_avg_gamma = res_beta_true.groupby(mode)['mean_response'].mean().reset_index() # average two gammas
    mean_response_beta_true = res_beta_true_avg_gamma['mean_response'].to_numpy()
    para_beta_true = res_beta_true_avg_gamma[mode].to_numpy() # should be the same as para_beta_false

    # compute the preferred beta
    pref_dire = neuron_bo_info['boi_pref_dire'].to_numpy()[0]
    if pref_dire < 180:
        mean_response_beta_pref = mean_response_beta_true
        mean_response_beta_npref = mean_response_beta_false
    else:
        mean_response_beta_pref = mean_response_beta_false
        mean_response_beta_npref = mean_response_beta_true

    fig, ax_para = plot_para_response(mean_response_beta_pref, mean_response_beta_npref, para_beta_false, mode=mode)
    return fig, ax_para
# #################### Hyperparameters ####################
module = 'E1'
neural_rank = 5
center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')

# #################### Main ####################
data = hkl.load(center_info_path)
bo_info, res_info, stim_info, unique_orientation = data['bo_info'], data['res_info'], data['stim_info'], data['unique_orientation']
res_shift_info, stim_shift_info = data['res_shift_info'], data['stim_shift_info']
res_size_info, stim_size_info = data['res_size_info'], data['stim_size_info']

#################### Different orientation ####################
nipor = nip.Neuron_Info_Processor()
nipor.load_data(bo_info, res_info, stim_info, module, unique_orientation)
neuron_bo_info, neuron_res_info = nipor.get_target_neuron_info(neural_rank)
boi = neuron_bo_info['boi'].iloc[0]

fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={'projection': 'polar'})
ax = ploter.plot_polar_boi(boi, nipor.boi_orim, ax=ax)
fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'eg_polar_boi.pdf'), bbox_inches='tight')
plt.show()

# # Preferred orientation distribution of BO unit
# bo_unit = nipor.bo_infom[nipor.bo_infom['is_bo']]
# plt.hist(bo_unit['boi_pref_dire'], bins=20)
# plt.show()

#################### Different shift/size ####################
nipor = nip.Neuron_Info_Processor()

### shift
nipor.load_data(bo_info, res_shift_info, stim_shift_info, module, unique_orientation)
neuron_bo_info, neuron_res_info = nipor.get_target_neuron_info(neural_rank)
fig_shift, ax_shift = get_res_para(neuron_bo_info, neuron_res_info, mode='shift')
fig_shift.savefig(os.path.join(FIGURE_DIR, 'eg_shift_response.svg'), bbox_inches='tight', format='svg')

### size
nipor.load_data(bo_info, res_size_info, stim_size_info, module, unique_orientation)

neuron_bo_info, neuron_res_info = nipor.get_target_neuron_info(neural_rank)
fig_size, ax_size = get_res_para(neuron_bo_info, neuron_res_info, mode='size')
fig_size.savefig(os.path.join(FIGURE_DIR, 'eg_size_response.svg'), bbox_inches='tight', format='svg')

plt.show()
