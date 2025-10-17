import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt
import border_ownership.neuron_info_processor as nip
from matplotlib.patches import Rectangle
from border_ownership.util import sci_notation
from scipy.stats import wilcoxon
from kitti_settings import *
import os

def compute_fr_squared(response):
    response = response - 1
    fr_squared = np.mean(response ** 2, axis=-1)
    fr_squared = np.mean(fr_squared, axis=-1)
    return fr_squared

def compute_avg_response(response, pref_dire):
    # response = np.array(response)
    # response = response - 1
    avg_response = np.array([(response[0] + response[1]) / 2, (response[2] + response[3]) / 2])
    if pref_dire < 180:
        avg_response = avg_response[::-1]
    avg_response = avg_response / np.mean(avg_response, keepdims=True)
    return avg_response

def get_avg_response(nipor, bo_info, module):
    # iterate through each neural_rank
    avg_response_list = []
    largest_rank = bo_info[module]['bo_only_rank'].dropna().max()
    for neural_rank in range(1, int(largest_rank) + 1):
        neuron_bo_info, neuron_res_info = nipor.get_target_neuron_info(neural_rank, rank_method='bo_only')
        pref_ori = nip.get_preferred_orientation(neuron_bo_info)
        _, response = nip.get_response_to_orientation(neuron_res_info, pref_ori)
        pref_dire = neuron_bo_info['boi_pref_dire'].iloc[0]
        avg_response = compute_avg_response(response, pref_dire)
        avg_response_list.append(avg_response)
    avg_response_arr = np.array(avg_response_list)
    return avg_response_arr

def compute_wilcoxon_p_values(avg_response_arr):
    _, _, n_time = avg_response_arr.shape
    p_values = np.zeros(n_time)
    for time_index in range(n_time):
        sample1 = avg_response_arr[:, 0, time_index]
        sample2 = avg_response_arr[:, 1, time_index]
        try:
            _, p_value = wilcoxon(sample1, sample2, zero_method='wilcox', correction=False)
        except ValueError:
            p_value = 1
        p_values[time_index] = p_value
    return p_values

def index_of_continuous_significance(p_values, threshold):
    '''
    Return the index of the first time point that is significant and all following time points are also significant.
    '''
    for i in range(len(p_values)):
        if p_values[i] < threshold:
            all_following_below_threshold = np.all(np.less(p_values[i:], threshold))
            if all_following_below_threshold:
                return i
    return None

#################### Hyperparameters ####################
module_list = ['R0', 'R1', 'R2', 'R3', 'E0', 'E1', 'E2', 'E3']
alpha = 0.05
center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')
avg_response_path = os.path.join(DATA_DIR_HOME, 'avg_response.hkl')

#################### Main ####################
data = hkl.load(center_info_path)
bo_info, res_info, stim_info, unique_orientation = data['bo_info'], data['res_info'], data['stim_info'], data['unique_orientation']

nipor = nip.Neuron_Info_Processor()

avg_response_dict = {}
for module in module_list:
    print('Processing module: {}'.format(module))
    nipor.load_data(bo_info, res_info, stim_info, module, unique_orientation)
    avg_response_arr = get_avg_response(nipor, bo_info, module)
    avg_response_dict[module] = avg_response_arr

hkl.dump(avg_response_dict, avg_response_path)

avg_response_dict = hkl.load(avg_response_path)

# compute the Wilcoxon signed-rank test
p_value_dict = {}
idx = {}
for module in module_list:
    avg_response_arr = avg_response_dict[module]
    p_value_dict[module] = compute_wilcoxon_p_values(avg_response_arr)
    idx[module] = index_of_continuous_significance(p_value_dict[module], alpha)

fig, axes = plt.subplots(2, 4, figsize=(10, 5))
axes = axes.flatten()
for i, module in enumerate(module_list):
    avg_response_arr = avg_response_dict[module]
    time_steps = np.arange(avg_response_arr.shape[-1])

    # Calculating mean and SEM across all units
    mean_response = np.mean(avg_response_arr, axis=0)
    sem_response = np.std(avg_response_arr, axis=0) / np.sqrt(len(avg_response_arr))

    # Plotting the time traces with error bands
    ax = axes[i]
    for j in range(2):
        ax.fill_between(time_steps, 
                        mean_response[j] - sem_response[j], 
                        mean_response[j] + sem_response[j], 
                        alpha=0.2)

        if i == 0:
            if j == 0:
                ax.plot(time_steps, mean_response[j], label='Preferred')
            else:
                ax.plot(time_steps, mean_response[j], label='Non-Pref.')
            ax.legend(fontsize=13)
        else:
            ax.plot(time_steps, mean_response[j])

    if idx[module] is not None:
        ax.axvline(idx[module], color='k', linestyle='--')
        ylim = ax.get_ylim()  # Get the current y-axis limits
        ax.text(idx[module], ylim[1], '*', ha='center', va='bottom', fontsize=15)  # Simple star
        ax.text(idx[module], ylim[0], f'{idx[module]}', ha='center', va='top', fontsize=13)  # X index annotation

    ax.set_title('{} \n n = {}'.format(module, avg_response_arr.shape[0]), fontsize=13)

# add x and y labels
x_label_id = [4, 5, 6, 7]
y_label_id = [0, 4]
for xi in x_label_id:
    axes[xi].set_xlabel('Time', fontsize=13)
for yi in y_label_id:
    axes[yi].set_ylabel('Unit activation', fontsize=13)

fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'temporal_trace.svg'), format='svg')
plt.show()
