import os
import numpy as np
import hickle as hkl
from border_ownership.util import compute_max_contour_distance_to_center, circular_average, compute_group_medians
from border_ownership.border_response_analysis import compute_BO, compute_bo_av, bo_permutation_test
from border_ownership.rf_finder import RF_Finder_Local_Sparse_Noise_Small_Memory_Center
from border_ownership.response_in_para import response_in_rf_paramemterization
import matplotlib.pyplot as plt
import scipy.io
from kitti_settings import *

def plot_bo_av_distribution(bo_av, title=''):
    """
    Plot the distribution of bo_av using a histogram.
    Parameters:
    - bo_av: array-like, Border Ownership Averaged values to be plotted.
    """
    plt.figure(figsize=(4, 4))
    # Make font larger
    plt.rcParams.update({'font.size': 14})
    # Plot histogram with updated appearance
    plt.hist(bo_av, bins=30, edgecolor='black', color='skyblue', linewidth=1.5)
    plt.title('Distribution of bo_av')
    plt.xlabel('bo_av')
    plt.ylabel('Frequency')
    # Make spines thicker
    ax = plt.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    # Remove grid
    ax.grid(False)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def load_responses():
    """Load the neural responses and the angles."""
    output_dark = hkl.load(output_dark_path)
    output_light = hkl.load(output_light_path)
    angle = hkl.load(label_path)['angle']
    return output_dark, output_light, angle

def filter_neurons_by_rf(rff, tot_neuron, smooth_sigma, z_thresh, rf_dis_thresh):
    """Filter neurons based on receptive field."""
    pass_rf_list = []
    for i in range(tot_neuron):
        zmap = rff.query_zmap(i, smooth_sigma=smooth_sigma, merge_black_white=True)
        max_distance = compute_max_contour_distance_to_center(zmap, z_thresh=z_thresh)
        if max_distance < rf_dis_thresh:
            pass_rf_list.append(i)
    return pass_rf_list

# Settings and parameters
output_mode = 'E1'
bo_mean_time_init = 5
bo_mean_time_final = 20
z_thresh = 1
smooth_sigma = 2
rf_dis_thresh = 20
significant_thre = 0.05
n_permutations = 10000
ERROR_TOL = 1e-5  # Adjust this based on your requirements

# Paths
output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center_length50.hkl')
output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center_length50.hkl')
label_path = os.path.join(DATA_DIR, 'rotating_square_label_all_length50.hkl')
center_heatmap_dir = os.path.join(DATA_DIR_HOME, 'heatmap_center_{}'.format(output_mode))

# Load heatmap and compute total neurons
rff = RF_Finder_Local_Sparse_Noise_Small_Memory_Center(z_thresh=z_thresh)
rff.load_center_heatmap(center_heatmap_dir)
tot_neuron = rff.heatmap_center.shape[-1]
print('Total number of neurons: {}'.format(tot_neuron))

# Filter neurons by receptive field
pass_rf_list = filter_neurons_by_rf(rff, tot_neuron, smooth_sigma, z_thresh, rf_dis_thresh)
print('Total number of neurons that pass rf test: {}'.format(len(pass_rf_list)))
print('neurons pass the rf test: ', pass_rf_list)

# Bav test
output_dark, output_light, angle = load_responses()
output_center, rf_para, alpha_rf, beta_rf, gamma_rf = response_in_rf_paramemterization(output_dark, output_light, angle)
output_module = output_center[output_mode] + 1
#output_module = output_module[..., pass_rf_list]

# Compute Bav and p-values
bo_av = compute_bo_av(output_module, alpha_rf)
permuted_bo_av, p_value = bo_permutation_test(output_module, alpha_rf, n_permutations=n_permutations)
pass_permutation_list = np.where(p_value <= significant_thre)[0]
print('neurons pass the permutation test: ', pass_permutation_list)

# Get BO neurons
bo_neuron = [value for value in pass_permutation_list if np.any(np.abs(pass_rf_list - value) < ERROR_TOL)]
print('BO neuron ID: \n {}\nBO neuron p value: \n{}\nB_av: {}'.format(bo_neuron, p_value[bo_neuron], bo_av[bo_neuron]))
print('Total number of bo neuron in {} is {}'.format(output_mode, len(bo_neuron)))

# save data
bo_av_file_name = os.path.join(DATA_DIR_HOME, 'bo_av_dis_{}.mat'.format(output_mode))
bo_av = bo_av[pass_rf_list] # only include neurons pass the permutation test
permuted_bo_av = permuted_bo_av[:, pass_rf_list] # only include neurons pass the permutation test
scipy.io.savemat(bo_av_file_name, {'bo_av': bo_av, 'shuffled_bo_av': permuted_bo_av})

# test save data
bo_av_load = scipy.io.loadmat(bo_av_file_name)
bo_av_load_origin = bo_av_load['bo_av']
bo_av_load_shuffle = bo_av_load['shuffled_bo_av']
print(bo_av_load_origin.shape)
print(bo_av_load_shuffle.shape)


# plot data
plot_bo_av_distribution(permuted_bo_av.flatten(), title='shuffled bav')
plot_bo_av_distribution(bo_av, title='true bav')

plt.show()

# compute the median distribution
median_shuffle = compute_group_medians(bo_av_load_shuffle, num_groups=10)
median_origin = compute_group_medians(bo_av_load_origin, num_groups=1)
print('median Bav of all shuffled {} neurons: {} \n median Bav of all true {} neurons: {}'.format(output_mode, median_shuffle, output_mode, median_origin))
