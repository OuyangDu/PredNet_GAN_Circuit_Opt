import numpy as np
from itertools import product
import hickle as hkl
import border_ownership.response_in_para as rip
from border_ownership.util import circular_average
import matplotlib.pyplot as plt

def find_neuron_index_by_central_RF(central_RF, image_shape, neural_response_shape):
    '''
    cetnral_RF (array, 2, int): indicate the central RF position in an image
    image_shape (array, 2, int): (width, height)
    neural_response_shape: the output shape of an prednet is (n_video, n_time, width, height, chs). This is asking for (width, height)
    '''
    return int(central_RF / image_shape * neural_response_shape)

def compute_BO(output_module, replace_nan=True):
    '''
    input:
      the output_module should be in the rf_parametereization (alpha, beta, gamma, t, chs), where chs also means neurons. Alpha means orientation.
    output:
      bo: array with shape (alpha, t, chs) indicating the bo value of neuron chs at angle alpha and time t
    '''
    res_mean_gamma = np.mean(output_module, axis=2) # average over gamma
    res_mean_beta_gamma = np.mean(res_mean_gamma, axis=1) # average also over beta as the denominator of the bo

    bo = np.zeros(res_mean_beta_gamma.shape)
    for i in range(output_module.shape[0]): # all angles
        for j in range(output_module.shape[-1]): # all neurons
            bo[i, :, j] = (res_mean_gamma[i, 1, :, j] - res_mean_gamma[i, 0, :, j]) / res_mean_beta_gamma[i, :, j] / 2.0 # negative means the neuron prefer the oppsite side, angle is larger than 180 degree

    if replace_nan: bo = np.nan_to_num(bo)
    return bo

def compute_BO_each_neuron(output_module, replace_nan=True):
    bo = compute_BO(output_module, replace_nan)
    bo_of_neuron = np.mean(bo, axis=(0, 1))
    return bo_of_neuron

def select_neuron_by_bo(bo, bo_order=[0], time_init=None, time_final=None):
    '''
    input:
      bo: the output of compute BO, with shape (alpha, t, chs) indicating the bo value of neuron chs at angle alpha and time t
      bo_order: list. 0 means quiring the id of neuron with largest bo. None means output the sorted neuron by their maximum bo angle
    output:
      angle_idx, neuron_idx: each has shape (bo_oder_len). angle_idx specifies id in the first rank of bo, neuron_idx specifies id in the last rank of bo
    '''
    bo_t = np.mean(bo[:, time_init:time_final], axis=1) # average over time
    bo_flat = bo_t.flatten()
    bo_flat_abs = np.abs(bo_flat)
    bo_idx_sort = np.argsort(bo_flat_abs) # from minimum to maximum
    bo_idx_sort = bo_idx_sort[::-1] # from maximum to minimum
    idx = np.unravel_index(bo_idx_sort, bo_t.shape)

    _, unique_neuron_id = np.unique(idx[1], return_index=True)
    unique_neuron_id = np.sort(unique_neuron_id) # keep the order in the original idx

    if bo_order is None:
        angle_idx, neuron_idx = idx[0][unique_neuron_id], idx[1][unique_neuron_id]
    else:
        angle_idx, neuron_idx = idx[0][unique_neuron_id][bo_order], idx[1][unique_neuron_id][bo_order]
    return angle_idx, neuron_idx

class BOI_Analyzer():
    def __init__(self, output_module, alpha_rf, bo_mean_time_init=None, bo_mean_time_final=None):
        '''
        input:
          the output_module should be in the rf_parametereization (alpha, beta, gamma, t, chs), where chs also means neurons
          alpha_rf: corresponding edge angle to y axis
          module_name: e.g. E0, E1 etc
          bo_mean_time_init, bo_mean_time_final: int, indicating the time window when selecting neuron by boi
        '''
        self.ouput_module = output_module # (alpha, beta, gamma, t, chs)
        self.alpha_rf = alpha_rf # (alpha)

        self.bo = compute_BO(output_module) # (alpha, time, chs)
        self.bo_mean_time_init = bo_mean_time_init
        self.bo_mean_time_final = bo_mean_time_final
        self.alpha_idx, self.neuron_idx = select_neuron_by_bo(self.bo, bo_order=None, time_init=bo_mean_time_init, time_final=bo_mean_time_final) # neuron_idx is the neural id ranked by the absolute value of boi at the maximum orientation. alpha_idx is the neural maximum boi orientation id. Both have shape (chs)

    def neural_boi_dist(self):
        '''
        a neuron's boi is defined as the maximum boi across all angle
        '''
        bo_t = np.mean( self.bo[:, self.bo_mean_time_init:self.bo_mean_time_final], axis=1 )
        bo_t = np.abs(bo_t)
        neural_boi = bo_t[self.alpha_idx, self.neuron_idx]
        return np.array(neural_boi)

    def neural_prefer_boi_angle_dist(self):
        prefer_angle = self.alpha_rf[self.alpha_idx] # preferred edge orientation. But we still need to convert it to direction. If the bo is negative, preferred orientation should be current edge orientation plus 180 degree

        bo_t = np.mean( self.bo, axis=1 )
        negative_boi =  bo_t[self.alpha_idx, self.neuron_idx] < 0
        prefer_angle[negative_boi] = prefer_angle[negative_boi] + 180
        return np.array(prefer_angle)

    def neural_boi_time_trace(self, neural_rank):
        time = np.arange(self.bo.shape[1])
        boi_time_trace = self.bo[self.alpha_idx[neural_rank], :, self.neuron_idx[neural_rank]] # boi can be negative because the preferred direction is in the oppsite direction
        if np.mean(boi_time_trace) < 0: boi_time_trace = - boi_time_trace # make it positive. So that for single neural, boi trace is the preferred side minus the non-preferred side
        return time, boi_time_trace

    def neural_boi_orientation(self, neural_rank):
        bo_t = np.mean(self.bo[:, self.bo_mean_time_init:self.bo_mean_time_final, self.neuron_idx[neural_rank]], axis=1)
        negative_bo = bo_t < 0
        alpha = self.alpha_rf.copy()
        alpha[negative_bo] = alpha[negative_bo] + 180

        return alpha, np.abs(bo_t)


def get_neural_id_by_rank(output_dark_path, output_light_path, label_path, output_mode, neural_rank, bo_mean_time_init=5, bo_mean_time_final=19):
    # load neural responses to rotating square
    # output neural id and the preferred edge orientation
    output_dark = hkl.load(output_dark_path)
    output_light = hkl.load(output_light_path)
    angle = hkl.load(label_path)['angle']
    output_center, rf_para, alpha_rf, beta_rf, gamma_rf = rip.response_in_rf_paramemterization(output_dark, output_light, angle)
    output_module = output_center[output_mode]
    output_module = output_module + 1 # avoid negative value
    # load analyzer
    boia = BOI_Analyzer(output_module, alpha_rf, bo_mean_time_init=bo_mean_time_init, bo_mean_time_final=bo_mean_time_final)
    return boia.neuron_idx[neural_rank], alpha_rf[boia.alpha_idx[neural_rank]]

def compute_bo_av(output_module, alpha_rf, time_window=None, with_angle=False):
    """
    Compute the average border ownership index (bo_av) from the given output module.
    do not normalize boi

    Parameters:
    - output_module (np.ndarray, shape is [num_ori, num_beta = 2, num_gamma = 2, time, number of neurons]): The primary data module.
    - alpha_rf (np.ndarray): The array of angles.

    Returns:
    - np.ndarray: The averaged border ownership index (bo_av).
    """
    bo = compute_BO(output_module, replace_nan=True)
    if time_window is None:
        bo_t = np.mean(bo, axis=1)
    else:
        bo_t = np.mean(bo[:, time_window[0]:time_window[1], :], axis=1)
    # bo_t = bo_t / np.max(np.abs(bo_t), axis=0)
    bo_t = np.nan_to_num(bo_t, nan=0.0)
    result = circular_average(bo_t, alpha_rf, with_angle=with_angle)

    if with_angle:
        bo_av, angle = result[0], result[1]
        bo_av = np.nan_to_num(bo_av, nan=0.0)
        return bo_av, angle
    else:
        bo_av = result
        bo_av = np.nan_to_num(bo_av, nan=0.0)
        return bo_av

def permute_beta_axis(arr, for_each_axis=[0, 2]):
    '''
    General function to permute the beta axis of the given array, dynamically looping
    over specified axes. By default, the beta axis is assumed to be the second one.
    
    Parameters:
        arr (np.ndarray): The input array of shape (n_ori, n_beta, n_gamma, t, last_dim)
        for_each_axis (list): List of axes indices over which to loop for permutation.  for_each_axis = [0, 2] means permute beta for each orientation and gamma
    
    Returns:
        np.ndarray: The array with the beta axis permuted.
    '''
    shape = arr.shape
    n_beta = shape[1]  # The beta axis is assumed to be the second one (index 1)
    permuted_arr = np.empty_like(arr)
    
    # Building indices for the axes to iterate over, excluding the beta axis
    iter_axes = [shape[axis] for axis in for_each_axis]
    
    full_slice = [slice(None)] * len(shape)  # start with full slices for all dimensions
    # Create an iterator for the cartesian product of the specified axes
    for indices in product(*(range(s) for s in iter_axes)):
        
        # Set the specific indices for the dimensions in for_each_axis
        for axis, index in zip(for_each_axis, indices):
            full_slice[axis] = index

        full_slice[1] = slice(None)  # ensure beta axis is fully included for permutation
        
        # Convert to tuple to use for numpy indexing
        index_tuple = tuple(full_slice)
        
        # Apply permutation on the beta axis while keeping other indices fixed
        permuted_arr[index_tuple] = np.random.permutation(arr[index_tuple])
    
    return permuted_arr

def bo_permutation_test(output_module, alpha_rf, time_window=None, n_permutations=5000, output_zscore_bav=False):
    """
    Conduct a permutation test to determine the statistical significance of bo_av.
    
    Parameters:
    - output_module (np.ndarray): The primary data module.
    - alpha_rf (np.ndarray): The array of angles.
    - n_permutations (int): Number of permutations to run. Default is 1000.

    Returns:
    - np.ndarray: Array of p-values indicating the statistical significance for each neuron.
    """
    # Compute the original bo_av without any shuffling
    original_bo_av = compute_bo_av(output_module, alpha_rf, time_window=time_window)
    
    # Storage for the permuted bo_av values
    permuted_bo_avs = np.zeros((n_permutations, original_bo_av.shape[0]))

    for i in range(n_permutations):
        # Create a copy of the output_module and shuffle the beta axis for each alpha value
        shuffled_module = permute_beta_axis(output_module)

        # Compute bo_av for the shuffled data
        permuted_bo_avs[i] = compute_bo_av(shuffled_module, alpha_rf, time_window=time_window)

    # Compute p-values for each neuron (one-tailed test)
    p_values = np.array([np.mean(permuted_bo_av >= orig_val)
                         for orig_val, permuted_bo_av in zip(original_bo_av, permuted_bo_avs.T)])

    # Ensure p-values are bounded between 0 and 1
    p_values = np.clip(p_values, 0, 1)

    if output_zscore_bav:
        # Compute the mean and standard deviation of the permuted bo_av values
        permuted_bo_avs_mean = np.mean(permuted_bo_avs, axis=0)
        permuted_bo_avs_std = np.std(permuted_bo_avs, axis=0)

        # Compute the z-scored bo_av values
        zscore_bavs = (original_bo_av - permuted_bo_avs_mean) / permuted_bo_avs_std

        return permuted_bo_avs, p_values, zscore_bavs

    return permuted_bo_avs, p_values
