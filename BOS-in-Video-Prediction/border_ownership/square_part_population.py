import numpy as np
import matplotlib.pyplot as plt
from border_ownership.ploter import error_bar_plot, bootstrap_confidence_intervals, removeOutliers

def remove_nan_neuron(data_square_part):
    data_square_part_no_nan = []
    for i in range(data_square_part.shape[0]):
        one_neuron = data_square_part[i, :, :]
        has_nan = np.isnan(one_neuron).any()

        if not has_nan:
            data_square_part_no_nan.append(one_neuron)
    return np.array(data_square_part_no_nan)

# gain correction
def gain_correction(data_square_part_no_nan):
    data_square_part_no_nan_with_center = data_square_part_no_nan[:, [0, 1]] # first row is preferred side, second row is non-preferred side
    mean_res_1 = np.mean(data_square_part_no_nan_with_center[:, :, 0], axis=1) # center edge only
    mean_res_2 = np.mean(data_square_part_no_nan_with_center[:, :, 1:-1], axis=(1, 2)) # two edge
    mean_res_all = np.mean(data_square_part_no_nan_with_center[:, :, -1], axis=1) # all edge

    gain_2 = mean_res_1 / mean_res_2
    gain_all = mean_res_1 / mean_res_all

    data_square_part_gain_corrected = data_square_part_no_nan_with_center.copy()
    data_square_part_gain_corrected[:, :, 1:-1] *= gain_2[:, np.newaxis, np.newaxis]
    data_square_part_gain_corrected[:, :, -1] *= gain_all[:, np.newaxis]
    data_square_part_change_gain_corrected = data_square_part_gain_corrected - data_square_part_gain_corrected[:, :, 0, np.newaxis] # substract the first column, i.e. compute the change from the center edge
    return data_square_part_change_gain_corrected

def switch_res_order(arr, input_res_name, target_res_name):
    name_to_index = {name: i for i, name in enumerate(input_res_name)}
    # Reordered array excluding 'CE_nonpref'
    arr_reordered = [arr[name_to_index[name]] for name in target_res_name]
    return np.array(arr_reordered)

def plot_hist(y_mean, ci_lower, ci_upper, input_res_name, target_res_name, fig=None, ax=None, bar_color='grey', y_label='Surround influence (a.u.)'):
    '''
    Plot the histogram of surround influence
    input:
        y_mean: (1d array) mean of surround influence
        ci_lower: (1d array) lower bound of confidence interval
        ci_upper: (1d array) upper bound of confidence interval
        input_res_name: (list) name of the input resolution
        target_res_name: (list) name of the target resolution
        fig: (matplotlib figure) figure to plot on
        ax: (matplotlib axis) axis to plot on
        bar_color: (str) color of the bar
    '''
    if fig is None: fig, ax = plt.subplots(figsize=(5, 3))

    y_mean = switch_res_order(y_mean, input_res_name, target_res_name)
    ci_lower = switch_res_order(ci_lower, input_res_name, target_res_name)
    ci_upper = switch_res_order(ci_upper, input_res_name, target_res_name)

    x = np.arange(len(y_mean))
    errors = [y_mean - ci_lower, ci_upper - y_mean]
    ax.bar(x, y_mean, yerr=errors, align='center', alpha=0.7, ecolor='black', capsize=3, color=bar_color, edgecolor='black', linewidth=2)
    ax.set_xticks(x)
    target_res_name = [s.split('_')[0] for s in target_res_name] # remove all chars after slash
    ax.set_xticklabels(target_res_name)
    ax.set_ylabel(y_label)

    ax.spines['bottom'].set_position(('data', 0))
    ax.xaxis.set_ticks_position('none') 
    ax.set_xticklabels([])
    return fig, ax

def plot_hist_si(y_mean, ci_lower, ci_upper, input_res_name, target_res_name, fig=None, ax=None, bar_color=['grey', 'white']):
    y_mean = switch_res_order(y_mean.T, input_res_name, target_res_name).T
    ci_lower = switch_res_order(ci_lower.T, input_res_name, target_res_name).T
    ci_upper = switch_res_order(ci_upper.T, input_res_name, target_res_name).T

    if fig is None: fig, ax = plt.subplots(figsize=(5, 3))

    x = np.arange(len(y_mean))
    errors = [y_mean - ci_lower, ci_upper - y_mean]

    num_bars, num_groups = y_mean.shape
    bar_width = 0.35
    x = np.arange(num_groups)

    for i in range(num_bars):
        errors = [y_mean[i] - ci_lower[i], ci_upper[i] - y_mean[i]]
        ax.bar(x + i * bar_width, y_mean[i], width=bar_width, yerr=errors, align='center', alpha=0.7, ecolor='black', capsize=3, color=bar_color[i], edgecolor='black')

    ax.set_xticks(x + bar_width / 2)
    target_res_name = [s.split('_')[0] for s in target_res_name] # remove all chars after slash
    ax.set_xticklabels(target_res_name)
    ax.set_ylabel('Surround influence (a.u.)')

    return fig, ax

def get_mean_ci(data_square_part_change, with_center=True):
    '''
    data_square_part_change: (array [:, n_condition = 4, :]): condition = 0 means with center edge
    '''
    if with_center: iterid = range(2)
    else: iterid = range(2, 4)

    y_mean, ci_lower, ci_upper = [], [], []
    for i in iterid:
        one_condition = data_square_part_change[:, i, :].T
        one_condition = [removeOutliers(v) for v in one_condition]
        y_mean_temp = [np.mean(v) for v in one_condition]
        ci_temp = [np.std(v) / np.sqrt(len(v)) * 1.96 for v in one_condition]
        ci_lower_temp = np.array(y_mean_temp) - np.array(ci_temp)
        ci_upper_temp = np.array(y_mean_temp) + np.array(ci_temp)
        y_mean = y_mean + y_mean_temp
        ci_lower = ci_lower + ci_lower_temp.tolist()
        ci_upper = ci_upper + ci_upper_temp.tolist()
    y_mean, ci_lower, ci_upper = np.array(y_mean), np.array(ci_lower), np.array(ci_upper)
    return y_mean, ci_lower, ci_upper
