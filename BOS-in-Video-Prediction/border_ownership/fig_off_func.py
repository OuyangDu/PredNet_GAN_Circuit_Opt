# all these functions are used to compute the half life of the neurons. They are more like one-time use functions so I didn't refactor them well
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, f_oneway, kruskal
import itertools
import pandas as pd
import seaborn as sns
from kitti_settings import *

def compute_crossings(res_s):
    half_max_abs_res_s = np.max(np.abs(res_s), axis=1, keepdims=True) / 2.0
    crossings = np.abs(res_s) < half_max_abs_res_s
    return crossings

def find_half_life(res_s, replace_none='none'):
    crossings = compute_crossings(res_s)
    sustained_idx = []
    for unit_crossings in crossings:
        for idx in range(len(unit_crossings)):
            if all(unit_crossings[idx:]):
                sustained_idx.append(idx)
                break
        else:
            if replace_none == 'remove': pass
            elif replace_none == 'none': sustained_idx.append(None)
            elif replace_none == 'max': sustained_idx.append(len(unit_crossings) - 1)
    return np.array(sustained_idx)

def compute_p_matrix(score, layer_order, test_method='ttest'):
    # Create all pairwise combinations of score keys
    key_pairs = list(itertools.combinations(score.keys(), 2))
    
    results = {}
    # Conduct t-test for each pair and store the p-value
    for key1, key2 in key_pairs:
        if test_method == 'ttest':
            _, p = ttest_ind(score[key1], score[key2])
        elif test_method == 'mannwhitneyu':
            _, p = mannwhitneyu(score[key1], score[key2], alternative='two-sided')
        elif test_method == 'wilcoxon':
            _, p = wilcoxon(score[key1], score[key2])
        results[(key1, key2)] = p

    # Initialize a matrix to hold the p-values
    num_keys = len(layer_order)
    p_mat = np.full((num_keys, num_keys), np.nan)
    
    # Fill the matrix with p-values
    for (key1, key2), p_value in results.items():
        i, j = layer_order[key1], layer_order[key2]
        p_mat[i, j] = p_value
        p_mat[j, i] = p_value  # Ensure the matrix is symmetric

    return p_mat

def remove_none_columns(data_dict):
    '''
    Remove the columns that have None values from a dictionary of lists. All lists have the same length. For example, if data_dict[key][i] = None, then all other data_dict[other_key][i] including data_dict[key][i] will be removed
    '''
    # Create a 2D NumPy array from the dict
    keys_sorted = sorted(data_dict.keys())
    data_values = [data_dict[key] for key in keys_sorted]
    data_array = np.array(data_values)

    # Find the columns without None values
    none_mask = np.vectorize(lambda x: x is None)(data_array)
    cols_to_keep = ~none_mask.any(axis=0)

    # Keep only the columns that do not have None values
    data_array_cleaned = data_array[:, cols_to_keep]

    # Convert the clean array back to a dictionary
    cleaned_data_dict = {key: data_array_cleaned[i, :].tolist() for i, key in enumerate(keys_sorted)}

    return cleaned_data_dict

def draw_lower_triangle_heatmap(p_mat, labels, fig=None, ax=None, cmap_label='p-value', cbar_kws={'label': 'p-value'}, heatmap_fmt='.2f', annot_kws={}):
    '''
    Draw a lower triangle heatmap with the p-values
    inputs:
        p_mat: a square matrix of p-values
        labels: a list of labels for the rows and columns of the matrix
    '''
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    mask_upper = np.triu(np.ones_like(p_mat, dtype=bool))

    sns.heatmap(p_mat, mask=mask_upper, annot=True, fmt=heatmap_fmt, cmap='coolwarm_r',
                cbar_kws=cbar_kws, ax=ax, annot_kws=annot_kws)

    for i in range(p_mat.shape[0]):
        for j in range(i+1, p_mat.shape[1]):
            ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, edgecolor='k', lw=1))

    for j in range(p_mat.shape[0]):
        ax.add_patch(plt.Rectangle((j, j), 1, 1, fill=False, edgecolor='k', lw=1))

    # Set the tick labels
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    return fig, ax

def get_significance_str(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'NA'

def half_life_avg_figoff_to_df(half_life_avg_figoff):
    # Flatten the nested dictionary into a list of tuples
    flattened_data = [
    (module, _type, value)
        for module, types in half_life_avg_figoff.items()
        for _type, values in types.items()
        for value in values
    ]

    # Create a DataFrame with the list of tuples
    df = pd.DataFrame(flattened_data, columns=["module", "type", "value"])
    return df
