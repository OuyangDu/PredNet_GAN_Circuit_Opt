# please run bo_non_bo_avg_fr_data.py first
import numpy as np
import os
import copy
import hickle as hkl
from scipy.stats import ranksums, ttest_ind
import seaborn as sns
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
from border_ownership.ploter import plot_seq_prediction, plot_layer_boxplot_helper, removeOutliers
from border_ownership.agent import Agent
from data_utils import SequenceGenerator
from kitti_settings import *

#################### Help functions ####################

def subset_min_median_diff(n_unit, n_repeat, bo_avg_res_module, nbo_avg_res_module):
    '''
    Randomly sample n_unit units from both groups and calculate the median difference between the two groups. Repeat this process n_repeat times and return the sample with the minimum median difference.
    input:
        n_unit: int, number of units to sample from each group
        n_repeat: int, number of times to repeat the sampling process
        bo_avg_res_module: np.array, average responses of the border ownership units
        nbo_avg_res_module: np.array, average responses of the non-border ownership units
    output:
        min_unit_id_bo: np.array, unit IDs of the border ownership units with the minimum median difference
        min_bo_sample: np.array, sample of the border ownership units with the minimum median difference
        min_unit_id_nbo: np.array, unit IDs of the non-border ownership units with the minimum median difference
        min_nbo_sample: np.array, sample of the non-border ownership units with the minimum median difference
        min_median_diff: float, minimum median difference between the two groups
    '''
    min_median_diff = np.inf
    min_unit_id_bo, min_unit_id_nbo = None, None
    min_bo_sample, min_nbo_sample = None, None
    
    for i in range(n_repeat):
        # Get the number of units in both groups
        n_unit_bo, n_unit_nbo = bo_avg_res_module.shape[0], nbo_avg_res_module.shape[0]
        # Randomly sample unit IDs for both groups
        unit_id_bo = np.random.choice(range(n_unit_bo), n_unit, replace=False)
        unit_id_nbo = np.random.choice(range(n_unit_nbo), n_unit, replace=False)
        
        # Fetch the samples for both groups using the unit IDs
        bo_sample = bo_avg_res_module[unit_id_bo]
        nbo_sample = nbo_avg_res_module[unit_id_nbo]

        m_median_d = (np.median(bo_sample) - np.median(nbo_sample))**2
        m_mean_d = (np.mean(bo_sample) - np.mean(nbo_sample))**2
        mmd = (m_median_d + m_mean_d) / 2

        if mmd < min_median_diff:
            min_median_diff = mmd
            min_unit_id_bo = unit_id_bo
            min_unit_id_nbo = unit_id_nbo
            min_bo_sample = bo_sample
            min_nbo_sample = nbo_sample

    return min_unit_id_bo, min_bo_sample, min_unit_id_nbo, min_nbo_sample, min_median_diff

def find_min_samples(bo_avg_res_module, nbo_avg_res_module, n_repeat, p_thresh=0.5):
    """
    Function to find the minimum number of units such that the statistical
    significance (p-value, both wilconxon rank-sum and t test) between bo_avg_res and nbo_avg_res samples is greater than p_thresh.
    
    Parameters:
    bo_avg_res_module: np.array, average responses of the border ownership units
    nbo_avg_res_module: np.array, average responses of the non-border ownership units
    n_repeat (int): Number of repetitions for sampling.
    
    Returns:
    tuple: Returns a tuple containing:
           - min_bo_sample (numpy.ndarray): Minimum sample of BO results meeting criteria.
           - min_nbo_sample (numpy.ndarray): Minimum sample of NBO results meeting criteria.
           - p (float): p-value of the statistical test at the stopping condition.
    """
    n_bo_unit = bo_avg_res_module.shape[0]
    n_nbo_unit = nbo_avg_res_module.shape[0]
    n_unit = min(n_bo_unit, n_nbo_unit)

    for n_unit_sample in range(n_unit, 0, -1):
        unit_id_bo, min_bo_sample, unit_id_nbo, min_nbo_sample, _ = subset_min_median_diff(
            n_unit_sample, n_repeat, bo_avg_res_module, nbo_avg_res_module
        )
        _, p = ranksums(min_bo_sample, min_nbo_sample)
        _, p_t = ttest_ind(min_bo_sample, min_nbo_sample)
        if p > p_thresh and p_t > p_thresh:
            break

    return min_bo_sample, min_nbo_sample, unit_id_bo, unit_id_nbo, p, n_unit_sample

# Function to remove outliers, for better visualization in stripplot
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Function to plot responses for 'bo' and 'nbo' types from data dictionaries
def plot_avg_response(bo_avg_res, nbo_avg_res, video_head, y_label='Mean Squared Response', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3))

    # Copy the data to avoid modifying the original data structures
    bo_avg_res_plot = copy.deepcopy(bo_avg_res)
    nbo_avg_res_plot = copy.deepcopy(nbo_avg_res)

    # Create a DataFrame from bo_avg_res data
    df_bo = pd.DataFrame({'module': k, 'value': v, 'type': 'bo'} 
                         for k, values in bo_avg_res_plot.items() for v in values)
    # Create a DataFrame from nbo_avg_res data
    df_nbo = pd.DataFrame({'module': k, 'value': v, 'type': 'nbo'} 
                          for k, values in nbo_avg_res_plot.items() for v in values)

    # Concatenate both DataFrames to create a single DataFrame
    df = pd.concat([df_bo, df_nbo])

    # Apply the function to each group to filter out outliers
    filtered_df = df.groupby(['type', 'module']).apply(remove_outliers, 'value').reset_index(drop=True)

    # Define the order for the categorical x-axis based on the 'module' column
    order = filtered_df['module'].drop_duplicates().sort_values().tolist()

    # Plotting the boxplot and stripplot
    color_palette = {'bo': 'red', 'nbo': 'blue'}

    sns.boxplot(x='module', y='value', hue='type', data=df, ax=ax, showcaps=False,
                boxprops={'facecolor':'None'}, flierprops={'visible': False},
                palette=color_palette, linewidth=1.5, dodge=True, order=order)

    sns.stripplot(x='module', y='value', hue='type', data=filtered_df, dodge=True, palette=color_palette,
                  jitter=True, ax=ax, alpha=0.6, size=4, order=order)

    # Set y-axis limits based on filtered data to exclude outliers from the view
    y_min = filtered_df['value'].min() - (filtered_df['value'].std() * 0.5)  # Optional padding
    y_max = filtered_df['value'].max() + (filtered_df['value'].std() * 0.5)
    ax.set_ylim(y_min, y_max)

    # Remove default legend and add customized legend
    ax.legend_.remove()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=['BO', 'Non-BO'], title="", fontsize=12)

    # Set the title and y-axis label from function parameters
    ax.set_title(video_head, fontsize=14)
    ax.set_ylabel(y_label)

    return ax

class Video_Avg_Res_Sampling():
    def __init__(self, video_head, data_path, n_subsample_repeat=1000):
        '''
        video_head: str, video head to analyze
        data_path: str, path to the data file
        n_subsample_repeat: int, number of times to repeat the subsampling process
        '''
        self.video_head = video_head
        self.data = hkl.load(data_path)
        self.module_list = self.data[video_head]['bo_avg_res'].keys()
        self.n_subsample_repeat = n_subsample_repeat

    def set_video_head(self, video_head):
        '''
        video_head: str, video head to analyze
        '''
        self.video_head = video_head

    def get_avg_res_unit(self, subsample_to_adjust_median=False):
        '''
        subsample_to_adjust_median: bool, whether to subsample to adjust the median difference
        outputs:
        bo_avg_res: dict, average responses of the border ownership units. Keys are modules
        nbo_avg_res: dict, average responses of the non-border ownership units. Keys are modules
        bo_id: dict, unit IDs of the border ownership units. Keys are modules
        nbo_id: dict, unit IDs of the non-border ownership units. Keys are modules
        n_bo_unit_dict: dict, number of border ownership units. Keys are modules
        n_nbo_unit_dict: dict, number of non-border ownership units. Keys are modules
        p_dict: dict, p-values of the wilconxon rank-sum tests. Keys are modules
        '''
        bo_avg_res = copy.deepcopy(self.data[self.video_head]['bo_avg_res'])
        nbo_avg_res = copy.deepcopy(self.data[self.video_head]['nbo_avg_res'])
        bo_id = copy.deepcopy(self.data[self.video_head]['bo_id'])
        nbo_id = copy.deepcopy(self.data[self.video_head]['nbo_id'])

        if subsample_to_adjust_median:
            return self._get_subsample_avg_res_unit(bo_avg_res, nbo_avg_res, bo_id, nbo_id)
        else:
            return self._get_full_avg_res_unit(bo_avg_res, nbo_avg_res, bo_id, nbo_id)

    def _get_full_avg_res_unit(self, bo_avg_res, nbo_avg_res, bo_id, nbo_id):
        '''
        bo_avg_res: dict, average responses of the border ownership units. Keys are modules
        nbo_avg_res: dict, average responses of the non-border ownership units. Keys are modules
        bo_id: dict, unit IDs of the border ownership units. Keys are modules
        nbo_id: dict, unit IDs of the non-border ownership units. Keys are modules
        '''
        p_dict = {}
        n_bo_unit_dict = {}
        n_nbo_unit_dict = {}
        for module in self.module_list:
            _, p_value = ranksums(bo_avg_res[module], nbo_avg_res[module])
            p_dict[module] = p_value

        for module in self.module_list:
            bo_id[module] = np.array(bo_id[module])
            nbo_id[module] = np.array(nbo_id[module])

            n_bo_unit_dict[module] = len(bo_avg_res[module])
            n_nbo_unit_dict[module] = len(nbo_avg_res[module])

        return bo_avg_res, nbo_avg_res, bo_id, nbo_id, n_bo_unit_dict, n_nbo_unit_dict, p_dict

    def _get_subsample_avg_res_unit(self, bo_avg_res, nbo_avg_res, bo_id, nbo_id):
        '''
        subsample units to adjust the median difference
        '''
        p_dict = {}
        n_bo_unit_dict = {}
        n_nbo_unit_dict = {}
        for module in self.module_list:
            bo_avg_res_module = np.array(bo_avg_res[module]);
            nbo_avg_res_module = np.array(nbo_avg_res[module])
            min_bo_sample, min_nbo_sample, unit_id_bo, unit_id_nbo, p, n_unit_sample = find_min_samples(bo_avg_res_module, nbo_avg_res_module, self.n_subsample_repeat)

            # save results
            bo_avg_res[module] = min_bo_sample; nbo_avg_res[module] = min_nbo_sample

            bo_id_module_full = np.array(bo_id[module])
            nbo_id_module_full = np.array(nbo_id[module])
            bo_id[module] = bo_id_module_full[unit_id_bo];
            nbo_id[module] = nbo_id_module_full[unit_id_nbo]

            p_dict[module] = p

            n_bo_unit_dict[module] = n_unit_sample
            n_nbo_unit_dict[module] = n_unit_sample

        return bo_avg_res, nbo_avg_res, bo_id, nbo_id, n_bo_unit_dict, n_nbo_unit_dict, p_dict

##################################################
video_head_list = ['translating', 'random', 'kitti']
data_path = os.path.join(DATA_DIR_HOME, 'bo_non_bo_response.hkl')
n_subsample_repeat = 1000

subsampled_unit_id = {}

viar = Video_Avg_Res_Sampling(video_head_list[0], data_path, n_subsample_repeat=n_subsample_repeat)

n_video = len(video_head_list)
fig, axes = plt.subplots(n_video, 2, figsize=(14, 3 * n_video))

for i, video_head in enumerate(video_head_list):
    print(f'Processing {video_head} video...')
    subsampled_unit_id[video_head] = {}

    viar.set_video_head(video_head)

    bo_avg_res, nbo_avg_res, bo_id, nbo_id, n_bo_unit_dict, n_nbo_unit_dict, p_dict = viar.get_avg_res_unit()
    ax = plot_avg_response(bo_avg_res, nbo_avg_res, video_head, y_label='Mean Squared Response', ax=axes[i, 0])
    pprint('full n_bo_unit_dict:')
    pprint(n_bo_unit_dict)
    pprint('full n_nbo_unit_dict:')
    pprint(n_nbo_unit_dict)

    bo_avg_res, nbo_avg_res, bo_id, nbo_id, n_bo_unit_dict, n_nbo_unit_dict, p_dict = viar.get_avg_res_unit(subsample_to_adjust_median=True)
    ax = plot_avg_response(bo_avg_res, nbo_avg_res, video_head, y_label='Mean Squared Response', ax=axes[i, 1])
    pprint('Subsampled n_bo_unit_dict:')
    pprint(n_bo_unit_dict)
    pprint('Subsampled n_nbo_unit_dict:')
    pprint(n_nbo_unit_dict)

    subsampled_unit_id[video_head]['bo'] = bo_id
    subsampled_unit_id[video_head]['nbo'] = nbo_id

hkl.dump(subsampled_unit_id, os.path.join(DATA_DIR_HOME, 'subsampled_to_adjust_median_unit_id.hkl'))
fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, f'bo_nbo_response.svg'), bbox_inches='tight') # Save the figure
plt.show()
