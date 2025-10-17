import border_ownership.neuron_info_processor as nip
from border_ownership.empty_square import plot_sequential_res_on_square, plot_sequential_square_part
import matplotlib.pyplot as plt
from scipy.stats import zscore
import copy
import pandas as pd
import numpy as np
from kitti_settings import *

def transform_keypoint_func(x):
    if x == ():
        return 'vanilla', None
    elif x == (0, 1, 2, 3, 4, 5, 6, 7): # center edge and full square
        return 'center_edge', 8
    elif x == (1, 2, 3, 4, 5, 6, 7): # no center edge and no bottom edge
        return 'no_center_edge', 8
    elif x[0] == None:
        return 'no_center_edge', None
    elif x[0] == 0:
        return ('center_edge', None) if len(x) == 1 else ('center_edge', int(x[1]))
    else:
        return 'no_center_edge', int(x[0])

def transform_keypoint(df):
    '''
    :param df: dataframe with column 'keep_keypoint_id'
    :return: dataframe with column 'square_type' and 'edge_id'
    '''
    df['keep_keypoint_id'] = df['keep_keypoint_id'].apply(tuple)
    df[['square_type', 'edge_id']] = df['keep_keypoint_id'].apply(lambda x: pd.Series(transform_keypoint_func(x)))
    df['edge_id'] = df['edge_id'].astype('UInt8')
    return df

def preprocess_neuron_info(neuron_info, module, neural_rank, rank_method='BOI'):
    """
    Load data and extract information for a target neuron.
    :param neuron_info: tuple containing all neuron data.
    :param module: string name of module.
    :param neural_rank: integer rank of target neuron.
    :return: Preprocessed neuron info.
    """
    bo_info, res_info, stim_info, unique_orientation = neuron_info
    nipor = nip.Neuron_Info_Processor()
    nipor.load_data(bo_info, res_info, stim_info, module, unique_orientation)
    neuron_bo_info, neuron_res_info = nipor.get_target_neuron_info(neural_rank, rank_method=rank_method)
    neuron_res_info['mean_response'] = neuron_res_info['response'].apply(lambda x: np.mean(x))
    return neuron_bo_info, neuron_res_info

def subset_to_response(subset):
    grouped = subset.groupby('edge_id', dropna=False)['mean_response'].mean().reset_index().sort_values(by='edge_id', na_position='first')
    return grouped['mean_response'].to_numpy()

def subset_to_image(subset, stim_info, image_gamma= False):
    trial_id = subset[subset['gamma'] == image_gamma].sort_values(by='edge_id', na_position='first')['trial_id'].to_numpy()
    image = stim_info[stim_info['trial_id'].isin(trial_id)]['image'].to_numpy()
    return image

def split_with_center_beta(neuron_df, stim_info, conditions):
    """
    Calculate response differences for each condition.
    :param neuron_df: DataFrame containing neuron response info.
    :return: List of normalized responses (center_edge_present, beta_condition).
    """
    split_res = []
    split_image = []
    for center_edge_present, beta_condition in conditions:
        subset = neuron_df[
            (neuron_df['square_type'] == 'center_edge' if center_edge_present else neuron_df['square_type'] == 'no_center_edge') &
            (neuron_df['beta'] == beta_condition)
        ]

        image = subset_to_image(subset, stim_info)
        normalized_response = subset_to_response(subset)

        split_image.append(copy.deepcopy(image))
        split_res.append(copy.deepcopy(normalized_response))

    return split_image, split_res

class Square_Part_Analyzer():
    def __init__(self, module, neural_rank, rank_method='bo_only'):
        '''
        :param module: string, name of module
        :param neural_rank: int, rank of neuron. If rank_method is BOI, then neural_rank is the rank of neuron in the BO module. Otherwise, it is the rank of neuron in all candidate neurons.
        :param rank_method: str. 'BOI' or 'bo_only'. 'BOI' means the rank of neuron in all candidate neurons. 'bo_only' means the rank of bo neuron, by their bav values
        :conditions: list of tuples, each tuple is (center_edge_present, preferred_side). center_edge_present is a bool, indicating whether the center edge is present. preferred_side is a bool, indicating whether the preferred side is the right side.
        '''
        self.module = module
        self.neural_rank = neural_rank
        self.rank_method = rank_method

        self.conditions = [
            (True, True),
            (True, False),
            (False, True),
            (False, False)
        ] # (center_edge_present, preferred_side)

    def _transform_condition_to_beta(self):
        '''
        transform condition to beta. The original condition is (center_edge_present, preferred_side). We want to transform it to (center_edge_present, beta_condition).
        '''
        pref_dir = self.neuron_info['boi_pref_dire'].iloc[0]

        if pref_dir < 180:
            condition_beta = self.conditions
        else:
            condition_beta = [(x[0], not x[1]) for x in self.conditions] # reverse the preferred side
        return condition_beta

    def load_data(self, data_info):
        '''
        data_info: dict, containing keys: bo_info, res_square_part_info, stim_square_part_info, unique_orientation
        output:
        self.neuron_info: dataframe, containing information of the neuron
        self.pref_ori: float, preferred orientation of the neuron
        self.split_image: length-4 list of numpy array. Each element is the images of one condition (self.conditions). The shape of each element is (8, image_width, image_height). 8 means 8 different square fragment, [None, 1, 2, 3, 4, 5, 6, 7] wehre None means the center edge only.
        self.split_res: length-4 list of numpy array. Each element is the responses (didn't substracting the center edge) of one condition (self.conditions). The shape of each element is (8, 8). 8 means 8 different square fragment, [None, 1, 2, 3, 4, 5, 6, 7] wehre None means the center edge only.
        self.zmap: numpy array, the zmap of the neuron. The shape is (image_width, image_height). Can you use to draw the cRF contour using ax.contourf.
        '''

        data_info_sub = (data_info['bo_info'], data_info['res_square_part_info'], data_info['stim_square_part_info'], data_info['unique_orientation'])
        self.neuron_info, neuron_res_info = preprocess_neuron_info(data_info_sub, self.module, self.neural_rank, rank_method=self.rank_method)
        self.pref_ori = nip.get_preferred_orientation(self.neuron_info)

        ### process neuron_res_info to res to different conditions
        neuron_res_info = neuron_res_info[neuron_res_info['orientation'] == self.pref_ori] # only use the preferred orientation
        neuron_res_info = transform_keypoint(neuron_res_info)

        condition_beta = self._transform_condition_to_beta() # convert condition ([center_edge_present, preferred_side]) to conditions_beta ([center_edge_present, beta])
        self.split_image, self.split_res = split_with_center_beta(neuron_res_info, data_info['stim_square_part_info'], condition_beta)

        ### zmap used for adding cRF
        _, self.zmap = nip.get_stimulus_and_rf(data_info['stim_square_part_info'], self.neuron_info, target_ori=self.pref_ori)

    def query(self, key):
        '''
        query neuron information. Here are the things that can be queried:
        'pref_ori': preferred orientation
        'neuron_id' (list [3]): the id of the neuron
        'boi' (list [n_dire]): the boi of the neuron in each direction
        'boi_abs_max' (float): the maximum absolute value of boi
        'boi_pref_dire' (float): the preferred direction of the neuron
        'boi_abs_rank' (int): the rank of the neuron according to the absolute value of boi
        'bav' (float): the average boi value of the neuron
        'bav_angle' (float): the direction of the bav
        'bav_pvalue' (float): the pvalue of the bav
        'is_bo' (bool): whether the neuron is a bo neuron
        '''
        if key == 'pref_ori':
            return self.pref_ori
        else:
            return self.neuron_info[key].iloc[0]

    def _get_index_by_condition(self, with_center_edge=True, pref_side=True, edge_id=None):
        '''
        get the index of the condition in self.split_image and self.split_res
        :param with_center_edge: bool, whether the center edge is present
        :param pref_side: bool
        :param edge_id: int, the id of the edge. If None, then edge_id = 0, which means the center edge.
        '''
        idx = self.conditions.index((with_center_edge, pref_side))
        edge_idx = edge_id if edge_id is not None else 0
        return idx, edge_idx

    def get_res_change(self, with_center_edge=True, pref_side=True, edge_id=None, is_zscore=False, mode='by_edge_id', substract=True):
        '''
        get the response change of the neuron. The response change is the response of each square fragment substracting the response of the center edge.
        :param with_center_edge: bool, whether the center edge is present
        :param pref_side: bool
        :param edge_id: int, the id of the edge. If None, then edge_id = 0, which means the center edge; if 8, full square (might not have center edge)
        :param is_zscore: bool, whether to zscore the response (before substraction)
        :param mode: string, 'by_edge_id', 'all', or 'all_by_name'. If 'by_edge_id', then return the response change of the specified condition. If 'all', then return the response change of all conditions. Each condition contains 8 edge ids from None, 1, ... to 7 and 8, where None means center edge, 1 to 7 are individual fragment, 8 is the full square. If 'all_by_name' then return the response change of all conditions but edge_id were averaged to edge name. Specifically, 'NC' is the average of id 1, 7; 'NE' is the average of id 2, 6; 'FC' is the average of id 3, 5; 'FE' is the average of id 4; 'CE' is the average of id None; 'all' is all fragment each condition is listed in the order of [0, NC, NE, FC, FE, all].
        :param substract: bool, whether to substract the response of the center edge
        '''
        if is_zscore:
            split_res_temp = self.zscore_split_res()
        else:
            split_res_temp = np.array(self.split_res)

        if mode == 'by_edge_id':
            idx, edge_idx = self._get_index_by_condition(with_center_edge, pref_side, edge_id)

            if substract:
                return split_res_temp[idx][edge_idx] - split_res_temp[idx][0]
            else:
                return split_res_temp[idx][edge_idx]

        elif mode == 'all':
            if substract:
                return split_res_temp - split_res_temp[:, 0, np.newaxis]
            else:
                return split_res_temp

        elif mode == 'all_by_name':
            if substract:
                split_res_change = split_res_temp - split_res_temp[:, 0, np.newaxis]
            else:
                split_res_change = split_res_temp

            split_res_change = self._avg_edge_to_edge_name(split_res_change)
            return split_res_change

    @staticmethod
    def _avg_edge_to_edge_name(split_res_changes):
        '''
        return the response change of all conditions but edge_id were averaged to edge name. Specifically, 'NC' is the average of id 1, 7; 'NE' is the average of id 2, 6; 'FC' is the average of id 3, 5; 'FE' is the average of id 4; 'CE' is the average of id None; 'all' is all fragment each condition is listed in the order of [0, NC, NE, FC, FE, all].
        input:
          split_res_changes: numpy array, shape (n_condition = 4, n_edge=9). The first edge is always the center edge, the 7 in between corresponds to edge id from 1 to 7, the last edge is the full square (might not have center edge).
        '''
        split_res_change_by_name = np.zeros((split_res_changes.shape[0], 6))
        split_res_change_by_name[:, 0] = split_res_changes[:, 0] # CE which should be 0
        split_res_change_by_name[:, 1] = np.mean(split_res_changes[:, [1, 7]], axis=1) # NC
        split_res_change_by_name[:, 2] = np.mean(split_res_changes[:, [2, 6]], axis=1) # NE
        split_res_change_by_name[:, 3] = np.mean(split_res_changes[:, [3, 5]], axis=1) # FC
        split_res_change_by_name[:, 4] = split_res_changes[:, 4] # FE
        split_res_change_by_name[:, 5] = split_res_changes[:, 8] # all
        return split_res_change_by_name

    def get_square_part_image(self, with_center_edge=True, pref_side=True, edge_id=None, mode='by_edge_id'):
        '''
        get the square part image of the neuron
        :param with_center_edge: bool, whether the center edge is present
        :param pref_side: bool
        :param edge_id: int, the id of the edge. If None, then edge_id = 0, which means the center edge. if 8, full square
        :param mode: string, 'by_condition' or 'all'. If 'by_condition', then return the response change of the specified condition. If 'all', then return the response change of all conditions.
        '''

        if mode == 'by_edge_id':
            idx, edge_idx = self._get_index_by_condition(with_center_edge, pref_side, edge_id)
            return self.split_image[idx][edge_idx]
        else:
            return self.split_image

    def zscore_split_res(self):
        '''
        zscore the split_res
        '''
        split_res_zscore = zscore(np.array(self.split_res), axis=None)
        return split_res_zscore

    def plot_sequential_res_on_square(self, is_zscore=False):
        '''
        plot the sequential response on the square. The sequential response is the response of each square fragment substracting the response of the center edge.
        :param is_zscore: bool, whether to zscore the response (before substraction)
        '''
        if is_zscore:
            split_res_temp = self.zscore_split_res()
        else:
            split_res_temp = np.array(self.split_res)

        split_res_change = [split_res_temp[i] - split_res_temp[i][0] for i in range(4)]
        split_res_change = np.array(split_res_change)
        fig, ax = plot_sequential_res_on_square(split_res_change[..., :-1], self.conditions) # -1 to remove the full square response change
        return fig, ax

    def plot_sequential_square_part_stim(self, with_cRF=True):
        split_image = np.array(self.split_image)[..., :-1] # -1 to remove the full square response change
        fig, axes = plot_sequential_square_part(split_image, self.conditions)
        for ax in np.array(axes).flatten():
            ax.contour(self.zmap, levels=[1], colors='white', linestyles='solid', linewidths=1)

        return fig, axes
