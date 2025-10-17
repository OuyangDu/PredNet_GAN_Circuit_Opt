import os
import hickle as hkl
import numpy as np
from border_ownership.ablation_visu import get_mse
from kitti_settings import *

class Ablation_Data_Reader():
    def __init__(self, data_head, random_id_list, data_save_dir=None):
        '''
        the data path should be like:
        data_path = os.path.join(data_save_dir, f"{data_head}_ablate_diff_num_units_random{rid}.hkl") or f"{data_head}_ablate_diff_num_units_random{rid}_10boot.hkl" if data_head is square or kitti
        Data must be generated from kitti_mul_para.py

        Args:
        data_head: str, the head of the data file name
        random_id_list: list of int, the random_id for random initialization
        '''
        if data_save_dir is None:
            data_save_dir = RESULTS_SAVE_DIR

        self.data = {}
        for rid in random_id_list:
            if data_head in {'kitti', 'square'}:
                data_path = os.path.join(data_save_dir, f"{data_head}_ablate_diff_num_units_random{rid}.hkl")
            else:
                data_path = os.path.join(data_save_dir, f"{data_head}_ablate_diff_num_units_random{rid}.hkl")
            self.data[rid] = hkl.load(data_path)

    def get_keys(self, module):
        '''
        get all possible keys for the data
        Args:
        module: str, the module name
        '''
        data_i = self.data[next(iter(self.data))] # pick one random_id to get the keys
        n_units, mse = get_mse(data_i, is_bo=True, module=module) # possible number of ablation units
        n_units = np.array(n_units); mse = np.array(mse) # shape of mse is (n_units, n_unit_reample_boot)
        unit_resample_id = mse.shape[1]

        possible_keys = {
            'video_resample_id': self.data.keys(),
            'n_units': n_units,
            'is_bo': [True, False],
            'unit_resample_id': np.arange(unit_resample_id),
        }
        return possible_keys

    def get_data(self, module, video_resample_id, n_units, is_bo, unit_resample_id=None):
        '''
        get the data for the given keys
        Args:
        video_resample_id: int, the video resample id
        n_units: int, the number of units
        is_bo: bool, whether the data is for BO
        unit_resample_id: int, the unit resample id
        module: str, the module name
        '''
        data_i = self.data[video_resample_id]
        n_unit_all, mse_all = get_mse(data_i, is_bo=is_bo, module=module)
        n_unit_all = np.array(n_unit_all); mse_all = np.array(mse_all) # shape of mse is (n_units, n_unit_reample_boot)
        n_unit_idx = list(n_unit_all).index(n_units)

        if unit_resample_id is None:
            return mse_all[n_unit_idx] # return all mse for the given n_units, for different resampling of a fixed number of ablation units
        return mse_all[n_unit_idx, unit_resample_id]

    def get_no_ablation(self, module, video_resample_id):
        '''
        get the no ablation mse
        Args:
        video_resample_id: int, the video resample id
        module: str, the module name
        '''
        data_i = self.data[video_resample_id]
        return data_i['no_ablation'][0]

def avg_video_num_unit_boot(adr, module, shuffle_is_bo=False, avg_video=True):
    '''
    average the data across video_resample_id and unit_resample_id
    Args:
    adr: Ablation_Data_Reader, the ablation data reader
    module: str, the module name
    shuffle_is_bo: bool, whether to shuffle is_bo
    avg_video: bool, whether to average across video_resample_id
    return:
    avg_data: dict, the average data across video_resample_id and unit_resample_id. avg_data should have three keys, n_units is the number of units ablated, True_rpmse is the relative prediction MSE for BO ablation, False_rpmse is the relative prediction MSE for non-BO ablation. The shape of n_units, True_rpmse and False_rpmse is [n_units]
    '''
    avg_data = {}
    all_keys = adr.get_keys(module)
    avg_data['n_units'] = all_keys['n_units']
    avg_data['True_rpmse'] = []
    avg_data['False_rpmse'] = []

    for n_units in all_keys['n_units']:

        to_be_avg_t_rpmse = []; to_be_avg_f_rpmse = []
        for rid in all_keys['video_resample_id']:
            no_ablation = adr.get_no_ablation(module, rid)

            if shuffle_is_bo:
                is_bo = np.random.choice([True, False])
            else:
                is_bo = True

            t_rpmse = (adr.get_data(module, rid, n_units, is_bo) - no_ablation) / no_ablation # shape is [n_unit_resample_boot]. rpmse means relative prediction MSE
            f_rpmse = (adr.get_data(module, rid, n_units, not is_bo) - no_ablation) / no_ablation # shape is [n_unit_resample_boot]

            t_rpmse = np.mean(t_rpmse) # mean over different bootstraping of num_units
            f_rpmse = np.mean(f_rpmse) # mean over different bootstraping of num_units

            to_be_avg_t_rpmse.append(t_rpmse)
            to_be_avg_f_rpmse.append(f_rpmse) # after looping over all rid, shape is [n_video_resample_boot]


        avg_data['True_rpmse'].append(to_be_avg_t_rpmse)
        avg_data['False_rpmse'].append(to_be_avg_f_rpmse)

    avg_data['True_rpmse'] = np.array(avg_data['True_rpmse']) # shape is [n_units, n_video_resample_boot]
    avg_data['False_rpmse'] = np.array(avg_data['False_rpmse'])

    if avg_video:
        avg_data['True_rpmse'] = np.mean(avg_data['True_rpmse'], axis=1) # shape is [n_units]
        avg_data['False_rpmse'] = np.mean(avg_data['False_rpmse'], axis=1)
    return avg_data

def compute_pdf(t_rpmse, f_rpmse):
    '''
    compute the prediction difference fraction (pdf)
    '''
    t_rpmse = np.array(t_rpmse)
    f_rpmse = np.array(f_rpmse)

    diff_rpmse = t_rpmse - f_rpmse
    tot_diff = np.sum(diff_rpmse)

    max_rpmse = np.max([np.abs(t_rpmse), np.abs(f_rpmse)], axis=0)
    tot_max = np.sum(max_rpmse)

    pdf = tot_diff / tot_max
    return pdf
