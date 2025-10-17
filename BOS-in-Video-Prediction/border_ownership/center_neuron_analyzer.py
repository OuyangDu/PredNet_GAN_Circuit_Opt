import numpy as np
import border_ownership.border_response_analysis as bra
import copy
import pandas as pd
import hickle as hkl

def save_center_neuron_only(save_path, res_ori_path, center_neuron_rf_path):
    npz_data = np.load(res_ori_path, allow_pickle=True)
    data = {key: npz_data[key] for key in npz_data.files}

    center_neuron_dict = hkl.load(center_neuron_rf_path)
    print(center_neuron_dict.keys())
    for module in center_neuron_dict.keys():
        print('working on module', module)
        center_neuron_id = center_neuron_dict[module]['neuron_id_list']
        center_neuron_id = np.array(center_neuron_id)
        print('number of center RF neurons', center_neuron_id.shape)
        data_module = []
        for cnid in center_neuron_id:
            data_module.append(data[module][:, :, cnid[0], cnid[1], cnid[2]].copy())
        data[module] = np.stack(data_module, axis=0)
    np.savez(save_path, **data)
    return data

class Center_RF_Neuron_Analyzer():
    def __init__(self, bav_pthresh=0.05):
        self.bav_pthresh = bav_pthresh

    def center_res_ori_shift_size_unique_checker(self):
        '''
        check if there's only one shift and size
        outputs:
          shift_unique: the unique shift
          size_unique: the unique size
        '''
        # confirm there's only one shift and size
        para = self.res_ori['para']
        keys = self.res_ori['key']
        shift_index = np.where(keys == 'shift')[0][0]
        size_index = np.where(keys == 'size')[0][0]

        shift_unique = np.unique(para[:, shift_index])
        size_unique = np.unique(para[:, size_index])

        # check if there's only one shift and size
        assert len(shift_unique) == 1, 'there must be only one shift'
        assert len(size_unique) == 1, 'there must be only one size'
        return shift_unique[0], size_unique[0]

    def _rearange_res_ori(self, responses, trial_conditions, parameter_names):
        '''
        rearrange the res_ori to be in the order of orientation, beta, gamma. This rearanged format can be used in BOI_Analyzer
        Assuming data initializing lines here, e.g.,
        responses = np.empty((n_neuron, n_trials, n_time))
        trial_conditions = np.empty((n_trials, n_para))
        parameter_names = np.array(['orientation', 'beta', 'gamma'])
        '''
        # 0. get shape parameters
        n_time, n_neuron = responses.shape[-1], responses.shape[0]
        # 1. Get indices of parameters
        orientation_idx = np.where(parameter_names == 'orientation')[0][0]
        beta_idx = np.where(parameter_names == 'beta')[0][0]
        gamma_idx = np.where(parameter_names == 'gamma')[0][0]

        # 2. Extract conditions
        orientation_values = trial_conditions[:, orientation_idx].astype(float)
        unique_orientations = np.unique(orientation_values)
        beta_values = trial_conditions[:, beta_idx].astype(bool)
        gamma_values = trial_conditions[:, gamma_idx].astype(bool)

        # 3. Create masks for beta and gamma
        unique_beta, unique_gamma = [False, True], [False, True]
        beta_masks = [beta_values == val for val in unique_beta]
        gamma_masks = [gamma_values == val for val in unique_gamma]
    
        # 4. Initialize the new array
        n_orientation = len(unique_orientations)
        n_beta = 2
        n_gamma = 2
        rearranged_res = np.empty((n_orientation, n_beta, n_gamma, n_time, n_neuron))

        # 5. Iterate and stack trials
        for orientation_index, orientation in enumerate(unique_orientations):
            # Creating orientation mask
            orientation_mask = orientation_values == orientation
            for beta_index, beta_mask in enumerate(beta_masks):
                for gamma_index, gamma_mask in enumerate(gamma_masks):
                    # Combine masks
                    combined_mask = orientation_mask & beta_mask & gamma_mask
                    # Rearrange and stack responses
                    rearranged_res[orientation_index, beta_index, gamma_index] = responses[:, combined_mask].transpose(1, 2, 0)
        return rearranged_res, unique_orientations, unique_beta, unique_gamma

    def _reformat_res_ori(self):
        ## reformat res_ori so that it can be used in border_response_analysis
        self.rearange_res = {}
        self.unique_orientation = {}
        self.unique_beta = {}
        self.unique_gamma = {}

        for module in self.module_name:
            responses = self.res_ori[module]
            trial_conditions = self.res_ori['para']
            parameter_names = self.res_ori['key']
            rearange_res, unique_orientation, unique_beta, unique_gamma = self._rearange_res_ori(responses, trial_conditions, parameter_names)
            self.rearange_res[module] = rearange_res
            self.unique_orientation[module] = unique_orientation
            self.unique_beta[module] = unique_beta
            self.unique_gamma[module] = unique_gamma
        return self.rearange_res, self.unique_orientation, self.unique_beta, self.unique_gamma

    def load_data(self, center_res_ori_path, center_neuron_rf_path, base_fr=1):
        self.res_ori = np.load(center_res_ori_path, allow_pickle=True)
        self.res_ori = {key: self.res_ori[key] for key in self.res_ori.keys()}
        self.center_neuron_rf = hkl.load(center_neuron_rf_path)
        no_module_keys = ['image', 'para', 'key']
        self.module_name = [name for name in self.res_ori.keys() if name not in no_module_keys]

        if base_fr is not None:
            for module in self.module_name:
                self.res_ori[module] = self.res_ori[module] + base_fr
                heatmap_list = self.center_neuron_rf[module]['heatmap_list']
                self.center_neuron_rf[module]['heatmap_list'] = np.array(heatmap_list) + base_fr

        self.rearange_res, self.unique_orientation, self.unique_beta, self.unique_gamma = self._reformat_res_ori()

        self.center_res_ori_shift_size_unique_checker() # check if there's only one shift and size

    def compute_boi(self, bo_mean_time_init=0, bo_mean_time_final=19):
        self.boi = {}
        for module in self.module_name:
            boi = bra.compute_BO(self.rearange_res[module], replace_nan=True)
            boi = np.mean(boi[:, bo_mean_time_init:bo_mean_time_final], axis=1) # average over time, shape is (n_orientaion, n_neuron)
            self.boi[module] = boi.transpose()
        return self.boi

    def compute_boi_abs_max_and_pref_dire(self, replace_boi=False, bo_mean_time_init=0, bo_mean_time_final=19):
        if (not hasattr(self, 'boi')) or replace_boi:
            self.compute_boi(bo_mean_time_init, bo_mean_time_final)

        self.boi_abs_max = {}
        self.boi_pref_dire = {}
        for module in self.module_name:
            boi_abs = np.abs(self.boi[module])
            index_of_max_abs = np.argmax(boi_abs, axis=1)
            boi_abs_max = [boi_abs[idx, index_of_max_abs[idx]] for idx in range(boi_abs.shape[0])]
            self.boi_abs_max[module] = np.array(boi_abs_max)

            pref_dire = []
            for idx, ioma in enumerate(index_of_max_abs):
                if self.boi[module][idx, ioma] < 0:
                    pref_dire.append(self.unique_orientation[module][ioma] + 180)
                else:
                    pref_dire.append(self.unique_orientation[module][ioma])

            self.boi_pref_dire[module] = np.array(pref_dire)

        return self.boi_abs_max, self.boi_pref_dire

    def compute_boi_abs_rank(self):
        self.boi_abs_rank = {}
        for module in self.module_name:
            self.boi_abs_rank[module] = self.boi_abs_max[module].size - np.argsort(np.argsort(self.boi_abs_max[module]))
        return self.boi_abs_rank

    def compute_bav(self, bav_mean_time_init=0, bav_mean_time_final=19):
        '''
        compute bav
        input:
            bav_mean_time_init: int, initial time point for averaging bav
            bav_mean_time_final: int, final time point for averaging bav
        output:
            bav: dict, bav for each module
        '''
        self.bav, self.bav_angle = {}, {}
        for module in self.module_name:
            bav, angle = bra.compute_bo_av(self.rearange_res[module], self.unique_orientation[module], time_window=[bav_mean_time_init, bav_mean_time_final], with_angle=True)
            self.bav[module] = bav.transpose()
            self.bav_angle[module] = angle.transpose()
        return self.bav, self.bav_angle

    def bav_permutation_test(self, n_permutation=5000, bav_mean_time_init=0, bav_mean_time_final=19):
        '''
        bav permutation test
        input:
            n_permutation: int, number of permutation
            bav_mean_time_init: int, initial time point for averaging bav
            bav_mean_time_final: int, final time point for averaging bav
        output:
            bav_pvalue: dict, pvalue for each module
        '''
        self.bav_pvalue = {}
        self.is_bo = {}
        self.zscore_bav = {}
        for module in self.module_name:
            _, pvalue = bra.bo_permutation_test(self.rearange_res[module], self.unique_orientation[module], time_window=[bav_mean_time_init, bav_mean_time_final], n_permutations=n_permutation)
            self.bav_pvalue[module] = pvalue
            self.is_bo[module] = pvalue < self.bav_pthresh
            print('Bav permutation test for module {} is done'.format(module))
        return self.bav_pvalue

    def compute_bo_only_rank(self):
        '''
        compute the rank of bo units by their bav value. Non-bo units are assigned with Nan. run bav_permutation_test to obtain self.is_bo and run compute_bav to obtain self.bav
        '''
        bo_only_rank = {}
        for module in self.module_name:
            ranks = pd.Series(self.bav[module][self.is_bo[module]]).rank(ascending=False, method='min').astype('UInt32').to_numpy()
            bo_only_rank[module] = np.full(self.bav[module].shape, np.nan, dtype='float64')
            bo_only_rank[module][self.is_bo[module]] = ranks
        self.bo_only_rank = bo_only_rank
        return copy.deepcopy(self.bo_only_rank)

    def export_neuron_info(self):
        if not hasattr(self, 'boi'): self.compute_boi()
        if not (hasattr(self, 'boi_abs_max') and hasattr(self, 'boi_pref_dire')): self.compute_boi_abs_max_and_pref_dire()
        if not hasattr(self, 'boi_abs_rank'): self.compute_boi_abs_rank()
        if not (hasattr(self, 'bav') and hasattr(self, 'bav_angle')): self.compute_bav()
        if not hasattr(self, 'bav_pvalue'): self.bav_permutation_test()
        if not hasattr(self, 'bo_only_rank'): self.compute_bo_only_rank()

        self.bo_info = {}
        self.res_info = {}
        for module in self.module_name:
            neuron_id = np.array(self.center_neuron_rf[module]['neuron_id_list'], dtype=int).copy()
            neuron_id = [tuple(row) for row in neuron_id]
            bo_info_dict = {
                'neuron_id': neuron_id,
                'boi': self.boi[module].copy(),
                'boi_abs_max': self.boi_abs_max[module].copy(),
                'boi_pref_dire': self.boi_pref_dire[module].copy(),
                'boi_abs_rank': self.boi_abs_rank[module].astype(int).copy(),
                'bav': self.bav[module].copy(),
                'bav_angle': self.bav_angle[module].copy(),
                'bav_pvalue': self.bav_pvalue[module].copy(),
                'is_bo': self.is_bo[module].copy(),
                'bo_only_rank': self.bo_only_rank[module].copy(),
                'heatmap': np.array(self.center_neuron_rf[module]['heatmap_list']),
                'rf': np.array(self.center_neuron_rf[module]['rf_list']),
            }
            self.bo_info[module] = pd.DataFrame({k: pd.Series(list(v)) for k, v in bo_info_dict.items()})
            self.res_info[module] = combine_res_data(self.res_ori[module].swapaxes(0, 1), self.res_ori['para'], self.res_ori['key'], np.array(self.center_neuron_rf[module]['neuron_id_list']).copy())

        self.stim_info = combine_stim_data(self.res_ori['image'], self.res_ori['para'], self.res_ori['key'])
        return self.bo_info, self.res_info, self.stim_info

    def combine_res_stim_data(self, center_res_path, base_fr=1):
        '''
        in load data, data have only different orientation. Here we combine data with any trial parameters
        '''
        res = np.load(center_res_path, allow_pickle=True)
        res = {key: res[key] for key in res.keys()}

        # add base fr
        if base_fr is not None:
            for module in self.module_name:
                res[module] = res[module] + base_fr

        res_info = {}
        # res info
        for module in self.module_name:
            res_info[module] = combine_res_data(res[module].swapaxes(0, 1), res['para'], res['key'], np.array(self.center_neuron_rf[module]['neuron_id_list']).copy())

        # stim info
        stim_info = combine_stim_data(res['image'], res['para'], res['key'])
        return res_info, stim_info

    def get_trial_res(module, trial_info):
        pass

def combine_stim_data(stim_image, trial_values, trial_key):
    '''
    Your data shapes:
    stim_image (num_trial, image_width, image_height, chs)
    trial_values (num_trial, n_trial_para)
    trial_keys (num_trial_para,)
    '''
    # Given your numpy arrays: stim_image, trial_values, and trial_key
    images_series = pd.Series([stim_image[i] for i in range(stim_image.shape[0])], name='image')
    trials_df = pd.DataFrame(trial_values, columns=[f'{i}' for i in trial_key])

    # Combine into a single DataFrame
    df = pd.concat([images_series, trials_df], axis=1)
    df.insert(0, 'trial_id', range(df.shape[0]))
    return df

def combine_res_data(neural_responses, trial_values, trial_keys, neural_id):
    '''
    Your data shapes:
    neural_responses (num_trial, num_neuron, n_time)
    trial_values (num_trial, n_trial_para)
    trial_keys (num_trial_para,)
    neural_id (num_neuron, 3)
    output:
        df: DataFrame, with columns: neuron_id, trial_id, response, trial_para1, trial_para2, ...
    '''
    num_trials, num_neurons, n_time = neural_responses.shape
    
    # Flatten the neural response to have one row per trial-neuron combination
    flattened_responses = neural_responses.reshape(num_trials * num_neurons, n_time)
    
    # Repeat trial values and neural ids to match the flattened response shape
    repeated_trial_values = np.repeat(trial_values, num_neurons, axis=0)
    repeated_neural_ids = np.tile(neural_id, (num_trials, 1))
    
    # Create the DataFrame
    repeated_neural_ids = [tuple(row) for row in repeated_neural_ids]
    df = pd.DataFrame({'neuron_id': repeated_neural_ids})
    df['trial_id'] = np.repeat(np.arange(num_trials), num_neurons)
    df['response'] = list(flattened_responses)
    
    # Add trial parameters
    for i, key in enumerate(trial_keys):
        df[key] = repeated_trial_values[:, i]

    return df
