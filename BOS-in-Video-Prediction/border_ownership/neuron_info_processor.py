# use the three info from center_rf_neuron_analyzer to compute other quantities.
import numpy as np
from border_ownership.rf_finder import RF_Finder_Local_Sparse_Noise
from border_ownership.util import sci_notation
import border_ownership.ploter as ploter

class Neuron_Info_Processor():
    def load_data(self, bo_info, res_info, stim_info, module, boi_ori_value):
        '''
        load the info dict from center_neuron_analyzer
        input:
            bo_info: dict, the bo_info (neural meta infomation) dict from center_neuron_analyzer.export_neuron_info(). It has module name as keys indicating the information about each module. Kyes under each module are the same, which are:
                'neuron_id' (list [3]): the id of the neuron
                'boi' (list [n_dire]): the boi of the neuron in each direction
                'boi_abs_max' (float): the maximum absolute value of boi
                'boi_pref_dire' (float): the preferred direction of the neuron
                'boi_abs_rank' (int): the rank of the neuron according to the absolute value of boi
                'bav' (float): the average boi value of the neuron
                'bav_angle' (float): the direction of the bav
                'bav_pvalue' (float): the pvalue of the bav
                'is_bo' (bool): whether the neuron is a bo neuron
                'bo_only_rank' (int): the rank of a bo neuron among all bo neurons. Ranked by bav
                'heatmap' (np.array [2, width, height]): the heatmap of the neuron. heatmap[0] is the response to black-pixel stim, heatmap[1] is the response to white-pixel stim
                'rf' (np.array [width, height]): the rf of the neuron. Binary value
            res_info: dict, the res_info dict (neural response profile) from center_neuron_analyzer.export_neuron_info(). It has module name as keys indicating the information about each module. Kyes under each module are the same, which are:
                'neuron_id' (list [3]): the id of the neuron
                'trial_id' (int): the id of the trial
                'response' (np.array [n_time]): the response of the neuron in a trial
                'dark_grey' (int): the grey value of the dark pixel
                'light_grey' (int): the grey value of the light pixel
                'orientaion (float): the orientation of the stimulus
                'beta' (float): the beta of the stimulus
                'gamma' (float): the gamma of the stimulus
                'shift' (int): shift value of the square
                'size' (int): size value of the square
            stim_info: dict, the stim_info dict. Including the stim image of different trials
                'trial_id' (int): the id of the trial
                'image' (np.array [width, height]): the stim image of different trials
                'dark_grey' (int): the grey value of the dark pixel
                'light_grey' (int): the grey value of the light pixel
                'orientaion (float): the orientation of the stimulus
                'beta' (float): the beta of the stimulus
                'gamma' (float): the gamma of the stimulus
                'shift' (int): shift value of the square
                'size' (int): size value of the square
            module: str, the name of the module. The input info actually include all modules, but usually we seperately process each module, so this would be the module you want to study. You can change the module by directly setting self.module = 'R0' for example.
            boi_dire_value (dict): this is the orientation corresponding to the boi list. This value should come from the unique_orientation in center_neuron_analyzer.export_neuron_info(). The keys are modules, the value are the directions. For example, {'R0': [0, 45, 90, 135], 'R1': [0, 45, 90, 135]}.
        '''
        self.bo_info = bo_info
        self.res_info = res_info
        self.stim_info = stim_info
        self.boi_ori_value = boi_ori_value

        self.bo_infom = bo_info[module]
        self.res_infom = res_info[module]
        self.stim_infom = stim_info
        self.boi_orim = boi_ori_value[module]

    def get_target_neuron_info(self, neuron_rank=0, rank_method='BOI', neuron_id=None):
        '''
        get the target neuron from the info dict. corrently only support by neuron rank
        The keys are the same as self.bo_info and res_info, but only contains the target neuron
        '''
        if rank_method == 'BOI':
            neuron_bo_info = self.bo_infom[self.bo_infom['boi_abs_rank'] == neuron_rank] # rank value starts from 1
        elif rank_method == 'bo_only':
            neuron_bo_info = self.bo_infom[self.bo_infom['bo_only_rank'] == neuron_rank]

        neuron_id = neuron_bo_info['neuron_id'].values[0]
        neuron_res_info = self.res_infom[self.res_infom['neuron_id'].apply(lambda x: np.array_equal(x, neuron_id))]
        return neuron_bo_info, neuron_res_info

def get_preferred_orientation(neuron_bo_info):
    pref_dire = neuron_bo_info['boi_pref_dire'].to_numpy()[0]
    return pref_dire - 180 if pref_dire >= 180 else pref_dire

def get_response_to_orientation(neuron_res_info, pref_ori):
    response_to_ori = neuron_res_info[neuron_res_info['orientation'] == pref_ori]
    return response_to_ori['trial_id'].to_numpy(), response_to_ori['response'].to_numpy()

def get_stimulus_and_rf(stim_info, neuron_bo_info, target_ori=0, rf_smooth_sigma=1):
    eg_img = stim_info[stim_info['orientation'] == target_ori]['image'].to_numpy()

    rflsn = RF_Finder_Local_Sparse_Noise()
    heatmap = neuron_bo_info.iloc[0]['heatmap']
    zmap = rflsn._heatmap_to_zmap(heatmap, smooth_sigma=rf_smooth_sigma, merge_black_white=True)
    return eg_img, zmap

def get_boi_details(neuron_bo_info):
    boi_abs_max = neuron_bo_info['boi_abs_max'].to_numpy()[0]
    boi = neuron_bo_info['boi'].to_numpy()[0]
    return boi_abs_max, boi

def init_a_neuron(nip, neural_rank, rf_smooth_sigma=1):
    # Main initialization for a neuron
    if neural_rank <= 0:
        neural_rank = nip.bo_infom.shape[0] + neural_rank
    neuron_bo_info, neuron_res_info = nip.get_target_neuron_info(neural_rank)
    pref_ori = get_preferred_orientation(neuron_bo_info)
    trial_id, response = get_response_to_orientation(neuron_res_info, pref_ori)
    eg_img, zmap = get_stimulus_and_rf(nip.stim_info, neuron_bo_info, rf_smooth_sigma=rf_smooth_sigma)
    eg_img = eg_img[0]
    # get the max BOI
    boi_abs_max = neuron_bo_info['boi_abs_max'].to_numpy()[0]
    # get BOI at different orientation
    boi = neuron_bo_info['boi'].to_numpy()[0]
    return neuron_bo_info, neuron_res_info, pref_ori, trial_id, response, eg_img, zmap, boi_abs_max, boi

def draw_one_neuron(nip, neural_rank, ax_rfi, ax_resi, ax_orii):
    neuron_bo_info, neuron_res_info, pref_ori, trial_id, response, eg_img, zmap, boi_abs_max, boi = init_a_neuron(nip, neural_rank)

    ax_rfi.imshow(eg_img)
    ax_rfi.contour(zmap, levels=[1], colors='white', linestyles='solid')
    ax_rfi.axis('off')

    time = np.arange(response[0].shape[0])
    for _, res in enumerate(response):
        ax_resi.plot(time, res)
    ax_resi.set_xlabel('Time \n (BOI: {})'.format(sci_notation(boi_abs_max)))
    ax_resi.set_ylabel('Unit activation')

    ploter.plot_polar_boi(boi, nip.boi_orim, ax=ax_orii)
    for spine in ax_orii.spines.values():
        spine.set_linewidth(1)
    return ax_rfi, ax_resi, ax_orii
