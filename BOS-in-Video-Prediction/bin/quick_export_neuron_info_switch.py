# draw example neural response
from border_ownership.center_neuron_analyzer import Center_RF_Neuron_Analyzer
import hickle as hkl
import os
from kitti_settings import *

# #################### Hyperparameters ####################
center_neuron_rf_path = os.path.join(DATA_DIR_HOME, 'center_neuron_dict.hkl')
center_res_ori_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_ori.npz')
center_res_shift_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_shift.npz')
center_res_size_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_size.npz')
center_res_square_part_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_square_part.npz')
center_res_switch_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_switch.npz')
center_res_switch_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_switch_grey.npz')
center_res_oscillate_square_ambiguous_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_oscillate_square_ambiguous.npz')
center_res_switch_square_flip_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_switch_square_flip.npz')
center_res_switch_grey_to_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_switch_grey_to_grey.npz')
center_res_switch_ambiguous_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_switch_ambiguous_grey.npz')
center_res_switch_grating_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_switch_grating_grey.npz')
center_res_switch_pixel_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_center_res_switch_pixel_grey.npz')
# #################### Main ####################
cn_analyzer = Center_RF_Neuron_Analyzer()
cn_analyzer.load_data(center_res_ori_path, center_neuron_rf_path)
bo_info, res_info, stim_info = cn_analyzer.export_neuron_info()
res_shift_info, stim_shift_info = cn_analyzer.combine_res_stim_data(center_res_shift_path)
# print(res_shift_info.keys())
# print(res_shift_info['R0'].shape)
# print(res_shift_info['R0'].columns)
# exit()
res_size_info, stim_size_info = cn_analyzer.combine_res_stim_data(center_res_size_path)
res_square_part_info, stim_square_part_info = cn_analyzer.combine_res_stim_data(center_res_square_part_path)
res_switch_info, stim_switch_info = cn_analyzer.combine_res_stim_data(center_res_switch_path)
res_switch_grey_info, stim_switch_grey_info = cn_analyzer.combine_res_stim_data(center_res_switch_grey_path)
res_oscillate_square_ambiguous_info, stim_oscillate_square_ambiguous_info = cn_analyzer.combine_res_stim_data(center_res_oscillate_square_ambiguous_path)
res_switch_square_flip_info, stim_switch_square_flip_info = cn_analyzer.combine_res_stim_data(center_res_switch_square_flip_path)
res_switch_grey_to_grey_info, stim_switch_grey_to_grey_info = cn_analyzer.combine_res_stim_data(center_res_switch_grey_to_grey_path)
res_switch_ambiguous_grey_info, stim_switch_ambiguous_grey_info = cn_analyzer.combine_res_stim_data(center_res_switch_ambiguous_grey_path)
res_switch_grating_grey_info, stim_switch_grating_grey_info = cn_analyzer.combine_res_stim_data(center_res_switch_grating_grey_path)
res_switch_pixel_grey_info, stim_switch_pixel_grey_info = cn_analyzer.combine_res_stim_data(center_res_switch_pixel_grey_path)

data = {
    'bo_info': bo_info,
    'res_info': res_info,
    'stim_info': stim_info,
    'unique_orientation': cn_analyzer.unique_orientation,
    'res_switch_info': res_switch_info,
    'stim_switch_info': stim_switch_info,
    'res_switch_grey_info': res_switch_grey_info,
    'stim_switch_grey_info': stim_switch_grey_info,
    'res_oscillate_square_ambiguous_info': res_oscillate_square_ambiguous_info,
    'stim_oscillate_square_ambiguous_info': stim_oscillate_square_ambiguous_info,
    'res_switch_square_flip_info': res_switch_square_flip_info,
    'stim_switch_square_flip_info': stim_switch_square_flip_info,
    'res_switch_grey_to_grey_info': res_switch_grey_to_grey_info,
    'stim_switch_grey_to_grey_info': stim_switch_grey_to_grey_info,
    'res_switch_ambiguous_grey_info': res_switch_ambiguous_grey_info,
    'stim_switch_ambiguous_grey_info': stim_switch_ambiguous_grey_info,
    'res_switch_grating_grey_info': res_switch_grating_grey_info,
    'stim_switch_grating_grey_info': stim_switch_grating_grey_info,
    'res_switch_pixel_grey_info': res_switch_pixel_grey_info,
    'stim_switch_pixel_grey_info': stim_switch_pixel_grey_info,
}

hkl.dump(data, os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')) # dumping can take a few minutes

# data = hkl.load(os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl'))
# print(data.keys())
