# please run pred_response_to_stim.py first to obtain the response to different orientation, also run prednet_rf_processing.py to obtain the center neuron dict
from border_ownership.center_neuron_analyzer import Center_RF_Neuron_Analyzer, save_center_neuron_only
import numpy as np
import matplotlib.pyplot as plt
from kitti_settings import *

# #################### Hyperparameters ####################
center_neuron_rf_path = os.path.join(DATA_DIR_HOME, 'center_neuron_dict.hkl')

res_ori_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_ori.npz')
res_shift_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_shift.npz')
res_size_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_size.npz')
res_square_part_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_square_part.npz')
res_switch_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_switch.npz')
res_switch_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_switch_grey.npz')
res_oscillate_square_ambiguous_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_oscillate_square_ambiguous.npz')
res_switch_square_flip_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_switch_square_flip.npz')
res_switch_grey_to_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_switch_grey_to_grey.npz')
res_switch_ambiguous_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_switch_ambiguous_grey.npz')
res_switch_grating_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_switch_grating_grey.npz')
res_switch_pixel_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_switch_pixel_grey.npz')

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
#################### Main ####################
# data = save_center_neuron_only(center_res_ori_path, res_ori_path, center_neuron_rf_path)
# data = save_center_neuron_only(center_res_shift_path, res_shift_path, center_neuron_rf_path)
# data = save_center_neuron_only(center_res_size_path, res_size_path, center_neuron_rf_path)
# data = save_center_neuron_only(center_res_square_part_path, res_square_part_path, center_neuron_rf_path)
# data = save_center_neuron_only(center_res_switch_path, res_switch_path, center_neuron_rf_path)
# data = save_center_neuron_only(center_res_switch_grey_path, res_switch_grey_path, center_neuron_rf_path)
# data = save_center_neuron_only(center_res_oscillate_square_ambiguous_path, res_oscillate_square_ambiguous_path, center_neuron_rf_path)
# data = save_center_neuron_only(center_res_switch_square_flip_path, res_switch_square_flip_path, center_neuron_rf_path)
# data = save_center_neuron_only(center_res_switch_grey_to_grey_path, res_switch_grey_to_grey_path, center_neuron_rf_path)
# data = save_center_neuron_only(center_res_switch_ambiguous_grey_path, res_switch_ambiguous_grey_path, center_neuron_rf_path)
# data = save_center_neuron_only(center_res_switch_grating_grey_path, res_switch_grating_grey_path, center_neuron_rf_path)
# data = save_center_neuron_only(center_res_switch_pixel_grey_path, res_switch_pixel_grey_path, center_neuron_rf_path)
