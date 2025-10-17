# run bo_non_bo_avg_fr_plot.py to get subsampled_to_adjust_median_unit_id.hkl
# also get center_neuron_info.hkl from export_center_neuron_info.py
import hickle as hkl
import os
import numpy as np
from kitti_settings import *

# load subsampled id
subsampled_unit_id_path = os.path.join(DATA_DIR_HOME, 'subsampled_to_adjust_median_unit_id.hkl')
subsampled_unit_id = hkl.load(subsampled_unit_id_path)

# load original bo_info
center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')
bo_info = hkl.load(center_info_path)['bo_info']

video_head_list = ['translating', 'random', 'kitti']
module_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']

for video_head in video_head_list:
    bo_info_subsampled = {}
    for module in module_list:
        bom = bo_info[module]
        bo_unit_id_m = subsampled_unit_id[video_head]['bo'][module]
        nbo_unit_id_m = subsampled_unit_id[video_head]['nbo'][module]
        merged_unit_id_m = np.concatenate((bo_unit_id_m, nbo_unit_id_m))
        tuple_unit_id_m = [tuple(x) for x in merged_unit_id_m]

        filtered_bom = bom[bom['neuron_id'].apply(lambda x: x in tuple_unit_id_m)]
        bo_info_subsampled[module] = filtered_bom
    hkl.dump(bo_info_subsampled, os.path.join(DATA_DIR_HOME, f'{video_head}_bo_info_subsampled.hkl'))
