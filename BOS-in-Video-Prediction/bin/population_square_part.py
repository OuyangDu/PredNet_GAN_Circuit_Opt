from border_ownership.square_part import Square_Part_Analyzer
import matplotlib.pyplot as plt
from border_ownership.ploter import error_bar_plot
from border_ownership.rf_finder import out_of_range
import numpy as np
import pandas as pd
import os
import hickle as hkl
from kitti_settings import *

#################### Hyperparameters ####################
module = 'E1'
outmost_distance = 20

center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')
res_square_part_path = os.path.join(DATA_DIR_HOME, 'res_square_part.hkl')
res_name = ['CE', 'NC', 'NE', 'FC', 'FE', 'all']

########## Population analysis ##########
data = hkl.load(center_info_path)

n_bo = np.sum(data['bo_info'][module]['is_bo'])

# select small rf
eg_rf = data['bo_info'][module]['rf'].iloc[0]
mask = out_of_range(eg_rf.shape[0], eg_rf.shape[1], outmost_distance)
small_rf = data['bo_info'][module]['rf'].apply(lambda x: np.all(x[mask] == 0))
data['bo_info'][module] = data['bo_info'][module][small_rf]

small_rf_id = data['bo_info'][module]['neuron_id']
data['res_square_part_info'][module] = data['res_square_part_info'][module][ data['res_square_part_info'][module]['neuron_id'].isin(small_rf_id) ]

neural_rank_list = data['bo_info'][module]['bo_only_rank'].dropna().unique()

res_all_neuron = []
for i, neural_rank in enumerate(neural_rank_list):
    print('processing neuron: {}/{} \t neural_rank: {}'.format(i, len(neural_rank_list), neural_rank))
    spa = Square_Part_Analyzer(module, neural_rank, rank_method='bo_only')
    spa.load_data(data)
    res_temp = spa.get_res_change(mode='all_by_name', is_zscore=False)
    res_all_neuron.append(res_temp)

res_all_neuron = np.array(res_all_neuron)

hkl.dump(res_all_neuron, res_square_part_path)

mean_res = np.mean(res_all_neuron, axis=0)
print(mean_res)

########## Load and draw
data = hkl.load(center_info_path)
data_square_part = hkl.load(res_square_part_path)

x = np.arange(data_square_part.shape[-1])
fig, ax = plt.subplots(figsize=(3, 3))
color_list = ['r', 'g', 'b', 'c', 'm', 'k']

data_square_part_no_nan = []
for i in range(data_square_part.shape[0]):
    one_neuron = data_square_part[i, :, :]
    has_nan = np.isnan(one_neuron).any()

    if not has_nan:
        data_square_part_no_nan.append(one_neuron)

data_square_part_no_nan = np.array(data_square_part_no_nan)

for i in range(4):
    one_condition = data_square_part_no_nan[:, i, :].T
    error_bar_plot(x, one_condition, fig=fig, ax=ax, label=res_name[i], error_mode='se', color=color_list[i])
plt.show()

