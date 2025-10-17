'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import hickle as hkl

from border_ownership.prednet_performance import PredNet_Evaluator, compute_pred_mse
from border_ownership.rf_finder import out_of_range
from border_ownership.bo_indices_masker import BO_Masker
from border_ownership.ablation import Ablation_Evaluator
from kitti_settings import *

test_file = 'square_bo_video_ori_x.hkl'
test_sources = 'square_bo_video_ori_sources.hkl'
data_dir = DATA_DIR_HOME

# data_dir = None
# test_file = 'new_X_test.hkl'
# test_sources = 'new_sources_test.hkl'

evaluator = Ablation_Evaluator(pixel_mask_distance=None)
n_bo_units = evaluator.bom.n_bo_units; n_non_bo_units = evaluator.bom.n_non_bo_units
min_units = {key: min(n_bo_units[key], n_non_bo_units[key]) for key in n_bo_units.keys()}
print('total number of bo units: {}; non bo units: {}'.format(n_bo_units, n_non_bo_units))

data = {}
n_boot = 5
module_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
for module in module_list:
    if module in ['E1', 'E2']: step_size = 20
    else: step_size = 5
    for is_bo in [True, False]:
        for n in range(1, min_units[module] + step_size, step_size):
            if n > min_units[module]: n = min_units[module] # include the end point
            num_units = {module: n}
            sample_method = 'random'
            # if is_bo: sample_method = 'rank'
            # else: sample_method = 'random'
            # num_units = {}
            evaluator.ablate_unit(num_units, is_bo=is_bo, sample_method=sample_method)
            mse_models = evaluator.compute_performance_boot(num_runs=n_boot, data_dir=data_dir, test_file=test_file, test_sources=test_sources)
            data[f'{is_bo}_{module}_{n}'] = mse_models
            print(f'{is_bo}_{module}_{n}: ', mse_models)

path = RESULTS_SAVE_DIR
if not os.path.exists(path): os.makedirs(path)
hkl.dump(data, os.path.join(RESULTS_SAVE_DIR, 'ablation_data_square.hkl'))
