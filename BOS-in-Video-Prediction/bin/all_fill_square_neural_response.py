import numpy as np
import os
from border_ownership.full_square_response import generate_square_and_neural_res, generate_square_and_neural_res_multiple_modes
from kitti_settings import *

n_time = 20
batch_size = 128
#para = {'dark_grey': [255//3], 'light_grey': [255 * 2 // 3], 'orientation': [60], 'beta': [False, True], 'gamma': [False, True], 'shift': [0], 'size': [50], 'image_width': [160], 'image_height': [128]}
#para = {'dark_grey': [255//3], 'light_grey': [255 * 2 // 3], 'orientation': np.linspace(0, 180, 10, endpoint=False), 'beta': [False, True], 'gamma': [False, True], 'shift': np.arange(-25, 25, 5), 'size': np.arange(10, 100, 20)}
para = {'dark_grey': [255//3], 'light_grey': [255 * 2 // 3], 'orientation': np.linspace(0, 180, 10, endpoint=False), 'beta': [False, True], 'gamma': [False, True], 'shift': [0], 'size': [50]}
prednet_json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
prednet_weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + 'prednet_kitti_weights.hdf5')
output_mode_list = ['E2', 'R1']

data_dir = os.path.join(DATA_DIR_HOME, 'full_response/')
if not os.path.exists(data_dir): os.makedirs(data_dir)

df = generate_square_and_neural_res_multiple_modes(n_time, batch_size, para, prednet_json_file, prednet_weights_file, output_mode_list)
path = os.path.join(data_dir, 'all_ori_neural_response.pkl')
df.to_pickle(path)
