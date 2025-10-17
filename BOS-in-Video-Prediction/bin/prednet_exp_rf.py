import os
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl

from border_ownership.rf_finder import RF_Finder_Local_Sparse_Noise
from border_ownership.rf_finder import RF_Finder_Local_Sparse_Noise_Small_Memory
from border_ownership.ploter import plot_seq_prediction
from border_ownership.agent import Agent, Agent_RF_Wraper

from kitti_settings import *

json_file = 'prednet_kitti_model.json'
json_file = os.path.join(WEIGHTS_DIR, json_file)
weights_file = 'prednet_kitti_weights.hdf5'
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + weights_file)

z_thresh = 1
num_frames = 4
input_shape = (128, 160, 3)
meta_batch_size = 32
output_mode = 'E2'
query_neural_id = (0, 0, 0)
heatmap_dir = os.path.join(DATA_DIR_HOME, 'heatmap_{}'.format(output_mode))

sub = Agent()
sub.read_from_json(json_file, weights_file)
sub_warp = Agent_RF_Wraper(sub, num_frames, output_mode, meta_batch_size=meta_batch_size)

#rff = RF_Finder_Local_Sparse_Noise(sub_warp, input_shape, z_thresh=z_thresh) # this is for large memory (more than 20 GB)
rff = RF_Finder_Local_Sparse_Noise_Small_Memory(heatmap_dir, sub_warp, input_shape, z_thresh=z_thresh)
rff.search_rf_all_neuron()
#rff.load_heatmap_all()
heatmap = rff.query_heatmap(query_neural_id)
plt.figure()
plt.title('White Heatmap')
plt.imshow(heatmap[0])
plt.colorbar()
plt.show()

plt.figure()
plt.title('Black Heatmap')
plt.imshow(heatmap[1])
plt.colorbar()
plt.show()

plt.figure()
plt.title('RF')
rf = rff.query_rf(query_neural_id)
plt.imshow(rf)
plt.show()

## even the heatmap can be larger than 10GB
#heatmap_dir = os.path.join(DATA_DIR_HOME, 'prednet_rf')
#heatmap_file_name = 'prednet_rf_heatmap_' + output_mode + '.hkl'
#if not os.path.exists(heatmap_dir): os.makedirs(heatmap_dir)
#heatmap_save_path = os.path.join(heatmap_dir, heatmap_file_name)
#hkl.dump(rff.heatmap_all, heatmap_save_path)
