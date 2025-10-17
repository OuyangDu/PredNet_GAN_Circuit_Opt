from keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from kitti_settings import *
from border_ownership.rf_finder import RF_Finder_Local_Sparse_Noise, RF_Finder_2D

input_shape = (32, 40, 3)
z_thresh = 2
query_neural_id = (2, 1, 2) # width id, heighth id and channel id. Note sometimes due to bad weight initialization this neuron may not fire at all (no rf). Change the neural id or rerun the code (for new random initialization) will solve this probelm.

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

rff = RF_Finder_Local_Sparse_Noise(model, input_shape, z_thresh)
rff.search_rf_all_neuron()

print(model.summary())

## test save
#import hickle as hkl
#output_mode = '0'
#heatmap_dir = os.path.join(DATA_DIR_HOME, 'cnn_rf')
#heatmap_file_name = 'cnn_rf_heatmap_' + output_mode + '.hkl'
#if not os.path.exists(heatmap_dir): os.makedirs(heatmap_dir)
#heatmap_save_path = os.path.join(heatmap_dir, heatmap_file_name)
#hkl.dump(rff.heatmap_all, heatmap_save_path)
#
## test load
#heatmap_all = hkl.load(heatmap_save_path)
#rff.load_heatmap(heatmap_all)
hm = rff.query_heatmap(query_neural_id)[0]
rf = rff.query_rf(query_neural_id)

plt.figure()
plt.imshow(hm)
plt.title('cnn: heatmap from experimental method')

plt.figure()
plt.imshow(rf)
plt.title('cnn: rf from experimental method')

layer_order = ['conv0', 'pool0', 'conv1']
layer_para = {
    'conv0': {'stride': (1, 1),
              'kernel_size': (3, 3),
              'padding': ('valide', 'valide'),
              'dilation': (1, 1)},
    'pool0': {'stride': (None, None),
              'kernel_size': (2, 2),
              'padding': ('valide', 'valide'),
              'dilation': (1, 1)},
    'conv1': {'stride': (1, 1),
              'kernel_size': (3, 3),
              'padding': ('valide', 'valide'),
              'dilation': (1, 1)},
}

input_width = (input_shape[0], input_shape[1])
query_neural_id = [
    [query_neural_id[0], query_neural_id[1]]
]
query_neural_id = np.array(query_neural_id)

rff = RF_Finder_2D(input_width, layer_order, layer_para)
rf = rff.search_rf_2d(query_neural_id)
image = np.zeros(input_width)
image[tuple(rf.T)] = 1

plt.figure()
plt.imshow(image)
plt.title('cnn: rf from theoretical method')
plt.show()
