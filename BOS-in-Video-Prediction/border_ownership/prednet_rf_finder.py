import numpy as np
import hickle as hkl
from border_ownership.rf_finder import RF_Finder_2D, RF_Finder_Local_Sparse_Noise, RF_Finder_Local_Sparse_Noise_Small_Memory_Center

# architecture of the prednet
PREDNET_LAYER_ORDER = ['conv0', 'pool0', 'conv1', 'pool1', 'conv2', 'pool2', 'conv3', 'pool3']
PREDNET_LAYER_PARA = {
    'conv0': {'stride': (1, 1),
              'kernel_size': (3, 3),
              'padding': ('same', 'same'),
              'dilation': (1, 1)},
    'pool0': {'stride': (None, None),
              'kernel_size': (2, 2),
              'padding': ('valide', 'valide'),
              'dilation': (1, 1)},
    'conv1': {'stride': (1, 1),
              'kernel_size': (3, 3),
              'padding': ('same', 'same'),
              'dilation': (1, 1)},
    'pool1': {'stride': (None, None),
              'kernel_size': (2, 2),
              'padding': ('valide', 'valide'),
              'dilation': (1, 1)},
    'conv2': {'stride': (1, 1),
              'kernel_size': (3, 3),
              'padding': ('same', 'same'),
              'dilation': (1, 1)},
    'pool2': {'stride': (None, None),
              'kernel_size': (2, 2),
              'padding': ('valide', 'valide'),
              'dilation': (1, 1)},
}

PREDNET_MODULE_STACKS = { # the conv and pool layer orders before each E module
    'E0': [],
    'E1': ['conv0', 'pool0'],
    'E2': ['conv0', 'pool0', 'conv1', 'pool1'],
    'E3': ['conv0', 'pool0', 'conv1', 'pool1', 'conv2', 'pool2'],
}

PREDNET_MODULE_SHAPE = { # the width, height, and chs of each module
    'E0': [128, 160, 6], 'E1': [64, 80, 96], 'E2': [32, 40, 192], 'E3': [16, 20, 384],
    'R0': [128, 160, 3], 'R1': [64, 80, 48], 'R2': [32, 40, 96], 'R3': [16, 20, 192],
    'A0': [128, 160, 3], 'A1': [64, 80, 48], 'A2': [32, 40, 96], 'A3': [16, 20, 192],
    'Ahat0': [128, 160, 3], 'Ahat1': [64, 80, 48], 'Ahat2': [32, 40, 96], 'Ahat3': [16, 20, 192],
}

def compute_rf_mask(module_name, query_neural_id='center', input_shape=(128, 160)):
    '''
    input:
      module_name: string. Can only be A0, A1, ... or E0, E1, ...
      query_neural_id: [[0, 1], [3, 4]] indicating the neuron's position in the input image. Each row is one neural, the final output is the union of all neurons' RFs.
    output:
      image: mask image. RF are indicated by 1, 0 otherwise
    '''
    if (isinstance(query_neural_id, str)) and (query_neural_id == 'center'):
        query_neural_id = [ width // 2 for width in PREDNET_MODULE_SHAPE[module_name][:-1] ]
        query_neural_id = np.array( [query_neural_id] )

    rff = RF_Finder_2D(input_shape, PREDNET_MODULE_STACKS[module_name], PREDNET_LAYER_PARA)
    rf = rff.search_rf_2d(query_neural_id)
    image = np.zeros(input_shape)
    image[tuple(rf.T)] = 1

    return image

def compute_center_rf_mask_exp(qid=0, center_heatmap_dir=None, heatmap_dir=None, generate_center_heatmap=False, generate_heatmap=False, z_thresh=1):
    '''
    compute the rf mask of center neurons.
    qid (int): center neuron id, basically is the channel id
    center_heatmap_dir (str): the dir keeps center neuron's heatmap
    generate_center_heatmap (bool): whether generating center heatmap
    heatmap_dir (str): the dir keeps all neuron's heatmap, if generate_center_heatmap is True, this para must be provided.
    generate_heatmap (bool): whether generating heatmap
    z_thresh (float): z threshold treated as the RF
    '''
    rff = RF_Finder_Local_Sparse_Noise_Small_Memory_Center(heatmap_dir, z_thresh=z_thresh)
    if generate_heatmap: rff.search_rf_all_neuron()
    if generate_center_heatmap: rff.keep_center_heatmap(center_heatmap_dir)
    rff.load_center_heatmap(center_heatmap_dir)

    rf = rff.query_rf(qid)
    return rf
