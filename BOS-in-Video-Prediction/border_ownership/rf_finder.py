import numpy as np
import scipy.stats as stats
import os
import hickle as hkl
from scipy.ndimage import gaussian_filter
from kitti_settings import *

def output_layer_width(input_width, kernel_size, padding, stride, dilation):
    width = (input_width - dilation * (kernel_size - 1) + 2 * padding - 1) / stride + 1 # check pytorch conv
    return int(width)

def out_of_range(width, height, distance):
    '''
    return a mask of the same size as the image, with True for pixels out of range
    input:
        width, height: the size of the image
        distance: the distance from the center to the outmost pixels
    output:
        mask: 2d array. True for pixels out of range
    '''
    yc, xc = np.indices((width, height)) - np.array([width/2., height/2.])[:, None, None]
    mask = np.hypot(xc, yc) > distance
    return mask

class RF_Finder_1D_1Layer():
    def __init__(self, input_width, kernel_size=1, padding=0, stride=1, dilation=1):
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.input_width = input_width
        self.kernel_size = kernel_size

    def prepare_rf_idx_check(self):
        self.output_idx_max = output_layer_width(self.input_width, self.kernel_size, self.padding, self.stride, self.dilation)
        return self.output_idx_max

    def rf_idx_checker(self, idx_list):
        # check wether i live within output_idx_max
        for idx in idx_list:
            if (idx < 0) or (idx >= self.output_idx_max):
                raise ValueError('query neural index is out of range')

    def remove_padding_idx(self, rf):
        '''remove index of rf outsides the input range due to padding'''
        correct_idx = (rf >= 0) & (rf < self.input_width)
        return rf[correct_idx]

    def move_back(self, idx_list):
        '''
        conv and pool layer have the same rf
        idx_list (int array, [n_query]): indices indicate neural positions in the same layer
        '''
        # assume only one channel
        self.prepare_rf_idx_check()
        self.rf_idx_checker(idx_list)

        rf_list = [] # get rf of all query neurons
        for idx in idx_list:
            rf = [idx * self.stride + self.dilation * k - self.padding for k in range(self.kernel_size)]
            rf_list.append(np.array(rf))

        rf = np.concatenate(rf_list)
        rf = np.unique(rf) # merge their rf

        rf = self.remove_padding_idx(rf)

        return rf

# Example layer order and layer_para can be see in test_rf_finder
class RF_Finder_1D():
    def __init__(self, input_width, layer_order, layer_para):
        '''
        input_width (int)
        layer_order: list of layer name. Each name must be different
        layer_para: a dict. layer names are the key. Each item is also a dict contains parameter values. Parameters including kernel_size, padding, stride, dilation
        '''
        self.input_width = input_width
        self.layer_order = layer_order.copy()
        self.layer_para = layer_para.copy()

    def _convert_stride_2_integer(self, one_layer_para):
        stride = one_layer_para['stride']
        if stride is None: one_layer_para['stride'] = one_layer_para['kernel_size']
        return stride

    def _convert_padding_2_integer(self, one_layer_para):
        padding = one_layer_para['padding']
        stride = one_layer_para['stride']
        dilation = one_layer_para['dilation']
        kernel_size = one_layer_para['kernel_size']

        if padding == 'valide':
            one_layer_para['padding'] = 0
        elif padding == 'same':
            '''if padding is same, output size is (input_size - 1)/stride + 1, hence we can obtain the desired padding size should be the following. see pool2d in keras for more information.'''
            one_layer_para['padding'] = ( (self.input_width - 1) * (stride - 1) + dilation * (kernel_size - 1) ) // 2
        return padding

    def _convert_all_para_2_integer(self):
        for layer_name in self.layer_order:
            self._convert_padding_2_integer(self.layer_para[layer_name])
            self._convert_stride_2_integer(self.layer_para[layer_name])

    def compute_layer_width(self):
        self.layer_width = [self.input_width]
        for i, layer_name in enumerate(self.layer_order):
            out_width = output_layer_width(self.layer_width[i], **self.layer_para[layer_name])
            self.layer_width.append(out_width)
        return self.layer_width

    def search_rf_1d(self, neural_list):
        '''
        neural_list: list of int indicating the query neural positions
        '''
        rf = neural_list
        self._convert_all_para_2_integer()

        layer_width = self.compute_layer_width()

        layer_width_rev = layer_width[::-1]
        for i, layer_name in enumerate( self.layer_order[::-1] ):
            rff = RF_Finder_1D_1Layer(input_width=layer_width_rev[i + 1], **self.layer_para[layer_name])
            rf = rff.move_back(rf)
        return rf

class RF_Finder_2D():
    def __init__(self, input_width, layer_order, layer_para):
        '''
        DO NOT CHANGE PARAMETER AFTER INIT
        input_width (size 2 int array)
        layer_order: list of layer name. Each name must be different
        layer_para: a dict. layer names are the key. Each item is also a dict contains parameter values. Parameters including kernel_size, padding, stride, dilation
        '''
        self.input_width = input_width
        self.layer_order = layer_order.copy()
        self.layer_para = layer_para.copy()
        self.split_layer_para()

    def split_layer_para(self):
        layer_para_w, layer_para_h = {}, {}
        for layer_name in self.layer_order:
            layer_para_w[layer_name] = {}
            layer_para_h[layer_name] = {}

            for para in self.layer_para[layer_name]:
                layer_para_w[layer_name][para] = self.layer_para[layer_name][para][0]
                layer_para_h[layer_name][para] = self.layer_para[layer_name][para][1]
        self.layer_para_w = layer_para_w
        self.layer_para_h = layer_para_h
        return layer_para_w, layer_para_h

    def search_rf_2d(self, neural_list):
        '''
        neural_list ([n_query, 2]): first column is for width, second column is for height
        '''
        rff_w = RF_Finder_1D(self.input_width[0], self.layer_order, self.layer_para_w)
        rf_w = rff_w.search_rf_1d(neural_list[:, 0])

        rff_h = RF_Finder_1D(self.input_width[1], self.layer_order, self.layer_para_h)
        rf_h = rff_h.search_rf_1d(neural_list[:, 1])

        rf = [(w, h) for w in rf_w for h in rf_h]
        return np.array(rf)

class RF_Finder_Local_Sparse_Noise():
    def __init__(self, model=None, input_shape=None, z_thresh=1):
        '''
        find the rf. model should have predict method with input and output interface same as keras
        z_thresh must be a positive number
        '''
        self.model = model
        self.input_shape = input_shape
        self.z_thresh = z_thresh

    def _make_input_image_batch(self):
        '''
        output:
          input_image_batch ((2, self.input_shape[0], self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[2])): each image has one dot (black or white). Image id is specified by the first two index. For example, image_i_j is in the position of input_image_batch[i, j]. image_i_j is a grey image except that (i, j) element is black/white
        '''
        input_image_batch = []
        for i in range(self.input_shape[0]):
            image_batch_row = self._make_input_image_batch_by_row(i)
            input_image_batch.append(image_batch_row.copy())
        input_image_batch = np.concatenate(input_image_batch, axis=1)
        return input_image_batch

    def _make_input_image_batch_by_row(self, row_id):
        '''
        input:
          row_id (int): the row index where the dots stimuli will shown up, iterately along differernt columns.
          input_image_batch (ndarray shape [2, self.input_shape[0], self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[2]], OR, None): image id is specified by the first two index. For example, image_i_j is in the position of input_image_batch[i, j]. image_i_j is a grey image except that (i, j) element is black/white
        output:
          input_image_batch (shape same above): image batch with appropriate dots stimuli
        '''
        input_image_batch = np.ones( (2, 1, self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[2]) ) * 0.5 

        for j in range(self.input_shape[1]):
            input_image_batch[0, 0, j, row_id, j] = 0 # make i j element of image_i_j to be black
            input_image_batch[1, 0, j, row_id, j] = 1 # make i j element of image_i_j to be black
        return input_image_batch

    def search_rf_all_neuron(self, scan_by_row=True):
        '''
        get heatmaps for all output neurons
        scan_by_row (bool): generate input image batches by showing dots row by row. A bit slower but can save a lot of memory
        '''
        if scan_by_row:
            self.heatmap_all = []
            for i in range(self.input_shape[0]):
                print('Computing response to row', i)
                input_image_batch = self._make_input_image_batch_by_row(i)
                input_image_batch = input_image_batch.reshape( (-1, *self.input_shape) ) # fllaten
                output = self.model.predict(input_image_batch)

                heatmap_row = output.reshape( (2, 1, self.input_shape[1], *output.shape[1:]) ) # recover shape. Basically split n_images to (2, 1, input_shape[1]) which is the size of 2 * one raw
                self.heatmap_all.append(heatmap_row.copy())
            self.heatmap_all = np.concatenate(self.heatmap_all, axis=1)
        else:
            input_image_batch = self._make_input_image_batch()
            input_image_batch = input_image_batch.reshape( (-1, *self.input_shape) ) # fllaten
            output = self.model.predict(input_image_batch)

            self.heatmap_all = output.reshape( (2, self.input_shape[0], self.input_shape[1], *output.shape[1:]) ) # recover shape
        return self.heatmap_all.copy()

    def load_heatmap(self, heatmap_all):
        '''
        besides generate heat map by the current instance, this class also allows directly read heat map generated from other instances and then perform query.
        '''
        self.heatmap_all = heatmap_all

    def query_heatmap(self, query_neural_id):
        '''
        input:
          query_neural_id (list, shape [3]): x, y and chs position of a query neuron
        output:
          heatmap (ndarray shape [2, img_width, img_height]): 2 corresponding to light dot and dark dot stimuli. Values are that single neural responses to each of the position the dot stimulus shown.
        '''
        heatmap = self.heatmap_all[:, :, :, query_neural_id[0], query_neural_id[1], query_neural_id[2]]
        return heatmap

    def _heatmap_to_zmap(self, heatmap, smooth_sigma=None, merge_black_white=False):
        zmap = stats.zscore(heatmap, axis=None)
        if smooth_sigma is not None:
            zmap[0] = gaussian_filter(zmap[0], sigma=smooth_sigma)
            zmap[1] = gaussian_filter(zmap[1], sigma=smooth_sigma)

        if merge_black_white:
            zmap = np.abs(zmap)
            zmap = np.maximum(zmap[0], zmap[1])

        return zmap

    def query_zmap(self, query_neural_id, smooth_sigma=None, merge_black_white=False):
        '''
        query_neural_id (int): neural channel id
        smooth_sigma (float or None): smooth black and white zmap
        merge_black_white (bool): if false, output is [2, width, height] for black zmap and white zmap seperately. If true, two zmap will take the absolute value, the output zmap is the maximum of two. Hence the shape is [width, height]. This merged zmap can be used to find the merged rf contour by thresholding the z score.
        '''
        heatmap = self.query_heatmap(query_neural_id)
        zmap = self._heatmap_to_zmap(heatmap)

        return zmap

    def _compute_rf(self, heatmap):
        zscore = stats.zscore(heatmap, axis=None)
        zscore_map_black = np.logical_or(
            (zscore[0] > self.z_thresh), (zscore[0] < -self.z_thresh)
        )
        zscore_map_white = np.logical_or(
            (zscore[1] > self.z_thresh), (zscore[1] < -self.z_thresh)
        )

        rf = np.logical_or(zscore_map_black, zscore_map_white)
        return rf

    def _compute_rf_list(self, heatmap_list):
        rf_list = []
        for heatmap in heatmap_list:
            rf = self._compute_rf(heatmap)
            rf_list.append(rf.copy())
        return rf_list

    def query_rf(self, query_neural_id):
        '''
        must has heatmap first
        '''
        heatmap = self.query_heatmap(query_neural_id)
        rf = self._compute_rf(heatmap)
        return rf

class RF_Finder_Local_Sparse_Noise_Small_Memory(RF_Finder_Local_Sparse_Noise):
    '''
    This is a version for which the computer memory is so small cannot even hold the heatmap. This class will save data of each row into a directory, and load them online
    '''
    def __init__(self, data_dir=None, model=None, input_shape=[128, 160, 3], z_thresh=1):
        self.data_dir = data_dir
        self.heatmap_all = None
        if data_dir is not None:
            if not os.path.exists(data_dir): os.makedirs(data_dir)

        super().__init__(model, input_shape, z_thresh)

    def search_rf_all_neuron(self):
        for i in range(self.input_shape[0]):

            print('Computing response to row', i)
            input_image_batch = self._make_input_image_batch_by_row(i)
            input_image_batch = input_image_batch.reshape( (-1, *self.input_shape) ) # fllaten
            output = self.model.predict(input_image_batch)

            heatmap_row = output.reshape( (2, 1, self.input_shape[1], *output.shape[1:]) ) # recover shape. Basically split n_images to (2, 1, input_shape[1]) which is the size of 2 * one raw

            file_path = os.path.join(self.data_dir, 'heatmap_row_{}.hkl'.format(str(i)) )
            hkl.dump(heatmap_row, file_path)


    def query_heatmap(self, query_neural_id, use_heatmap_all=True):
        '''
        input:
          query_neural_id (list, shape [3]): x, y and chs position of a query neuron
          use_heatmap_all (bool): if true, use the heatmap_all attribute if it's avaiable. Otherwise, load heatmap from file
        output:
          heatmap (ndarray shape [2, img_width, img_height]): 2 corresponding to light dot and dark dot stimuli. Values are that single neural responses to each of the position the dot stimulus shown.
        '''
        query_neural_id_list = [query_neural_id]
        heatmap_list = self.query_heatmap_list(query_neural_id_list, use_heatmap_all)
        return heatmap_list[0]

    def query_heatmap_list(self, query_neural_id_list, use_heatmap_all=True):
        '''
        input:
            query_neural_id_list (list, shape [n, 3]): x, y and chs position of a query neuron
          use_heatmap_all (bool): if true, use the heatmap_all attribute if it's avaiable. Otherwise, load heatmap from file
        output:
            heatmap_list (list of ndarray shape [2, img_width, img_height]): 2 corresponding to light dot and dark dot stimuli. Values are that single neural responses to each of the position the dot stimulus shown.
        '''
        num_query_neural_ids = len(query_neural_id_list)
        heatmap_mat = np.zeros((num_query_neural_ids, 2, self.input_shape[0], self.input_shape[1]))

        if (self.heatmap_all is not None) and use_heatmap_all:
            for query_neural_id in range(num_query_neural_ids):
                heatmap_mat[query_neural_id] = self.heatmap_all[:, :, :, query_neural_ids[query_neural_id][0], query_neural_ids[query_neural_id][1], query_neural_ids[query_neural_id][2]]
        else:
            for i in range(self.input_shape[0]):
                file_path = os.path.join(self.data_dir, 'heatmap_row_{}.hkl'.format(str(i)) )
                heatmap_row = hkl.load(file_path)
                for idx, qni in enumerate(query_neural_id_list):
                    heatmap_mat[idx, :, i, :] = heatmap_row[:, 0, :, qni[0], qni[1], qni[2]]
        heatmap_list = [heatmap_mat[i] for i in range(num_query_neural_ids)]
        return heatmap_list

    def load_heatmap_all(self):
        self.heatmap_all = []
        for i in range(self.input_shape[0]):
            file_path = os.path.join(self.data_dir, 'heatmap_row_{}.hkl'.format(str(i)) )
            heatmap_row = hkl.load(file_path)
            self.heatmap_all.append(heatmap_row.copy())
        self.heatmap_all = np.concatenate(self.heatmap_all, axis=1)

    def get_neural_id_for_null_chs(self, neuron_width, neuron_height, outmost_distance=20, img_width=None, img_height=None):
        if img_width is None: img_width = self.input_shape[0]
        if img_height is None: img_height = self.input_shape[1]

        i_dis_img_ratio = outmost_distance / img_width
        j_dis_img_ratio = outmost_distance / img_height
        i_range = [int(neuron_width * (0.5 - i_dis_img_ratio)), int(neuron_width * (0.5 + i_dis_img_ratio))]
        j_range = [int(neuron_height * (0.5 - j_dis_img_ratio)), int(neuron_height * (0.5 + j_dis_img_ratio))]
        id_list_a_chs_null = []
        for i, j in np.ndindex(neuron_width, neuron_height):
            if i < i_range[1] and i > i_range[0] and j < j_range[1] and j > j_range[0]:
                id_list_a_chs_null.append([i, j, 0])
        id_list_a_chs_null = np.array(id_list_a_chs_null, dtype=np.int32)
        return id_list_a_chs_null

    def obtain_central_RF_neuron(self, outmost_distance=20, verbose=True):
        '''
        find neurons RF that contains image center, and not too large
        input:
            ring_radius (int): the radius of the ring that is not allowed to be in the RF. Unit is pixel
        output:
            neuron_id_list (list): list of neuron id that satisfy the condition
            heatmap_list (list): list of heatmap of the corresponding neuron
            rf_list (list): list of rf of the corresponding neuron
        '''
        ## load one example row to get the shape of the neuron
        file_path = os.path.join(self.data_dir, 'heatmap_row_{}.hkl'.format(str(0)) )
        heatmap_row = hkl.load(file_path)

        neuron_width = heatmap_row.shape[3]
        neuron_height = heatmap_row.shape[4]
        neuron_chs = heatmap_row.shape[-1]

        img_center = self.input_shape[0] // 2, self.input_shape[1] // 2

        id_list_a_chs = self.get_neural_id_for_null_chs(neuron_width, neuron_height, outmost_distance)

        neuron_id_list, heatmap_list, rf_list = [], [], []
        mask = out_of_range(self.input_shape[0], self.input_shape[1], outmost_distance)
        # for k in range(neuron_chs): #TEMP
        for k in range(neuron_chs):
            print('processing chs {}/{}'.format(k, neuron_chs-1))
            id_list_a_chs[:, 2] = k

            heatmap_list_chs = self.query_heatmap_list(id_list_a_chs)
            rf_list_chs = self._compute_rf_list(heatmap_list_chs)

            for rf_idx, rf in enumerate(rf_list_chs):
                if (rf[img_center[0], img_center[1]] == 1) & np.all(rf[mask] == 0):
                    neuron_id_list.append(tuple(id_list_a_chs[rf_idx]))
                    heatmap_list.append(heatmap_list_chs[rf_idx].copy())
                    rf_list.append(rf.copy())
        return neuron_id_list, heatmap_list, rf_list

class RF_Finder_Local_Sparse_Noise_Small_Memory_Center(RF_Finder_Local_Sparse_Noise_Small_Memory):
    '''
    only keep neurons with cRF at image center.
    '''
    def load_center_heatmap(self, center_data_dir):
        file_path = os.path.join(center_data_dir, 'heatmap_center.hkl')
        self.heatmap_center = hkl.load(file_path)

    def query_heatmap(self, query_id):
        '''
        data_dir: dir contains heatmap_center.hkl. Should match to keep_center_neuron_heatmap
        query_id (int): index at center
        '''
        return self.heatmap_center[..., query_id]

    def keep_center_heatmap(self, center_data_dir):
        '''
        output:
          heatmap_center (shape [2, imshape[0], imshape[1], n_center_neuron])
        '''
        # load example heatmap row
        file_path = os.path.join(self.data_dir, 'heatmap_row_{}.hkl'.format(str(0)) )
        heatmap_row = hkl.load(file_path)

        # obtain parameters
        neuron_width = heatmap_row.shape[3]
        neuron_height = heatmap_row.shape[4]
        center = [neuron_width // 2, neuron_height // 2]
        n_center_neuron = heatmap_row.shape[-1]

        # create heatmap_center
        heatmap_center = np.zeros( (2, self.input_shape[0], self.input_shape[1], n_center_neuron) )
        for i in range(self.input_shape[0]):
            file_path = os.path.join(self.data_dir, 'heatmap_row_{}.hkl'.format(str(i)) )
            heatmap_row = hkl.load(file_path)
            heatmap_center[:, i, :] = heatmap_row[:, 0, :, center[0], center[1]]

        # save heatmap_center
        if not os.path.exists(center_data_dir): os.makedirs(center_data_dir)
        file_path = os.path.join(center_data_dir, 'heatmap_center.hkl')
        hkl.dump(heatmap_center, file_path)
