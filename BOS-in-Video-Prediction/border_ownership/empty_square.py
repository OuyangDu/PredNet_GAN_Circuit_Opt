# plotting help functions about empty square
import numpy as np
import matplotlib.pyplot as plt
from border_ownership.ploter import add_ax_contour, simpleaxis
from border_ownership.square_stimuli_generator import Square_Generator, convert_strip_id_by_beta
from border_ownership.agent import Agent
import border_ownership.response_in_para as rip
import matplotlib.colors as mcolors
from kitti_settings import *
import os

def keep_to_strip(keep_list):
    '''
    converting keep list to strip list
    keep_list (list of int): int from 0 to 7 indicating which edges will be kept
    '''
    full_list = list(range(0, 8))
    strip_list = [x for x in full_list if x not in keep_list]
    return strip_list

def batch_keep_to_strip(batch_keep_list):
    batch_strip_list = []
    for keep_list in batch_keep_list:
        strip_list = keep_to_strip(keep_list)
        batch_strip_list.append(strip_list)
    return batch_strip_list

class Empty_Square():
    '''
    attribute:
      empty_square_image_ (list of images): each image corresponding to one strip_id
    '''
    def __init__(self, edge_ori=0, background_grey=255 * 2 // 3, beta=True, strip_id_list=[[None]], width=160, height=128, length_square=50, n_frames=20, strip_id_mode='strip'):
        '''
        strip_id_mode (str): keep of strip
        '''
        self.edge_ori = edge_ori
        self.background_grey = background_grey
        self.beta = beta
        self.width = width
        self.height = height
        self.length_square = length_square
        self.n_frames = n_frames
        self.beta = beta

        if strip_id_mode == 'keep':
            strip_id_list_conv = batch_keep_to_strip(strip_id_list)
            self.strip_id_list = strip_id_list_conv
        else:
            self.strip_id_list = strip_id_list

        self.prednet_json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
        self.prednet_weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + 'prednet_kitti_weights.hdf5')

        self.video = self._generate_empty_square_video()
        self.empty_square_image_ = [v[0] for v in self.video]

    def _generate_empty_square_video(self):
        rsg = Square_Generator(background_grey=self.background_grey, square_grey=None, length_square=self.length_square)
        _, _, video_batch = rsg.generate_strip_square_video_list(edge_ori=self.edge_ori, beta=self.beta, strip_id_list=self.strip_id_list, n_frames=self.n_frames, pixel_format='1')
        return video_batch

    def get_neural_response(self, module_name, neural_id):
        sub = Agent()
        sub.read_from_json(self.prednet_json_file, self.prednet_weights_file)

        #for v in self.video:
        #    plt.figure()
        #    plt.imshow(v[0])
        #    plt.show()

        output_square = sub.output(self.video, output_mode=module_name, batch_size=32, is_upscaled=False) # is_upscaled false assumes that input pixel ranges from 0 to 1

        output_square = {'null': output_square}
        output_square_center = rip.keep_central_neuron(output_square)
        neural_res = output_square_center['null'] # [video.shape[0], n_time, n_neuron]
        return neural_res[..., neural_id]

    def get_avg_neural_response(self, module_name, neural_id, time_window=[5, 20]):
        neural_res = self.get_neural_response(module_name, neural_id)
        return np.mean(neural_res[:, time_window[0]:time_window[1]], axis=1)


    def set_edge_ori(self, edge_ori):
        self.edge_ori = edge_ori
        self.video = self._generate_empty_square_video()
        self.empty_square_image_ = [v[0] for v in self.video]

    def set_background_grey(self, background_grey):
        self.background_grey = background_grey
        self.video = self._generate_empty_square_video()
        self.empty_square_image_ = [v[0] for v in self.video]

    def set_strip_id_list(self, strip_id_list, mode='strip'):
        if mode == 'keep':
            strip_id_list_conv = batch_keep_to_strip(strip_id_list)
            self.strip_id_list = strip_id_list_conv
        else:
            self.strip_id_list = strip_id_list

        self.video = self._generate_empty_square_video()
        self.empty_square_image_ = [v[0] for v in self.video]

    def set_beta(self, beta):
        self.beta = beta
        self.video = self._generate_empty_square_video()
        self.empty_square_image_ = [v[0] for v in self.video]

def plot_neural_res(pos_image, neg_image, pos_single_neural_res, neg_single_neural_res, strip_name, center_heatmap_dir, neural_id, n_frames=20):
    c_pos = 'tab:green' # response curve figure
    c_neg = 'tab:blue'

    fig = plt.figure(figsize=(4, 3))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])

    ax.imshow(pos_image)
    ax.set_title(c_pos)
    add_ax_contour(ax, neural_id, center_heatmap_dir, linestyle='solid')

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(neg_image)
    ax.set_title(c_neg)
    add_ax_contour(ax, neural_id, center_heatmap_dir, linestyle='solid')

    time = np.arange(n_frames)
    ax = fig.add_subplot(gs[:, 1:])
    ax.plot(time, pos_single_neural_res, color=c_pos)
    ax.plot(time, neg_single_neural_res, color=c_neg)
    ax.set_xlabel('time step')
    ax.set_ylabel('neural response')

    return fig, ax

def empty_square_part_image(es, with_center=True, beta=True):
    if with_center:
        batch_keep_list = [[0, i] for i in range(0, 8)]
    else:
        batch_keep_list = [[i] for i in range(1, 8)]
        batch_keep_list.insert(0, [])

    es.set_strip_id_list(batch_keep_list, mode='keep')
    es.set_beta(beta)
    return es.empty_square_image_

def sequential_id_to_mat_id(beta):
    '''
    convert sequential id to the corresponding positions of a 3 by 3 matrix
    '''
    sequential_to_mat_betaF = {
        0: (1, 0),
        1: (0, 0),
        2: (0, 1),
        3: (0, 2),
        4: (1, 2),
        5: (2, 2),
        6: (2, 1),
        7: (2, 0),
    }

    sequential_to_mat_betaT = {
        0: (1, 2),
        1: (2, 2),
        2: (2, 1),
        3: (2, 0),
        4: (1, 0),
        5: (0, 0),
        6: (0, 1),
        7: (0, 2),
    }

    if beta:
        sequential_to_mat = sequential_to_mat_betaT
    else:
        sequential_to_mat = sequential_to_mat_betaF
    return sequential_to_mat

def plot_empty_square_part_image(image, beta, with_rf_contour=False, center_heatmap_dir=None, neural_id=None):
    sequential_to_mat = sequential_id_to_mat_id(beta)

    fig, ax = plt.subplots(3, 3, figsize=(4, 4))
    for i, im in enumerate(image):
        index = sequential_to_mat[i]
        ax[index].imshow(im)
        ax[index].set_title(i)
        #idx = i // 3
        #idy = i % 3
        #ax[idx, idy].imshow(im)
        #ax[idx, idy].set_title(i)
        if with_rf_contour: add_ax_contour(ax[index], neural_id, center_heatmap_dir)
        simpleaxis(ax[index])
    simpleaxis(ax[1, 1])
    return fig, ax

def compute_empty_square_part_neural_res(es, with_center=True, beta=True, output_mode='R2', neural_id=0, output_change=True):
    '''
    compute the averaged neural response to empty squares part. Run show_empty_square_part_image to show the corresponding input images for the output neural response
    '''
    if with_center:
        batch_keep_list = [[0, i] for i in range(0, 8)]
    else:
        batch_keep_list = [[i] for i in range(1, 8)]
        batch_keep_list.insert(0, [])

    es.set_strip_id_list(batch_keep_list, mode='keep')
    es.set_beta(beta)

    neural_res = es.get_avg_neural_response(output_mode, neural_id)
    if output_change:
        neural_res_change = neural_res - neural_res[0]
        return neural_res_change
    else:
        return neural_res

def plot_sequential_square_part(square_part_image, config_list):
    '''
    square_part_image (list): a list of four with [with_center, beta] value specified by config_list. Each element itself is a ndarray contains 8 elements that matches to the batch_keep_list in compute_empty_square_part_neural_res
    config_list (list): a list of four. Each element is a length-2 list, first element indicating with_center second indicating beta
    '''
    fig, axes = plt.subplots(6, 6, figsize=(6, 6))
    ax1 = axes[:3, :3]
    ax2 = axes[:3, 3:]
    ax3 = axes[3:, :3]
    ax4 = axes[3:, 3:]
    axes = [ax1, ax2, ax3, ax4]
    for i, square_part in enumerate(square_part_image):
        ax = axes[i]

        beta = config_list[i][1]
        sequential_to_mat = sequential_id_to_mat_id(beta)
        for j in range(8):
            ax[sequential_to_mat[j]].imshow(square_part[j])

    [simpleaxis(axi) for axi in np.array(axes).flatten()]

    return fig, axes

def plot_sequential_res_on_square(neural_response_change, config_list, symmetric_color=True):
    '''
    neural_response_change (list): a list of four with [with_center, beta] value specified by config_list. Each element itself is a ndarray contains 8 elements that matches to the batch_keep_list in compute_empty_square_part_neural_res
    config_list (list): a list of four. Each element is a length-2 list, first element indicating with_center second indicating beta
    '''

    # colorbar options
    vcenter = 0
    if symmetric_color:
        rmax = np.max(np.abs(neural_response_change))
        vmin, vmax = -rmax, rmax
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        vmin = np.min(np.array(neural_response_change).flatten())
        vmax = np.max(np.array(neural_response_change).flatten())
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    mat_fig = np.zeros((3, 3))
    fig, ax = plt.subplots(2, 2, figsize=(3, 3))
    for i, nrc in enumerate(neural_response_change):
        beta = config_list[i][1]
        sequential_to_mat = sequential_id_to_mat_id(beta)
        for j in range(8): mat_fig[sequential_to_mat[j]] = nrc[j]
        idx, idy = i // 2, i % 2

        # im = ax[idx, idy].imshow(mat_fig, cmap='bwr', vmin=vmin, vmax=vmax)
        im = ax[idx, idy].imshow(mat_fig, cmap='bwr', norm=norm)

        simpleaxis(ax[idx, idy])
        ax[idx, idy].grid(color='k', linestyle='-', linewidth=.5)
        ax[idx, idy].set_xticks(ticks=np.arange(-.5, 3, 1))
        ax[idx, idy].set_yticks(ticks=np.arange(-.5, 3, 1))

        ax[idx, idy].set_title(config_list[i], fontsize=6)

    fig.subplots_adjust(bottom=0.1)
    cbar_ax = fig.add_axes([0.17, 0.0, 0.7, 0.04])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar_ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, labelsize=6)

    return fig, ax
