import os
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt
from border_ownership.square_stimuli_generator import Square_Generator
from border_ownership.agent import Agent
from border_ownership.ploter import plot_seq_prediction
import border_ownership.response_in_para as rip
import border_ownership.border_response_analysis as bra

from kitti_settings import *

json_file = 'prednet_kitti_model.json'
json_file = os.path.join(WEIGHTS_DIR, json_file)
weights_file = 'prednet_kitti_weights.hdf5'
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + weights_file)
#output_mode = 'prediction'
output_mode = 'E2'

# create a grey image
width, height = 160, 128
dark_grey = 255 // 3
light_grey = 255 * 2 // 3
gamma = False
length_square = 50
n_frames = 10
pixel_format = '1'
shift_dis_list = np.arange(-30, 30, 2)
neural_id = 8 # This neural prefer boi at the chosen edge orientation. Check boi_to_diff_ori first, make sure neural id and edge_ori are match. Then run this code
edge_ori = 144
#neural_id = 169
#edge_ori = 126

def get_output(
        edge_ori, background_grey, square_grey,
        beta,
        shift_dis_list,
        n_frames,
        pixel_format,
        output_mode
):
    rsg = Square_Generator(background_grey=background_grey, square_grey=square_grey)
    edge_ori, shift_dis_list, video_batch = rsg.generate_shift_square_video_list(edge_ori=edge_ori, beta=beta, shift_dis_list=shift_dis_list, n_frames=n_frames, pixel_format=pixel_format)

    sub = Agent()
    sub.read_from_json(json_file, weights_file)
    output_square = sub.output(video_batch, output_mode=output_mode, batch_size=10, is_upscaled=False) # is_upscaled false assumes that input pixel ranges from 0 to 1
    output_square = {'X': video_batch, output_mode: output_square}
    return output_square

output_dark_square = get_output(edge_ori, light_grey, dark_grey,
    beta=gamma,
    shift_dis_list=shift_dis_list,
    n_frames=n_frames,
    pixel_format=pixel_format,
    output_mode=output_mode)
output_light_square = get_output(edge_ori, dark_grey, light_grey,
    beta=(not gamma),
    shift_dis_list=shift_dis_list,
    n_frames=n_frames,
    pixel_format=pixel_format,
    output_mode=output_mode)

# only keep the center neurons
output_dark_square_center = rip.keep_central_neuron(output_dark_square)
output_dark_square_center[output_mode] = output_dark_square_center[output_mode][..., neural_id]
output_light_square_center = rip.keep_central_neuron(output_light_square)
output_light_square_center[output_mode] = output_light_square_center[output_mode][..., neural_id]

# output data
is_exist = os.path.exists(DATA_DIR)
if not is_exist: os.makedirs(DATA_DIR)

output_path = os.path.join(DATA_DIR, 'shifting_square_dark_prednet_{}_{}_{}.hkl'.format(output_mode, neural_id, gamma))
hkl.dump(output_dark_square_center, output_path)
output_path = os.path.join(DATA_DIR, 'shifting_square_light_prednet_{}_{}_{}.hkl'.format(output_mode, neural_id, gamma))
hkl.dump(output_light_square_center, output_path)

label = {'output_mode': output_mode, 'neural_id': neural_id, 'edge_dir': edge_ori, 'shift': shift_dis_list}
label_path = os.path.join(DATA_DIR, 'shifting_square_label_{}_{}_{}.hkl'.format(output_mode, neural_id, gamma))
hkl.dump(label, label_path)
