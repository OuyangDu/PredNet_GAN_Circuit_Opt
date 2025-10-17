import os
import numpy as np
import matplotlib.pyplot as plt
import hickle as hkl

from border_ownership.agent import Agent, Agent_RF_Wraper
from border_ownership.ploter import plot_seq_prediction
from border_ownership.square_stimuli_generator import Square_Generator
import border_ownership.response_in_para as rip

from kitti_settings import *

#json_file = 'prednet_kitti_model.json'
#weights_file = 'prednet_kitti_weights.hdf5'
#file_name_tail = ''

## Untrained
json_file = 'prednet_kitti_model_untrain.json'
weights_file = 'prednet_kitti_weights_untrain.hdf5'
file_name_tail = 'untrain'

### untrained small
json_file = 'prednet_kitti_model_untrain_small.json'
weights_file = 'prednet_kitti_weights_untrain_small.hdf5'
file_name_tail = 'untrain_small'
json_file = os.path.join(WEIGHTS_DIR, json_file)
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + weights_file)

#output_mode = ['prediction']
#output_mode = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3', 'A0', 'A1', 'A2', 'A3', 'Ahat0', 'Ahat1', 'Ahat2', 'Ahat3']
#output_mode = ['R0', 'R1', 'R2', 'R3']
output_mode = ['E2']

# create a grey image
width, height = 160, 128
dark_grey = 255 // 3
light_grey = 255 * 2 // 3
length_square = 50
n_direction_line = 10 # it's better to be even to include 90 degree direction
n_frames = 20
pixel_format = '1' # pixel value ranges from 0 to 1

rsg = Square_Generator(background_grey=light_grey, square_grey=dark_grey)
edge_dir_list, angle_list, video_batch = rsg.generate_rotated_square_video_list(n_direction_line=n_direction_line, n_frames=n_frames, pixel_format=pixel_format)

sub = Agent()
sub.read_from_json(json_file, weights_file)
output_dark_square = sub.output_multiple(video_batch, output_mode=output_mode, batch_size=100, is_upscaled=False) # is_upscaled false assumes that input pixel ranges from 0 to 1
output_dark_square['X'] = video_batch

rsg = Square_Generator(background_grey=dark_grey, square_grey=light_grey)
edge_dir_list, angle_list, video_batch = rsg.generate_rotated_square_video_list(n_direction_line=n_direction_line, n_frames=n_frames, pixel_format=pixel_format)
output_light_square = sub.output_multiple(video_batch, output_mode=output_mode, batch_size=100, is_upscaled=False) # is_upscaled false assumes that input pixel ranges from 0 to 1
output_light_square['X'] = video_batch

#if output_mode[0] == 'prediction':
#    fig, gs = plot_seq_prediction(video_batch[0], output_dark_square['prediction'][0])
#    plt.show()

# only keep the center neurons
output_dark_square_center = rip.keep_central_neuron(output_dark_square)
output_light_square_center = rip.keep_central_neuron(output_light_square)

# output data
is_exist = os.path.exists(DATA_DIR)
if not is_exist: os.makedirs(DATA_DIR)

output_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center_length{}_{}.hkl'.format(length_square, file_name_tail))
hkl.dump(output_dark_square_center, output_path)
output_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center_length{}_{}.hkl'.format(length_square, file_name_tail))
hkl.dump(output_light_square_center, output_path)

label = {'edge_dir': edge_dir_list, 'angle': angle_list}
label_path = os.path.join(DATA_DIR, 'rotating_square_label_all_length{}_{}.hkl'.format(length_square, file_name_tail))
hkl.dump(label, label_path)
