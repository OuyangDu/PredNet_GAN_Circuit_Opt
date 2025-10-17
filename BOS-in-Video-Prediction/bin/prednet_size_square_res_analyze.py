# analyze the BO neural response, same as shift. prednet_shif...analyze and this file may should be merged. But since same codes only repeated twice so I didn't merge them.
import os
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt
from border_ownership.prednet_rf_finder import compute_rf_mask
from kitti_settings import *

# parameters
output_mode = 'E2'
gamma = False # True or False
time_cut_init = 5; time_cut_final = 10 # in training the prednet, number of time steps (nt) is 10
neural_id = 115 # 8 or 169 for current datasets
head = 'size' # please use the same neural_id and head as the ones used in prednet_to_size_square_keep_center

# file path. To create data file, run prednet_to_shift_square_keep_center.py
name_tail = '{}_{}_{}'.format(output_mode, neural_id, gamma)
output_dark_path = os.path.join(DATA_DIR, head + '_square_dark_prednet_{}.hkl'.format(name_tail))
output_light_path = os.path.join(DATA_DIR, head + '_square_light_prednet_{}.hkl'.format(name_tail))
label_path = os.path.join(DATA_DIR, head + '_square_label_{}.hkl'.format(name_tail))

# load data
output_dark = hkl.load(output_dark_path)
output_light = hkl.load(output_light_path)
label = hkl.load(label_path)
output_mode = label['output_mode']

# plot the input stimuli
target_square_length = 60
square_length = hkl.load(label_path)['size']
target_length_id = np.where(square_length==target_square_length)[0][0] # the video id when the square length equals to the target
rf_mask = compute_rf_mask(output_mode, query_neural_id='center')

c_light = 'tab:blue'
c_dark = 'tab:green'

fig = plt.figure(figsize=(13, 4.8))

# show the input image
video_dark_batch = output_dark['X']
video_light_batch = output_light['X']

ax_dark = fig.add_subplot(231)
stimulus = video_dark_batch[target_length_id, 0] + rf_mask[..., np.newaxis] # add the RF
stimulus = np.clip(stimulus, a_min=0, a_max=1)
ax_dark.imshow(stimulus)
ax_dark.set_title(c_dark + '\nsquare length {} pixels'.format(target_square_length))

ax_light = fig.add_subplot(234)
stimulus = video_light_batch[target_length_id, 0] + rf_mask[..., np.newaxis] # add the RF
stimulus = np.clip(stimulus, a_min=0, a_max=1)
ax_light.imshow(stimulus)
ax_light.set_title(c_light + '\nsquare length {} pixels'.format(target_square_length))

# plot the response time courses for target square
output_module_dark = output_dark[output_mode]
output_module_light = output_light[output_mode]
output_module_dark += 1
output_module_light += 1

ax_res = fig.add_subplot(132)
time = np.arange(output_module_dark.shape[1])
ax_res.scatter(time, output_module_dark[target_length_id, :], color=c_dark)
ax_res.plot(time, output_module_dark[target_length_id, :], color=c_dark)
ax_res.scatter(time, output_module_light[target_length_id, :], color=c_light)
ax_res.plot(time, output_module_light[target_length_id, :], color=c_light)
ax_res.set_xlabel('time step')
ax_res.set_ylabel('Firing rate')

# plot the averaged neural response as a function of square length
neural_res_dark = np.mean(output_module_dark[:, time_cut_init:time_cut_final], axis=1)
neural_res_light = np.mean(output_module_light[:, time_cut_init:time_cut_final], axis=1)

ax_shift = fig.add_subplot(133)
ax_shift.scatter(square_length, neural_res_dark, color=c_dark)
ax_shift.plot(square_length, neural_res_dark, color=c_dark)
ax_shift.scatter(square_length, neural_res_light, color=c_light)
ax_shift.plot(square_length, neural_res_light, color=c_light)
ax_shift.set_xlabel('square_length')
ax_shift.set_ylabel('Firing rate')

fig.tight_layout()
plt.show()
