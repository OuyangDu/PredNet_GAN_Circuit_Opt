# analyze the BO neural response 
import os
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt
from border_ownership.prednet_rf_finder import compute_rf_mask
from kitti_settings import *


# parameters
output_mode = 'E2'
neural_id = 8 # 8 or 169 for current datasets
gamma = False # True or False
time_cut_init = 0; time_cut_final = 10 # in training the prednet, number of time steps (nt) is 10

# file path. To create data file, run prednet_to_shift_square_keep_center.py
name_tail = '{}_{}_{}'.format(output_mode, neural_id, gamma)
output_dark_path = os.path.join(DATA_DIR, 'shifting_square_dark_prednet_{}.hkl'.format(name_tail))
output_light_path = os.path.join(DATA_DIR, 'shifting_square_light_prednet_{}.hkl'.format(name_tail))
label_path = os.path.join(DATA_DIR, 'shifting_square_label_{}.hkl'.format(name_tail))

# load data
output_dark = hkl.load(output_dark_path)
output_light = hkl.load(output_light_path)
label = hkl.load(label_path)
output_mode = label['output_mode']

# plot the input stimuli
shifted_pixel = -30
shift = hkl.load(label_path)['shift']
zero_shift_video = np.where(shift==shifted_pixel)[0][0] # the video id when the shifting is zero
rf_mask = compute_rf_mask(output_mode, query_neural_id='center')

c_light = 'tab:blue'
c_dark = 'tab:green'

fig = plt.figure(figsize=(13, 4.8))

# show the input image
video_dark_batch = output_dark['X']
video_light_batch = output_light['X']

ax_dark = fig.add_subplot(231)
stimulus = video_dark_batch[zero_shift_video, 0] + rf_mask[..., np.newaxis] # add the RF
stimulus = np.clip(stimulus, a_min=0, a_max=1)
ax_dark.imshow(stimulus)
ax_dark.set_title(c_dark + '\nshifted {} pixels'.format(shifted_pixel))

ax_light = fig.add_subplot(234)
stimulus = video_light_batch[zero_shift_video, 0] + rf_mask[..., np.newaxis] # add the RF
stimulus = np.clip(stimulus, a_min=0, a_max=1)
ax_light.imshow(stimulus)
ax_light.set_title(c_light + '\nshifted {} pixels'.format(shifted_pixel))

output_module_dark = output_dark[output_mode]
output_module_light = output_light[output_mode]
output_module_dark += 1
output_module_light += 1

# plot the response time courses for shift = 0
ax_res = fig.add_subplot(132)
time = np.arange(output_module_dark.shape[1])
ax_res.scatter(time, output_module_dark[zero_shift_video, :], color=c_dark)
ax_res.plot(time, output_module_dark[zero_shift_video, :], color=c_dark)
ax_res.scatter(time, output_module_light[zero_shift_video, :], color=c_light)
ax_res.plot(time, output_module_light[zero_shift_video, :], color=c_light)
ax_res.set_xlabel('time step')
ax_res.set_ylabel('Firing rate')

# plot the averaged neural response as a function of shift
neural_res_dark = np.mean(output_module_dark[:, time_cut_init:time_cut_final], axis=1)
neural_res_light = np.mean(output_module_light[:, time_cut_init:time_cut_final], axis=1)

ax_shift = fig.add_subplot(133)
ax_shift.scatter(shift, neural_res_dark, color=c_dark)
ax_shift.plot(shift, neural_res_dark, color=c_dark)
ax_shift.scatter(shift, neural_res_light, color=c_light)
ax_shift.plot(shift, neural_res_light, color=c_light)
ax_shift.set_xlabel('Pixel shifted')
ax_shift.set_ylabel('Firing rate')

fig.tight_layout()
plt.show()
