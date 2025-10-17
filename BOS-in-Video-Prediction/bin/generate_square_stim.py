from border_ownership.full_square_response import combine_parameters, square_generator_bo_batch, convert_grey_img_to_rgb_video_list
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from border_ownership.ploter import show_center_region
from border_ownership.util import imshow_central_region
from border_ownership.switching_stimuli import Switching_Square_Stimuli, Switching_Square_to_Grey_Stimuli, Switching_Square_Flip, Oscillating_Square_Ambiguous, Switching_Grey_to_Grey_Stimuli, Switching_Ambiguous_Grey_Stimuli, Switching_Grating_Grey_Stimuli, Switching_Pixel_Grey_Stimuli
from kitti_settings import *

#################### Hypoerparameter ####################
n_time = 1
switch_t0_len = 4; switch_t1_len = 16
para_ori = {'dark_grey': [255//3], 'light_grey': [255 * 2 // 3], 'orientation': np.linspace(0, 180, 10, endpoint=False), 'beta': [False, True], 'gamma': [False, True], 'shift': [0], 'size': [50]}

para_size = {'dark_grey': [255//3], 'light_grey': [255 * 2 // 3], 'orientation': np.linspace(0, 180, 10, endpoint=False), 'beta': [False, True], 'gamma': [False, True], 'shift': [0], 'size': np.arange(10, 90, 10)}

para_shift = {'dark_grey': [255//3], 'light_grey': [255 * 2 // 3], 'orientation': np.linspace(0, 180, 10, endpoint=False), 'beta': [False, True], 'gamma': [False, True], 'shift': np.arange(-30, 30, 10), 'size': [50]}

keep_keypoint_id_no_center = [[i] for i in range(1, 8)]
keep_keypoint_id_no_center.insert(0, [None])
keep_keypoint_id_no_center_full_square = [[i for i in range(1, 8)]]

keep_keypoint_id_center = [[0, i] for i in range(1, 8)]
keep_keypoint_id_center.insert(0, [0])
keep_keypoint_id_full_square = [[i for i in range(8)]]

keep_keypoint_id = keep_keypoint_id_no_center + keep_keypoint_id_no_center_full_square + keep_keypoint_id_center + keep_keypoint_id_full_square

para_square_part = {'dark_grey': [255//3], 'light_grey': [255 * 2 // 3], 'orientation': np.linspace(0, 180, 10, endpoint=False), 'beta': [False, True], 'gamma': [False, True], 'shift': [0], 'size': [50], 'keep_keypoint_id': keep_keypoint_id}

video_ori_path = os.path.join(DATA_DIR_HOME, 'square_bo_video_ori.npz')
video_size_path = os.path.join(DATA_DIR_HOME, 'square_bo_video_size.npz')
video_shift_path = os.path.join(DATA_DIR_HOME, 'square_bo_video_shift.npz')
video_square_part_path = os.path.join(DATA_DIR_HOME, 'square_bo_video_square_part.npz')
video_path = [video_ori_path, video_size_path, video_shift_path, video_square_part_path]
switch_stimuli_path = os.path.join(DATA_DIR_HOME, 'switching_stimuli.hkl')
switch_grey_stimuli_path = os.path.join(DATA_DIR_HOME, 'switching_grey_stimuli.hkl')
oscillate_square_ambiguous_path = os.path.join(DATA_DIR_HOME, 'oscillate_square_ambiguous_stimuli.hkl')
switch_square_flip_path = os.path.join(DATA_DIR_HOME, 'switch_square_flip_stimuli.hkl')
switch_grey_to_grey_path = os.path.join(DATA_DIR_HOME, 'switch_grey_to_grey_stimuli.hkl')
switch_ambiguous_grey_path = os.path.join(DATA_DIR_HOME, 'switch_ambiguous_grey_stimuli.hkl')
switch_grating_grey_path = os.path.join(DATA_DIR_HOME, 'switch_grating_grey_stimuli.hkl')
switch_pixel_grey_path = os.path.join(DATA_DIR_HOME, 'switch_pixel_grey_stimuli.hkl')

#################### Main ####################
df_ori = combine_parameters(para_ori)
df_size = combine_parameters(para_size)
df_shift = combine_parameters(para_shift)
df_square_part = combine_parameters(para_square_part)
df = [df_ori, df_size, df_shift, df_square_part]

# generate squares with different orientations, shifts, sizes, and parts
for vpi, dfi in zip(video_path, df):
    img = square_generator_bo_batch(dfi)
    # #### !!! This is very important, make sure the central images are the same for the same beta. After visual inspection, please comment this part to generate video
    # for im in img:
    #     show_center_region(im, region_size=(20, 20))
    # ###
    video = convert_grey_img_to_rgb_video_list(img, n_frames=n_time) # (n_samples, n_time, n_x, n_y, n_channel)
    np.savez(vpi, video=video, para=dfi.to_numpy(), key=dfi.columns.values)

# # generate switching stimuli. In this paper this is called square-ambiguous stimuli
# sss = Switching_Square_Stimuli()
# sss.generate_stimuli_prototype_batch(switch_t0_len, switch_t1_len, df_ori.to_dict('records'))
# sss.dump(switch_stimuli_path)

# # generate flip sitmuli
# ssf = Switching_Square_Flip()
# ssf.generate_stimuli_prototype_batch(switch_t0_len, switch_t1_len, df_ori.to_dict('records'))
# ssf.dump(switch_square_flip_path)

# # generate ambiguous to grey stimuli
# sags = Switching_Ambiguous_Grey_Stimuli()
# sags.generate_stimuli_prototype_batch(switch_t0_len, switch_t1_len, df_ori.to_dict('records'))
# sags.dump(switch_ambiguous_grey_path)

# # generate grating to grey stimuli
# sgrat = Switching_Grating_Grey_Stimuli()
# sgrat.generate_stimuli_prototype_batch(switch_t0_len, switch_t1_len, df_ori.to_dict('records'))
# sgrat.dump(switch_grating_grey_path)

# # generate single pixel to grey stimuli
# spgs = Switching_Pixel_Grey_Stimuli()
# spgs.generate_stimuli_prototype_batch(switch_t0_len, switch_t1_len, df_ori.to_dict('records'))
# spgs.dump(switch_pixel_grey_path)


## show example videos
# def video_show(video):
#     n_frame = video.shape[0]
#     fig, ax = plt.subplots(1, n_frame)
#     for i in range(n_frame):
#         ax[i].imshow(video[i])
#         for spine in ax[i].spines.values():
#             spine.set_visible(False)
#         ax[i].tick_params(bottom=False, left=False)
#         ax[i].set_xticks([])
#         ax[i].set_yticks([])
#     plt.show()

# for i in range(len(spgs)):
#     videoi = spgs[i]
#     video_show(videoi[0])

# video = ssgs[10]
# print(video.shape)
# for im in video[0]:
#     plt.figure()
#     plt.imshow(im)
# plt.show()

# data = np.load(video_path[-1], allow_pickle=True)
# for i in data['video']:
#     plt.imshow(i[0])
#     plt.show()
# print(data.files)
# print(data['video'].shape)
# print(data['para'].shape)
# print(data['para'])
# print(data['key'])
