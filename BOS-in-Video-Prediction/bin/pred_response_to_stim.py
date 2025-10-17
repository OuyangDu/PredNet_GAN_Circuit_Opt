# get the prednet response to the stimuli of different parameters
import os
import numpy as np
import hickle as hkl
from border_ownership.agent import Agent
from border_ownership.switching_stimuli import Switching_Square_Stimuli, Switching_Square_to_Grey_Stimuli, Switching_Square_Flip, Oscillating_Square_Ambiguous, Switching_Grey_to_Grey_Stimuli, Switching_Ambiguous_Grey_Stimuli, Switching_Grating_Grey_Stimuli, Switching_Pixel_Grey_Stimuli
from border_ownership.ploter import plot_seq_prediction
import matplotlib.pyplot as plt
from kitti_settings import *

#################### Hyperparameters ####################
n_time = 20
# n_time = 58
batch_size = 128
# output_mode = ['R0', 'R1', 'R2', 'R3',  'E0', 'E1', 'E2', 'E3']
output_mode = ['E1']

video_ori_path = os.path.join(DATA_DIR_HOME, 'square_bo_video_ori.npz')
video_size_path = os.path.join(DATA_DIR_HOME, 'square_bo_video_size.npz')
video_shift_path = os.path.join(DATA_DIR_HOME, 'square_bo_video_shift.npz')
video_square_part_path = os.path.join(DATA_DIR_HOME, 'square_bo_video_square_part.npz')
video_switch_stimuli_path = os.path.join(DATA_DIR_HOME, 'switching_stimuli.hkl')
video_switch_grey_stimuli_path = os.path.join(DATA_DIR_HOME, 'switching_grey_stimuli.hkl')
video_oscillate_square_ambiguous_path = os.path.join(DATA_DIR_HOME, 'oscillate_square_ambiguous_stimuli.hkl')
video_switch_square_flip_path = os.path.join(DATA_DIR_HOME, 'switch_square_flip_stimuli.hkl')
video_switch_grey_to_grey_path = os.path.join(DATA_DIR_HOME, 'switch_grey_to_grey_stimuli.hkl')
video_switch_ambiguous_grey_path = os.path.join(DATA_DIR_HOME, 'switch_ambiguous_grey_stimuli.hkl')
video_switch_grating_grey_path = os.path.join(DATA_DIR_HOME, 'switch_grating_grey_stimuli.hkl')
video_switch_pixel_grey_path = os.path.join(DATA_DIR_HOME, 'switch_pixel_grey_stimuli.hkl')

res_ori_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_ori.npz')
res_size_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_size.npz')
res_shift_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_shift.npz')
res_square_part_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_square_part.npz')
res_switch_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_switch.npz')
res_switch_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_switch_grey.npz')
res_oscillate_square_ambiguous_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_oscillate_square_ambiguous.npz')
res_switch_square_flip_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_switch_square_flip.npz')
res_switch_grey_to_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_switch_grey_to_grey.npz')
res_switch_ambiguous_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_switch_ambiguous_grey.npz')
res_switch_grating_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_switch_grating_grey.npz')
res_switch_pixel_grey_path = os.path.join(DATA_DIR_HOME, 'square_bo_res_switch_pixel_grey.npz')

prednet_json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
prednet_weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + 'prednet_kitti_weights.hdf5')

#################### Main ####################

sub = Agent()
sub.read_from_json(prednet_json_file, prednet_weights_file)

def get_response_stim(stim, sub, stim_mode='image'):
    '''
    stim_mode: 'image' or 'video'. If image, this function will repeat the image to form a video. If video, this function will do nothing.
    '''
    response = {om : [] for om in output_mode}
    for i in np.arange(0, stim.shape[0], batch_size):
        print('processing video batch {}'.format(i))
        img_batch = stim[i:i + batch_size]

        if stim_mode == 'image':
            video_batch = np.repeat(img_batch, n_time, axis=1)
        elif stim_mode == 'video':
            video_batch = img_batch
        else:
            raise ValueError('stim_mode must be image or video')

        ##### check the prediction
        # # video_batch = video_batch[0].reshape(1, *video_batch[0].shape)
        # video_batch = video_batch[0:4]
        # response_batch = sub.output_multiple(video_batch, output_mode=['prediction'], batch_size=batch_size, is_upscaled=False)
        # for i in range(4):
        #     plot_seq_prediction(video_batch[i], response_batch['prediction'][i])
        #     plt.show()
        #### check prediction end

        response_batch = sub.output_multiple(video_batch, output_mode=output_mode, batch_size=batch_size, is_upscaled=False)
        [response[om].append(response_batch[om]) for om in output_mode]

    return response

# # respones to different square orientations
# data_ori = np.load(video_ori_path, allow_pickle=True)
# response_ori = get_response_stim(data_ori['video'], sub)
# np.savez(res_ori_path, image=data_ori['video'][:, 0], para=data_ori['para'], key=data_ori['key'],  **response_ori)
# del data_ori
# del response_ori

# # respones to different square sizes
# data_size = np.load(video_size_path, allow_pickle=True)
# response_size = get_response_stim(data_size['video'], sub)
# np.savez(res_size_path, image=data_size['video'][:, 0], para=data_size['para'], key=data_size['key'],  **response_size)
# del data_size

# data_shift = np.load(video_shift_path, allow_pickle=True)
# response_shift = get_response_stim(data_shift['video'], sub)
# np.savez(res_shift_path, image=data_shift['video'][:, 0], para=data_shift['para'], key=data_shift['key'],  **response_shift)
# del data_shift

# data = np.load(video_square_part_path, allow_pickle=True)
# response = get_response_stim(data['video'], sub)
# np.savez(res_square_part_path, image=data['video'][:, 0], para=data['para'], key=data['key'],  **response)
# del data

# sss = Switching_Square_Stimuli() # square-ambiguous
# sss.load(video_switch_stimuli_path)
# response_switch = get_response_stim(sss, sub, stim_mode='video')
# im_np = [np.array(videoi[0, 0]) for videoi in sss]
# im_np = np.stack(im_np, axis=0)
# para = sss.export_para_df()
# np.savez(res_switch_path, image=im_np, para=para.to_numpy(), key=para.columns.values,  **response_switch)
# del sss
# del response_switch

# ssf = Switching_Square_Flip()
# ssf.load(video_switch_square_flip_path)
# response_ssf = get_response_stim(ssf, sub, stim_mode='video')
# im_np = [np.array(videoi[0, 0]) for videoi in ssf]
# im_np = np.stack(im_np, axis=0)
# para = ssf.export_para_df()
# np.savez(res_switch_square_flip_path, image=im_np, para=para.to_numpy(), key=para.columns.values,  **response_ssf)
# del ssf
# del response_ssf

# sags = Switching_Ambiguous_Grey_Stimuli()
# sags.load(video_switch_ambiguous_grey_path)
# response_sags = get_response_stim(sags, sub, stim_mode='video')
# im_np = [np.array(videoi[0, 0]) for videoi in sags]
# im_np = np.stack(im_np, axis=0)
# para = sags.export_para_df()
# np.savez(res_switch_ambiguous_grey_path, image=im_np, para=para.to_numpy(), key=para.columns.values,  **response_sags)
# del sags
# del response_sags

# sgrat = Switching_Grating_Grey_Stimuli()
# sgrat.load(video_switch_grating_grey_path)
# response_sgrat = get_response_stim(sgrat, sub, stim_mode='video')
# im_np = [np.array(videoi[0, 0]) for videoi in sgrat]
# im_np = np.stack(im_np, axis=0)
# para = sgrat.export_para_df()
# np.savez(res_switch_grating_grey_path, image=im_np, para=para.to_numpy(), key=para.columns.values,  **response_sgrat)

# spgs = Switching_Pixel_Grey_Stimuli()
# spgs.load(video_switch_pixel_grey_path)
# response_spgs = get_response_stim(spgs, sub, stim_mode='video')
# im_np = [np.array(videoi[0, 0]) for videoi in spgs]
# im_np = np.stack(im_np, axis=0)
# para = spgs.export_para_df()
# np.savez(res_switch_pixel_grey_path, image=im_np, para=para.to_numpy(), key=para.columns.values,  **response_spgs)
