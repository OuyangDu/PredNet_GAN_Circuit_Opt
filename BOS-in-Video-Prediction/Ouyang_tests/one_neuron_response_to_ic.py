import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from drawing_pacman import border_kaniza_sqr as k_sqr
from drawing_pacman import non_kaniza_sqr as non_k_sqr
from drawing_square_image import create_static_video_from_image as pic2video

#### PredNet demo. You need to put border_ownership.agent.py file in the correct path so that import works.
from border_ownership.agent import Agent

## turn off CUDA if your GPU is not available or too new > 3060
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

WEIGHTS_DIR = '../model_data_keras2/'
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')

output_mode = ['prediction', 'E0','E2'] # a list of output modes. Can be 'prediction', 'E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3', 'A0', 'A1', 'A2', 'A3', 'Ahat0', 'Ahat1', 'Ahat2', 'Ahat3'. If prediction, output prediction; if others, output the units responses
agent = Agent()
agent.read_from_json(json_file, weights_file)

###############################################################################
#creating video for Kanizsa Square and Control Square
ic_image=k_sqr(width=48,r=12)
control_image= non_k_sqr(width=48,r=12)
 # video dimension
 # (n_video, n_time_step, height=128, width, 3). Make sure your video has correct shape
ic_videos=pic2video(ic_image)
control_videos=pic2video(control_image)

#show image
ic_image.show()
control_image.show()

###############################################################################
# Getting Outputs
ic_output = agent.output_multiple(ic_videos, output_mode=output_mode, is_upscaled=False) # the output is a dictionary with keys as output_mode. is_upscaled=False means the input frame ranges from 0 to 1; is_upscaled=True means the input frame ranges from 0 to 255.
control_output = agent.output_multiple(control_videos, output_mode=output_mode, is_upscaled=False)
#################################################################################
"""
print("ic_output E2 shape:")
print(ic_output['E2'].shape) # The shape of E0 (n_video, n_time_step, E0_height, E0_width, E0_channel). The last three channels describes the shape of E0 module.
print("ic_output E2_0:")
print(ic_output['E2'][0,0,16,20,181])
print(ic_output['E2'][0,1,16,20,181])
print(ic_output['E2'][0,2,16,20,181])
print(ic_output['E2'][0,3,16,20,181])
print(ic_output['E2'][0,4,16,20,181])
print("control_output E2 shape:")
print(control_output['E2'].shape)
print("control_output E2_0:")
print(control_output['E2'][0,0,16,20,181])
print(control_output['E2'][0,1,16,20,181])
print(control_output['E2'][0,2,16,20,181])
print(control_output['E2'][0,3,16,20,181])
print(control_output['E2'][0,4,16,20,181])
"""
""""
# visualize one prediction frame and one actual frame
#Illutionary output prediction
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(ic_output['prediction'][0, 0])
plt.title('Prediction')
plt.subplot(1, 2, 2)
plt.imshow(ic_videos[0, 4])
plt.title('Actual')
plt.show()

#Control output prediction
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(control_output['prediction'][0, 0])
plt.title('Prediction')
plt.subplot(1, 2, 2)
plt.imshow(control_videos[0, 4])
plt.title('Actual')
plt.show()"""