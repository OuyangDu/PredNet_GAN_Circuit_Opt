# a simple code to compute the shape of prednet in each module
import os
import numpy as np

from border_ownership.agent import Agent, Agent_RF_Wraper

from kitti_settings import *

json_file = 'prednet_kitti_model.json'
json_file = os.path.join(WEIGHTS_DIR, json_file)
weights_file = 'prednet_kitti_weights.hdf5'
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + weights_file)
num_frames = 2
input_shape = (128, 160, 3)
output_mode = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3', 'A0', 'A1', 'A2', 'A3', 'Ahat0', 'Ahat1', 'Ahat2', 'Ahat3']

im = np.zeros( (1, *input_shape) )

sub = Agent()
sub.read_from_json(json_file, weights_file)
sub_warp = Agent_RF_Wraper(sub, num_frames, output_mode)
for om in output_mode:
    sub_warp.output_mode = om
    output = sub_warp.predict(im)
    print(output.shape)
    print('The shape of {} is {}'.format(om, output.shape[1:]))
