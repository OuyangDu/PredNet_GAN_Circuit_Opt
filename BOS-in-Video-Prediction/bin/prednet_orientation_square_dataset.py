# create a dataset about prednet rsponses to orientation square
import numpy as np
import os

from kitti_settings import *
from border_ownership.dataset import Prednet_Ori_Square_Dataset

time_cut = [0, 10] # we havn't implemented this but 5 seems a good time cut read from the figure
output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center.hkl')
output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center.hkl')
label_path = os.path.join(DATA_DIR, 'rotating_square_label_all.hkl')

posd = Prednet_Ori_Square_Dataset(output_dark_path, output_light_path, label_path, output_mode='E2')
X, Y, label_name = posd.output_data(time_cut, time_processing='average')
#X, Y, label_name = prednet_orientation_square_dataset(output_dark_path, output_light_path, label_path, output_mode='E2')
print(label_name)
print(X.shape, Y.shape)
