import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt

from kitti_settings import *

import border_ownership.response_in_para as rip

output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all.hkl')
output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all.hkl')
label_path = os.path.join(DATA_DIR, 'rotating_square_label_all.hkl')

output_dark = hkl.load(output_dark_path)
output_light = hkl.load(output_light_path)

output_dark_center = rip.keep_central_neuron(output_dark)
output_light_center = rip.keep_central_neuron(output_light)

output_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center.hkl')
hkl.dump(output_dark_center, output_path)
output_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center.hkl')
hkl.dump(output_light_center, output_path)
