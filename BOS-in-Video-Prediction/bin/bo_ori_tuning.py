# test the orientation tuning of a typical bo cell
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt

from kitti_settings import *

import border_ownership.response_in_para as rip
import border_ownership.border_response_analysis as bra
import border_ownership.orientation_analysis as oa
import border_ownership.ploter as ploter

neural_rank = 0 # select neuron with rank of boi, from largest (0) to smalest

output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet.hkl')
output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet.hkl')
label_path = os.path.join(DATA_DIR, 'rotating_square_label.hkl')

output_dark = hkl.load(output_dark_path)
output_light = hkl.load(output_light_path)
angle = hkl.load(label_path)['angle']

output_rf, rf_para, alpha_rf, beta_rf, gamma_rf = rip.response_in_rf_paramemterization(output_dark, output_light, angle)
output_center = rip.keep_central_neuron(output_rf) # (alpha, beta, gamma, t, width, height, chs) for pixel, (alpha, beta, gamma, t, chs) for the rest

output_mode = 'R2'
output_module = output_center[output_mode]
if output_mode[0] == 'R':
    output_module = output_module + 1 # avoid negative value

bo = bra.compute_BO(output_module)

angle_idx, neuron_idx = bra.select_neuron_by_bo(bo, bo_order=None)

# plot orientation tuning
fig = plt.figure()
ax_tuning = fig.add_subplot()
tuning = oa.ori_tuning(output_module)
ax_tuning.scatter(alpha_rf, tuning[:, neuron_idx[neural_rank]])
ax_tuning.plot(alpha_rf, tuning[:, neuron_idx[neural_rank]])
ax_tuning.axhline(0, linestyle='--')

plt.show()
