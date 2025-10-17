import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt

from kitti_settings import *

import border_ownership.response_in_para as rip
import border_ownership.border_response_analysis as bra
import border_ownership.orientation_analysis as oa
from border_ownership.prednet_rf_finder import compute_rf_mask
import border_ownership.ploter as ploter

time_cut = [5, 19] # we havn't implemented this but 5 seems a good time cut read from the figure
neural_rank = 0 # select neuron with rank of boi, from largest (0) to smalest
length_square = 50
output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center_length{}.hkl'.format(length_square))
output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center_length{}.hkl'.format(length_square))
label_path = os.path.join(DATA_DIR, 'rotating_square_label_all_length{}.hkl'.format(length_square))
## Full R2 neurons
#output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet.hkl')
#output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet.hkl')
#label_path = os.path.join(DATA_DIR, 'rotating_square_label.hkl')

output_dark = hkl.load(output_dark_path)
output_light = hkl.load(output_light_path)
angle = hkl.load(label_path)['angle']

output_center, rf_para, alpha_rf, beta_rf, gamma_rf = rip.response_in_rf_paramemterization(output_dark, output_light, angle)
#output_center = rip.keep_central_neuron(output_rf) # (alpha, beta, gamma, t, width, height, chs) for pixel, (alpha, beta, gamma, t, chs) for the rest

output_mode = 'E2'
output_module = output_center[output_mode]
output_module = output_module + 1 # avoid negative value

boia = bra.BOI_Analyzer(output_module, alpha_rf, bo_mean_time_init=time_cut[0], bo_mean_time_final=time_cut[1])

# plot boi distribution
neural_boi = boia.neural_boi_dist()
plt.figure()
plt.hist(neural_boi, bins=20)
pref_angle = boia.neural_prefer_boi_angle_dist()
plt.figure()
plt.hist(pref_angle)

# plot neural response
if output_mode[0] == 'A' or output_mode[0] == 'E':
    rf_mask = compute_rf_mask(output_mode, query_neural_id='center')
else:
    rf_mask = None

# add RF mask to output_module_X
fig = ploter.plot_example_neuron_response_and_input(output_center['X'], output_module, boia.alpha_idx[neural_rank], boia.neuron_idx[neural_rank], rf_mask=rf_mask)

time, boi_time_trace = boia.neural_boi_time_trace(neural_rank)
plt.figure()
plt.plot(time, boi_time_trace)

# plot boi as a function of orientation
angle, boi = boia.neural_boi_orientation(neural_rank)
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.scatter(np.deg2rad(angle), boi)
ax.vlines(np.deg2rad(angle), 0, boi)

plt.show()
