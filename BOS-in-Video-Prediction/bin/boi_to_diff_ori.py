# plot a single cell response to different edge orientation
# remember to run prednet_to_rotating_sqaure_keep_center to generate prednet response files
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt

from kitti_settings import *

import border_ownership.response_in_para as rip
import border_ownership.border_response_analysis as bra
from border_ownership.prednet_rf_finder import compute_rf_mask
import border_ownership.ploter as ploter

neural_rank = 0 # select neuron with rank of boi, from largest (0) to smalest
bo_mean_time_init = 5; bo_mean_time_final = 20

### trained
output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center_length50.hkl')
output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center_length50.hkl')
label_path = os.path.join(DATA_DIR, 'rotating_square_label_all_length50.hkl')

### untrained
#length_square = 50
#file_name_tail = 'untrain'
#output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center_length{}_{}.hkl'.format(length_square, file_name_tail))
#output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center_length{}_{}.hkl'.format(length_square, file_name_tail))
#label_path = os.path.join(DATA_DIR, 'rotating_square_label_all_length{}_{}.hkl'.format(length_square, file_name_tail))

### untrained small
#length_square = 50
#file_name_tail = 'untrain_small'
#output_dark_path = os.path.join(DATA_DIR, 'rotating_square_dark_prednet_all_center_length{}_{}.hkl'.format(length_square, file_name_tail))
#output_light_path = os.path.join(DATA_DIR, 'rotating_square_light_prednet_all_center_length{}_{}.hkl'.format(length_square, file_name_tail))
#label_path = os.path.join(DATA_DIR, 'rotating_square_label_all_length{}_{}.hkl'.format(length_square, file_name_tail))

output_dark = hkl.load(output_dark_path)
output_light = hkl.load(output_light_path)
angle = hkl.load(label_path)['angle']
output_center, rf_para, alpha_rf, beta_rf, gamma_rf = rip.response_in_rf_paramemterization(output_dark, output_light, angle)
#output_center = rip.keep_central_neuron(output_rf) # (alpha, beta, gamma, t, width, height, chs) for pixel, (alpha, beta, gamma, t, chs) for the rest

output_mode = 'E2'
output_module = output_center[output_mode]
output_module = output_module + 1 # avoid negative value

boia = bra.BOI_Analyzer(output_module, alpha_rf, bo_mean_time_init=bo_mean_time_init, bo_mean_time_final=bo_mean_time_final)

angle, boi = boia.neural_boi_orientation(neural_rank)
print('Rank {} in module {} \nneural id is: {} \nthe preferred angle is {}'.format(neural_rank, output_mode, boia.neuron_idx[neural_rank], angle[np.argmax(boi)]) )
print('Its maximum BOI is {}'.format(np.max(np.abs(boi))))

#if output_mode[0] == 'A' or output_mode[0] == 'E':
#    rf_mask = compute_rf_mask(output_mode, query_neural_id='center')
#else:
#    rf_mask = None
rf_mask = None

for i in range(alpha_rf.shape[0]):
    fig, _, ax_res = ploter.plot_example_neuron_response_and_input(output_center['X'], output_module, i, boia.neuron_idx[neural_rank], rf_mask=rf_mask)
    # neuron_id = 8
    # fig, _, ax_res = ploter.plot_example_neuron_response_and_input(output_center['X'], output_module, i, neuron_id, rf_mask=rf_mask)
    ax_res.set_title('alpha = {}'.format(alpha_rf[i]))
    fig.tight_layout()
    fig.savefig('./figure/' + output_mode + '_ori_' + str(i))
#fig, _, _ = ploter.plot_example_neuron_response_and_input(output_center['X'], output_module, boia.alpha_idx[neural_rank], boia.neuron_idx[neural_rank], rf_mask=rf_mask)
#for i in range(alpha_rf.shape[0]):
#    fig, _, _ = ploter.plot_example_neuron_response_and_input(output_center['X'], output_module, i, 119, rf_mask=rf_mask)
plt.show()

angle, boi = boia.neural_boi_orientation(neural_rank)
fig = plt.figure()
ax = fig.add_subplot(projection='polar')
ax.scatter(np.deg2rad(angle), boi)
ax.vlines(np.deg2rad(angle), 0, boi)
fig.savefig('./figure/' + output_mode + '_ori_boi')

plt.show()
