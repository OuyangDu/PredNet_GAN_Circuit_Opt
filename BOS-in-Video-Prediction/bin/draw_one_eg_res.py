
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt
import border_ownership.neuron_info_processor as nip
from border_ownership.border_response_analysis import compute_BO
from matplotlib.patches import Rectangle
from border_ownership.util import sci_notation
from kitti_settings import *
import os

# #################### Hyperparameters ####################
module = 'E2'
neural_rank = 13 # E2, -186: non-BOS unit shown in bottom of Figure 1D. E2, 13: BOS unit shown in top of Figure 1D
center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')

# #################### Main ####################
data = hkl.load(center_info_path)
bo_info, res_info, stim_info, unique_orientation = data['bo_info'], data['res_info'], data['stim_info'], data['unique_orientation']

nipor = nip.Neuron_Info_Processor()
nipor.load_data(bo_info, res_info, stim_info, module, unique_orientation)
if neural_rank <= 0:
    neural_rank = nipor.bo_infom.shape[0] + neural_rank
neuron_bo_info, neuron_res_info = nipor.get_target_neuron_info(neural_rank)

### get the preferred orientation of the neuron
pref_ori = nip.get_preferred_orientation(neuron_bo_info)

### draw the four squares
four_square, zmap = nip.get_stimulus_and_rf(stim_info, neuron_bo_info, target_ori=pref_ori)
height, width = four_square[0].shape[:2]

four_color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, ax = plt.subplots(2, 2, figsize=(2, 2))
ax = ax.flatten()
for i in range(4):
    ax[i].imshow(four_square[i])
    ax[i].axis('off')
    ax[i].add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor=four_color[i], lw=6))
    ax[i].contour(zmap, levels=[1], colors='white', linestyles='solid', linewidths=1)
fig.savefig(os.path.join(FIGURE_DIR, 'four_square.pdf'), bbox_inches='tight')

### response
trial_id, response = nip.get_response_to_orientation(neuron_res_info, pref_ori)
fig, ax = plt.subplots(figsize=(3, 3))
time = np.arange(response[0].shape[0])
for res in response:
    ax.plot(time, res)
ax.set_xlabel('Time')
ax.set_ylabel('Unit activation')

### Compute BOI at time steps, BOI without shift +1, ratio of mean response to non-preferred BOS over the means response to preferred BOS
print(response.shape)
res_module = [[response[0], response[1]], [response[2], response[3]]]
res_module = np.array(res_module)[None, :, :, :, None]
boi = compute_BO(res_module) # BOI as a function of time
print('BOI:', boi)
res_module -= 1  # BOI wihout shift +1
boi_no_shift = compute_BO(res_module)
print('BOI without shift +1:', boi_no_shift)
boi_ratio = (response[2] + response[3] - 2).mean() / (response[0] + response[1] - 2).mean()
print('BOI ratio:', boi_ratio)
exit()

bav = neuron_bo_info['bav'].iloc[0]
bav_pvalue = neuron_bo_info['bav_pvalue'].iloc[0]
bav_str = r'$B_{av}$'
ax.text(0.5, 0.8, f'{bav_str}: {sci_notation(bav)} \n p = {bav_pvalue}', ha='center', va='center', fontsize=12, transform=ax.transAxes)
fig.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, 'response.svg'), bbox_inches='tight')
plt.show()

print(neuron_bo_info['bav'])
print(neuron_bo_info['bav_pvalue'])
# plt.figure(figsize=(3, 3))
# plt.hist(bo_info[module]['boi_abs_max'], bins=50)
# plt.show()
