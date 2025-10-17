import os
import hickle as hkl
import pprint
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from border_ownership.ablation_visu import get_mse
from border_ownership.ploter import plot_layer_boxplot_helper, plot_layer_violin_helper
from border_ownership.ablation_stat import Ablation_Data_Reader, avg_video_num_unit_boot, compute_pdf
from kitti_settings import *

########################################
# module_list = ['E0', 'E1', 'E2', 'E3']
# module_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
module_list = ['E0']
random_id = [0, 1, 2, 3, 4, 5] # random_id for random initialization
# adr = Ablation_Data_Reader('square', random_id)
adr = Ablation_Data_Reader('random', random_id)
pdf = {}

for module in module_list:
    avg_data_temp = avg_video_num_unit_boot(adr, module, shuffle_is_bo=False) # average across video_resample_id and unit_resample_id. avg_data_temp should have three keys, n_units is the number of units ablated, True_rpmse is the relative prediction MSE for BO ablation, False_rpmse is the relative prediction MSE for non-BO ablation. The shape of n_units, True_rpmse and False_rpmse is [n_units]
    pdf[module] = compute_pdf(avg_data_temp['True_rpmse'], avg_data_temp['False_rpmse'])

pdf_shuffle = defaultdict(list)
for _ in range(100):
    for module in module_list:
        avg_data_temp = avg_video_num_unit_boot(adr, module, shuffle_is_bo=False) # average across video_resample_id and unit_resample_id. avg_data_temp should have three keys, n_units is the number of units ablated, True_rpmse is the relative prediction MSE for BO ablation, False_rpmse is the relative prediction MSE for non-BO ablation. The shape of n_units, True_rpmse and False_rpmse is [n_units]
        pdf[module] = compute_pdf(avg_data_temp['True_rpmse'], avg_data_temp['False_rpmse'])

    pdf_shuffle = defaultdict(list)
    for _ in range(n_shuffle):
        for module in module_list:
            avg_data_temp = avg_video_num_unit_boot(adr, module, shuffle_is_bo=True)
            pdf_shuffle_temp = compute_pdf(avg_data_temp['True_rpmse'], avg_data_temp['False_rpmse'])
            pdf_shuffle[module].append(pdf_shuffle_temp)
    return pdf, pdf_shuffle

def compute_quantile_and_p_value(numbers_list, single_number, tail_mode='two-tailed'):
    quantile = np.sum(np.array(numbers_list) <= single_number) / len(numbers_list)

    if tail_mode == 'two-tailed':
        p_value = 2 * min(quantile, 1 - quantile)
    elif tail_mode == 'greater':
        p_value = 1 - quantile
    elif tail_mode == 'less':
        p_value = quantile
    else:
        raise ValueError("tail_mode should be 'two-tailed', 'greater', or 'less'.")

    return quantile, p_value

########################################
n_shuffle = 10
# module_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
module_list = ['E0', 'E1', 'E2', 'E3']
# module_list = ['R0', 'R1', 'R2', 'R3']
random_id = [0, 1, 2, 3, 4, 5, 6] # random_id for random initialization
data_head_list = ['square', 'trans', 'random', 'kitti']
data_head_xlabel = ['Static', 'Trans', 'Random', 'KITTI']
data_save_path = os.path.join(RESULTS_SAVE_DIR, 'ablation_pdf.hkl')

# data = defaultdict(dict)
# for dh in data_head_list:
#     adr = Ablation_Data_Reader(dh, random_id)
#     pdf, pdf_shuffle = get_pdf_and_shuffle(module_list, adr, n_shuffle=n_shuffle)
#     data[dh]['pdf'] = dict(pdf)
#     data[dh]['pdf_shuffle'] = dict(pdf_shuffle)

# data = dict(data)
# hkl.dump(data, data_save_path)

data = hkl.load(data_save_path)

font_size = 12
plt.rcParams.update({'font.size': font_size})
plt.rcParams['axes.labelsize'] = font_size  # Adjust the 14 to your desired font size
plt.rcParams['xtick.labelsize'] = font_size  # X-axis tick label size
plt.rcParams['ytick.labelsize'] = font_size  # Y-axis tick label size

n_module = len(module_list)
p_value = {}
fig, axes = plt.subplots(1, n_module, figsize=(n_module * 2 + 2, 2))
# for mi, module in enumerate(module_list):
for mi, key in enumerate(data_head_list):
    pdf_shuffle = {module: data[key]['pdf_shuffle'][module] for module in module_list}
    pdf = {module: data[key]['pdf'][module] for module in module_list}

    ########## compute p value
    p_value[key] = {}
    for module in module_list:
        quantile, p_value_temp = compute_quantile_and_p_value(pdf_shuffle[module], pdf[module], tail_mode='two-tailed')
        p_value[key][module] = p_value_temp

    ########## plot
    layer_order = {module: i for i, module in enumerate(module_list)}
    plot_layer_violin_helper(pdf_shuffle, layer_order=layer_order, ax=axes[mi], fig=fig, color_palette=['lightgray'])
    axes[mi].scatter(np.arange(len(module_list)), [pdf[module] for module in module_list], color='r', s=50, label='True', zorder=3)
    axes[mi].hlines(0, -0.5, 3.5, color='k', linestyle='-')

    axes[mi].set_title(data_head_xlabel[mi])
    # axes[mi].set_xticks(np.arange(len(module_list)))
    # axes[mi].set_xticklabels(data_head_xlabel, rotation=45)
    if mi == 0:
        axes[mi].set_ylabel('PDF')
    else:
        axes[mi].set_ylabel('')
    axes[mi].set_xlabel('')

########## print p value
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(p_value)

fig.savefig(os.path.join(RESULTS_SAVE_DIR, 'ablation_pdf_test.svg'), bbox_inches='tight')
plt.show()
