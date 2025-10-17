from border_ownership.square_part import Square_Part_Analyzer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import hickle as hkl
from kitti_settings import *

#################### Hyperparameters ####################
module = 'E2'
neural_rank = 10

center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')

################### Main ####################
data = hkl.load(center_info_path)

spa = Square_Part_Analyzer(module, neural_rank, rank_method='BOI')
spa.load_data(data)


fig, ax = spa.plot_sequential_res_on_square(is_zscore=False)
fig.savefig(os.path.join(FIGURE_DIR, 'square_part_res.svg'), format='svg')
fig, ax = spa.plot_sequential_square_part_stim()

res = spa.get_res_change(mode='all_by_name', is_zscore=False)
print(res)
plt.show()
