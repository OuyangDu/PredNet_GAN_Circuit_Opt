# !!! This algorithm will remove units who never reach half max response. This can be due to the neuron doesn't response at all, or the half-life is longer than observation time (16 steps).
import hickle as hkl
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, f_oneway, kruskal
from border_ownership.persistent_ploter import format_axis
import seaborn as sns
import os
from collections import defaultdict
from border_ownership.ploter import plot_layer_boxplot_helper, plot_layer_violin_helper
from border_ownership.fig_off_func import *
from kitti_settings import *

#################### Hyperparameters ####################
module_list = ['R0', 'R1', 'R2', 'R3', 'E0', 'E1', 'E2', 'E3']
data_save_path = os.path.join(DATA_DIR_HOME, 'pers_figoff_res_diff.hkl')
keys = ['s', 'sags', 'sgrat', 'spgs']
new_labels = ['Square-Ambiguous', 'Ambiguous-Off', 'Grating-Off', 'Pixel-Off']
layer_order = {'s': 0, 'sags': 1, 'sgrat': 2, 'spgs': 3}
color_palette = {'s': 'tab:red', 'sags': 'tab:blue', 'sgrat': 'tab:green', 'spgs': 'tab:purple'}
test_method = ''

#################### Main ####################
data = hkl.load(data_save_path)
time = data['time']
switch_time = data['switch_time']

##################### compute half-life for each unit and each stimulus
half_life = defaultdict(dict)
for module in module_list:
    for key in keys:
        res_s = data[key][module][:, switch_time:]
        half_life[module][key] = find_half_life(res_s)
    half_life[module] = remove_none_columns(half_life[module])

#################### half-life distribution for all modules
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
axes = axes.flatten()
for i, module in enumerate(module_list):
    ax = axes[i]
    n_units = len(half_life[module]['s'])
    fig, ax = plot_layer_boxplot_helper(half_life[module], layer_order, fig=fig, ax=ax, jitter=0.1, jitter_s=30, jitter_alpha=0.4, jitter_color=color_palette, color=(0.3, 0.3, 0.3))
    if i == 0: is_first_ax = True
    else: is_first_ax = False
    if i in [4, 5, 6, 7]: is_last_row = True
    else: is_last_row = False
    format_axis(ax, title='{} \n n = {}'.format(module, n_units), xlabel= 'Stimulus Type', ylabel='Half Life Time', is_first_ax=is_first_ax, is_last_row=is_last_row, sci_ytick=False)

    if is_last_row: ax.set_xticklabels(new_labels, fontsize=12, rotation=45)
    else: ax.set_xticklabels([])
fig.savefig(os.path.join(FIGURE_DIR, 'half_life_time.svg'), bbox_inches='tight')

#################### half-life distribution for E2 only
module = 'E2'
fig, ax = plt.subplots(figsize=(3, 3))
fig, ax = plot_layer_violin_helper(half_life[module], layer_order, fig=fig, ax=ax, color_palette=color_palette)
ax.set_xlabel('Stimulus Type')
ax.set_ylabel('Half Life Time')
ax.set_xticklabels(new_labels, fontsize=12, rotation=45)
fig.savefig(os.path.join(FIGURE_DIR, 'half_life_time_E2.svg'), bbox_inches='tight')

#################### compute p-matrix for each module
p_mat_dict = {}
for i, module in enumerate(module_list):
    # Generate all unique pairs of keys for comparison
    hlm = half_life[module]
    p_mat_module = compute_p_matrix(hlm, layer_order, test_method='wilcoxon')
    p_mat_dict[module] = p_mat_module

#################### draw pmat for all modules
fig, axes = plt.subplots(2, 4, figsize=(12, 5))
for i, module in enumerate(module_list):
    ax = axes[i // 4, i % 4]
    n_units = len(half_life[module]['s'])
    fig, ax = draw_lower_triangle_heatmap(p_mat_dict[module], new_labels, fig=fig, ax=ax, cbar_kws={'label': ''}, heatmap_fmt='.1e', annot_kws={'fontsize': 7})
    if i == 0: is_first_ax = True
    else: is_first_ax = False
    if i in [4, 5, 6, 7]: is_last_row = True
    else: is_last_row = False
    format_axis(ax, title='{} \n n = {}'.format(module, n_units), xlabel= 'Stimulus Type', ylabel='p-value', is_first_ax=is_first_ax, is_last_row=is_last_row)
    ax.set_ylabel('')
    ax.set_xlabel('')

    ax.set_yticklabels([])
    colorbar = ax.collections[0].colorbar
    colorbar.outline.set_visible(False)  # Hides the colorbar outline
    colorbar.ax.set_title('') # Removes the title of the colorbar

    if is_last_row: ax.set_xticklabels(new_labels, fontsize=12, rotation=45)
    else: ax.set_xticklabels([])
fig.savefig(os.path.join(FIGURE_DIR, 'half_life_time_p_mat.svg'))

#################### p_mat for E2 only
module = 'E2'
p_mat = p_mat_dict[module]
fig, ax = plt.subplots(figsize=(3, 3))
fig, ax = draw_lower_triangle_heatmap(p_mat, new_labels, fig=fig, ax=ax)
fig.savefig(os.path.join(FIGURE_DIR, 'half_life_time_p_mat_{}.svg'.format(module)), bbox_inches='tight')

#################### Compute the average half-life for each module, and pvalues within each module
figoff_stim = ['sags', 'sgrat', 'spgs']
color_palette = {'s': 'tab:red', 'figoff': 'tab:blue'}

half_life_avg_figoff = defaultdict(dict)
# average figure-off which includes sags, sgrat and spgs
for module in module_list:
    half_life_avg_figoff[module]['s'] = half_life[module]['s']
    half_life_avg_figoff[module]['figoff'] = np.nanmean([half_life[module][stim] for stim in figoff_stim], axis=0)

### compute the wilcoxon test on each module
p_dict = {}

for module, types in half_life_avg_figoff.items():
    _, p_value = wilcoxon(types['s'], types['figoff'])
    p_dict[module] = p_value

print('\n')
print('p_value within each module comparing square_ambiguous to figoff: \n {}'.format(p_dict))

##### Half-life across modules in a single plot
# make the x label which includes the module name, stars indicating the significance level and the number of units
x_labels_half_life = []
for module in module_list:
    module_str = module
    signif_str = get_significance_str(p_dict[module])
    n_units = len(half_life_avg_figoff[module]['s'])
    x_label = '{} \n{} \n{}'.format(module_str, signif_str, n_units)
    x_labels_half_life.append(x_label)

df = half_life_avg_figoff_to_df(half_life_avg_figoff)

# start plotting
fig, ax = plt.subplots(figsize=(7, 3))

sns.stripplot(x='module', y='value', hue='type', data=df, dodge=True, palette=color_palette, jitter=True, ax=ax, alpha=0.6, size=4)
sns.boxplot(x='module', y='value', hue='type', data=df, ax=ax, showcaps=False,
            boxprops={'facecolor': 'None'}, flierprops={'visible': False},
            palette=color_palette, linewidth=1.5)
ax.set_xticklabels(x_labels_half_life)
ax.legend_.remove()

# Customize legend with new names
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=['Square-Ambiguous', 'Figure-Off'], title="", fontsize=12)
ax.set_ylabel('Half-Life')
fig.savefig(os.path.join(FIGURE_DIR, 'half_life_time_across_modules.svg'))

#################### compute whether different modules have different half-life
for key in ['s', 'figoff']:
    half_life_key = {module: half_life_avg_figoff[module][key] for module in module_list}
    del half_life_key['R3']
    # del half_life_key['R0']
    # del half_life_key['R1']
    # del half_life_key['R2']
    value_key = list(half_life_key.values())
    _, p_value_kruskal = kruskal(*value_key)
    _, p_value_foneway = f_oneway(*value_key)
    print('p_value_{} across different modules: kruskal = {}; f_oneway = {}'.format(key, p_value_kruskal, p_value_foneway))

plt.show()
