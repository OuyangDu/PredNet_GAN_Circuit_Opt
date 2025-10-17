# This script loads per-neuron averaged response pairs to compute mean
# Illusory Contour Index (ICI) for Up Dark and Down Dark conditions,
# loads precomputed Border Ownership Indices (BOI) for each neuron,
# calculates the mean BOI across rectangle sizes per neuron, and generates
# scatter plots: ICI (Up vs Down) and BOI vs ICI (Up & Down), colored by significance.

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

# Load averaged responses
with open('E2_rf10_per_neuron_avg_ab.pkl', 'rb') as f:
    avg_responses = pkl.load(f)

# Load significance map
with open('significant_neurons_dark.pkl', 'rb') as f:
    dark_map = pkl.load(f)

# Compute mean ICI for Up Dark and Down Dark
ici_mean = {}
for nid, conds in avg_responses.items():
    up = np.nan
    dn = np.nan
    if 'Up Dark' in conds:
        a, b = conds['Up Dark']['a_mean'], conds['Up Dark']['b_mean']
        if (a + b) != 0:
            up = (a - b) / (a + b)
    if 'Down Dark' in conds:
        a, b = conds['Down Dark']['a_mean'], conds['Down Dark']['b_mean']
        if (a + b) != 0:
            dn = (a - b) / (a + b)
    ici_mean[nid] = (up, dn)

# Load BOI values
with open('E2_rf10_per_neuron_rect_boi.pkl', 'rb') as f:
    neuron_boi = pkl.load(f)

# Compute mean BOI per neuron
boi_mean = {}
for nid, entries in neuron_boi.items():
    vals = [e['boi'] for e in entries]
    boi_mean[nid] = np.nanmean(vals) if vals else np.nan

# Organize data into significance categories
categories = {cat: {'up': [], 'dn': [], 'boi': []}
              for cat in ['Up only','Down only','Both','Non-significant']}
for nid, (up, dn) in ici_mean.items():
    bm = boi_mean.get(nid, np.nan)
    if np.isnan(up) or np.isnan(dn) or np.isnan(bm):
        continue
    if nid in dark_map:
        passed = dark_map[nid]
        if isinstance(passed, list):   cat = 'Both'
        elif passed == 'Up Dark':     cat = 'Up only'
        elif passed == 'Down Dark':   cat = 'Down only'
        else:                          cat = 'Non-significant'
    else:
        cat = 'Non-significant'
    categories[cat]['up'].append(up)
    categories[cat]['dn'].append(dn)
    categories[cat]['boi'].append(bm)

# Define colors
colors = {
    'Up only': 'tab:blue',
    'Down only': 'tab:orange',
    'Both': 'tab:green',
    'Non-significant': 'grey'
}

# 1) ICI scatter: Up Dark vs Down Dark
plt.figure(figsize=(6,6))
for cat, data in categories.items():
    plt.scatter(data['up'], data['dn'], c=colors[cat], label=cat, alpha=0.7)
plt.xlabel('Up Dark Mean ICI')
plt.ylabel('Down Dark Mean ICI')
plt.title('Mean ICI: Up Dark vs Down Dark')
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()

# 2) BOI vs ICI: side-by-side plots
fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# BOI vs Up Dark ICI
for cat, data in categories.items():
    axs[0].scatter(data['up'], data['boi'], c=colors[cat], label=cat, alpha=0.7)
axs[0].set_xlabel('Up Dark Mean ICI')
axs[0].set_ylabel('Mean BOI')
axs[0].set_title('Mean BOI vs Up Dark Mean ICI')
axs[0].legend()

# BOI vs Down Dark ICI
for cat, data in categories.items():
    axs[1].scatter(data['dn'], data['boi'], c=colors[cat], label=cat, alpha=0.7)
axs[1].set_xlabel('Down Dark Mean ICI')
axs[1].set_title('Mean BOI vs Down Dark Mean ICI')
axs[1].legend()

plt.tight_layout()
plt.show()