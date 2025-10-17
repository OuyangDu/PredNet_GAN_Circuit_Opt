# Uses NEW files:
#   - E2_rf10_per_neuron_response_pair.pkl  (has per-size a/b for each condition)
#   - wilcoxon_dark_pvals_and_significance.pkl  (aka "result_map"; produced by your Wilcoxon script)
#
# Still uses:
#   - E2_rf10_per_neuron_rect_boi.pkl  (unchanged)
#
# What changes:
#   1) Compute a_mean/b_mean directly from the pairs file (per neuron & condition)
#   2) Compute ICI per neuron from those means: (a_mean - b_mean) / (a_mean + b_mean)
#   3) Build significance categories from the Wilcoxon result_map file

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

# ---------- Load NEW a/b pairs ----------
with open('E2_rf10_per_neuron_response_pair.pkl', 'rb') as f:
    pairs = pkl.load(f)  # { nid: [ {condition, a, b, width, height, r, ...}, ... ] }

# ---------- Load NEW Wilcoxon result_map ----------
# Primary expected name produced by earlier script:
RESULT_MAP_CANDIDATES = [
    'wilcoxon_dark_pvals_and_significance.pkl',
    'result_map.pkl',  # fallback if you saved under this name
]
result_map = None
for path in RESULT_MAP_CANDIDATES:
    try:
        with open(path, 'rb') as f:
            result_map = pkl.load(f)
            break
    except FileNotFoundError:
        continue
if result_map is None:
    raise FileNotFoundError(
        "Could not find Wilcoxon result map. Looked for:\n" +
        "\n".join(f"  - {p}" for p in RESULT_MAP_CANDIDATES)
    )

# ---------- Compute mean a/b and ICI per neuron for Up/Down Dark ----------
conds_of_interest = ['Up Dark', 'Down Dark']

ici_mean = {}  # nid -> (ICI_up_dark, ICI_down_dark)
a_mean  = {}  # nid -> (a_up_mean, a_dn_mean)
b_mean  = {}  # nid -> (b_up_mean, b_dn_mean)

for nid, recs in pairs.items():
    means = {}
    for cond in conds_of_interest:
        a_vals = [float(r['a']) for r in recs
                  if r.get('condition') == cond and np.isfinite(r.get('a', np.nan)) and np.isfinite(r.get('b', np.nan))]
        b_vals = [float(r['b']) for r in recs
                  if r.get('condition') == cond and np.isfinite(r.get('a', np.nan)) and np.isfinite(r.get('b', np.nan))]
        if len(a_vals) > 0 and len(b_vals) > 0:
            a_m = float(np.nanmean(a_vals))
            b_m = float(np.nanmean(b_vals))
            denom = a_m + b_m
            ici  = (a_m - b_m) / denom if denom != 0 else 0.0
        else:
            a_m, b_m, ici = np.nan, np.nan, np.nan
        means[cond] = (a_m, b_m, ici)

    a_up,  b_up,  ici_up = means['Up Dark']
    a_dn,  b_dn,  ici_dn = means['Down Dark']
    a_mean[nid] = (a_up, a_dn)
    b_mean[nid] = (b_up, b_dn)
    ici_mean[nid] = (ici_up, ici_dn)

# ---------- Load BOI (unchanged) ----------
with open('E2_rf10_per_neuron_rect_boi.pkl', 'rb') as f:
    neuron_boi = pkl.load(f)

boi_mean = {}
for nid, entries in neuron_boi.items():
    vals = [e.get('boi', np.nan) for e in entries]
    boi_mean[nid] = np.nanmean(vals) if len(vals) > 0 else np.nan

# ---------- Build significance categories from result_map ----------
# result_map structure (per your Wilcoxon script):
#   result_map[nid] = {
#       'Up Dark':   {'p': float, 'significant': bool, 'n_pairs': int},
#       'Down Dark': {'p': float, 'significant': bool, 'n_pairs': int},
#       '_summary':  {'labels_for_plot': 'Up Dark' | 'Down Dark' | ['Up Dark','Down Dark'] | None}
#   }
def labels_from_result(info_for_nid):
    # Prefer the summary if present
    lbl = None
    if isinstance(info_for_nid, dict):
        summ = info_for_nid.get('_summary', {})
        lbl = summ.get('labels_for_plot', None)
        if lbl is not None:
            return lbl
        # Otherwise, derive from flags
        up_sig   = bool(info_for_nid.get('Up Dark',   {}).get('significant', False))
        down_sig = bool(info_for_nid.get('Down Dark', {}).get('significant', False))
        if up_sig and down_sig:
            return ['Up Dark', 'Down Dark']
        elif up_sig:
            return 'Up Dark'
        elif down_sig:
            return 'Down Dark'
    return None

categories = {cat: {'up': [], 'dn': [], 'boi': []}
              for cat in ['Up only', 'Down only', 'Both', 'Non-significant']}

for nid, (up, dn) in ici_mean.items():
    bm = boi_mean.get(nid, np.nan)
    if np.isnan(up) or np.isnan(dn) or np.isnan(bm):
        continue

    # Map to category based on Wilcoxon result_map
    info = result_map.get(nid, None)
    lbl = labels_from_result(info)
    if isinstance(lbl, list):               cat = 'Both'
    elif lbl == 'Up Dark':                  cat = 'Up only'
    elif lbl == 'Down Dark':                cat = 'Down only'
    else:                                   cat = 'Non-significant'

    categories[cat]['up'].append(up)
    categories[cat]['dn'].append(dn)
    categories[cat]['boi'].append(bm)

# ---------- Plotting (unchanged) ----------
colors = {
    'Up only': 'tab:blue',
    'Down only': 'tab:orange',
    'Both': 'tab:green',
    'Non-significant': 'grey'
}

# 1) ICI scatter: Up Dark vs Down Dark
plt.figure(figsize=(6, 6))
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