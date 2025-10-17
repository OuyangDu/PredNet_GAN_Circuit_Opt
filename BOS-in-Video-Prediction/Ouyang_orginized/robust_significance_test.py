# wilcoxon_dark_from_pairs.py
import pickle as pkl
import numpy as np
from scipy.stats import wilcoxon

PAIRS_PATH = 'E2_rf10_per_neuron_response_pair.pkl'     # produced by your new script
OUT_PATH   = 'wilcoxon_dark_pvals_and_significance.pkl'
CONDS      = ['Up Dark', 'Down Dark']                   # test these conditions
ALPHA      = 0.05
MIN_PAIRS  = 3

with open(PAIRS_PATH, 'rb') as f:
    neuron_pairs = pkl.load(f)

result_map = {}  # nid -> {cond: {'p', 'significant', 'n_pairs'}, '_summary': {...}}

for nid, records in neuron_pairs.items():
    per_neuron = {}
    sig_labels = []

    for cond in CONDS:
        # collect a/b across sizes for this condition
        a_list, b_list = [], []
        for r in records:
            if r.get('condition') == cond:
                a = float(r.get('a', np.nan))
                b = float(r.get('b', np.nan))
                if np.isfinite(a) and np.isfinite(b):
                    a_list.append(a)
                    b_list.append(b)

        a_vec = np.asarray(a_list, dtype=float)
        b_vec = np.asarray(b_list, dtype=float)

        # ensure paired, drop any residual NaNs (defensive)
        m = (~np.isnan(a_vec)) & (~np.isnan(b_vec))
        a_use, b_use = a_vec[m], b_vec[m]
        n_pairs = int(a_use.size)

        if n_pairs < MIN_PAIRS:
            per_neuron[cond] = {'p': float('nan'), 'significant': False, 'n_pairs': n_pairs}
            continue

        diffs = a_use - b_use
        if np.allclose(diffs, 0):
            pval = 1.0
        else:
            try:
                _, pval = wilcoxon(a_use, b_use, alternative='two-sided', zero_method='pratt')
            except Exception:
                _, pval = wilcoxon(a_use, b_use)

        sig = bool(np.isfinite(pval) and pval < ALPHA)
        if sig:
            sig_labels.append(cond)

        per_neuron[cond] = {'p': float(pval), 'significant': sig, 'n_pairs': n_pairs}

    # summary label compatible with your existing plotting categories
    if len(sig_labels) == 1:
        labels_for_plot = sig_labels[0]                         # 'Up Dark' or 'Down Dark'
    elif len(sig_labels) >= 2:
        labels_for_plot = [c for c in CONDS if c in sig_labels] # ['Up Dark','Down Dark']
    else:
        labels_for_plot = None

    # only store neurons that had at least one dark condition processed
    if any(c in per_neuron for c in CONDS):
        per_neuron['_summary'] = {'labels_for_plot': labels_for_plot}
        result_map[nid] = per_neuron

# save results
with open(OUT_PATH, 'wb') as f:
    pkl.dump(result_map, f)

# print counts
sig_counts   = {c: 0 for c in CONDS}
tested_counts= {c: 0 for c in CONDS}
both_sig = 0
any_sig  = 0
for nid, info in result_map.items():
    per_sig = []
    for cond in CONDS:
        if cond in info:
            if info[cond]['n_pairs'] >= MIN_PAIRS and np.isfinite(info[cond]['p']):
                tested_counts[cond] += 1
            if info[cond]['significant']:
                sig_counts[cond] += 1
                per_sig.append(cond)
    if per_sig:
        any_sig += 1
    if all(info.get(c, {}).get('significant', False) for c in CONDS):
        both_sig += 1

print(f"Saved Wilcoxon results to: {OUT_PATH}")
for cond in CONDS:
    print(f"{cond}: {sig_counts[cond]} significant (out of {tested_counts[cond]} tested)")
print(f"Both conditions significant: {both_sig}")
print(f"Any condition significant:  {any_sig}")