import pickle as pkl
import numpy as np
from scipy.stats import mannwhitneyu

# Load data
with open('E2_rf10_per_neuron_response_pair.pkl', 'rb') as f:
    neuron_response_pairs = pkl.load(f)

conditions = ['Up Dark', 'Down Dark', 'Up Light', 'Down Light']

# Prepare results
results = {
    cond: {
        'neuron_ids': [],
        'p_values': [],
        'significant': []
    }
    for cond in conditions
}

alpha = 0.05  # Significance level

for cond in conditions:
    for neuron_id, records in neuron_response_pairs.items():
        # Filter only for this condition
        filtered = [(entry['a'], entry['b']) for entry in records if entry['condition'] == cond]
        if len(filtered) < 3:
            continue  # Not enough data

        a_vals, b_vals = map(np.array, zip(*filtered))

        # Mann–Whitney U (Wilcoxon rank-sum) test, but catch identical-data errors
        try:
            stat, p_val = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
            is_significant = p_val < alpha
        except ValueError:
            # raised when all values are identical
            p_val = np.nan
            is_significant = False

        # Store results
        results[cond]['neuron_ids'].append(neuron_id)
        results[cond]['p_values'].append(p_val)
        results[cond]['significant'].append(is_significant)

# Print per‐condition significance summary
for cond in conditions:
    total = len(results[cond]['neuron_ids'])
    sig_count = np.sum(results[cond]['significant'])
    print(f"{cond}: {sig_count}/{total} neurons passed (p < {alpha})")

# Build sets of significant neurons per condition
significant_sets = {
    cond: set(nid for nid, sig in zip(results[cond]['neuron_ids'], results[cond]['significant']) if sig)
    for cond in conditions
}

# --- Save dark conditions results ---
dark_conds = ['Up Dark', 'Down Dark']
dark_union = set.union(*[significant_sets[c] for c in dark_conds])
dark_results = {}
for nid in dark_union:
    passed = [c for c in dark_conds if nid in significant_sets[c]]
    dark_results[nid] = passed if len(passed) > 1 else passed[0]

with open('significant_neurons_dark.pkl', 'wb') as f:
    pkl.dump(dark_results, f)
print(f"Saved {len(dark_results)} dark‐condition neurons to 'significant_neurons_dark.pkl'")

# --- Save light conditions results ---
light_conds = ['Up Light', 'Down Light']
light_union = set.union(*[significant_sets[c] for c in light_conds])
light_results = {}
for nid in light_union:
    passed = [c for c in light_conds if nid in significant_sets[c]]
    light_results[nid] = passed if len(passed) > 1 else passed[0]

with open('significant_neurons_light.pkl', 'wb') as f:
    pkl.dump(light_results, f)
print(f"Saved {len(light_results)} light‐condition neurons to 'significant_neurons_light.pkl'")