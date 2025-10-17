import pickle as pkl
import numpy as np
from scipy.stats import ttest_rel, shapiro, wilcoxon
import matplotlib.pyplot as plt

# Load data
with open('E2_rf10_per_neuron_response_pair.pkl', 'rb') as f:
    neuron_response_pairs = pkl.load(f)

conditions = ['Up Dark', 'Down Dark', 'Up Light', 'Down Light']

# Prepare results
results = {cond: {'neuron_ids': [], 'p_values': [], 'normal': [], 'significant': [],
                  'test_used': []} for cond in conditions}

alpha = 0.05  # Significance level

for cond in conditions:
    for neuron_id, records in neuron_response_pairs.items():
        # Filter only for this condition
        filtered = [(entry['a'], entry['b']) for entry in records if entry['condition'] == cond]
        if len(filtered) < 3:
            continue  # Not enough data to run meaningful test

        a_vals, b_vals = zip(*filtered)
        a_vals = np.array(a_vals)
        b_vals = np.array(b_vals)

        # Check normality of differences
        diff = a_vals - b_vals
        stat, p_normal = shapiro(diff)
        is_normal = p_normal > alpha

        # Select appropriate test
        if is_normal:
            t_stat, p_val = ttest_rel(a_vals, b_vals)
            test_used = 't-test'
            is_significant = p_val < alpha
        else:
            try:
                w_stat, p_val = wilcoxon(a_vals, b_vals)
                test_used = 'wilcoxon'
                is_significant = p_val < alpha
            except ValueError:
                p_val = np.nan
                test_used = 'wilcoxon'
                is_significant = False

        results[cond]['neuron_ids'].append(neuron_id)
        results[cond]['p_values'].append(p_val)
        results[cond]['normal'].append(is_normal)
        results[cond]['significant'].append(is_significant)
        results[cond]['test_used'].append(test_used)

# Print summary and normality info
for cond in conditions:
    sig = np.sum(results[cond]['significant'])
    total = len(results[cond]['significant'])
    norm_count = np.sum(results[cond]['normal'])
    print(f"{cond}: {sig}/{total} neurons show significant difference (based on normality + test, alpha={alpha})")
    #print(f"{cond}: {norm_count}/{total} neurons passed normality test")

# Analyze overlap of significant neurons
significant_sets = {
    cond: set([nid for nid, sig in zip(results[cond]['neuron_ids'], results[cond]['significant']) if sig])
    for cond in conditions
}

# Pairwise overlap report
print("\nOverlap of significant neurons across conditions:")
for i in range(len(conditions)):
    for j in range(i + 1, len(conditions)):
        cond1, cond2 = conditions[i], conditions[j]
        overlap = significant_sets[cond1] & significant_sets[cond2]
        print(f"{cond1} ∩ {cond2}: {len(overlap)} neurons overlap")

# Intersection across all four conditions
all_intersection = set.intersection(*significant_sets.values())
print(f"Intersection of all four conditions: {len(all_intersection)} neurons overlap")

# Union of all significant neurons
all_union = set.union(*significant_sets.values())
print(f"Union of all significant neurons across conditions: {len(all_union)} neurons")

# Save union IDs to file
with open('significant_neuron_ids_union.pkl', 'wb') as f:
    pkl.dump(all_union, f)

print("Union of significant neuron IDs saved to 'significant_neuron_ids_union.pkl'")
