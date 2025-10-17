import pickle as pkl
import numpy as np

PATH = "wilcoxon_dark_pvals_and_significance.pkl"

with open(PATH, "rb") as f:
    res = pkl.load(f)

print(f"[info] Loaded {PATH}. Type={type(res).__name__}")

if not isinstance(res, dict) or len(res) == 0:
    print("[warn] Result is not a non-empty dict; nothing to rank.")
    raise SystemExit

example = next(iter(res.values()))
print(f"[info] Example entry type={type(example).__name__}")

def top3_flat(r):
    items = []
    for nid, info in r.items():
        if isinstance(info, dict):
            p = info.get("p", np.nan)
            if np.isfinite(p):
                items.append((nid, float(p)))
    items.sort(key=lambda x: x[1])
    return items[:3]

def top3_conditional(r, cond):
    items = []
    for nid, info in r.items():
        if isinstance(info, dict) and cond in info:
            p = info[cond].get("p", np.nan)
            if np.isfinite(p):
                items.append((nid, float(p)))
    items.sort(key=lambda x: x[1])
    return items[:3]

# Decide shape and print
if isinstance(example, dict) and "p" in example:
    # Flat: single Dark Up-vs-Down test
    t3 = top3_flat(res)
    print("\nTop 3 neurons (Dark Up vs Down):")
    if not t3:
        print("  (no finite p-values found)")
    for nid, p in t3:
        print(f"  neuron {nid}: p = {p:.3e}")
else:
    # Per-condition: print each
    for cond in ["Up Dark", "Down Dark"]:
        t3 = top3_conditional(res, cond)
        print(f"\nTop 3 neurons for {cond}:")
        if not t3:
            print("  (no finite p-values found)")
        for nid, p in t3:
            print(f"  neuron {nid}: p = {p:.3e}")