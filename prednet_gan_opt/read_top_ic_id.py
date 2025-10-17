#!/usr/bin/env python3
"""
read_top_ic_id.py

Reads E2_rf10_per_neuron_response_pair.pkl and prints the top-3 neuron IDs
(for condition "Down Dark") where 'significant' is True. If p-values exist,
results are sorted by ascending p-value; otherwise original order is used.

Usage (default path/condition):
  python read_top_ic_id.py

List available conditions in the PKL:
  python read_top_ic_id.py --list

Verbose debugging (prints why nothing matched; for Schema A also prints the 'significant' array):
  python read_top_ic_id.py --debug

Override anything:
  python read_top_ic_id.py --pkl "D:\\some\\other\\path.pkl" --condition "Down Dark" --top 5
"""

import argparse
import os
import pickle
from typing import Any, Dict, Iterable, List, Tuple

# ---------- Config: default path ----------
DEFAULT_PKL = r"C:\Users\ThinkPad\Documents\Chisel\BOS-in-Video-Prediction\Ouyang_tests\E2_rf10_per_neuron_response_pair.pkl"


# ---------- Utilities ----------
def load_pickle(pkl_path: str):
    with open(pkl_path, "rb") as f:
        try:
            data = pickle.load(f)
        except Exception:
            f.seek(0)
            data = pickle.load(f, encoding="latin1")
    return data

def _as_float_or_inf(x: Any) -> float:
    try:
        v = float(x)
        if not (v == v) or v is None:  # NaN -> inf
            return float("inf")
        return v
    except Exception:
        return float("inf")

def _truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    return s in ("1", "true", "t", "yes", "y")

def _norm(s: Any) -> str:
    """Normalize condition label: lowercase, remove non-alnum (spaces/_/ - all ignored)."""
    if s is None:
        return ""
    s = str(s).lower()
    return "".join(ch for ch in s if ch.isalnum())

def _get_pval_from_dict(d: Dict[str, Any]) -> float:
    for k in ("p_value", "pval", "p", "p_val"):
        if k in d:
            return _as_float_or_inf(d[k])
    return float("inf")


# ---------- Condition discovery (for --list / debugging) ----------
def discover_conditions(data: Any) -> List[str]:
    conds = set()

    # Schema A: results[cond] = {...}
    if isinstance(data, dict):
        hit = False
        for k, v in data.items():
            if isinstance(v, dict) and ("neuron_ids" in v or "significant" in v or "p_values" in v):
                conds.add(str(k))
                hit = True
        if hit:
            return sorted(conds)

    # Schema B: { neuron_id: [records...] }
    if isinstance(data, dict):
        for _, records in data.items():
            if isinstance(records, list):
                for r in records:
                    if isinstance(r, dict) and "condition" in r:
                        conds.add(str(r.get("condition")))
        if conds:
            return sorted(conds)

    # Schema C: [ {neuron_id, condition, ...}, ... ]
    if isinstance(data, list):
        for r in data:
            if isinstance(r, dict) and "condition" in r:
                conds.add(str(r.get("condition")))
        return sorted(conds)

    return sorted(conds)


def map_condition_key(data: Any, wanted: str) -> str:
    """Return the actual condition key in Schema A that best matches 'wanted' by normalized comparison."""
    if not isinstance(data, dict):
        return wanted
    wanted_norm = _norm(wanted)
    for key in data.keys():
        if _norm(key) == wanted_norm:
            return key
    return wanted


# ---------- Core extraction ----------
def extract_significant_ids(data: Any, condition: str, debug: bool = False) -> List[Tuple[Any, float]]:
    out: List[Tuple[Any, float]] = []
    cond_norm = _norm(condition)

    # ----- Schema A: results[cond] with parallel lists -----
    if isinstance(data, dict):
        cond_key = map_condition_key(data, condition)
        block = data.get(cond_key)
        if isinstance(block, dict) and ("neuron_ids" in block or "significant" in block):
            neuron_ids = block.get("neuron_ids", [])
            significant = block.get("significant", [])
            p_values = block.get("p_values", block.get("pvals", [float("inf")] * len(neuron_ids)))
            L = min(len(neuron_ids), len(significant), len(p_values))

            if debug:
                # Print Schema A debug, including the 'significant' array as requested
                try:
                    sig_list_preview = list(significant[:50]) if hasattr(significant, "__getitem__") else list(significant)
                except Exception:
                    sig_list_preview = list(significant)
                true_count = sum(_truthy(x) for x in significant) if hasattr(significant, "__iter__") else 0
                print(f"[DEBUG] Schema A block found for key='{cond_key}'. Keys: {list(block.keys())}")
                print(f"[DEBUG] neuron_ids: len={len(neuron_ids)} | p_values: len={len(p_values)} | significant: type={type(significant).__name__}, len={len(significant)}")
                print(f"[DEBUG] significant (first 50): {sig_list_preview}")
                print(f"[DEBUG] significant TRUE count: {true_count}")

            for i in range(L):
                if _truthy(significant[i]):
                    out.append((neuron_ids[i], _as_float_or_inf(p_values[i])))

            if debug:
                print(f"[DEBUG] Schema A kept {len(out)} significant entries after filtering.")
            return out  # return even if empty (definitive for Schema A)

    # ----- Schema B: { neuron_id: [records...] } -----
    if isinstance(data, dict):
        total_records = 0
        cond_match = 0
        sig_values_seen = set()
        sample_rows = []

        for neuron_id, records in data.items():
            if isinstance(records, list):
                for r in records:
                    if not isinstance(r, dict):
                        continue
                    total_records += 1
                    if _norm(r.get("condition")) == cond_norm:
                        cond_match += 1
                        sig_values_seen.add(str(r.get("significant")))
                        if _truthy(r.get("significant")):
                            out.append((neuron_id, _get_pval_from_dict(r)))
                            if len(sample_rows) < 5:
                                sample_rows.append(dict(neuron_id=neuron_id, **{k: r.get(k) for k in ("condition","significant","p_value")}))

        if debug:
            print(f"[DEBUG] Schema B: total_records={total_records}, condition_matches={cond_match}, collected={len(out)}")
            print(f"[DEBUG] Schema B: raw 'significant' values seen for matching condition: {sorted(sig_values_seen)}")
            if sample_rows:
                print(f"[DEBUG] Schema B: sample of up to 5 collected rows: {sample_rows[:5]}")

        if out:
            return out

    # ----- Schema C: [ {neuron_id, condition, significant, p_value, ...}, ... ] -----
    if isinstance(data, list):
        for row in data:
            if not isinstance(row, dict):
                continue
            if _norm(row.get("condition")) == cond_norm and _truthy(row.get("significant")):
                nid = row.get("neuron_id")
                pval = _get_pval_from_dict(row)
                out.append((nid, pval))
        if debug:
            print(f"[DEBUG] Schema C: collected {len(out)} rows with significant==True for condition '{condition}'.")
        return out

    return out


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Print top-N significant neuron IDs for a condition from a results PKL.")
    ap.add_argument("--pkl", default=DEFAULT_PKL, help=f"Path to PKL file (default: {DEFAULT_PKL})")
    ap.add_argument("--condition", default="Down Dark", help='Condition to filter (default "Down Dark")')
    ap.add_argument("--top", type=int, default=3, help="How many IDs to print (default 3)")
    ap.add_argument("--list", action="store_true", help="List available conditions and exit.")
    ap.add_argument("--debug", action="store_true", help="Print debug info; for Schema A prints the full 'significant' array preview.")
    args = ap.parse_args()

    if not os.path.isfile(args.pkl):
        raise FileNotFoundError(f"Pickle not found: {args.pkl}")

    data = load_pickle(args.pkl)

    if args.list:
        conds = discover_conditions(data)
        if not conds:
            print("(no condition labels found in this PKL)")
        else:
            print("Available conditions:")
            for c in conds:
                print(" -", c)
        return

    rows = extract_significant_ids(data, args.condition, debug=args.debug)
    # Sort by p-value (inf -> end)
    rows.sort(key=lambda tup: tup[1])

    # Print only neuron IDs
    top_rows = rows[: args.top]
    if not top_rows:
        if args.debug:
            print(f"[DEBUG] No significant entries found for condition '{args.condition}'.")
            print("[DEBUG] Try --list to see available condition labels.")
        return

    for nid, _ in top_rows:
        print(nid)


if __name__ == "__main__":
    main()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Print top-N significant neuron IDs for a condition from a results PKL.")
    ap.add_argument("--pkl", default=DEFAULT_PKL, help=f"Path to PKL file (default: {DEFAULT_PKL})")
    ap.add_argument("--condition", default="Down Dark", help='Condition to filter (default "Down Dark")')
    ap.add_argument("--top", type=int, default=3, help="How many IDs to print (default 3)")
    ap.add_argument("--list", action="store_true", help="List available conditions and exit.")
    ap.add_argument("--debug", action="store_true", help="Print debug info; for Schema A prints the full 'significant' array preview.")
    args = ap.parse_args()

    if not os.path.isfile(args.pkl):
        raise FileNotFoundError(f"Pickle not found: {args.pkl}")

    data = load_pickle(args.pkl)

    if args.list:
        conds = discover_conditions(data)
        if not conds:
            print("(no condition labels found in this PKL)")
        else:
            print("Available conditions:")
            for c in conds:
                print(" -", c)
        return

    rows = extract_significant_ids(data, args.condition, debug=args.debug)
    # Sort by p-value (inf -> end)
    rows.sort(key=lambda tup: tup[1])

    # Print only neuron IDs
    top_rows = rows[: args.top]
    if not top_rows:
        if args.debug:
            print(f"[DEBUG] No significant entries found for condition '{args.condition}'.")
            print("[DEBUG] Try --list to see available condition labels.")
        return

    for nid, _ in top_rows:
        print(nid)


if __name__ == "__main__":
    main()