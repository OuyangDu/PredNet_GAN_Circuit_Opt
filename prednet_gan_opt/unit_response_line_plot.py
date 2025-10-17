# plot_opt_log.py
# Usage (CLI):
#   python plot_opt_log.py --csv C:\path\to\opt_log.csv
#   python plot_opt_log.py --csv C:\path\to\opt_log.csv --save C:\somewhere\run_plot.png
# Or inside Python:
#   from plot_opt_log import plot_opt_log
#   plot_opt_log(r"C:\path\to\opt_log.csv")  # auto-saves next to the CSV

import os
import csv
import math
import argparse
from typing import Optional, List, Dict

import matplotlib.pyplot as plt

def _read_opt_log(path: str) -> Dict[str, List[float]]:
    """
    Reads an opt_log.csv with header, at least: step,best,median,min
    (extra columns like median,std are ignored).
    Returns dict of lists keyed by column name (lowercased).
    """
    cols: Dict[str, List[float]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError("Empty CSV: no header row found.")

        header_norm = [h.strip().lower() for h in header]
        for h in header_norm:
            cols[h] = []

        for row in reader:
            if not row:
                continue
            for h, val in zip(header_norm, row):
                v = val.strip()
                if h == "step":
                    try:
                        cols[h].append(int(v))
                    except ValueError:
                        cols[h].append(int(float(v)))
                else:
                    try:
                        cols[h].append(float(v))
                    except ValueError:
                        cols[h].append(math.nan)

    required = {"step", "best", "median", "min"}
    if not required.issubset(cols.keys()):
        missing = required - set(cols.keys())
        raise ValueError(f"opt_log missing required columns: {sorted(missing)}")

    return cols


def plot_opt_log(opt_log_path: str, save: Optional[str] = None, show: bool = True, title: Optional[str] = None):
    """
    Plot best/median/min over steps and shade area between min and best.

    Behavior change: If 'save' is None, auto-save next to the CSV as '<csvstem>_plot.png'.
    """
    data = _read_opt_log(opt_log_path)
    x = data["step"]
    y_best = data["best"]
    y_median = data["median"]
    y_min  = data["min"]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, y_best, label="best", linewidth=1, color="tab:blue") 
    ax.plot(x, y_median, label="median", linewidth=1, color="tab:blue") 
    ax.plot(x, y_min, label="min", linewidth=1, color="tab:blue")

    ax.fill_between(x, y_min, y_best, alpha=0.2)

    ax.set_xlabel("step")
    ax.set_ylabel("score")
    if title is None:
        stem = os.path.splitext(os.path.basename(opt_log_path))[0]
        title = f"{stem} — best/median/min"
    ax.set_title(title)

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()

    # --- NEW: default save path = same dir as CSV ---
    if save is None:
        csv_dir  = os.path.dirname(os.path.abspath(opt_log_path))
        csv_stem = os.path.splitext(os.path.basename(opt_log_path))[0]
        save = os.path.join(csv_dir, f"{csv_stem}_plot.png")

    # Ensure parent directory exists (safe even if it already does)
    os.makedirs(os.path.dirname(save), exist_ok=True)

    fig.savefig(save, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _main():
    p = argparse.ArgumentParser(description="Plot best/median/min from opt_log.csv with shaded min↔best area.")
    p.add_argument("--csv", required=True, help="Path to opt_log.csv")
    p.add_argument("--save", default=None, help="Optional output image path; if omitted, auto-saves next to the CSV")
    p.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    p.add_argument("--title", default=None, help="Optional plot title")
    args = p.parse_args()

    plot_opt_log(args.csv, save=args.save, show=not args.no_show, title=args.title)

if __name__ == "__main__":
    _main()