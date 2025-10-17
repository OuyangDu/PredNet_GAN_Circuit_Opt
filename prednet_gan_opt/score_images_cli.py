# Dummy scorer to test the file/CLI bridge (no PredNet yet)
import os, argparse, numpy as np

p = argparse.ArgumentParser()
p.add_argument("--in_batch", required=True)
p.add_argument("--out_csv", required=True)
args = p.parse_args()

arr = np.load(args.in_batch)  # expect [B,3,224,224] in 0..1
if arr.ndim != 4 or arr.shape[1] != 3:
    raise ValueError(f"Expected [B,3,H,W], got {arr.shape}")

scores = arr.mean(axis=(1,2,3)).astype("float32")  # one scalar per image

os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
with open(args.out_csv, "w", newline="") as f:
    f.write("idx,score\n")
    for i, s in enumerate(scores):
        f.write(f"{i},{float(s)}\n")

print(f"Saved {args.out_csv} with {len(scores)} scores.")