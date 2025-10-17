# -*- coding: utf-8 -*-
"""
score_images_cli_prednet.py  (NHWC-only + (y, x, ch) indexing)

This scorer:
  1) Reads a batch of images saved as .npy [B, 3, 224, 224] in 0..1
  2) Optional RGB->grayscale (still 3 channels)
  3) Fits each image to PredNet size (128x160) via:
       - center crop ("crop")
       - resize-to-width then center-crop height ("preserve_ar")
  4) Builds a sequence per image: [burn-in gray] + [T repeats/jitters]
  5) Runs PredNet (BOS Agent) and extracts responses
  6) Reduces time, then **expects NHWC** features and indexes **[:, y, x, ch]**
  7) Writes CSV "idx,score"

Usage (PredNet env):
  python score_images_cli_prednet.py ^
    --in_batch expt\batch_0000.npy ^
    --out_csv  expt\scores_prednet_0000.csv ^
    --output_mode R0 --target_y 10 --target_x 5 --target_ch 0 ^
    --pred_h 128 --pred_w 160 --prep_mode crop --to_gray --reduce mean --save_debug
"""
import os
import sys
import json
import argparse
from typing import Optional

import numpy as np
from PIL import Image


# ============================ CLI / argument parsing ===========================

def parse_args():
    p = argparse.ArgumentParser(description="PredNet scorer (Agent) CLI")

    # I/O
    p.add_argument("--in_batch", required=True,
                   help="Path to .npy batch with shape [B,3,224,224], float32 in [0,1].")
    p.add_argument("--out_csv", required=True,
                   help="Where to write scores CSV (format: 'idx,score').")

    # Which PredNet tensor + target unit (order: y, x, ch)
    p.add_argument("--output_mode", default="R0",
                   help="PredNet tensor to read (e.g., R0, E0, prediction).")
    p.add_argument("--target_y",  type=int, default=10, help="Row (y) index.")
    p.add_argument("--target_x",  type=int, default=5,  help="Column (x) index.")
    p.add_argument("--target_ch", type=int, default=0,  help="Channel (ch) index.")

    # PredNet input geometry
    p.add_argument("--pred_h", type=int, default=128, help="PredNet input height.")
    p.add_argument("--pred_w", type=int, default=160, help="PredNet input width.")

    # Temporal
    p.add_argument("--T", type=int, default=8,
                   help="Stimulus frames per sequence (repeats).")
    p.add_argument("--burnin", type=int, default=2,
                   help="Neutral gray warm-up frames.")
    p.add_argument("--reduce", choices=["last","mean","mean_stim","last_stim"],
                   default="mean_stim",
                   help="Time reduction: over all frames or stimulus-only.")
    p.add_argument("--jitter_px", type=int, default=0,
                   help="±pixels of per-frame circular shift (0 disables).")

    # Spatial pre-processing
    p.add_argument("--prep_mode", choices=["crop","preserve_ar"], default="crop",
                   help="'crop' or 'preserve_ar' aspect strategy.")
    p.add_argument("--to_gray", action="store_true",
                   help="Convert RGB→grayscale, then replicate to 3 channels.")

    # BOS repo + model files
    p.add_argument("--bos_repo",
                   default=r"C:\Users\ThinkPad\Documents\Chisel\BOS-in-Video-Prediction",
                   help="Path to BOS-in-Video-Prediction repo.")
    p.add_argument("--json_path", default=None,
                   help="Optional path to prednet_kitti_model.json.")
    p.add_argument("--weights_path", default=None,
                   help="Optional path to prednet_kitti_weights.hdf5.")

    # Debug
    p.add_argument("--save_debug", action="store_true",
                   help="Write *_debug.json with shapes/settings.")

    return p.parse_args()


# ============================== Image preprocessing ===========================

def rgb_to_grayscale_bhwc(imgs_bhwc: np.ndarray) -> np.ndarray:
    gray = (0.299 * imgs_bhwc[..., 0] +
            0.587 * imgs_bhwc[..., 1] +
            0.114 * imgs_bhwc[..., 2]).astype(np.float32)
    gray = np.clip(gray, 0.0, 1.0)
    return np.repeat(gray[..., None], 3, axis=-1)


def center_crop_batch(imgs_bhwc: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    B, H, W, C = imgs_bhwc.shape
    if H < out_h or W < out_w:
        raise ValueError(f"Cannot center-crop {out_h}x{out_w} from {H}x{W}.")
    top  = (H - out_h) // 2
    left = (W - out_w) // 2
    return imgs_bhwc[:, top:top+out_h, left:left+out_w, :]


def resize_width_then_center_crop_height(imgs_bhwc: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    B, H, W, C = imgs_bhwc.shape
    frames_uint8 = (imgs_bhwc * 255.0).astype(np.uint8)
    out_list = []
    for i in range(B):
        im = Image.fromarray(frames_uint8[i])
        new_h = int(round(H * (out_w / float(W))))
        im = im.resize((out_w, new_h), resample=Image.BILINEAR)
        top = max(0, (new_h - out_h) // 2)
        im = im.crop((0, top, out_w, top + out_h))
        out_list.append(np.asarray(im, dtype=np.uint8))
    out = np.stack(out_list, axis=0).astype(np.float32) / 255.0
    return out


def preprocess_images(imgs_bhwc: np.ndarray, out_h: int, out_w: int, mode: str) -> np.ndarray:
    if mode == "crop":
        return center_crop_batch(imgs_bhwc, out_h, out_w)
    elif mode == "preserve_ar":
        return resize_width_then_center_crop_height(imgs_bhwc, out_h, out_w)
    else:
        raise ValueError(f"Unknown --prep_mode '{mode}'.")


def make_sequence(imgs_bhwc: np.ndarray, T: int, burnin: int, jitter_px: int = 0) -> np.ndarray:
    B, H, W, C = imgs_bhwc.shape
    gray = np.full((B, burnin, H, W, C), 0.5, dtype=np.float32)
    seq  = np.tile(imgs_bhwc[:, None, ...], (1, T, 1, 1, 1)).astype(np.float32)
    if jitter_px > 0:
        for b in range(B):
            for t in range(T):
                dy = np.random.randint(-jitter_px, jitter_px + 1)
                dx = np.random.randint(-jitter_px, jitter_px + 1)
                seq[b, t] = np.roll(np.roll(seq[b, t], dy, axis=0), dx, axis=1)
    return np.concatenate([gray, seq], axis=1)


# ========================== PredNet (Agent) construction ======================

def build_agent(bos_repo: str, json_path: Optional[str], weights_path: Optional[str]):
    sys.path.append(bos_repo)
    from border_ownership.agent import Agent

    if json_path is None:
        json_path = os.path.join(bos_repo, "model_data_keras2", "prednet_kitti_model.json")
    if weights_path is None:
        weights_path = os.path.join(bos_repo, "model_data_keras2", "tensorflow_weights", "prednet_kitti_weights.hdf5")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing PredNet JSON at: {json_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing PredNet weights at: {weights_path}")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    agent = Agent()
    agent.read_from_json(json_path, weights_path)
    return agent


# ======================== Time reduction (no layout changes) ===================

def reduce_over_time(feats: np.ndarray, T_total: int, burnin: int, how: str) -> np.ndarray:
    """
    Collapse time along the axis whose size equals T_total (= burnin + T).
    If no such axis exists, feats is returned unchanged.
    NOTE: This function DOES NOT permute channel/layout. NHWC is expected downstream.
    """
    if feats.ndim < 4:
        return feats

    t_axes = [i for i, s in enumerate(feats.shape) if s == T_total]
    if not t_axes:
        return feats
    t_ax = t_axes[0]

    def slice_time(x, start, end):
        sl = [slice(None)] * x.ndim
        sl[t_ax] = slice(start, end)
        return x[tuple(sl)]

    if how == "last":
        sl = [slice(None)] * feats.ndim
        sl[t_ax] = -1
        return feats[tuple(sl)]
    elif how == "mean":
        return feats.mean(axis=t_ax)
    elif how == "last_stim":
        sl = [slice(None)] * feats.ndim
        sl[t_ax] = T_total - 1
        return feats[tuple(sl)]
    elif how == "mean_stim":
        stim = slice_time(feats, burnin, T_total)
        return stim.mean(axis=t_ax)
    else:
        raise ValueError(f"Unknown reduce mode: {how}")


# ============================== NHWC-only indexing ============================

def index_scores_nhwc(feats_nhwc: np.ndarray, y: int, x: int, ch: int) -> np.ndarray:
    """
    Strict NHWC indexing with (y, x, ch). No conversion is attempted.
    Expects feats_nhwc shape: [B, H, W, C].
    """
    if feats_nhwc.ndim != 4:
        raise ValueError(f"Expected 4D NHWC features, got {feats_nhwc.shape}")
    B, H, W, C = feats_nhwc.shape
    if not (0 <= y < H and 0 <= x < W and 0 <= ch < C):
        raise IndexError(f"(y,x,ch)=({y},{x},{ch}) out of bounds for [B,H,W,C]={[B,H,W,C]}")
    return feats_nhwc[:, y, x, ch]


# ==================================== main ====================================

def main():
    args = parse_args()

    # 1) Load input batch and convert to BHWC (for preproc only)
    batch = np.load(args.in_batch)
    if batch.ndim != 4 or batch.shape[1] != 3:
        raise ValueError(f"Expected [B,3,H,W], got {batch.shape}")
    imgs = np.clip(batch, 0.0, 1.0).transpose(0, 2, 3, 1).astype(np.float32)  # NCHW -> NHWC (images only)

    if args.to_gray:
        imgs = rgb_to_grayscale_bhwc(imgs)

    # 2) Spatial fit to PredNet size (still NHWC images)
    imgs = preprocess_images(imgs, args.pred_h, args.pred_w, args.prep_mode)  # [B,H,W,3]

    # 3) Build sequences
    seq = make_sequence(imgs, T=args.T, burnin=args.burnin, jitter_px=args.jitter_px)
    T_total = args.T + args.burnin

    # 4) Run PredNet
    agent = build_agent(args.bos_repo, args.json_path, args.weights_path)
    out = agent.output_multiple(seq, output_mode=[args.output_mode], is_upscaled=False)
    feats = out[args.output_mode]  # may include time axis; layout provided by Agent

    # 5) Reduce time (NO layout conversion; NHWC REQUIRED from Agent)
    feats_reduced = reduce_over_time(np.array(feats), T_total, args.burnin, args.reduce)

    # 6) Index (y, x, ch) under NHWC-only assumption
    scores = index_scores_nhwc(feats_reduced, args.target_y, args.target_x, args.target_ch).astype(np.float32)

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        f.write("idx,score\n")
        for i, s in enumerate(scores):
            f.write(f"{i},{float(s)}\n")

    # Optional debug
    if args.save_debug:
        dbg = {
            "in_batch": args.in_batch,
            "out_csv": args.out_csv,
            "bos_repo": args.bos_repo,
            "json_path": args.json_path,
            "weights_path": args.weights_path,
            "output_mode": args.output_mode,
            "pred_input_hw": [args.pred_h, args.pred_w],
            "prep_mode": args.prep_mode,
            "to_gray": bool(args.to_gray),
            "T": args.T, "burnin": args.burnin, "T_total": T_total,
            "reduce": args.reduce,
            "jitter_px": args.jitter_px,
            "target": {"y": args.target_y, "x": args.target_x, "ch": args.target_ch},
            "feats_shape": tuple(np.array(feats).shape),
            "feats_reduced_shape": tuple(np.array(feats_reduced).shape),
            "index_order": "[:, y, x, ch]",
            "nhwc_only": True
        }
        with open(os.path.splitext(args.out_csv)[0] + "_debug.json", "w") as g:
            json.dump(dbg, g, indent=2)

    print(f"Saved {args.out_csv} with {len(scores)} scores.")


if __name__ == "__main__":
    main()