# pred_net_check.py — unified checker for PNG (overlay+score) and NPY batch (score)
# STRICT (y, x, ch) ordering throughout; no fallback indexing.
import os
import sys
import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Any
from pathlib import Path

# ========================= Import BOS repo (Agent) =========================
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')

FALLBACK_REPO = Path(r"C:\Users\ThinkPad\Documents\Chisel\BOS-in-Video-Prediction")

_here = Path(__file__).resolve().parent
candidates = [FALLBACK_REPO, _here, _here.parent, _here.parent.parent]

added = None
for pth in candidates:
    pkg = pth / "border_ownership"
    if pkg.exists() and (pkg / "agent.py").exists():
        sys.path.insert(0, str(pth))
        added = pth
        break

if added is None:
    print("[import] Could not locate 'border_ownership' package.")
    for pth in candidates: print("   -", pth)
    raise SystemExit(1)
else:
    print(f"[import] Added to sys.path: {added}")

from border_ownership.agent import Agent  # noqa: E402


# ============================ Viz utilities ============================
def _extent_for_origin(H: int, W: int, origin: str):
    if origin == 'upper':
        return (-0.5, W - 0.5, H - 0.5, -0.5)
    elif origin == 'lower':
        return (-0.5, W - 0.5, -0.5, H - 0.5)
    else:
        raise ValueError("origin must be 'upper' or 'lower'")


def show_tight_boundary(binary_img, img_bkgrd, origin='upper', color='red', lw=1,
                        save: Optional[str] = None, verbose: bool = True):
    M = np.asarray(binary_img)
    B = np.asarray(img_bkgrd)

    if verbose:
        print(f"[overlay] mask dtype/shape: {M.dtype}, {M.shape}")
        print(f"[overlay] bkgrd dtype/shape: {B.dtype}, {B.shape}")

    if M.ndim == 3:
        M = M.any(axis=-1)
    M = M.astype(bool)

    if M.shape != B.shape[:2]:
        if verbose:
            print(f"[overlay] resizing mask {M.shape} → {B.shape[:2]} (nearest)")
        pil_mask = Image.fromarray(M.astype(np.uint8) * 255)
        Ht, Wd = B.shape[:2]
        pil_mask = pil_mask.resize((Wd, Ht), resample=Image.NEAREST)
        M = np.array(pil_mask) >= 128

    H, W = B.shape[:2]
    extent = _extent_for_origin(H, W, origin)

    fig, ax = plt.subplots()
    cmap = 'gray' if B.ndim == 2 else None
    ax.imshow(B, origin=origin, interpolation='nearest', extent=extent, cmap=cmap)

    if M.any():
        ax.contour(M.astype(float), levels=[0.5], colors=[color], linewidths=lw,
                   origin=origin, extent=extent)

    ax.axis('off')

    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True) if os.path.dirname(save) else None
        if os.path.isdir(save):
            save = os.path.join(save, "overlay.png")
        fig.savefig(save, dpi=200, bbox_inches='tight', pad_inches=0)
        print(f"[overlay] saved → {save}")

    if not plt.isinteractive():
        plt.show()
    plt.close(fig)


# ============================ MATCH scorer pre/post ============================
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
    frames_uint8 = np.clip(imgs_bhwc * 255.0, 0, 255).astype(np.uint8)
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
        raise ValueError(f"Unknown --prep_mode '{mode}'")


def make_sequence(imgs_bhwc: np.ndarray, T: int, burnin: int) -> np.ndarray:
    """Return [B, burnin+T, H, W, 3] float32 in [0,1] with exact 0.5 burn-in."""
    B, H, W, C = imgs_bhwc.shape
    gray = np.full((B, burnin, H, W, C), 0.5, dtype=np.float32)
    seq  = np.tile(imgs_bhwc[:, None, ...], (1, T, 1, 1, 1)).astype(np.float32)
    return np.concatenate([gray, seq], axis=1)


def reduce_over_time(feats: np.ndarray, T_total: int, burnin: int, how: str) -> np.ndarray:
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


# ---------- Layout enforcement to NHWC (so we can index strictly by y,x,ch) ----------
def ensure_nhwc(feats: np.ndarray) -> np.ndarray:
    """
    Ensure 4D features are in [B, H, W, C] (NHWC). If they appear to be [B, C, H, W],
    transpose to NHWC. This enables strict (y, x, ch) indexing with no fallback.
    """
    if feats.ndim != 4:
        raise ValueError(f"[layout] Expected feats with 4 dims [B,*,*,*], got {feats.shape}")
    B, A, B2, C_or_W = feats.shape  # not used; just to visualize dims while coding

    # Heuristic: if the last dimension is much larger than the second dimension,
    # assume current is NHWC. Otherwise, if the second dimension is small (typical C)
    # and the last dimension looks like a spatial W, we consider it NCHW and transpose.
    # Safer practical rule: if the second dim is <= 512 and the last dim is > 512, treat as NCHW.
    # But PredNet feature maps won't have W > 512 at these sizes, so we'll do a simpler check:
    # if last dim < second dim, likely NCHW -> transpose.
    if feats.shape[-1] < feats.shape[1]:
        # NCHW -> NHWC
        feats = np.transpose(feats, (0, 2, 3, 1))
    # else assume already NHWC
    return feats


def index_scores(feats_nhwc: np.ndarray, y: int, x: int, ch: int) -> np.ndarray:
    """
    STRICT (y, x, ch) indexing.
    Expects feats in NHWC: [B, H, W, C]. No fallback path.
    """
    if feats_nhwc.ndim != 4:
        raise ValueError(f"[index] Expected NHWC [B,H,W,C], got {feats_nhwc.shape}")
    return feats_nhwc[:, y, x, ch]


def _normalize_layer_name(layer: str) -> str:
    layer = str(layer).strip().strip("'").strip('"').upper()
    if layer in {'E0', 'E1', 'E2', 'E3'}:
        return layer
    if layer.isdigit():
        i = int(layer)
        if 0 <= i <= 3:
            return f"E{i}"
    raise ValueError("Layer must be E0/E1/E2/E3 or 0..3.")


def find_neuron_index(bo_info_layer: dict, y: int, x: int, ch: int) -> int:
    """Find index for a neuron specified strictly by (y, x, ch)."""
    neuron_ids = bo_info_layer['neuron_id']
    if hasattr(neuron_ids, 'iloc'):
        getter = lambda i: neuron_ids.iloc[i]
        n = len(neuron_ids)
    else:
        getter = lambda i: neuron_ids[i]
        n = len(neuron_ids)
    target = (int(y), int(x), int(ch))
    for i in range(n):
        nid = getter(i)
        if tuple(nid) == target:
            return i
    raise ValueError(f"Neuron (y={y}, x={x}, ch={ch}) not found.")


def load_prednet_agent(weights_dir: str) -> Agent:
    wd = Path(weights_dir)
    if wd.is_file(): wd = wd.parent
    json_file = wd / 'prednet_kitti_model.json'
    weights_file = wd / 'tensorflow_weights' / 'prednet_kitti_weights.hdf5'
    if not json_file.exists() or not weights_file.exists():
        raise FileNotFoundError(
            "[prednet] Missing model files.\n"
            f"  JSON:    {json_file}\n"
            f"  Weights: {weights_file}\n"
        )
    print(f"[prednet] json: {json_file}\n[prednet] weights: {weights_file}")
    agent = Agent()
    agent.read_from_json(str(json_file), str(weights_file))
    print("[prednet] agent ready.")
    return agent


# ============================ Main ============================
def main():
    p = argparse.ArgumentParser(description="Check PredNet unit response using PNG (overlay+score) and/or NPY batch (score). Strict (y,x,ch).")
    # Inputs (one or both)
    p.add_argument("--pnf", default=None, help="Path to image.png (e.g., best_####.png). If given: overlay + score 1 image.")
    p.add_argument("--in_batch", default=None, help="Path to .npy with shape [B,3,224,224] in [0,1]. If given: score batch.")

    # RF data and layer/unit
    p.add_argument("--data", required=True, help="Path to center_neuron_info_radius10.pkl")
    p.add_argument("--layer", required=True, help="Layer (E0/E1/E2/E3 or 0..3)")
    p.add_argument("--target_id_one",   type=int, required=True, help="Neuron y index")
    p.add_argument("--target_id_two",   type=int, required=True, help="Neuron x index")
    p.add_argument("--target_id_three", type=int, required=True, help="Neuron channel index (ch)")

    # Overlay
    p.add_argument("--save_overlay", default=None, help="Save overlay PNG path or directory (only if --pnf provided)")
    p.add_argument("--origin", default="upper", choices=["upper", "lower"])
    p.add_argument("--lw", type=float, default=1.0)
    p.add_argument("--color", default="red")
    p.add_argument("--no_display", action="store_true")

    # PredNet runtime — MATCH SCORER DEFAULTS
    p.add_argument("--T", type=int, default=8, help="Stimulus frames")
    p.add_argument("--burnin", type=int, default=2, help="Burn-in frames (gray 0.5)")
    p.add_argument("--reduce", choices=["last","mean","mean_stim","last_stim"],
                   default="mean_stim", help="Time reduction policy")

    # Spatial prep — MATCH SCORER DEFAULTS
    p.add_argument("--pred_h", type=int, default=128, help="PredNet input height")
    p.add_argument("--pred_w", type=int, default=160, help="PredNet input width")
    p.add_argument("--prep_mode", choices=["crop","preserve_ar"], default="crop")
    p.add_argument("--to_gray", action="store_true", help="Convert RGB→grayscale then replicate to 3ch")

    # Weights dir
    p.add_argument("--weights_dir",
                   default=r"C:\Users\ThinkPad\Documents\Chisel\BOS-in-Video-Prediction\model_data_keras2",
                   help="Directory containing prednet_kitti_model.json and tensorflow_weights/...")

    # Optional CSV output for batch scores (default: auto-generate next to in_batch)
    p.add_argument("--out_csv_batch", default=None,
                   help="If set and --in_batch is given, write idx,score CSV to this path. "
                        "If not set, will auto-write <in_batch>_scores.csv")

    args = p.parse_args()

    if args.pnf is None and args.in_batch is None:
        raise SystemExit("Provide at least one input: --pnf <png> and/or --in_batch <npy>")

    # Strictly bind CLI → (y, x, ch)
    y = int(args.target_id_one)
    x = int(args.target_id_two)
    ch = int(args.target_id_three)
    layer = _normalize_layer_name(args.layer)
    print(f"[unit] (y, x, ch) = ({y}, {x}, {ch})  layer = {layer}")

    # Load RF info for overlay
    with open(args.data, 'rb') as f:
        data = pkl.load(f)
    bo_layer = data['bo_info'][layer]
    idx = find_neuron_index(bo_layer, y, x, ch)
    rf_mask = np.asarray(bo_layer['rf'][idx]).astype(bool)
    print(f"[neuron] matched rf index: {idx} for (y={y}, x={x}, ch={ch})")

    # PredNet
    agent = load_prednet_agent(args.weights_dir)
    T_total = args.T + args.burnin

    # =================== PNG path (overlay + single-image score) ===================
    if args.pnf is not None:
        print(f"[png] {args.pnf}")
        img = Image.open(args.pnf).convert('RGB')
        img_u8 = np.asarray(img)  # H×W×3 uint8

        # Overlay
        save_path = args.save_overlay
        if save_path is None:
            root, ext = os.path.splitext(args.pnf)
            save_path = root + "_overlay.png"
        elif os.path.isdir(save_path):
            base = os.path.splitext(os.path.basename(args.pnf))[0]
            save_path = os.path.join(save_path, f"{base}_overlay.png")
        show_tight_boundary(rf_mask, img_u8, origin=args.origin, color=args.color, lw=args.lw,
                            save=save_path, verbose=True)
        if args.no_display:
            print("[overlay] no_display set")
        # --- also save an RF-only contour image (mask over itself) ---
        # Build an output path alongside the first overlay
        if args.save_overlay is None:
            root, ext = os.path.splitext(args.pnf)
            save_path_rf = root + "_overlay_rf.png"
        elif os.path.isdir(args.save_overlay):
            base = os.path.splitext(os.path.basename(args.pnf))[0]
            save_path_rf = os.path.join(args.save_overlay, f"{base}_overlay_rf.png")
        else:
        # If user passed a specific file path, append "_rf" before the extension
            base_no_ext, ext = os.path.splitext(args.save_overlay)
            save_path_rf = base_no_ext + "_rf" + ext

        # Plot the contour on the RF mask itself
        show_tight_boundary(
                            rf_mask,                 # contour mask
                            rf_mask,                 # background = mask (rf-only view)
                            origin=args.origin,
                            color=args.color,
                            lw=args.lw,
                            save=save_path_rf,
                            verbose=True,
                          )

        # Score single image with scorer-identical prep
        img_f = (img_u8.astype(np.float32) / 255.0)[None, ...]  # [1,H,W,3], [0,1]
        if args.to_gray:
            img_f = rgb_to_grayscale_bhwc(img_f)
        img_f = preprocess_images(img_f, args.pred_h, args.pred_w, args.prep_mode)  # -> [1,Hw,Ww,3]
        seq = make_sequence(img_f, T=args.T, burnin=args.burnin)                     # -> [1,T_total,Hw,Ww,3]
        print(f"[png] seq shape={seq.shape}, minmax=({seq.min():.3f},{seq.max():.3f})")

        out = agent.output_multiple(seq, output_mode=[layer], is_upscaled=False)
        feats = np.array(out[layer])
        feats_reduced = reduce_over_time(feats, T_total=T_total, burnin=args.burnin, how=args.reduce)
        feats_nhwc = ensure_nhwc(feats_reduced)  # enforce NHWC
        png_score = float(index_scores(feats_nhwc, y=y, x=x, ch=ch)[0])  # STRICT (y,x,ch)
        print(f"[png result] reduce={args.reduce}  score={png_score:.6f}")

    # =================== NPY batch path (score) ===================
    if args.in_batch is not None:
        print(f"[batch] {args.in_batch}")
        batch = np.load(args.in_batch)  # [B,3,224,224] in [0,1]
        if batch.ndim != 4 or batch.shape[1] != 3:
            raise ValueError(f"[batch] Expected shape [B,3,H,W], got {batch.shape}")

        # NCHW -> BHWC; clamp and ensure float32
        imgs = np.clip(batch, 0.0, 1.0).transpose(0, 2, 3, 1).astype(np.float32)  # [B,224,224,3]
        if args.to_gray:
            imgs = rgb_to_grayscale_bhwc(imgs)
        imgs = preprocess_images(imgs, args.pred_h, args.pred_w, args.prep_mode)  # [B,Hw,Ww,3]
        seq  = make_sequence(imgs, T=args.T, burnin=args.burnin)                  # [B,T_total,Hw,Ww,3]
        print(f"[batch] seq shape={seq.shape}, minmax=({seq.min():.3f},{seq.max():.3f})")

        out   = agent.output_multiple(seq, output_mode=[layer], is_upscaled=False)
        feats = np.array(out[layer])
        feats_reduced = reduce_over_time(feats, T_total=T_total, burnin=args.burnin, how=args.reduce)
        feats_nhwc = ensure_nhwc(feats_reduced)  # enforce NHWC
        scores = index_scores(feats_nhwc, y=y, x=x, ch=ch).astype(np.float32)  # [B] STRICT (y,x,ch)

        # Print summary
        Bn = scores.shape[0]
        print(f"[batch result] N={Bn}  reduce={args.reduce}")
        print(f"[batch result] mean={scores.mean():.6f}  std={scores.std():.6f}  "
              f"min={scores.min():.6f}  max={scores.max():.6f}")
        if Bn <= 16:
            print("[batch scores]", scores.tolist())
        else:
            head = ", ".join(f"{v:.4f}" for v in scores[:8])
            tail = ", ".join(f"{v:.4f}" for v in scores[-8:])
            print(f"[batch scores] head: [{head}]  ...  tail: [{tail}]")

        # Always write CSV (default next to in_batch if not provided)
        if args.out_csv_batch:
            out_csv = args.out_csv_batch
        else:
            root, _ = os.path.splitext(args.in_batch)
            out_csv = root + "_scores.csv"

        os.makedirs(os.path.dirname(out_csv), exist_ok=True) if os.path.dirname(out_csv) else None
        with open(out_csv, "w", newline="") as f:
            f.write("idx,score\n")
            for i, s in enumerate(scores):
                f.write(f"{i},{float(s)}\n")
        print(f"[batch] wrote CSV → {out_csv}")


if __name__ == "__main__":
    main()