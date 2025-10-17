# -*- coding: utf-8 -*-
r"""
optimize_prednet_unit_cli.py — Phase 3 (GAN -> PredNet scorer loop)
Now supports --gan {fc6,biggan}. For BigGAN we use circuit_toolkit.utils.GAN_utils.BigGAN_wrapper.

Run (examples):
  # fc6 (default)
  python optimize_prednet_unit_cli.py --mode test --steps 3 --batch 40

  # BigGAN: optimize both noise+class embedding (256-D total)
  python optimize_prednet_unit_cli.py --gan biggan --steps 60 --batch 28 --optimizer auto

  # BigGAN: optimize noise only
  python optimize_prednet_unit_cli.py --gan biggan --part noise --steps 60 --batch 28

  # BigGAN: optimize class embedding only
  python optimize_prednet_unit_cli.py --gan biggan --part class --steps 60 --batch 28
"""

import os, csv, subprocess, datetime, math, argparse
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pytorch_pretrained_biggan import BigGAN as PTBigGAN  # add this import

# -------------------- Toolkit imports (GAN + optional CMA-ES) --------------------

from circuit_toolkit.GAN_utils import upconvGAN
try:
    # Optional BigGAN wrapper (only needed when --gan biggan)
    from circuit_toolkit.GAN_utils import loadBigGAN, BigGAN_wrapper
    HAS_BIGGAN = True
except Exception:
    HAS_BIGGAN = False

try:
    from circuit_toolkit.Optimizers import CholeskyCMAES
    HAS_CMA = True
except Exception:
    HAS_CMA = False

try:
    from circuit_toolkit.Optimizers import HessCMAES
    HAS_HESS = True
except Exception:
    HAS_HESS = False

# ========================= Defaults (override via CLI) =========================
GLUE_DIR   = r"C:\Users\ThinkPad\Documents\Chisel\prednet_gan_opt"
SCORER     = os.path.join(GLUE_DIR, "score_images_cli_prednet.py")
PRED_ENV   = "border_ownership"

# PredNet scorer knobs
OUTPUT_MODE= "E2"
TARGET_Y, TARGET_X, TARGET_CH = 16, 21, 5
PRED_H, PRED_W = 128, 160
PREP_MODE  = "crop"            # 'crop' or 'preserve_ar'
TO_GRAY    = False             # default: COLOR (override with --gray)
REDUCE     = "mean_stim"
T_FRAMES   = 8
BURNIN     = 2
JITTER_PX  = 0

# GAN + search (defaults; may be overridden by --gan and other CLI flags)
DEFAULT_GAN   = "fc6"          # or "biggan"
DEFAULT_PART  = "all"          # for biggan: {all, noise, class}
LATENT_DIM    = 4096           # fc6; BigGAN will switch to 256
BATCH         = 40
STEPS         = 3
INIT_SIGMA    = 3.0            # fc6 default (BigGAN will auto reduce)
TOPK          = 4
TRUNCATION    = 0.7            # BigGAN default truncation for sampling

SEED: Optional[int] = None     # e.g., 123

# =============================== CLI parsing ===============================
def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 optimizer loop (GAN -> PredNet scorer).")
    p.add_argument("--mode", choices=["run","test"], default="run",
                   help="test = verbose; run = quieter.")
    p.add_argument("--gan", choices=["fc6","biggan"], default=DEFAULT_GAN,
                   help="Which generator to use.")
    p.add_argument("--part", choices=["all","noise","class"], default=DEFAULT_PART,
                   help="BigGAN only: optimize both halves (all) or only noise/class embedding.")
    p.add_argument("--trunc", type=float, default=TRUNCATION,
                   help="BigGAN truncation (ψ); ignored for fc6.")
    p.add_argument("--steps", type=int, default=STEPS, help="Number of optimization steps.")
    p.add_argument("--batch", type=int, default=BATCH, help="Population size per step.")
    # Scorer knobs
    p.add_argument("--output_mode", default=OUTPUT_MODE)
    p.add_argument("--target_y",  type=int, default=TARGET_Y)
    p.add_argument("--target_x",  type=int, default=TARGET_X)
    p.add_argument("--target_ch", type=int, default=TARGET_CH)
    p.add_argument("--prep_mode", choices=["crop","preserve_ar"], default=PREP_MODE)
    p.add_argument("--reduce", default=REDUCE, choices=["mean","last","mean_stim","last_stim"])
    p.add_argument("--burnin", type=int, default=BURNIN)
    p.add_argument("--tframes", type=int, default=T_FRAMES)
    p.add_argument("--jitter_px", type=int, default=JITTER_PX)
    # Gray/color
    group = p.add_mutually_exclusive_group()
    group.add_argument("--gray",  action="store_true",  help="Force grayscale->3ch.")
    group.add_argument("--color", action="store_true",  help="Force color (no grayscale).")
    # Opt
    p.add_argument("--seed", type=int, default=SEED if SEED is not None else -1)
    p.add_argument("--optimizer", choices=["auto","cholesky","hessian","fallback"],
                   default="auto",
                   help="Optimizer choice. 'auto' prefers HessCMAES, then CholeskyCMAES, else fallback.")
    # Expert: override init sigma
    p.add_argument("--init_sigma", type=float, default=-1.0,
                   help="Override initial sigma. If <0, use per-GAN default.")
    return p.parse_args()

# =============================== Utils ===============================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def make_run_dir(base: str) -> str:
    ts = datetime.datetime.now().strftime("run_%Y-%m-%d_%H%M")
    run = os.path.join(base, ts); ensure_dir(run); return run

def to_uint8_img(tchw: torch.Tensor) -> Image.Image:
    t = tchw.detach().cpu().clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)

def save_tensor_grid(imgs_0to1: torch.Tensor, path: str, nrow: Optional[int] = None):
    imgs = imgs_0to1.detach().cpu().clamp(0, 1)
    B, C, H, W = imgs.shape
    if nrow is None: nrow = int(math.ceil(math.sqrt(B)))
    ncol = int(math.ceil(B / nrow))
    canvas = Image.new("RGB", (nrow * W, ncol * H))
    k = 0
    for j in range(ncol):
        for i in range(nrow):
            if k >= B: break
            arr = (imgs[k].permute(1, 2, 0).numpy() * 255.0).astype('uint8')
            canvas.paste(Image.fromarray(arr), (i * W, j * H))
            k += 1
    canvas.save(path)

def fit_batch_to_224(imgs: torch.Tensor) -> torch.Tensor:
    B, C, H, W = imgs.shape
    if (H, W) == (224, 224): return imgs
    side = min(H, W); top = (H-side)//2; left = (W-side)//2
    cropped = imgs[:, :, top:top+side, left:left+side]
    return F.interpolate(cropped, size=(224, 224), mode="bilinear", align_corners=False)

def prep_for_prednet_preview_torch(
    imgs_0to1: torch.Tensor, pred_h: int, pred_w: int,
    prep_mode: str = "crop", to_gray: bool = False
) -> torch.Tensor:
    x = imgs_0to1.detach().clone().clamp(0, 1)
    B, C, H, W = x.shape
    if to_gray:
        w = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1,3,1,1)
        gray = (x * w).sum(dim=1, keepdim=True)
        x = gray.expand(-1, 3, -1, -1)
    if prep_mode == "crop":
        top  = max(0, (H - pred_h) // 2); left = max(0, (W - pred_w) // 2)
        x = x[:, :, top:top+pred_h, left:left+pred_w]
    elif prep_mode == "preserve_ar":
        new_h = int(round(H * (pred_w / float(W))))
        x = F.interpolate(x, size=(new_h, pred_w), mode="bilinear", align_corners=False)
        top = max(0, (new_h - pred_h) // 2)
        x = x[:, :, top:top+pred_h, :]
    else:
        raise ValueError(f"Unknown prep_mode '{prep_mode}'")
    return x.clamp(0, 1)

def build_command_for_scorer(in_path: str, out_path: str, args, want_gray: bool) -> List[str]:
    cmd = [
        "conda", "run", "-n", PRED_ENV, "python", SCORER,
        "--in_batch", in_path,
        "--out_csv",  out_path,
        "--output_mode", args.output_mode,
        "--target_y", str(args.target_y), "--target_x", str(args.target_x), "--target_ch", str(args.target_ch), 
        "--pred_h", str(PRED_H), "--pred_w", str(PRED_W),
        "--prep_mode", args.prep_mode,
        "--reduce", args.reduce,
        "--T", str(args.tframes), "--burnin", str(args.burnin),
        "--jitter_px", str(args.jitter_px),
    ]
    if want_gray:
        cmd.append("--to_gray")
    return cmd

# ---------- shape normalizers ----------
def normalize_center(arr: np.ndarray, latent_dim: int, verbose: bool=False) -> Optional[np.ndarray]:
    a = np.asarray(arr)
    if a.size == latent_dim:
        return a.reshape(1, latent_dim).astype(np.float32)
    if a.ndim == 2 and (a.shape[0] == latent_dim or a.shape[1] == latent_dim):
        v = a.mean(axis=1) if a.shape[0] == latent_dim else a.mean(axis=0)
        return v.reshape(1, latent_dim).astype(np.float32)
    return None

def normalize_codes(arr: np.ndarray, batch: int, latent_dim: int, verbose: bool=False) -> Optional[np.ndarray]:
    a = np.asarray(arr, dtype=np.float32)
    if a.shape == (batch, latent_dim): return a
    if a.shape == (latent_dim, batch): return a.T.copy()
    if a.ndim == 1 and a.size == latent_dim: return np.tile(a.reshape(1, latent_dim), (batch, 1))
    if a.size == batch * latent_dim: return a.reshape(batch, latent_dim)
    return None

# ---------------- CMA-ES hooks ----------------
def cma_sample(opt, batch: int, center: np.ndarray, sigma: float,
               latent_dim: int, verbose: bool=False) -> np.ndarray:
    if hasattr(opt, "sample"):
        try:
            arr = opt.sample(batch, center=center)
            norm = normalize_codes(arr, batch, latent_dim, verbose=verbose)
            if norm is not None: return norm
        except Exception:
            pass
    if hasattr(opt, "ask"):
        for sig in [(batch, center), (batch,), tuple()]:
            try:
                arr = opt.ask(*sig)
                norm = normalize_codes(arr, batch, latent_dim, verbose=verbose)
                if norm is not None: return norm
            except Exception:
                continue
    mu = center.reshape(1, latent_dim).astype(np.float32) if center is not None and center.size == latent_dim \
         else np.zeros((1, latent_dim), np.float32)
    return (mu + sigma * np.random.randn(batch, latent_dim).astype(np.float32))

def evolve_population(opt, scores, codes, center, sigma, latent_dim, topk=4, verbose=False):
    if opt is not None and hasattr(opt, "step_simple"):
        try:
            nxt = opt.step_simple(scores, codes)
            arr = np.asarray(nxt, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == latent_dim:
                if arr.shape[0] == codes.shape[0]:  # same pop size
                    return arr, center, sigma, "pop_step_simple"
                # adapt popsize
                if arr.shape[0] > codes.shape[0]:
                    arr = arr[:codes.shape[0]]
                else:
                    reps = int(np.ceil(codes.shape[0] / arr.shape[0]))
                    arr = np.tile(arr, (reps, 1))[:codes.shape[0]]
                return arr, center, sigma, "pop_step_simple_adapted"
            cen = normalize_center(arr, latent_dim, verbose=verbose)
            if cen is not None:
                new_codes = cen + sigma * np.random.randn(codes.shape[0], latent_dim).astype(np.float32)
                return new_codes, cen, sigma, "center_step_simple"
        except Exception:
            pass
    if opt is not None and hasattr(opt, "update_simple"):
        try:
            cen = opt.update_simple(scores, codes)
            cen = normalize_center(cen, latent_dim, verbose=verbose)
            if cen is not None:
                new_codes = cen + sigma * np.random.randn(codes.shape[0], latent_dim).astype(np.float32)
                return new_codes, cen, sigma, "center_update_simple"
        except Exception:
            pass
    if opt is not None and hasattr(opt, "tell"):
        try:
            try:    opt.tell(codes, scores)
            except TypeError: opt.tell(scores, codes)
            cen = getattr(opt, "center", None)
            cen = normalize_center(cen, latent_dim, verbose=verbose)
            if cen is not None:
                new_codes = cen + sigma * np.random.randn(codes.shape[0], latent_dim).astype(np.float32)
                return new_codes, cen, sigma, "center_tell"
        except Exception:
            pass
    if opt is not None and (hasattr(opt, "ask") or hasattr(opt, "sample")):
        try:
            arr = cma_sample(opt, codes.shape[0],
                             center if center is not None else np.zeros((1,latent_dim),np.float32),
                             sigma, latent_dim, verbose=verbose)
            return arr, center, sigma, "pop_ask_sample"
        except Exception:
            pass
    top_idx = np.argsort(scores)[-topk:]
    cen = codes[top_idx].mean(axis=0, keepdims=True).astype(np.float32)
    sigma = max(0.5, sigma * 0.98)
    new_codes = cen + sigma * np.random.randn(codes.shape[0], latent_dim).astype(np.float32)
    return new_codes, cen, sigma, "fallback_topk"

# =============================== Generator init ===============================
def init_generator(gan_name: str, part: str, trunc: float, device: torch.device):
    gan_name = gan_name.lower()
    if gan_name == "fc6":
        G = upconvGAN("fc6").to(device).eval()
        for p in G.parameters(): p.requires_grad_(False)
        latent_dim, init_sigma, mask, out_size = 4096, 3.0, None, 256
        return G, latent_dim, init_sigma, mask, out_size

    if gan_name == "biggan":
        # 1) Load the pretrained BigGAN *module* on CPU
        biggan = PTBigGAN.from_pretrained("biggan-deep-256")
        biggan.eval().to(device)          # device is cpu in your script
        for p in biggan.parameters(): 
            p.requires_grad_(False)

        # 2) Wrap it — DO NOT call .to()/.eval() on the wrapper (not an nn.Module)
        G = BigGAN_wrapper(biggan)

        # 3) Try to set truncation if the wrapper exposes it (optional)
        if hasattr(G, "set_truncation") and callable(G.set_truncation):
            try: G.set_truncation(trunc)
            except Exception: pass
        elif hasattr(G, "truncation"):
            try: G.truncation = float(trunc)
            except Exception: pass

        # 4) Return config
        latent_dim = 256        # 128 z + 128 class-embed
        init_sigma = 0.2
        if part == "all":
            mask = None
        elif part == "noise":
            m = np.zeros(latent_dim, dtype=np.float32); m[:128] = 1.0
            mask = m
        elif part == "class":
            m = np.zeros(latent_dim, dtype=np.float32); m[128:] = 1.0
            mask = m
        else:
            raise ValueError(f"Unknown part '{part}' for BigGAN")
        out_size = 256
        return G, latent_dim, init_sigma, mask, out_size

    raise ValueError(f"Unknown GAN '{gan_name}'")

def apply_mask_codes(codes: np.ndarray, mask: Optional[np.ndarray], center: np.ndarray) -> np.ndarray:
    """
    If mask is provided (BigGAN part-wise optimization), enforce fixed halves by
    mixing current population with 'center' on the fixed dimensions.
    mask=1 -> free to optimize; mask=0 -> keep center dimension.
    """
    if mask is None: return codes
    assert center.shape == (1, codes.shape[1])
    return codes * mask[None, :] + center * (1.0 - mask[None, :])

# =============================== Main ===============================
def main():
    args = parse_args()
    VERBOSE = (args.mode == "test")

    expt_dir = os.path.join(GLUE_DIR, "expt"); ensure_dir(expt_dir)
    run_dir = make_run_dir(expt_dir)
    print(f"[run] outputs -> {run_dir}")
    print("Output mode:", args.output_mode, args.target_y, args.target_x, args.target_ch)
    print("GAN:", args.gan, "| BigGAN part:", args.part if args.gan=="biggan" else "-")

    # Seed
    if args.seed is not None and args.seed >= 0:
        if VERBOSE: print(f"[seed] {args.seed}")
        np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Device (stick to CPU unless you know your scorer path is GPU-safe)
    device = torch.device("cpu")

    # Init generator & per-GAN defaults
    G, LATENT_DIM, default_sigma, part_mask, out_size = init_generator(
        gan_name=args.gan, part=args.part, trunc=args.trunc, device=device
    )
    # Init sigma selection
    sigma = default_sigma if args.init_sigma < 0 else float(args.init_sigma)

    # Optimizer selection (population_size=args.batch)
    center = np.zeros((1, LATENT_DIM), np.float32)
    opt = None
    opt_name = "fallback"

    def _try_ctor(Cls, pop):
        for kw in ({"population_size": pop}, {"popsize": pop}, {"lambda_": pop}, {}):
            try:
                return Cls(space_dimen=LATENT_DIM, init_sigma=sigma, init_code=center.copy(), **kw)
            except TypeError:
                continue
        return Cls(space_dimen=LATENT_DIM, init_sigma=sigma)

    if args.optimizer in ("auto", "hessian") and HAS_HESS:
        opt = _try_ctor(HessCMAES, args.batch); opt_name = "HessCMAES"
    elif args.optimizer in ("auto", "cholesky") and HAS_CMA:
        opt = _try_ctor(CholeskyCMAES, args.batch); opt_name = "CholeskyCMAES"

    if VERBOSE:
        if opt is not None:
            print(f"[opt] Using {opt_name}.")
        else:
            print("[opt] CMA-ES not available; using fallback (top-K mean).")

    # If optimizer imposes its own popsize, respect it
    batch = args.batch
    if opt is not None:
        opt_pop = getattr(opt, "lambda_", getattr(opt, "popsize", None))
        if isinstance(opt_pop, int) and opt_pop > 0 and opt_pop != batch:
            if VERBOSE:
                print(f"[opt] Adjusting batch to optimizer popsize: {batch} -> {opt_pop}")
            batch = opt_pop

    # Gray intent
    want_gray = TO_GRAY
    if args.gray:  want_gray = True
    if args.color: want_gray = False

    # Logs
    log_path = os.path.join(run_dir, "opt_log.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "best", "mean", "median", "std", "min",])

    scores_all = os.path.join(run_dir, "scores_all.csv")
    with open(scores_all, "w", newline="") as f:
        f.write("step,idx,score\n")

    # Initialize first gen
    codes = center + sigma * np.random.randn(batch, LATENT_DIM).astype(np.float32)
    if part_mask is not None:
        codes = apply_mask_codes(codes, part_mask, center)

    # ========================= main loop =========================
    for step in range(args.steps):
        # Enforce mask each generation (keeps fixed half equal to center)
        if part_mask is not None:
            codes = apply_mask_codes(codes, part_mask, center)

        z = torch.from_numpy(codes).to(device)

        # Generate images
        with torch.inference_mode():
            imgs = G.visualize(z).clamp(0, 1)    # [B,3,H,W]
        # Optional: keep your 224 preview crop (neutral across fc6/BigGAN)
        imgs = fit_batch_to_224(imgs)

        # Save grids & scorer preview
        save_tensor_grid(imgs, os.path.join(run_dir, f"gen_{step:04d}.png"))
        pred_in_imgs = prep_for_prednet_preview_torch(
            imgs, PRED_H, PRED_W, prep_mode=args.prep_mode, to_gray=want_gray
        )
        save_tensor_grid(pred_in_imgs, os.path.join(run_dir, f"pred_input_{step:04d}.png"))

        # Save batch for scorer
        in_path  = os.path.join(run_dir, f"batch_{step:04d}.npy")
        np.save(in_path, imgs.cpu().numpy().astype(np.float32))

        # Call scorer (cross-env)
        out_path = os.path.join(run_dir, f"scores_{step:04d}.csv")
        cmd = build_command_for_scorer(in_path, out_path, args, want_gray)
        if VERBOSE: print(f"[step {step}] calling scorer:\n  {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Read scores
        scores = np.loadtxt(out_path, delimiter=",", skiprows=1, usecols=1).astype(np.float32)
        best_i = int(scores.argmax()); best_score = float(scores[best_i]); mean_score = float(scores.mean())
        median_score = float(np.median(scores))
        std_score = float(scores.std())
        min_score = float(scores.min())
        with open(scores_all, "a", newline="") as f:
            for i, s in enumerate(scores):
                f.write(f"{step},{i},{float(s)}\n")

        # ----------- Update population for next generation -----------
        codes, center, sigma, upd_mode = evolve_population(opt, scores, codes, center, sigma, LATENT_DIM, topk=TOPK, verbose=VERBOSE)
        if part_mask is not None:
            # After update, enforce fixed dims again
            codes = apply_mask_codes(codes, part_mask, center)
        if VERBOSE and upd_mode:
            print(f"[opt] update mode: {upd_mode}")

        # Save best image and log
        to_uint8_img(pred_in_imgs[best_i]).save(os.path.join(run_dir, f"best_{step:04d}.png"))
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([step, f"{best_score:.6f}", f"{mean_score:.6f}", f"{median_score:.6f}", f"{std_score:.6f}", f"{min_score:.6f}"])
        print(f"[step {step}] best {best_score:.4f} | "
              f"mean {mean_score:.4f} | median {median_score:.4f} | "
              f"std {std_score:.4f} | min {min_score:.4f}")

    print(f"[done] run folder: {run_dir}")

if __name__ == "__main__":
    main()