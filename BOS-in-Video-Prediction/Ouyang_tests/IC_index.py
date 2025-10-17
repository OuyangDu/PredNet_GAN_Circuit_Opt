# Import image generators and utilities
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle as pkl
from PIL import Image, ImageDraw
from drawing_square_image import *
from drawing_pacman import *

# Load neuron info (contains neuron coordinates for receptive field R<10 pixels)
with open('center_neuron_info_radius10.pkl', 'rb') as file:
    data = pkl.load(file)

# NOTE: ensure create_static_video_from_two_images is imported from your utilities
# from your_video_utils import create_static_video_from_two_images

# Get script directory and add parent to path for imports
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# PredNet agent import
from border_ownership.agent import Agent


def main():
    # Parameters
    width = 48       # image square side length
    r = 12           # kernel radius for inducers
    orientations = [0, 2]  # orientations: 0=up, 2=down

    # Define greyscale colors
    light_grey_value = 255 * 2 // 3
    dark_grey_value = 255 // 3
    light_grey = (light_grey_value,) * 3
    dark_grey = (dark_grey_value,) * 3

    # Storage for videos and PredNet responses
    videos = {}
    responses = {}

    # Define background/inducer color combos
    color_combos = [
        (light_grey, dark_grey),  # light background, dark inducer
        (dark_grey, light_grey)   # dark background, light inducer
    ]

    # 1) Generate videos for each orientation and color combo
    for ori in orientations:
        for bg_col, ind_col in color_combos:
            key = f"ori{ori}_bg{bg_col[0]}_ind{ind_col[0]}"
            # Circle and square images with matching colors/orientation
            img_circle = circle_sqr(r=r, width=width, orientation=ori,
                                     circle_color=ind_col, background_color=bg_col)
            ksqr = border_kaniza_sqr(r=r, width=width, orientation=ori,
                                     pacman_color=ind_col, background_color=bg_col)
            nonksqr = non_kaniza_sqr(r=r, width=width, orientation=ori,
                                     pacman_color=ind_col, background_color=bg_col)
            # Create two-frame videos
            videos[f"video_{key}_ksqr"] = create_static_video_from_two_images(img_circle, ksqr)
            videos[f"video_{key}_nonksqr"] = create_static_video_from_two_images(img_circle, nonksqr)

    # 2) Initialize PredNet agent on CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    WEIGHTS_DIR = '../model_data_keras2/'
    json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
    weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights', 'prednet_kitti_weights.hdf5')
    agent = Agent()
    agent.read_from_json(json_file, weights_file)

    # 3) Run all videos through PredNet, capture full E2 responses
    for vid_key, vid in videos.items():
        resp = agent.output_multiple(vid, output_mode=['E2'], is_upscaled=False)
        # resp['E2'] has shape (1, T, F, H, W)
        responses[vid_key] = resp['E2']

    # Debug: print full response shapes
    for k, arr in responses.items():
        print(f"{k}: full response shape = {arr.shape}")

    # -------------------------------------------------------------------------
    # Extract receptive field (R<10 pixels) responses
    # -------------------------------------------------------------------------
    selective_responses = {}
    neuron_ids = data['bo_info']['E2']['neuron_id']  # list of (f, i, j)
    for vid_key, full_resp in responses.items():
        T = full_resp.shape[1]  # number of time frames
        N = len(neuron_ids)     # number of receptive field neurons
        sel = np.zeros((T, N))  # container: (time x neurons)
        for idx, (f, i, j) in enumerate(neuron_ids):
            sel[:, idx] = full_resp[0, :, f, i, j]
        selective_responses[vid_key] = sel

    # Debug: print selective response shapes
    for k, arr in selective_responses.items():
        print(f"{k}: selective response shape = {arr.shape}")

    # -------------------------------------------------------------------------
    # Calculate Illusory Contour Index (ICI) at 6th time frame (index=6)
    # -------------------------------------------------------------------------
    time_idx = 6
    # Helper to fetch selective response at time_idx
    def resp(key_suffix):
        return selective_responses[f"video_{key_suffix}"][time_idx]

    # Compose condition suffixes
    dark_bg = str(dark_grey_value)
    light_bg = str(light_grey_value)
    dark_ind = light_grey_value
    light_ind = dark_grey_value
    keys = {
        'ups_k': f"ori0_bg{dark_bg}_ind{dark_ind}_ksqr",
        'ups_n': f"ori0_bg{dark_bg}_ind{dark_ind}_nonksqr",
        'dns_k': f"ori2_bg{dark_bg}_ind{dark_ind}_ksqr",
        'dns_n': f"ori2_bg{dark_bg}_ind{dark_ind}_nonksqr",
        'upl_k': f"ori0_bg{light_bg}_ind{light_ind}_ksqr",
        'upl_n': f"ori0_bg{light_bg}_ind{light_ind}_nonksqr",
        'dnl_k': f"ori2_bg{light_bg}_ind{light_ind}_ksqr",
        'dnl_n': f"ori2_bg{light_bg}_ind{light_ind}_nonksqr"
    }
    # Extract arrays for each condition
    a = resp(keys['ups_k']); b = resp(keys['ups_n'])
    c = resp(keys['dns_k']); d = resp(keys['dns_n'])
    e = resp(keys['upl_k']); f_ = resp(keys['upl_n'])
    g = resp(keys['dnl_k']); h = resp(keys['dnl_n'])

    # Numerator and denominator
    num = abs(a - b) + abs(c - d) + abs(e - f_) + abs(g - h)
    den = (a + b + 2) + (c + d + 2) + (e + f_ + 2) + (g + h + 2 )
    with np.errstate(divide='ignore', invalid='ignore'):
        ICI = np.where(den != 0, num / den, 0)
    print("Computed ICI values for each neuron at frame 6:", ICI)

    # -------------------------------------------------------------------------
    # Scatter plot: BOI index vs ICI index for each neuron
    # -------------------------------------------------------------------------
    # BOI values from pickle: data['bo_info']['E2']['boi'] is a pandas Series of lists
    boi_series = data['bo_info']['E2']['boi']  # Series length N
    print("ICI size:",len(ICI))
    print("BOI size:",len(boi_series))
    boi_vals = [boi_series.iloc[i][5] for i in range(len(boi_series))]
    plt.figure()
    plt.scatter(boi_vals, ICI)
    plt.xlabel('BOI index')
    plt.ylabel('ICI index')
    plt.title('BOI vs ICI scatter (frame 6)')
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # (Commented out) Sum and plot selective responses per background
    # -------------------------------------------------------------------------
    # ... previous commented sum-plot code ...

    # -------------------------------------------------------------------------
    # Commented-out image plotting since verified
    # -------------------------------------------------------------------------
    # ... previous commented image-plot code ...

if __name__ == '__main__':
    main()
