# File: generate
#
#E2_rf10_per_neuron_rect_response_pair.pkl
#E2_rf10_per_neuron_rect_boi.pkl
#
# This script generates static videos of rectangles of varying sizes and orientations
# under dark/light backgrounds, runs them through a PredNet model to extract neural
# activations (E2 layer), normalizes each neuron's response across the four
# background/orientation conditions per size, computes a border ownership index (BOI),
# and saves both the normalized responses and BOI values to pickle files.

import pickle as pkl
import numpy as np
import os
from drawing_pacman import circle_rec, rec_generator
from drawing_square_image import create_static_video_from_two_images
from border_ownership.agent import Agent

# Load receptive field neuron info
with open('center_neuron_info_radius10.pkl', 'rb') as f:
    data = pkl.load(f)
    neuron_ids = data['bo_info']['E2']['neuron_id']

# Prepare PredNet agent (disable GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
WEIGHTS_DIR = '../model_data_keras2/'
agent = Agent()
agent.read_from_json(
    os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json'),
    os.path.join(WEIGHTS_DIR, 'tensorflow_weights', 'prednet_kitti_weights.hdf5')
)

# Stimulus parameters
IMAGE_SIZE   = (160, 128)
ORIENTATIONS = [0, 2]        # 0=Up, 2=Down
FRAME_IDX    = 6             # frame index for response extraction

# Rectangle dimensions and radii to test
widths  = [48, 52, 56, 60, 64, 68, 72, 76, 80]
heights = [48, 47, 46, 45, 44, 43, 42, 41, 40]
radii   = [12, 13, 14, 15, 16, 17, 18, 19, 20]

# Define greyscale colors based on thirds
light_grey_value = 255 * 2 // 3
dark_grey_value  = 255 // 3
light_grey       = (light_grey_value,) * 3
dark_grey        = (dark_grey_value,) * 3

# Color combinations: (background, fill, label)
color_combos = [
    (dark_grey,  light_grey,  'Dark'),
    (light_grey, dark_grey,   'Light')
]

# Initialize storage for normalized responses and BOI values
neuron_response_pairs = {nid: [] for nid in neuron_ids}
neuron_boi = {nid: [] for nid in neuron_ids}

# Loop through rectangle sizes and radii
for width, height, RADIUS in zip(widths, heights, radii):
    # Collect raw responses at FRAME_IDX for each of the 4 conditions
    condition_infos = []
    for ori in ORIENTATIONS:
        for bg_color, fill_color, bgcolor_label in color_combos:
            # Generate stimuli images
            circle_img = circle_rec(
                image_size=IMAGE_SIZE,
                orientation=ori,
                width=width,
                height=height,
                circle_color=fill_color,
                background_color=bg_color,
                r=RADIUS
            )
            rect_img = rec_generator(
                image_size=IMAGE_SIZE,
                orientation=ori,
                width=width,
                height=height,
                background_color=bg_color,
                rect_fill_color=fill_color
            )

            # Build static video and run through PredNet
            video = create_static_video_from_two_images(circle_img, rect_img)
            resp = agent.output_multiple(video, output_mode=['E2'], is_upscaled=False)['E2'] + 1

            # Extract frame-6 responses for each neuron
            raw_vals = np.zeros(len(neuron_ids))
            for idx, (f, i, j) in enumerate(neuron_ids):
                raw_vals[idx] = resp[0, FRAME_IDX, f, i, j]

            # Label condition
            cond_label = 'Up' if ori == 0 else 'Down'
            full_label = f"Rect_{width}x{height}_r{RADIUS}_{bgcolor_label}_{cond_label}"
            condition_infos.append({
                'label':    full_label,
                'bg':       bgcolor_label,
                'ori':      cond_label,
                'raw_vals': raw_vals
            })

    # Compute per-neuron max across the 4 conditions
    all_raw = np.vstack([ci['raw_vals'] for ci in condition_infos])  # shape (4, N)
    max_response = np.nanmax(all_raw, axis=0)

    # Normalize and collect responses
    norm_vals_dict = {}
    for ci in condition_infos:
        norm_vals = ci['raw_vals'] / max_response
        norm_vals_dict[(ci['bg'], ci['ori'])] = norm_vals
        for idx, nid in enumerate(neuron_ids):
            neuron_response_pairs[nid].append({
                'condition':     ci['label'],
                'width':         width,
                'height':        height,
                'radius':        RADIUS,
                'response_norm': float(norm_vals[idx])
            })

    # Compute BOI for each neuron: (|up_dark - down_light| + |up_light - down_dark|)/sum
    up_dark_vals   = norm_vals_dict.get(('Dark', 'Up'))
    down_light_vals= norm_vals_dict.get(('Light', 'Down'))
    up_light_vals  = norm_vals_dict.get(('Light', 'Up'))
    down_dark_vals = norm_vals_dict.get(('Dark', 'Down'))
    for idx, nid in enumerate(neuron_ids):
        ud = up_dark_vals[idx]
        dl = down_light_vals[idx]
        ul = up_light_vals[idx]
        dd = down_dark_vals[idx]
        denom = ud + dl + ul + dd
        if denom != 0:
            boi = ((ud - dl) + (ul - dd)) / denom
        else:
            boi = np.nan
        neuron_boi[nid].append({
            'width':    width,
            'height':   height,
            'radius':   RADIUS,
            'boi':      float(boi)
        })

# Save normalized response pairs and BOI to pickles
with open('E2_rf10_per_neuron_rect_response_pair.pkl', 'wb') as f:
    pkl.dump(neuron_response_pairs, f)
with open('E2_rf10_per_neuron_rect_boi.pkl', 'wb') as f:
    pkl.dump(neuron_boi, f)

print(f"Saved normalized responses and BOI for {len(neuron_ids)} neurons across {len(widths)} sizes.")
