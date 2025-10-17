import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- Variable definitions ---
# a: raw E2 response amplitude to Kanizsa (illusory contour) stimulus
# a_val: normalized Kanizsa response at FRAME_IDX used for ICI calculation
# b: raw E2 response amplitude to non-Kanizsa (control) stimulus
# b_val: normalized non-Kanizsa response at FRAME_IDX used for ICI calculation
# x_val: normalized cross stimulus response at FRAME_IDX used for ICI calculation
# FRAME_IDX: index of the frame used for computing ICI (0-based)

# --- Shape & significance maps & params ---
ORIENTATIONS = [0, 2]
WIDTH      = 48
HEIGHT     = 48
RADIUS     = 12
FRAME_IDX  = 6

# --- Load dark-significant neurons and full response pairs ---
with open('significant_neurons_dark.pkl', 'rb') as f:
    dark_map = pkl.load(f)
with open('E2_rf10_per_neuron_response_pair.pkl', 'rb') as f:
    pairs = pkl.load(f)

# --- Load neuron_ids for full normalization pipeline ---
with open('center_neuron_info_radius10.pkl', 'rb') as f:
    data = pkl.load(f)
neuron_ids = data['bo_info']['E2']['neuron_id']
N = len(neuron_ids)

# --- Definition of Illusory Contour Index (ICI) ---
# Note: compute_ici will receive normalized values (a_val, b_val)
def compute_ici(a, b):
    return (a - b) / (a + b)

# --- Initial ICI scatter for Kanizsa vs non-Kanizsa (old method) ---
def make_scatter_old(condition_map, cond_up, cond_down, title):
    up_x, up_y, dn_x, dn_y, both_x, both_y = [], [], [], [], [], []
    for nid, passed in condition_map.items():
        recs = pairs.get(nid, [])
        r_up = next((r for r in recs if r['condition']==cond_up
                     and r['orientation'] in ORIENTATIONS
                     and r['width']==WIDTH
                     and r['height']==HEIGHT
                     and r['r']==RADIUS), None)
        r_down = next((r for r in recs if r['condition']==cond_down
                       and r['orientation'] in ORIENTATIONS
                       and r['width']==WIDTH
                       and r['height']==HEIGHT
                       and r['r']==RADIUS), None)
        if not (r_up and r_down):
            continue
        ici_up   = compute_ici(r_up['a'], r_up['b'])
        ici_down = compute_ici(r_down['a'], r_down['b'])
        if isinstance(passed, list):
            both_x.append(ici_up); both_y.append(ici_down)
        elif passed == cond_up:
            up_x.append(ici_up);   up_y.append(ici_down)
        else:
            dn_x.append(ici_up);   dn_y.append(ici_down)
    plt.figure(figsize=(5,5))
    plt.scatter(up_x, up_y, c='tab:blue',   label=f'{cond_up} only',    alpha=0.7)
    plt.scatter(dn_x, dn_y, c='tab:orange', label=f'{cond_down} only', alpha=0.7)
    plt.scatter(both_x, both_y, c='tab:green', label='both significant',alpha=0.7)
    plt.axis('equal')
    plt.xlabel(f'ICI ({cond_up})')
    plt.ylabel(f'ICI ({cond_down})')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot the old ICI scatter
make_scatter_old(dark_map, 'Up Dark', 'Down Dark', 'ICI: Up Dark vs Down Dark')

# --- Predictive pipeline: generate stimuli, run PredNet, normalize, compute ICI ---
from drawing_pacman import circle_rec, border_kaniza_rec, non_kaniza_rec, kaniza_cross_rec
from drawing_square_image import create_static_video_from_two_images
from border_ownership.agent import Agent

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Load PredNet model
global WEIGHTS_DIR
WEIGHTS_DIR = '../model_data_keras2/'
agent = Agent()
agent.read_from_json(
    os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json'),
    os.path.join(WEIGHTS_DIR, 'tensorflow_weights', 'prednet_kitti_weights.hdf5')
)

# Colors & thickness values
dark_grey_value  = 255 // 3
light_grey_value = 255 * 2 // 3
dark_grey  = (dark_grey_value,)*3
light_grey = (light_grey_value,)*3
THICKNESS_DIV = 2

# Containers for normalized frame responses
k_norm_responses = {ori: {} for ori in ORIENTATIONS}
n_norm_responses = {ori: {} for ori in ORIENTATIONS}
x_norm_responses = {ori: {} for ori in ORIENTATIONS}
x_ici_responses  = {ori: {} for ori in ORIENTATIONS}

for ori in ORIENTATIONS:
    # Generate stimuli images
    circle_img     = circle_rec((160,128), ori, WIDTH, HEIGHT, light_grey, dark_grey, RADIUS)
    kanizsa_img    = border_kaniza_rec((160,128), ori, WIDTH, HEIGHT, light_grey, dark_grey, RADIUS)
    nonkanizsa_img = non_kaniza_rec((160,128), ori, WIDTH, HEIGHT, light_grey, dark_grey, RADIUS)
    cross_img      = kaniza_cross_rec((160,128), ori, WIDTH, HEIGHT, light_grey, dark_grey, RADIUS, THICKNESS_DIV)

    # Display stimuli side by side
    fig, axs = plt.subplots(1,4,figsize=(12,3))
    for ax, img, ttl in zip(axs, [circle_img, kanizsa_img, nonkanizsa_img, cross_img],
                             ['Circle','Kanizsa Rec','Non-Kanizsa Rec','Kanizsa Cross']):
        ax.imshow(img); ax.set_title(ttl); ax.axis('off')
    plt.suptitle(f'Orientation {ori}')
    plt.tight_layout(); plt.show()

    # Build videos and run through PredNet
    vk = create_static_video_from_two_images(circle_img, kanizsa_img)
    vn = create_static_video_from_two_images(circle_img, nonkanizsa_img)
    vx = create_static_video_from_two_images(circle_img, cross_img)
    rk = agent.output_multiple(vk, output_mode=['E2'], is_upscaled=False)['E2'] + 1
    rn = agent.output_multiple(vn, output_mode=['E2'], is_upscaled=False)['E2'] + 1
    rx = agent.output_multiple(vx, output_mode=['E2'], is_upscaled=False)['E2'] + 1

    # Extract per-frame responses for each neuron
    T = rk.shape[1]
    k_sel = np.zeros((T, N)); n_sel = np.zeros((T, N)); x_sel = np.zeros((T, N))
    for idx, (f,i,j) in enumerate(neuron_ids):
        k_sel[:,idx] = rk[0,:,f,i,j]
        n_sel[:,idx] = rn[0,:,f,i,j]
        x_sel[:,idx] = rx[0,:,f,i,j]

    # Normalize peak response across all three conditions (frames 4:)
    max_vals = np.maximum(
        np.maximum(k_sel[4:].max(axis=0), n_sel[4:].max(axis=0)),
        x_sel[4:].max(axis=0)
    )
    k_norm = k_sel / max_vals
    n_norm = n_sel / max_vals
    x_norm = x_sel / max_vals

        # Store normalized frame-6 values and compute cross ICI
    for idx, nid in enumerate(neuron_ids):
        # collect normalized responses
        k_val = k_norm[FRAME_IDX, idx]
        n_val = n_norm[FRAME_IDX, idx]
        x_val = x_norm[FRAME_IDX, idx]
        k_norm_responses[ori][nid] = k_val
        n_norm_responses[ori][nid] = n_val
        x_norm_responses[ori][nid] = x_val
        # compute ICI between Kanizsa and Cross for ALL neurons
        x_ici_responses[ori][nid] = compute_ici(k_val, x_val)

# After you’ve computed k_norm, n_norm, x_norm for each orientation,
# store the sums in a dict:
sum_responses = {}
time = np.arange(T)

for ori in ORIENTATIONS:
    # ... your existing code to build k_norm_responses, etc. ...
    # (i.e. everything up through computing k_norm, n_norm, x_norm)
    sum_responses[ori] = {
        'Kanizsa': k_norm.sum(axis=1),
        'Non-Kanizsa': n_norm.sum(axis=1),
        'Cross': x_norm.sum(axis=1)
    }

# Now plot all orientations side by side:
labels = {0: 'Up Dark', 2: 'Down Dark'}
fig, axes = plt.subplots(1, len(ORIENTATIONS), figsize=(4*len(ORIENTATIONS), 4), sharey=True)

for ax, ori in zip(axes, ORIENTATIONS):
    sums = sum_responses[ori]
    ax.plot(time, sums['Kanizsa'],    label='Kanizsa')
    ax.plot(time, sums['Non-Kanizsa'], label='Non-Kanizsa')
    ax.plot(time, sums['Cross'],      label='Cross')
    ax.set_title(labels[ori])             # e.g. “Up Dark” or “Down Dark”
    ax.set_xlabel('Frame')
    if ori == ORIENTATIONS[0]:
        ax.set_ylabel('Sum of normalized response')
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# --- Scatter of prednet ICI (Dark-Up vs Dark-Down) ---
# Compute ICI at FRAME_IDX for Up Dark (orientation 0) vs Down Dark (orientation 2)
up_only_x, up_only_y = [], []
dn_only_x, dn_only_y = [], []
both_x, both_y     = [], []
other_x, other_y   = [], []
for nid in neuron_ids:
    a_up = k_norm_responses[0].get(nid)
    b_up = n_norm_responses[0].get(nid)
    a_dn = k_norm_responses[2].get(nid)
    b_dn = n_norm_responses[2].get(nid)
    if None in (a_up, b_up, a_dn, b_dn):
        continue
    ici_up = compute_ici(a_up, b_up)
    ici_dn = compute_ici(a_dn, b_dn)
    if nid in dark_map:
        passed = dark_map[nid]
        if isinstance(passed, list):
            both_x.append(ici_up); both_y.append(ici_dn)
        elif passed == 'Up Dark':
            up_only_x.append(ici_up); up_only_y.append(ici_dn)
        else:
            dn_only_x.append(ici_up); dn_only_y.append(ici_dn)
    else:
        other_x.append(ici_up); other_y.append(ici_dn)
plt.figure(figsize=(5,5))
plt.scatter(up_only_x,    up_only_y,    c='tab:blue',   label='Up Dark only',    alpha=0.7)
plt.scatter(dn_only_x,    dn_only_y,    c='tab:orange', label='Down Dark only', alpha=0.7)
plt.scatter(both_x,       both_y,       c='tab:green',  label='both significant',alpha=0.7)
plt.scatter(other_x,      other_y,      c='grey',       label='not significant', alpha=0.7)
plt.axis('equal')
plt.xlabel('ICI (Up Dark)')
plt.ylabel('ICI (Down Dark)')
plt.title('PredNet ICI: Dark-Up vs Dark-Down at t=6')
plt.legend(); plt.tight_layout(); plt.show()

# --- Scatter of new cross ICI (no unity line) ---
# Include neurons not significant alongside dark_map categories
up_x, up_y, dn_x, dn_y, both_x, both_y = [], [], [], [], [], []
other_x, other_y = [], []
for nid in neuron_ids:
    # Check if cross ICI exists for both orientations
    if nid in x_ici_responses[0] and nid in x_ici_responses[2]:
        xi_up   = x_ici_responses[0][nid]
        xi_down = x_ici_responses[2][nid]
        if nid in dark_map:
            passed = dark_map[nid]
            if isinstance(passed, list):
                both_x.append(xi_up); both_y.append(xi_down)
            elif passed == 'Up Dark':
                up_x.append(xi_up);   up_y.append(xi_down)
            else:
                dn_x.append(xi_up);   dn_y.append(xi_down)
        else:
            other_x.append(xi_up); other_y.append(xi_down)
# Plot cross ICI scatter with non-significant points
plt.figure(figsize=(5,5))
plt.scatter(up_x,    up_y,    c='tab:blue',   label='Up Dark only',    alpha=0.7)
plt.scatter(dn_x,    dn_y,    c='tab:orange', label='Down Dark only',  alpha=0.7)
plt.scatter(both_x,  both_y,  c='tab:green',  label='both significant',alpha=0.7)
plt.scatter(other_x, other_y, c='grey',       label='not significant', alpha=0.7)
plt.axis('equal')
plt.xlabel('Cross ICI (up)')
plt.ylabel('Cross ICI (down)')
plt.title('Cross ICI: Up vs Down Dark at t=6')
plt.legend(); plt.tight_layout(); plt.show()
