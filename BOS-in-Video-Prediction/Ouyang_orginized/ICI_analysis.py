import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle as pkl
from scipy.stats import spearmanr
from drawing_pacman import border_kaniza_rec, non_kaniza_rec, circle_rec
from drawing_square_image import create_static_video_from_two_images

# Load receptive field neuron info
with open('center_neuron_info_radius10.pkl', 'rb') as file:
    data = pkl.load(file)
    neuron_ids = data['bo_info']['E2']['neuron_id']

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from border_ownership.agent import Agent

# Define color values
light_grey_value = 255 * 2 // 3
dark_grey_value = 255 // 3
light_grey = (light_grey_value,) * 3
dark_grey = (dark_grey_value,) * 3

# Define configurations
orientations = [0, 2]
widths = [48, 52, 56, 60, 64, 68, 72, 76, 80]
heights = [48, 47, 46, 45, 44, 43, 42, 41, 40]
radii = [12, 13, 14, 15, 16, 17, 18, 19, 20]
shape_triplets = list(zip(widths, heights, radii))

# Define color combos as (background, inducer) with labels
color_combos = [
    ((dark_grey, 'bg_dark'), (light_grey, 'ind_light')),
    ((light_grey, 'bg_light'), (dark_grey, 'ind_dark'))
]

# Storage dictionaries
responses = {}
images = {}
icis = {}
neuron_response_pairs = {neuron: [] for neuron in neuron_ids}  # Updated to store condition-tagged dicts

# Prepare PredNet agent
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
WEIGHTS_DIR = '../model_data_keras2/'
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights', 'prednet_kitti_weights.hdf5')
agent = Agent()
agent.read_from_json(json_file, weights_file)

# First loop: generate videos and compute responses
for ori in orientations:
    for (bg_color, bg_label), (ind_color, ind_label) in color_combos:
        for width, height, r in shape_triplets:
            base_key = f"video_ori{ori}_{bg_label}_{ind_label}_w{width}_h{height}_r{r}"

            circle_img = circle_rec(image_size=(160, 128), orientation=ori, width=width, height=height,
                                    circle_color=ind_color, background_color=bg_color, r=r)
            kanizsa_img = border_kaniza_rec(image_size=(160, 128), orientation=ori, width=width, height=height,
                                            pacman_color=ind_color, background_color=bg_color, r=r)
            nonkanizsa_img = non_kaniza_rec(image_size=(160, 128), orientation=ori, width=width, height=height,
                                            pacman_color=ind_color, background_color=bg_color, r=r)

            video_ksqr = create_static_video_from_two_images(circle_img, kanizsa_img)
            video_nonksqr = create_static_video_from_two_images(circle_img, nonkanizsa_img)

            resp_k = agent.output_multiple(video_ksqr, output_mode=['E2'], is_upscaled=False)['E2'] + 1
            resp_n = agent.output_multiple(video_nonksqr, output_mode=['E2'], is_upscaled=False)['E2'] + 1

            T = resp_k.shape[1]
            N = len(neuron_ids)
            k_sel = np.zeros((T, N))
            n_sel = np.zeros((T, N))
            for idx, (f, i, j) in enumerate(neuron_ids):
                k_sel[:, idx] = resp_k[0, :, f, i, j]
                n_sel[:, idx] = resp_n[0, :, f, i, j]

            max_vals = np.maximum(k_sel[4:].max(axis=0), n_sel[4:].max(axis=0))
            k_norm = k_sel / max_vals
            n_norm = n_sel / max_vals

            responses[base_key] = (k_norm, n_norm)
            images[base_key] = (circle_img, kanizsa_img, nonkanizsa_img)

            # Compute ICI at time index 6
            time_idx = 6
            a = k_norm[time_idx]  # Kanizsa
            b = n_norm[time_idx]  # Non-Kanizsa
            ici = (a - b) / (a + b)
            icis[base_key] = ici

            # Determine condition tag
            if ori == 0 and bg_label == 'bg_dark':
                condition = 'Up Dark'
            elif ori == 2 and bg_label == 'bg_dark':
                condition = 'Down Dark'
            elif ori == 0 and bg_label == 'bg_light':
                condition = 'Up Light'
            elif ori == 2 and bg_label == 'bg_light':
                condition = 'Down Light'
            else:
                condition = 'Unknown'

            # Collect per-neuron response pairs with condition tag
            for idx, neuron in enumerate(neuron_ids):
                neuron_response_pairs[neuron].append({
                    'condition':    condition,
                    'orientation':  ori,
                    'width':        width,
                    'height':       height,
                    'r':            r,
                    'a':            a[idx],
                    'b':            b[idx]
                })

# Save per-neuron Kanizsa and Non-Kanizsa response pairs with condition tags
output_file = 'E2_rf10_per_neuron_response_pair.pkl'
with open(output_file, 'wb') as f:
    pkl.dump(neuron_response_pairs, f)

print(f"Per-neuron response pairs saved to: {output_file}")

# Second loop: plotting
for base_key, (k_norm, n_norm) in responses.items():
    T = k_norm.shape[0]
    time = np.arange(T)

    plt.figure(figsize=(8, 4))
    plt.plot(time[1:], k_norm[1:].sum(axis=1), label='Kanizsa')
    plt.plot(time[1:], n_norm[1:].sum(axis=1), label='Non-Kanizsa')
    plt.xlabel('Time Frame')
    plt.ylabel('Summed Normalized Response')
    plt.title(f"Summed Response vs Time\n{base_key}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    circle_img, kanizsa_img, nonkanizsa_img = images[base_key]
    fig, axs = plt.subplots(1, 3, figsize=(9, 3))
    axs[0].imshow(circle_img)
    axs[0].set_title("Circle")
    axs[1].imshow(kanizsa_img)
    axs[1].set_title("Kanizsa Rec")
    axs[2].imshow(nonkanizsa_img)
    axs[2].set_title("Non-Kanizsa Rec")
    for ax in axs:
        ax.axis('off')
    plt.suptitle(base_key)
    plt.tight_layout()
    plt.show()

# Compute and plot mean ± std ICI for each neuron
ici_matrix = np.array(list(icis.values()))  # shape: (num_conditions, num_neurons)
ici_means = ici_matrix.mean(axis=0)
ici_stds = ici_matrix.std(axis=0)

plt.figure(figsize=(10, 5))
plt.errorbar(np.arange(len(ici_means)), ici_means, yerr=ici_stds, fmt='o', ecolor='gray', capsize=3)
plt.xlabel('Neuron Index')
plt.ylabel('ICI (mean ± std)')
plt.title('Neuron-wise ICI Consistency Across All Inputs')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Plot by condition subset
conditions = {
    'Up Dark': [k for k in icis if 'ori0' in k and 'bg_dark' in k],
    'Down Dark': [k for k in icis if 'ori2' in k and 'bg_dark' in k],
    'Up Light': [k for k in icis if 'ori0' in k and 'bg_light' in k],
    'Down Light': [k for k in icis if 'ori2' in k and 'bg_light' in k]
}

for title, keys in conditions.items():
    submatrix = np.array([icis[k] for k in keys])
    means = submatrix.mean(axis=0)
    stds = submatrix.std(axis=0)

    plt.figure(figsize=(10, 5))
    plt.errorbar(np.arange(len(means)), means, yerr=stds, fmt='o', ecolor='gray', capsize=3)
    plt.xlabel('Neuron Index')
    plt.ylabel('ICI (mean ± std)')
    plt.title(f'Neuron-wise ICI Consistency: {title}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Additional plot: raw ICI values scatter for each neuron across inputs
plt.figure(figsize=(12, 6))
condition_colors = {
    'Up Dark': 'tab:blue',
    'Down Dark': 'tab:orange',
    'Up Light': 'tab:green',
    'Down Light': 'tab:red'
}

for condition, keys in conditions.items():
    submatrix = np.array([icis[k] for k in keys])
    color = condition_colors.get(condition, 'gray')
    for row in submatrix:
        plt.scatter(np.arange(len(row)), row, alpha=0.4, label=condition, color=color)

# Avoid duplicate labels in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.xlabel('Neuron Index')
plt.ylabel('ICI value')
plt.title('ICI Index per Neuron Across All Input Conditions')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print("Normalized responses, ICI, and per-neuron response pairs computed and saved.")
