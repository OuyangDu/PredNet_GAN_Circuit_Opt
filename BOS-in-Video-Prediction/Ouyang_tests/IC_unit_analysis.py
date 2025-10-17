#####################################################
#This file creates videos of 2 pictures and looks at the 
#  Compares ICI with for up and down orientaion.
#   and different colors
#  Calculate BOI index myself for up down and fliped color
# only the canidate neurons with the reciptive field of radius less than 10 pixels
#####################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys
import pickle as pkl
from PIL import Image, ImageDraw
from drawing_square_image import *
from drawing_pacman import *

# Load neuron info
with open('center_neuron_info_radius10.pkl', 'rb') as file:
    data = pkl.load(file)

# Load significant neuron union IDs
with open('significant_neuron_ids_union.pkl', 'rb') as file:
    significant_neuron_ids_union = pkl.load(file)

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from border_ownership.agent import Agent

def main():
    # Parameters
    width = 48
    r = 12
    orientations = [0, 2]
    light_grey_value = 255 * 2 // 3
    dark_grey_value = 255 // 3
    light_grey = (light_grey_value,) * 3
    dark_grey = (dark_grey_value,) * 3

    videos = {}
    responses = {}

    color_combos = [
        (light_grey, dark_grey),
        (dark_grey, light_grey)
    ]

    for ori in orientations:
        for bg_col, ind_col in color_combos:
            key = f"ori{ori}_bg{bg_col[0]}_ind{ind_col[0]}"
            img_circle = circle_sqr(r=r, width=width, orientation=ori,
                                     circle_color=ind_col, background_color=bg_col)
            ksqr = border_kaniza_sqr(r=r, width=width, orientation=ori,
                                     pacman_color=ind_col, background_color=bg_col)
            nonksqr = non_kaniza_sqr(r=r, width=width, orientation=ori,
                                     pacman_color=ind_col, background_color=bg_col)
            fillsqr = square_generator(width=width, orientation=ori,
                                     square_fill_color=ind_col, background_color=bg_col)
            videos[f"video_{key}_ksqr"] = create_static_video_from_two_images(img_circle, ksqr)
            videos[f"video_{key}_nonksqr"] = create_static_video_from_two_images(img_circle, nonksqr)
            videos[f"video_{key}_fillsqr"] = create_static_video_from_two_images(img_circle, fillsqr)

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    WEIGHTS_DIR = '../model_data_keras2/'
    json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
    weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights', 'prednet_kitti_weights.hdf5')
    agent = Agent()
    agent.read_from_json(json_file, weights_file)

    for vid_key, vid in videos.items():
        resp = agent.output_multiple(vid, output_mode=['E2'], is_upscaled=False)
        responses[vid_key] = resp['E2']+1

    selective_responses = {}
    neuron_ids = data['bo_info']['E2']['neuron_id']
    for vid_key, full_resp in responses.items():
        T = full_resp.shape[1]
        N = len(neuron_ids)
        sel = np.zeros((T, N))
        for idx, (f, i, j) in enumerate(neuron_ids):
            sel[:, idx] = full_resp[0, :, f, i, j]
        selective_responses[vid_key] = sel

    N = len(neuron_ids)
    max_vals = np.full(N, -np.inf)
    for resp in selective_responses.values():
        max_vals = np.maximum(max_vals, resp.max(axis=0))
    for vid_key in selective_responses:
        selective_responses[vid_key] /= max_vals

    neuron_id_to_index = {nid: idx for idx, nid in enumerate(neuron_ids)}
    significant_indices = [neuron_id_to_index[nid] for nid in significant_neuron_ids_union if nid in neuron_id_to_index]

    time_idx = 6
    def resp(key_suffix):
        return selective_responses[f"video_{key_suffix}"][time_idx]

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
        'dnl_n': f"ori2_bg{light_bg}_ind{light_ind}_nonksqr",
        'ups_f': f"ori0_bg{dark_bg}_ind{dark_ind}_fillsqr",
        'dns_f': f"ori2_bg{dark_bg}_ind{dark_ind}_fillsqr",
        'upl_f': f"ori0_bg{light_bg}_ind{light_ind}_fillsqr",
        'dnl_f': f"ori2_bg{light_bg}_ind{light_ind}_fillsqr"
    }

    ICI_ups = (resp(keys['ups_k']) - resp(keys['ups_n'])) / (resp(keys['ups_k']) + resp(keys['ups_n']))
    ICI_dns = (resp(keys['dns_k']) - resp(keys['dns_n'])) / (resp(keys['dns_k']) + resp(keys['dns_n']))
    ICI_upl = (resp(keys['upl_k']) - resp(keys['upl_n'])) / (resp(keys['upl_k']) + resp(keys['upl_n']))
    ICI_dnl = (resp(keys['dnl_k']) - resp(keys['dnl_n'])) / (resp(keys['dnl_k']) + resp(keys['dnl_n']))

    print("Length of ICI (ups):", len(ICI_ups[significant_indices]))
    print("Length of ICI (dns):", len(ICI_dns[significant_indices]))
    print("Length of ICI (upl):", len(ICI_upl[significant_indices]))
    print("Length of ICI (dnl):", len(ICI_dnl[significant_indices]))

    # Split into neurons with any ICI = 0 vs. all non-zero ICIs
    bor_ici_neurons = []
    nonbor_ici_neurons = []
    for idx in significant_indices:
        condition1 = (ICI_ups[idx] == 0 or ICI_dns[idx] == 0)
        condition2 = (ICI_upl[idx] == 0 or ICI_dnl[idx] == 0)
        if condition1 or condition2:
            bor_ici_neurons.append(idx)
        else:
            nonbor_ici_neurons.append(idx)


    print(f"Neurons with at least one ICI = 0 (bor_ici_neurons): {len(bor_ici_neurons)}")
    print(f"Neurons with all ICI ≠ 0 (nonbor_ici_neurons): {len(nonbor_ici_neurons)}")

    # Calculate BOI index as a scalar and per neuron
    ups_f_response = resp(keys['ups_f'])
    dns_f_response = resp(keys['dns_f'])
    upl_f_response = resp(keys['upl_f'])
    dnl_f_response = resp(keys['dnl_f'])
    BOI = (np.abs(ups_f_response - dnl_f_response) + np.abs(upl_f_response - dns_f_response)) / (ups_f_response + dns_f_response + upl_f_response + dnl_f_response)

    print("BOI index (scalar):", BOI)
   
    print("BOI index (non_bor):", BOI[nonbor_ici_neurons])

    # Box plot of BOI values
    plt.figure()
    plt.boxplot([
        BOI,
        BOI[bor_ici_neurons],
        BOI[nonbor_ici_neurons]
    ], labels=['All RF', 'BOR ICI', 'Non-BOR'], patch_artist=True)
    plt.title('BOI Index Distribution')
    plt.ylabel('BOI')
    plt.tight_layout()
    plt.show()

     # Scatter plot: ICI_ups vs ICI_dns
    plt.figure()
    plt.scatter([ICI_ups[i] for i in bor_ici_neurons], [ICI_dns[i] for i in bor_ici_neurons], color='red', label='BOR ICI Neurons')
    plt.scatter([ICI_ups[i] for i in nonbor_ici_neurons], [ICI_dns[i] for i in nonbor_ici_neurons], color='blue', label='Non-BOR ICI Neurons')
    plt.xlabel('ICI (ups)')
    plt.ylabel('ICI (dns)')
    plt.title('ICI: Up Dark vs Down Dark')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Scatter plot: ICI_upl vs ICI_dnl
    plt.figure()
    plt.scatter([ICI_upl[i] for i in bor_ici_neurons], [ICI_dnl[i] for i in bor_ici_neurons], color='red', label='BOR ICI Neurons')
    plt.scatter([ICI_upl[i] for i in nonbor_ici_neurons], [ICI_dnl[i] for i in nonbor_ici_neurons], color='blue', label='Non-BOR ICI Neurons')
    plt.xlabel('ICI (dns)')
    plt.ylabel('ICI (dnl)')
    plt.title('ICI: Down Dark vs Down Light')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
