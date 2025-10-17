#####################################################
#This file creates videos of 2 pictures and looks at the 
#  Compares ICI with for up and down orientaion.
#   and different colors
#  Calculate BOI index myself for up down and fliped color
# only the canidate neurons with the reciptive field of radius less than 10 pixels
#####################################################
# Import image generators and utilities
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys
import pickle as pkl
from PIL import Image, ImageDraw
from drawing_square_image import *
from drawing_pacman import *

# Load neuron info (contains neuron coordinates for receptive field R<10 pixels)
with open('center_neuron_info_radius10.pkl', 'rb') as file:
    data = pkl.load(file)

# nOtE: ensure create_static_video_from_two_images is imported from your utilities
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
            fillsqr = square_generator(width=width, orientation=ori,
                                     square_fill_color=ind_col, background_color=bg_col)
            # Create two-frame videos
            videos[f"video_{key}_ksqr"] = create_static_video_from_two_images(img_circle, ksqr)
            videos[f"video_{key}_nonksqr"] = create_static_video_from_two_images(img_circle, nonksqr)
            videos[f"video_{key}_fillsqr"] = create_static_video_from_two_images(fillsqr, fillsqr)

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
        responses[vid_key] = resp['E2']+1

    # Debug: print full response shapes
    #for k, arr in responses.items():
        #print(f"{k}: full response shape = {arr.shape}")

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

    # -------------------------------------------------------------------------
    # Normalize each neuron's response across all videos and time steps
    # -------------------------------------------------------------------------
    # Step 1: Initialize max response tracker per neuron
    N = len(neuron_ids)
    max_vals = np.full(N, -np.inf)

    # Step 2: Find maximum per neuron across all videos and time steps
    for resp in selective_responses.values():
        max_vals = np.maximum(max_vals, resp.max(axis=0))

    # Step 3: Normalize each neuron's responses in each video
    for vid_key in selective_responses:
        selective_responses[vid_key] /= max_vals

    # Debug: print selective response shapes
    #for k, arr in selective_responses.items():
        #print(f"{k}: selective response shape = {arr.shape}")


    # -------------------------------------------------------------------------
    # Compute average response over time for each neuron in each fillsqr video
    # -------------------------------------------------------------------------

    fillsqr_keys = {
        'ups_f' : f"video_ori0_bg{dark_grey_value}_ind{light_grey_value}_fillsqr",  # ups_f
        'dns_f' : f"video_ori2_bg{dark_grey_value}_ind{light_grey_value}_fillsqr",  # dns_f
        'upl_f' : f"video_ori0_bg{light_grey_value}_ind{dark_grey_value}_fillsqr",  # upl_f
        'dnl_f' : f"video_ori2_bg{light_grey_value}_ind{dark_grey_value}_fillsqr"   # dnl_f
    }

    # Dictionary to store time-averaged responses per neuron
    avg_fill_responses = {}

    for name,video_key in fillsqr_keys.items():
        resp_matrix = selective_responses[video_key]         # shape (T, N)
        avg_response = resp_matrix.mean(axis=0)        # shape (N,), average over time
        avg_fill_responses[name] = avg_response

    i_ = avg_fill_responses['ups_f']
    j_ = avg_fill_responses['dns_f']
    k_ = avg_fill_responses['upl_f']
    l_ = avg_fill_responses['dnl_f']
    #print("Shape of avg_response",l_.shape)
    #print(l_)

    # BOI index top fill sqr is light and bottom fill square is dark
    boi_ul=(k_-j_)/(k_+j_)
    # BOI index top fill sqr is dark and bottom fill square is light
    boi_us=(i_-l_)/(i_+l_)
    # BOI that looks at both
    BOI=(abs(k_-j_)+abs(i_-l_))/((i_+l_)+(k_+j_))
    # Plot to compare the 2 BOI indecies
    plt.figure()
    plt.scatter(boi_ul, boi_us)
    plt.xlabel('BOI index up filled sqr is light')
    plt.ylabel('BOI index up filled sqr is dark')
    plt.title('BOI_1 vs BOI_2 (time is averaged)')
    plt.tight_layout()
    plt.show()

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
        'ups_f': f"ori0_bg{dark_bg}_ind{dark_ind}_fillsqr",
        'dns_f': f"ori2_bg{dark_bg}_ind{dark_ind}_fillsqr",

        'upl_k': f"ori0_bg{light_bg}_ind{light_ind}_ksqr",
        'upl_n': f"ori0_bg{light_bg}_ind{light_ind}_nonksqr",
        'dnl_k': f"ori2_bg{light_bg}_ind{light_ind}_ksqr",
        'dnl_n': f"ori2_bg{light_bg}_ind{light_ind}_nonksqr",
        'upl_f': f"ori0_bg{light_bg}_ind{light_ind}_fillsqr",
        'dnl_f': f"ori2_bg{light_bg}_ind{light_ind}_fillsqr"

    }
    # Extract arrays for each condition
    a = resp(keys['ups_k']); b = resp(keys['ups_n'])
    c = resp(keys['dns_k']); d = resp(keys['dns_n'])
    e = resp(keys['upl_k']); f_ = resp(keys['upl_n'])
    g = resp(keys['dnl_k']); h = resp(keys['dnl_n'])


    

    # Numerator and denominator
    num = abs(a - b) + abs(c - d) + abs(e - f_) + abs(g - h)
    den = (a + b ) + (c + d ) + (e + f_ ) + (g + h )
    with np.errstate(divide='ignore', invalid='ignore'):
        ICI = np.where(den != 0, num / den, 0)
    #print("Computed ICI values for each neuron at frame 6:", ICI)

    # calculate ICI for up and down orientation
        # dark background, up orientation
    num_up_s= (a-b)
    den_up_s= (a+b)
    ICI_up_s=num_up_s/den_up_s

        # dark background, down orientation
    num_dn_s= (c-d)
    den_dn_s= (c+d)
    ICI_dn_s=num_dn_s/den_dn_s

        #light background, up orientation
    num_up_l= (e-f_)
    den_up_l= (e+f_)
    ICI_up_l=num_up_l/den_up_l

        #light background, dn orientation
    num_dn_l= (g-h)
    den_dn_l= (g+h)
    ICI_dn_l=num_dn_l/den_dn_l

    #IC_BOI (up-down)/(up+down) for kanizsa sqr, inducers are the same color
    IC_BOI= (abs(a-c)+abs(e-g))/(a+c+e+g)


    # -------------------------------------------------------------------------
    # Scatter plot: BOI index vs ICI index for each neuron
    # -------------------------------------------------------------------------
    # BOI values from pickle: data['bo_info']['E2']['boi'] is a pandas Series of lists
    boi_series = data['bo_info']['E2']['boi']  # Series length N
    #print("ICI size:",len(ICI))
    #print("BOI size:",len(boi_series))
    #boi_vals = [boi_series.iloc[i][5] for i in range(len(boi_series))]
    plt.figure()
    plt.scatter(BOI, ICI)
    plt.xlabel('BOI index')
    plt.ylabel('ICI index')
    plt.title('BOI (time-avg) vs ICI (frame 6)')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(IC_BOI, BOI)
    plt.xlabel('IC_BOI index')
    plt.ylabel('BOI index')
    plt.title('BOI (time-avg) vs IC_BOI (frame 6)')
    plt.tight_layout()
    plt.show()

    ######################################
    # ICI compare for up and down, light background
    ######################################
    plt.figure()
    plt.scatter(ICI_up_l, ICI_dn_l)
    plt.xlabel('ICI index up light background')
    plt.ylabel('ICI index down light background')
    plt.title('up/down for light background (frame 6)')
    plt.tight_layout()
    plt.show()

    ######################################
    # ICI compare for up and down, dark background
    ######################################
    plt.figure()
    plt.scatter(ICI_up_s, ICI_dn_s)
    plt.xlabel('ICI index up dark background')
    plt.ylabel('ICI index down dark background')
    plt.title('up/down for dark background (frame 6)')
    plt.tight_layout()
    plt.show()

    ######################################
    # ICI compare for up ori, light and dark background
    ######################################
    plt.figure()
    plt.scatter(ICI_up_s, ICI_up_l)
    plt.xlabel('ICI index up dark background')
    plt.ylabel('ICI index up light background')
    plt.title('dark/light background for up  (frame 6)')
    plt.tight_layout()
    plt.show()

   

     ######################################
    # ICI compare for down ori, light and dark background
    ######################################
    plt.figure()
    plt.scatter(ICI_dn_s, ICI_dn_l)
    plt.xlabel('ICI index up dark background')
    plt.ylabel('ICI index up light background')
    plt.title('dark/light background for down (frame 6)')
    plt.tight_layout()
    plt.show()


    # -------------------------------------------------------------------------
    # Sum of normalized responses across all neurons per time step
    # Plotting line graphs for ups_k, ups_n, and ups_f
    # -------------------------------------------------------------------------

    # Condition keys
    keys_to_plot = {
    'ups_k': f"video_ori0_bg{dark_grey_value}_ind{light_grey_value}_ksqr",
    'ups_n': f"video_ori0_bg{dark_grey_value}_ind{light_grey_value}_nonksqr",
    'ups_f': f"video_ori0_bg{dark_grey_value}_ind{light_grey_value}_fillsqr"
}

    # Prepare time vector
    time_steps = selective_responses[keys_to_plot['ups_k']].shape[0]
    time = np.arange(time_steps)

    # Plotting
    plt.figure()
    for label, key in keys_to_plot.items():
        summed = selective_responses[key].sum(axis=1)  # sum over neurons
        plt.plot(time, summed, label=label)

    plt.xlabel('Time frame')
    plt.ylabel('Summed normalized response')
    plt.title('Summed responses over time for ups_k, ups_n, ups_f')
    plt.legend()
    plt.tight_layout()
    plt.show()
    # -------------------------------------------------------------------------
    # Identify “potential IC” units: those with non-zero ICI in all four conditions
    # -------------------------------------------------------------------------

# -------------------------------------------------------------------------
    # Identify “potential IC” units on light and dark backgrounds, and record their indices
    # -------------------------------------------------------------------------
    potential_ic_l = []      
    potential_ic_l_idx = []  # will hold their integer positions
    potential_ic_s = []
    potential_ic_s_idx = []

    for idx, nid in enumerate(neuron_ids):
        # light background: both up & down non-zero
        if ICI_up_l[idx] != 0 and ICI_dn_l[idx] != 0:
            potential_ic_l.append(nid)
            potential_ic_l_idx.append(idx)

        # dark background: both up & down non-zero
        if ICI_up_s[idx] != 0 and ICI_dn_s[idx] != 0:
            potential_ic_s.append(nid)
            potential_ic_s_idx.append(idx)

    print(f"Found {len(potential_ic_l)} potential IC_l units in E2")
    print(f"Found {len(potential_ic_s)} potential IC_s units in E2")
    
    # -------------------------------------------------------------------------
    # Scatter plot for potential_ic_l: ICI_up_l vs ICI_dn_l
    # -------------------------------------------------------------------------
    x_l = ICI_up_l[potential_ic_l_idx]
    y_l = ICI_dn_l[potential_ic_l_idx]

    plt.figure()
    plt.scatter(x_l, y_l)
    plt.xlabel('ICI_up_l')
    plt.ylabel('ICI_dn_l')
    plt.title('Potential IC_l units (non-zero ICI up/down on light bg)')
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Scatter plot for potential_ic_s: ICI_up_s vs ICI_dn_s
    # -------------------------------------------------------------------------
    x_s = ICI_up_s[potential_ic_s_idx]
    y_s = ICI_dn_s[potential_ic_s_idx]

    plt.figure()
    plt.scatter(x_s, y_s)
    plt.xlabel('ICI_up_s')
    plt.ylabel('ICI_dn_s')
    plt.title('Potential IC_s units (non-zero ICI up/down on dark bg)')
    plt.tight_layout()
    plt.show()
# -------------------------------------------------------------------------
    # Cross‐condition scatter for union of IC_l and IC_s units
    # -------------------------------------------------------------------------
    # Compute the intersect of both index lists
    intersect_idx = sorted(set(potential_ic_l_idx) & set(potential_ic_s_idx))
    # Compute the union of both index lists
    union_idx = sorted(set(potential_ic_l_idx) | set(potential_ic_s_idx))
    print(f"Total units in union: {len(union_idx)}")

    # Print the total number of units in the union
    print(f"Total units in intersection: {len(intersect_idx)}")

    # 1) ICI_up_s vs ICI_up_l for all units in the intersect
    x1 = ICI_up_s[union_idx]
    y1 = ICI_up_l[union_idx]

    plt.figure()
    plt.scatter(x1, y1)
    plt.xlabel('ICI_up_s')
    plt.ylabel('ICI_up_l')
    plt.title('Intersection IC units: ICI_up_s vs ICI_up_l (frame 6)')
    plt.tight_layout()
    plt.show()

    # 2) ICI_dn_l vs ICI_dn_s for all units in the intersect
    x2 = ICI_dn_s[union_idx]
    y2 = ICI_dn_l[union_idx]

    plt.figure()
    plt.scatter(x2, y2)
    plt.xlabel('ICI_dn_s')
    plt.ylabel('ICI_dn_l')
    plt.title('Intersection IC units: ICI_dn_s vs ICI_dn_l (frame 6)')
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Scatter plot: BOI vs. (ICI_up_s + ICI_dn_s) for potential_ic_s units
    # -------------------------------------------------------------------------
    # Gather BOI and summed ICI for those “S‐units”
    boi_vals_s    = BOI[potential_ic_s_idx]
    ici_sums_s    = ICI_up_s[potential_ic_s_idx] + ICI_dn_s[potential_ic_s_idx]

    # Optional: print how many points we’ll plot
    print(f"Plotting {len(potential_ic_s_idx)} points: BOI vs (ICI_up_s + ICI_dn_s)")

    # Create the scatter plot
    plt.figure()
    plt.scatter(boi_vals_s, ici_sums_s)
    plt.xlabel('BOI index (time‐avg fill sqr)')
    plt.ylabel('ICI_up_s + ICI_dn_s (frame 6 sum)')
    plt.title('BOI vs. Sum of ICI (dark bg) for Potential IC_s Units')
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Scatter plot: BOI vs. (ICI_up_l + ICI_dn_l) for potential_ic_l units
    # -------------------------------------------------------------------------
    # Gather BOI and summed ICI for those “L‐units”
    boi_vals_l = BOI[potential_ic_l_idx]
    ici_sums_l = ICI_up_l[potential_ic_l_idx] + ICI_dn_l[potential_ic_l_idx]

    # Optional debug
    print(f"Plotting {len(potential_ic_l_idx)} points: BOI vs (ICI_up_l + ICI_dn_l)")

    # Create the scatter plot
    plt.figure()
    plt.scatter(boi_vals_l, ici_sums_l)
    plt.xlabel('BOI index (time‐avg fill sqr)')
    plt.ylabel('ICI_up_l + ICI_dn_l (frame 6 sum)')
    plt.title('BOI vs. Sum of ICI (light bg) for Potential IC_l Units')
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Scatter plot: BOI vs. IC_BOI for union of IC_l and IC_s units
    # -------------------------------------------------------------------------
  

    # Extract BOI and IC_BOI for those units
    x_u = BOI[union_idx]
    y_u = IC_BOI[union_idx]

    plt.figure()
    plt.scatter(x_u, y_u)
    plt.xlabel('BOI index (time‐avg fill sqr)')
    plt.ylabel('IC_BOI index (frame 6)')
    plt.title('BOI vs. IC_BOI for Union of Potential IC Units')
    plt.tight_layout()
    plt.show()
    #-----------------------------------------
    # for intersect index
    #-----------------------------------------
    # Extract BOI and IC_BOI for those units
    x_i = BOI[intersect_idx]
    y_i = IC_BOI[intersect_idx]

    plt.figure()
    plt.scatter(x_i, y_i)
    plt.xlabel('BOI index (time‐avg fill sqr)')
    plt.ylabel('IC_BOI index (frame 6)')
    plt.title('BOI vs. IC_BOI for intersect of Potential IC Units')
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Count how many union IC units are BOS units (is_bo == True)
    # -------------------------------------------------------------------------
    # Assume `union_idx` is already defined as:
    # union_idx = sorted(set(potential_ic_l_idx) | set(potential_ic_s_idx))

    # Grab the boolean array of BOS flags for E2
    bo_flags = np.array(data['bo_info']['E2']['is_bo'])  # shape (N,)
    n_bos = np.sum(bo_flags)
    print("Number of BOS units:", n_bos)
    # Subset to just our union indices
    union_bo_flags = bo_flags[union_idx]

    # Count how many are True
    n_bos_in_union = np.sum(union_bo_flags)
    print(f"{n_bos_in_union} out of {len(union_idx)} union IC units are BOS units")

    # (Optional) List their neuron IDs
    #bos_union_ids = [ neuron_ids[i] for i, flag in zip(union_idx, union_bo_flags) if flag ]
    #print("Neuron IDs of BOS units in union:", bos_union_ids)
   
    # Subset to just our intersect indices
    intersect_bo_flags = bo_flags[intersect_idx]

    # Count how many are True
    n_bos_in_intersect = np.sum(intersect_bo_flags)
    print(f"{n_bos_in_intersect} out of {len(intersect_idx)} intersect IC units are BOS units")

    # (Optional) List their neuron IDs
    #bos_union_ids = [ neuron_ids[i] for i, flag in zip(union_idx, union_bo_flags) if flag ]
    #print("Neuron IDs of BOS units in union:", bos_union_ids)

   # -------------------------------------------------------------------------
    # Box plot of BOI across all E2 neurons
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots()
    ax.boxplot([BOI[potential_ic_l_idx], BOI[potential_ic_s_idx],BOI],
                labels=['IC_l', 'IC_s','BOI'],
                patch_artist=True)

    # Increase major y-ticks to, say, 10 bins:
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))

    # (Optionally) add minor ticks for finer gridlines:
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(which='major', axis='y', linestyle='-', alpha=0.7)
    ax.grid(which='minor', axis='y', linestyle='--', alpha=0.4)

    ax.set_ylabel('BOI index')
    ax.set_title('BOI: IC_l , IC_s units')
    plt.tight_layout()
    plt.show()
   
  
if __name__ == '__main__':
    main()
