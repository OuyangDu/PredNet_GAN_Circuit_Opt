import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def true_bounds(mask: np.ndarray):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None, None
    r, c = np.where(mask)
    return (r.min(), c.min()), (r.max(), c.max())

def show_with_bbox(img2d, mask, color='r', lw=2, origin='upper'):
    """Show img2d (2-D) and draw a rectangle around True pixels in mask."""
    (r0, c0), (r1, c1) = true_bounds(mask)
    fig, ax = plt.subplots()
    ax.imshow(img2d, cmap='gray', vmin=0, vmax=1, origin=origin)

    if r0 is not None:
        # width/height in pixels; +1 so the box includes the last pixel index
        w = c1 - c0 + 1
        h = r1 - r0 + 1
        rect = Rectangle((c0, r0), w, h, fill=False, edgecolor=color, linewidth=lw)
        ax.add_patch(rect)

    ax.set_axis_off()
    plt.show()


def show_tight_boundary(binary_img):
    """Display a binary 2-D array and draw a tight outline that aligns with pixels."""
    M = np.asarray(binary_img, dtype=bool)
    H, W = M.shape

    # Use 'upper' image convention (row 0 at top) and align pixel edges via extent
    extent = (-0.5, W - 0.5, H - 0.5, -0.5)  # left, right, top, bottom for origin='upper'

    fig, ax = plt.subplots()
    ax.imshow(M, cmap='gray', vmin=0, vmax=1, origin='upper',
              interpolation='nearest', extent=extent)

    if M.any():
        ax.contour(M.astype(float), levels=[0.5], colors='red', linewidths=1,
                   origin='upper', extent=extent)

    ax.axis('off')
    plt.show()

with open('center_neuron_info_radius10.pkl', 'rb') as file:
    data = pkl.load(file)

# data is a dict with keys:
# 'bo_info': meta information, such as neuron id, bav etc.
# 'res_info': response information, units' responses to different square orientations.
# 'stim_info': stimulus information, such as stimulus, orientation etc.
# 'unique_orientation': unique orientation, ignore this
print(data.keys())

# bo_info is a dict with keys:
# 'E0': E0 module's bo_info
# 'E1': E1 module's bo_info
# 'E2': E2 module's bo_info
# 'E3': E3 module's bo_info
print(data['bo_info'].keys())

# data['bo_info']['E0'] is a dict with keys:
# 'neuron_id': a 3D tuple, continas neuron id
# 'BOI': BOI value at each orientation
# 'boi_abs_max': maximum BOI value
# 'boi_pref_dire': preferred BOI direction of the neuron
# 'boi_abs_rank': 
# 'bav': bav value
# 'bav_angle': the angle of circular averaged boi
# 'bav_p_value': p value of bav. If smaller than 0.05, the neuron is a BOS unit.
# 'is_bo': whether the neuron is a BOS unit, by bav_p_value < 0.05
# 'bo_only_rank': the rank of the neuron among all BOS units
# 'heatmap': heatmap of the neuron's response to different sparse noise stimuli (can be used to estimate cRF size)
# 'rf': cRF
print(data['bo_info']['E2'].keys())

for mode in ['E0', 'E1', 'E2', 'E3']:
    print(mode)
    #print("number of BOS units: ", np.sum(data['bo_info'][mode]['is_bo']))
    print("number of canidate:", np.size(data['bo_info'][mode]))

print("E_0 id?", data['bo_info']['E0']['neuron_id'][50])

# grab the BOI array for module E2'
print(data['bo_info']['E2'].shape)

# the code below prints the neuron_id of 0th neuron canadate
print(data['bo_info']['E2']['neuron_id'].iloc[0])


# iloc is the 0th id
print("BOI at Orientation",data['bo_info']['E2']['boi'].iloc[0][5])

print(data['bo_info']['E2'].keys())
print("orientation:",data['unique_orientation']['E2'][5])

import matplotlib.pyplot as plt
plt.figure()
print(data['stim_info']['orientation'])
plt.imshow(data['stim_info']['image'][20])
plt.show()


boi_E2 = data['bo_info']['E2']['boi']
print("BOI array keys:", boi_E2)



# plot boundary box around cRF
(top,left),(bottom,right) = true_bounds(data['bo_info']['E2']['rf'][50])
print("top-left True:", (top,left))
print("bottom-right True:", (bottom,right))
w = right - left + 2
h = bottom - top +2
print("cRF:", data['bo_info']['E2']['rf'][50].shape)

fig, ax = plt.subplots()                # create an Axes object
ax.imshow(data['bo_info']['E2']['rf'][50], cmap='gray', origin='upper')
# Rectangle takes (x, y) = (left, top), width, height
ax.add_patch(Rectangle((left-1, top-1), w, h, fill=False, edgecolor='r', linewidth=1))
plt.show()


img=data['bo_info']['E2']['rf'][50]
show_tight_boundary(img)
neuron_ids = data['bo_info']['E2']['neuron_id']
for nid in neuron_ids:
    print(nid) 