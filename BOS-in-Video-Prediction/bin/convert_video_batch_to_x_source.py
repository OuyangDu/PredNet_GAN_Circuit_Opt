# get the square video
import os
import matplotlib.pyplot as plt
import numpy as np
import hickle as hkl
from border_ownership.full_square_response import transform_batch_video_to_x_source
from kitti_settings import *

n_frames = 20
video_ori_path = os.path.join(DATA_DIR_HOME, 'square_bo_video_ori.npz')
data_ori = np.load(video_ori_path, allow_pickle=True)
videos = np.repeat(data_ori['video'], n_frames, axis=1)# shape is (40, n_frames, 128, 160, 3) = (n_videos, n_frames, height, width, n_channels)
print(videos.shape)
x, sources = transform_batch_video_to_x_source(videos)
print(x.shape, len(sources))

hkl.dump(x, os.path.join(DATA_DIR_HOME, 'square_bo_video_ori_x.hkl'))
hkl.dump(sources, os.path.join(DATA_DIR_HOME, 'square_bo_video_ori_sources.hkl'))

x = hkl.load(os.path.join(DATA_DIR_HOME, 'square_bo_video_ori_x.hkl'))
sources = hkl.load(os.path.join(DATA_DIR_HOME, 'square_bo_video_ori_sources.hkl'))

# print first 10 sources
print(sources[:10])
# plot first 10 frames
n_show = 10
fig, axes = plt.subplots(1, n_show, figsize=(15, 2))
for i in range(n_show):
    axes[i].imshow(x[i])
plt.show()
