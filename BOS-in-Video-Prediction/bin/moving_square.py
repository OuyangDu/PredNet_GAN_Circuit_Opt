import numpy as np
from border_ownership.moving_square import SquareVideoCreator, visualize_video_frames
import hickle as hkl
import os
import math
from border_ownership.full_square_response import transform_batch_video_to_x_source
import random
import matplotlib.pyplot as plt
from kitti_settings import *

creator = SquareVideoCreator()
translating_videos_array = creator.create_multiple_translating_videos()
visualize_video_frames(translating_videos_array[10], "Translating Video Example")
random_squares_videos_array = creator.create_multiple_random_squares_videos()
visualize_video_frames(random_squares_videos_array[10], title="Random Squares Video")
print(random_squares_videos_array.shape)

hkl.dump(translating_videos_array, os.path.join(DATA_DIR_HOME, 'translating_videos_array.hkl'))
hkl.dump(random_squares_videos_array, os.path.join(DATA_DIR_HOME, 'random_squares_videos_array.hkl'))

# convert to x and sources
translating_x, translating_sources = transform_batch_video_to_x_source(translating_videos_array)
random_squares_x, random_squares_sources = transform_batch_video_to_x_source(random_squares_videos_array)

hkl.dump(translating_x, os.path.join(DATA_DIR_HOME, 'square_bo_video_translating_x.hkl'))
hkl.dump(translating_sources, os.path.join(DATA_DIR_HOME, 'square_bo_video_translating_sources.hkl'))
hkl.dump(random_squares_x, os.path.join(DATA_DIR_HOME, 'square_bo_video_random_x.hkl'))
hkl.dump(random_squares_sources, os.path.join(DATA_DIR_HOME, 'square_bo_video_random_sources.hkl'))

x = hkl.load(os.path.join(DATA_DIR_HOME, 'square_bo_video_random_x.hkl'))
sources = hkl.load(os.path.join(DATA_DIR_HOME, 'square_bo_video_random_sources.hkl'))

# print first 10 sources
print(sources[:10])
# plot first 10 frames
n_show = 10
fig, axes = plt.subplots(1, n_show, figsize=(15, 2))
for i in range(n_show):
    axes[i].imshow(x[i])
plt.show()

