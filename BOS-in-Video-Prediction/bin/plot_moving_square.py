import hickle as hkl
import os
import numpy as np
import matplotlib.pyplot as plt
from kitti_settings import DATA_DIR_HOME

random_x = hkl.load(os.path.join(DATA_DIR_HOME, 'square_bo_video_random_x.hkl'))
rotating_x = hkl.load(os.path.join(DATA_DIR_HOME, 'square_bo_video_rotating_x.hkl'))

plt.figure()
plt.imshow(random_x[0])
plt.axis('off')

plt.figure()
plt.imshow(rotating_x[302])
plt.axis('off')
plt.show()

# rotating square stimuli seems wrong
x = hkl.load(os.path.join(DATA_DIR_HOME, 'square_bo_video_rotating_x.hkl'))
sources = hkl.load(os.path.join(DATA_DIR_HOME, 'square_bo_video_rotating_sources.hkl'))

# print first 10 sources
print(sources[:10])
# plot first 10 frames
n_show = 20
fig, axes = plt.subplots(1, n_show, figsize=(15, 2))
for i in range(n_show):
    axes[i].imshow(x[200 + i])
plt.show()
