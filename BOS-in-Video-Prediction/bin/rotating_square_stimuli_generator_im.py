import numpy as np
import matplotlib.pyplot as plt
from border_ownership.rotating_square_stimuli_generator import Square_Generator

# create a grey image
width, height = 160, 128
light_grey = 255 // 3
dark_grey = 255 * 2 // 3
length_square = 50
n_direction_line = 3 # it's better to be even to include 90 degree direction
n_frames = 5
pixel_format = '1'

rsg = Square_Generator(background_grey=light_grey, square_grey=dark_grey)
edge_dir_list, angle_list, video_batch = rsg.generate_rotated_square_video_list(n_direction_line=n_direction_line, n_frames=n_frames, pixel_format=pixel_format)
print(edge_dir_list, angle_list)
print(video_batch.shape)
for i in range(video_batch.shape[0]):
    plt.imshow(video_batch[i, 0])
    plt.show()

rsg = Square_Generator(background_grey=dark_grey, square_grey=light_grey)
edge_dir_list, angle_list, video_batch = rsg.generate_rotated_square_video_list(n_direction_line=n_direction_line, n_frames=n_frames, pixel_format=pixel_format)
for i in range(video_batch.shape[0]):
    plt.imshow(video_batch[i, 0])
    plt.show()
