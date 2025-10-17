from border_ownership.moving_square import SquareVideoCreator, visualize_video_frames
import matplotlib.pyplot as plt
import os
from data_utils import SequenceGenerator
import hickle as hkl
from kitti_settings import *

creator = SquareVideoCreator()
translating_videos_array = creator.create_multiple_translating_videos()
fig, ax = visualize_video_frames(translating_videos_array[::10], "Translating Video Examples")
fig.savefig(os.path.join(FIGURE_DIR, 'translating_video_example.svg'))

random_squares_videos_array = creator.create_multiple_random_squares_videos()
fig, ax = visualize_video_frames(random_squares_videos_array[::10], title="Random Squares Video Examples")
fig.savefig(os.path.join(FIGURE_DIR, 'random_squares_video_example.svg'))

test_file_path = os.path.join(KITTI_DATA_DIR, 'new_X_test.hkl')
test_sources_path = os.path.join(KITTI_DATA_DIR, 'new_sources_test.hkl')
generator = SequenceGenerator(test_file_path, test_sources_path, nt=20, sequence_start_mode='unique', data_format='channels_last', shuffle=False, N_seq=None)
kitti_video = generator.create_all()
fig, ax = visualize_video_frames(kitti_video[::10], title="KITTI Video Examples")
fig.savefig(os.path.join(FIGURE_DIR, 'kitti_video_example.svg'))
plt.show()
