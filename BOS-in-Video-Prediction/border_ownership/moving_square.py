import numpy as np
from PIL import Image, ImageDraw
from border_ownership.full_square_response import transform_batch_video_to_x_source
import random
import matplotlib.pyplot as plt
from kitti_settings import *

class SquareVideoCreator:
    def __init__(self, image_size=(160, 128), background_grey=255//3, square_grey=255*2//3, square_size=50):
        self.image_size = image_size
        self.background_grey = background_grey
        self.square_grey = square_grey
        self.square_size = square_size

    def generate_grey_square_img(self, square_center):
        img = Image.new(mode='L', size=self.image_size, color=self.background_grey)
        drawer = ImageDraw.Draw(img)
        half_square = self.square_size // 2
        upper_left = (square_center[0] - half_square, square_center[1] - half_square)
        bottom_right = (upper_left[0] + self.square_size, upper_left[1] + self.square_size)
        drawer.rectangle([upper_left, bottom_right], fill=self.square_grey)
        return img

    def calculate_square_position(self, frame_number, total_frames, direction):
        mid_frame = total_frames // 2
        center_x, center_y = self.image_size[0] // 2, self.image_size[1] // 2
        direction_rad = np.radians(direction)
        total_distance = max(self.image_size) / 2
        move_per_frame = total_distance / mid_frame
        movement_distance = (frame_number - mid_frame) * move_per_frame
        position_x = center_x + movement_distance * np.cos(direction_rad)
        position_y = center_y - movement_distance * np.sin(direction_rad)
        return int(position_x), int(position_y)

    def create_moving_square_video(self, direction, total_frames=20):
        frames = []
        for frame_num in range(total_frames):
            square_pos = self.calculate_square_position(frame_num, total_frames, direction)
            frame = self.generate_grey_square_img(square_pos)
            frames.append(frame)
        return frames

    def create_multiple_translating_videos(self, num_videos=40, total_frames=20):
        directions = np.linspace(0, 360, num_videos, endpoint=False)
        videos = []
        for direction in directions:
            frames = self.create_moving_square_video(direction, total_frames)
            video = np.stack([np.repeat(np.array(frame)[..., np.newaxis], 3, axis=2) for frame in frames])
            videos.append(video)
        return np.array(videos)

    def generate_random_squares_video(self, total_frames=20, num_squares=5, max_square_size=50, edge_width=2):
        frames = []
        squares = []

        for _ in range(num_squares):
            size = random.randint(10, max_square_size)
            x = random.randint(0, self.image_size[0] - size)
            y = random.randint(0, self.image_size[1] - size)
            dx = random.randint(-2, 2)
            dy = random.randint(-2, 2)
            squares.append({'position': (x, y), 'size': size, 'dx': dx, 'dy': dy})

        for _ in range(total_frames):
            img = Image.new('L', self.image_size, color=self.background_grey)
            drawer = ImageDraw.Draw(img)

            for square in squares:
                x, y = square['position']
                size = square['size']
                drawer.rectangle([x, y, x + size, y + size], fill=self.square_grey, outline='black', width=edge_width)
                square['position'] = (max(min(x + square['dx'], self.image_size[0] - size), 0), 
                                      max(min(y + square['dy'], self.image_size[1] - size), 0))

            frames.append(img)

        return frames

    def create_multiple_random_squares_videos(self, num_videos=40, total_frames=20, max_square_size=50, edge_width=2):
        videos = []

        for _ in range(num_videos):
            num_squares = random.randint(1, 5)
            frames = self.generate_random_squares_video(total_frames, num_squares, max_square_size, edge_width)
            video = np.stack([np.repeat(np.array(frame)[..., np.newaxis], 3, axis=2) for frame in frames])
            videos.append(video)

        return np.array(videos)

# Visualization function (can be kept outside the class as a utility function)
def visualize_video_frames(frames, title="Video Frames", cmap='gray', vmin=0, vmax=255):
    num_videos = frames.shape[0]
    num_frames = frames.shape[1]
    fig, axes = plt.subplots(num_videos, num_frames, figsize=(num_frames * 1.5, num_videos * 2), gridspec_kw={'wspace': 0, 'hspace': 0})
    fig.suptitle(title, fontsize=20)
    fig.tight_layout(pad=0)
    fig.subplots_adjust(wspace=0, hspace=0)

    if num_videos == 1:
        axes = [axes]
    if num_frames == 1:
        axes = [[axes]]

    for vi in range(num_videos):
        for fi in range(num_frames):
            axes[vi][fi].imshow(frames[vi, fi], cmap=cmap, vmin=0, vmax=255)
            axes[vi][fi].spines['top'].set_visible(True)
            axes[vi][fi].set_xticks([])
            axes[vi][fi].set_yticks([])
    return fig, axes
