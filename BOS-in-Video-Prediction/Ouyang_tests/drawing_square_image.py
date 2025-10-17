"""
Goal:
    The goal of this code is to gain familarity with creating 
Border Ownership Stimuli: i.e squares with an edge in the center.
Method:
    Trying to use the function: square_generator_bo() 
    This function is from full_square_response.py
"""

"Import the possible libararies I might need"

import math
import sys
import os
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Get the directory of the current script
current_dir = os.path.dirname(__file__)
# Get the common parent directory (one level up)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the parent directory to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from border_ownership.full_square_response import square_generator_bo

def create_static_video_from_image(pil_img, n_frames=5, n_video=1):
    """
    Create a video from a static PIL image.
    
    Input:
    - pil_img (PIL.Image): A static image. If not already in grayscale ('L'), it will be converted.
    - n_frames (int): Number of frames per video sequence.
    - n_video (int): Number of video sequences to generate.
    
    Output:
    - video (numpy array): Video batch of shape (n_video, n_frames, height, width) with pixel values in [0, 1].
    """
   # Ensure the image is in RGB
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
    # Convert PIL image to a numpy array and normalize pixel values to [0, 1]
    img_np = np.asarray(pil_img, dtype=np.float32)
    if img_np.max() > 1.0001:
        img_np = img_np / 255.0
    
    
    # Get image dimensions
    height, width,channels = img_np.shape
    
    # Create a single video sequence by repeating the image along the time dimension.
    # This gives an array of shape (n_frames, height, width)
    video_single = np.repeat(img_np[ np.newaxis, :, :], n_frames, axis=0)
    
    # Repeat the video sequence for n_video times to create a batch.
    # Final shape will be (n_video, n_frames, height, width)
    video_batch = video_single[np.newaxis, ...]
    
    return video_batch

def create_static_video_from_two_images(pil_img1, pil_img2, n_frames=5, m_frames=5, n_video=1):
    """
    Create a video from two static PIL images. The first image is repeated for `n_frames`
    and the second is repeated for `m_frames`. The two segments are concatenated along 
    the time axis and then replicated to create a video batch of shape 
    (n_video, n_frames+m_frames, height, width, channels) with pixel values in [0, 1].
    
    Parameters:
      - pil_img1 (PIL.Image): The first static image.
      - pil_img2 (PIL.Image): The second static image.
      - n_frames (int): Number of frames to show the first image.
      - m_frames (int): Number of frames to show the second image.
      - n_video (int): Number of video sequences to generate (batch size).
      
    Returns:
      - video (numpy array): Video batch of shape 
        (n_video, n_frames+m_frames, height, width, channels) with pixel values in [0, 1].
    """
    # Ensure both images are in RGB (or convert to the desired mode)
    if pil_img1.mode != 'RGB':
        pil_img1 = pil_img1.convert('RGB')
    if pil_img2.mode != 'RGB':
        pil_img2 = pil_img2.convert('RGB')
    
    # Convert PIL images to NumPy arrays and normalize to [0, 1] (assuming original 0-255 range)
    img_np1 = np.asarray(pil_img1, dtype=np.float32)
    if img_np1.max() > 1.0001:
        img_np1 = img_np1 / 255.0
        
    img_np2 = np.asarray(pil_img2, dtype=np.float32)
    if img_np2.max() > 1.0001:
        img_np2 = img_np2 / 255.0
    
    # Optionally, you might want to check that both images have the same shape.
    if img_np1.shape != img_np2.shape:
        raise ValueError("The two images must have the same dimensions and number of channels.")
    
    # Get image dimensions
    height, width, channels = img_np1.shape
    
    # Create a video segment for the first image by repeating it along a new time axis.
    # The resulting shape will be (n_frames, height, width, channels)
    video_seq1 = np.repeat(img_np1[np.newaxis, ...], n_frames, axis=0)
    
    # Create a video segment for the second image.
    video_seq2 = np.repeat(img_np2[np.newaxis, ...], m_frames, axis=0)
    
    # Concatenate the two segments along the time dimension.
    # The resulting shape will be (n_frames + m_frames, height, width, channels)
    video_single = np.concatenate([video_seq1, video_seq2], axis=0)
    
    # Replicate the video sequence to create a batch of n_video identical sequences.
    # Final shape: (n_video, n_frames+m_frames, height, width, channels)
    video_batch = video_single[np.newaxis, ...]
    
    return video_batch




def play_video_matplotlib(video, fps=30):
    """
    Play a video using matplotlib.

    Input:
    - video (numpy array): A video sequence of shape (n_frames, height, width, 3)
      with pixel values normalized in [0, 1].
    - fps (int): Frames per second for the animation.
    """
    fig, ax = plt.subplots()
    # Display the first frame
    im = ax.imshow(video[0])
    ax.axis('off')  # Turn off axis

    def update(frame):
        im.set_data(video[frame])
        return [im]

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=range(len(video)), interval=1000/fps, blit=True
    )
    
    plt.show()



if __name__ == "__main__":
    # Here are the prameters for generating square function
    # square_generator_bo(dark_grey=255//3, light_grey=255 * 2 // 3, orientation=0, beta=True, gamma=True, shift=0, size=50, image_width=160, image_height=128, keep_keypoint_id=[], base_grey=128)

    """sqr_Image1 = square_generator_bo(orientation=0,beta=True,gamma=True,size=50)
    sqr_Image1.show()
    sqr_Image2 = square_generator_bo(orientation=0,beta=False,gamma=False,size=50)
    sqr_Image2.show()"""
    sqr_Image3 = square_generator_bo(orientation=90,beta=True,gamma=True,size=50)
    #sqr_Image3.show()
    sqr_Image4 = square_generator_bo(orientation=90,beta=False,gamma=False,size=50)
    #sqr_Image4.show()
    print()

    """
    #Test creat video from one pic
    video_batch=create_static_video_from_image(sqr_Image4,)
    print("Video shape:", video_batch.shape)
    
    """
    # Test creat video from 2 pics
    video_batch=create_static_video_from_two_images(sqr_Image4,sqr_Image3)
    print(video_batch.shape)
    print(video_batch)

    # try and play the video
     # Since we have one video sequence, extract it (shape: (n_frames, height, width, 3))
    video = video_batch[0]
    
    # Play the video using matplotlib
    play_video_matplotlib(video, fps=30)

