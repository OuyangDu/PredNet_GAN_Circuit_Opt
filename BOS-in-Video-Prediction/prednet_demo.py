import numpy as np
import matplotlib.pyplot as plt
import os

def create_moving_square_videos(n_video=2, n_time_step=20, height=128, width=160, square_size=20):
    """
    Create videos of squares moving from left to right
    
    Args:
        n_video (int): Number of videos to generate
        n_time_step (int): Number of frames per video 
        height (int): Height of each frame
        width (int): Width of each frame
        square_size (int): Size of the square in pixels
        
    Returns:
        numpy array of shape (n_video, n_time_step, height, width, 3)
    """
    # Initialize empty videos array
    videos = np.zeros((n_video, n_time_step, height, width, 3))

    # Generate moving squares
    for vid in range(n_video):
        # Calculate starting y position randomly but keep square fully visible
        y_pos = np.random.randint(square_size, height-square_size)
        
        for t in range(n_time_step):
            # Calculate x position moving from left to right
            x_pos = int((width-square_size) * t/n_time_step)
            
            # Draw white square
            videos[vid, t, 
                  y_pos:y_pos+square_size,
                  x_pos:x_pos+square_size] = 1.0
            
    return videos


#### PredNet demo. You need to put border_ownership.agent.py file in the correct path so that import works.
from border_ownership.agent import Agent

## turn off CUDA if your GPU is not available or too new > 3060
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

WEIGHTS_DIR = './model_data_keras2/'
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/prednet_kitti_weights.hdf5')

output_mode = ['prediction', 'E0'] # a list of output modes. Can be 'prediction', 'E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3', 'A0', 'A1', 'A2', 'A3', 'Ahat0', 'Ahat1', 'Ahat2', 'Ahat3'. If prediction, output prediction; if others, output the units responses
agent = Agent()
agent.read_from_json(json_file, weights_file)

videos = create_moving_square_videos() # (n_video, n_time_step, height=128, width, 3). Make sure your video has correct shape

output = agent.output_multiple(videos, output_mode=output_mode, is_upscaled=False) # the output is a dictionary with keys as output_mode. is_upscaled=False means the input frame ranges from 0 to 1; is_upscaled=True means the input frame ranges from 0 to 255.

print(output['prediction'].shape) # The shape of prediction (n_video, n_time_step, height, width, 3)
print(output['E0'].shape) # The shape of E0 (n_video, n_time_step, E0_height, E0_width, E0_channel). The last three channels describes the shape of E0 module.

# visualize one prediction frame and one actual frame
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(output['prediction'][0, 5])
plt.title('Prediction')
plt.subplot(1, 2, 2)
plt.imshow(videos[0, 5])
plt.title('Actual')
plt.show()