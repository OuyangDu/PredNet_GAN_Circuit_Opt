import math
import os
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import itertools
from border_ownership.square_stimuli_generator import Square_Generator
from border_ownership.response_in_para import keep_central_neuron
from border_ownership.agent import Agent
from kitti_settings import WEIGHTS_DIR

def convert_rf_params_to_natural_params(orientation, beta, gamma, dark_grey, light_grey):
    """
    Convert receptive field parameters to natural parameters.
    
    Input:
    - orientation (float): Value from 0 to 180.
    - beta (bool): if true, square is in the upper-half plane, false in the lower-half.
    - gamma (bool): if true, light grey would be in the upper-half, false otherwise.
    - dark_grey (int): Dark grey value, ranges from 0 to 255.
    - light_grey (int): Light grey value.
    
    Output:
    - angle (float): Angle in degrees.
    - background_grey (int): Background grey value.
    - square_grey (int): Square grey value.
    """
    if (orientation > 180) or (orientation < 0):
        print('orientation must within 0 to 180')
        raise ValueError
    if dark_grey > light_grey:
        print('grey value of dark grey must be smaller than light grey')
        raise ValueError

    angle = orientation if beta else orientation + 180
    if (gamma and beta) or (not gamma and not beta):
        background_grey = dark_grey
        square_grey = light_grey
    else:
        background_grey = light_grey
        square_grey = dark_grey
    return angle, background_grey, square_grey

def shift_image(img, angle, shift_amount, fill_color=128):
    """
    Shift an image by a specified amount along a given angle.
    
    Input:
    - img (PIL Image): Input image.
    - angle (float): Angle in degrees.
    - shift_amount (int): Shift amount. Angle will be firstly converted to be smaller than 180. Imagine a vector with converted angle, direction of this vector is the positive direction of shift.
    - fill_color (int): Fill color for shifted pixels.
    
    Output:
    - final_img (PIL Image): Shifted image.
    """
    width, height = img.size
    angle_rad = np.deg2rad(angle % 180)
    y_sd, x_sd = int(shift_amount * np.cos(angle_rad)), -int(shift_amount * np.sin(angle_rad))
    im_np = np.asarray(img)
    im_shift = np.ones(im_np.shape) * fill_color
    for i in range(im_np.shape[0]):
        for j in range(im_np.shape[1]):
            idx, idy = i + x_sd, j + y_sd
            if 0 <= idx < height and 0 <= idy < width:
                im_shift[idx, idy] = im_np[i, j]
    final_img = Image.fromarray(np.uint8(im_shift))
    return final_img

def rotate_image(img, angle, fill_color=128):
    """
    Rotate an image by a specified angle.
    
    Input:
    - img (PIL Image): Input image.
    - angle (float): Angle in degrees.
    - fill_color (int): Fill color for rotated pixels.
    
    Output:
    - final_img (PIL Image): Rotated image.
    """
    width, height = img.size
    rotate_center = (width // 2, height // 2)
    final_img = img.rotate(angle, fillcolor=fill_color, center=rotate_center, resample=Image.BICUBIC)
    return final_img

def _check_img_pixel_format(img):
    '''
    check the image pixel format.
    input:
        img: PIL Image class. Grey image
    '''
    pixel_val = img.getpixel( (0, 0) )
    if type(pixel_val) == int:
        pixel_format = '255'
    else:
        pixel_format = '1'
    return pixel_format

def generate_grey_square_img(width, height, background_grey, square_grey, length_square):
    """
    Generate a grey square image with specified parameters.
    
    Input:
    - width (int): Width of the image.
    - height (int): Height of the image.
    - background_grey (int): Background grey value.
    - square_grey (int): Square grey value.
    - length_square (int): Length of the square side.
    
    Output:
    - img (PIL Image): Generated square image.
    """
    if length_square < 0:
        raise ValueError('Square size must be positive')

    img = Image.new(mode='L', size=(width, height), color=background_grey)
    center = (width // 2, height // 2)
    upper_left = (center[0], center[1] - length_square // 2)
    bottom_right = (center[0] + length_square, center[1] + length_square // 2)
    drawer = ImageDraw.Draw(img)
    drawer.rectangle([upper_left, bottom_right], fill=square_grey)
    return img

def square_generator_bo(dark_grey=255//3, light_grey=255 * 2 // 3, orientation=0, beta=True, gamma=True, shift=0, size=50, image_width=160, image_height=128, keep_keypoint_id=[], base_grey=128):
    """
    Generate a square image with specified parameters.
    
    Input:
    - dark_grey (int): Dark grey value.
    - light_grey (int): Light grey value.
    - orientation (float): Orientation value from 0 to 180.
    - beta (bool): Orientation type.
    - gamma (bool): Orientation type.
    - shift (int): Shift amount.
    - size (int): Size of the square.
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.
    - keep_keypoint_id (key point id): List of key points to be masked. from 0 to 7 are left_center, upper_left, top_center, upper_right, right_center, bottom_right, bottom_center, bottom_left
    - base_grey (int): Base grey value. This would be used if keep_keypoint_id is not empty.
    - fill_color_mode (str): Fill color mode. 'background' or 'square'.
    
    Output:
    - img (PIL Image): Generated square image.
    """

    angle, background_grey, square_grey = convert_rf_params_to_natural_params(orientation, beta, gamma, dark_grey, light_grey)
    img = generate_grey_square_img(image_width, image_height, background_grey, square_grey, size)

    key_points = get_square_key_points(image_width, image_height, size)
    img = keep_key_points(img, key_points, keep_keypoint_id, base_grey=base_grey)

    if len(keep_keypoint_id) == 0:
        fill_color = background_grey
    else:
        fill_color = base_grey

    img = rotate_image(img, angle, fill_color=fill_color)
    img = shift_image(img, angle, shift, fill_color=fill_color)
    return img

def get_square_key_points(im_width, im_height, length_square):
    '''
    get the key points of the square.
    input:
        im_width: width of the image
        im_height: height of the image
        length_square: length of the square
    output:
        square_key_points: a dictionary containing the key points of the square
    '''
    center = (im_width // 2 - 0.5, im_height // 2 - 0.5) # -0.5 to make sure the center is in the center of the pixel.

    # four coners
    upper_left = (center[0], center[1] - length_square // 2)
    bottom_right = (center[0] + length_square, center[1] + length_square // 2)
    upper_right = (upper_left[0] + length_square, upper_left[1])
    bottom_left = (upper_left[0], upper_left[1] + length_square)

    # Edge centers
    top_center = (center[0] + length_square//2, upper_left[1])
    bottom_center = (center[0] + length_square//2, bottom_left[1])
    left_center = (upper_left[0], center[1])
    right_center = (bottom_right[0], center[1])

    square_key_points = [left_center, upper_left, top_center, upper_right, right_center, bottom_right, bottom_center, bottom_left]
    # print(square_key_points)
    return square_key_points

def apply_radial_gaussian_multiple_points(image, points, sigma=5, base_grey=128):
    np_image = np.array(image).astype(int)
    np_image = np_image - base_grey
    x, y = np.meshgrid(np.arange(np_image.shape[1]), np.arange(np_image.shape[0]))
    gaussian_sum = np.zeros_like(x, dtype=float)

    # Sum Gaussian contributions for each point
    for point in points:
        d = np.sqrt((x - point[0])**2 + (y - point[1])**2)
        gaussian_sum += np.exp(-(d ** 2 / (2 * sigma ** 2)))
        
    # Normalizing the Gaussian sum to prevent overflow when multiplying
    # gaussian_sum = gaussian_sum / len(points)
    
    # Apply the accumulated Gaussian to the image
    np_image[:, :] = np_image[:, :] * gaussian_sum
        
    np_image = np_image + base_grey
    np_image = np.clip(np_image, 0, 255)
    result_image = Image.fromarray(np_image.astype('uint8'))
    return result_image

def add_grey_value_to_image(img, grey_value):
    np_img = np.array(img)
    np_img = np.clip(np_img + grey_value, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)

def keep_key_points(img, key_points, keep_keypoint_id, base_grey=128):
    '''
    keep the key points of the square, in another words, these key points will be masked with a gaussian function.
    '''
    if len(keep_keypoint_id) == 0:
        return img
    elif keep_keypoint_id[0] is None:
        img = Image.new(img.mode, img.size, base_grey)
        return img

    select_key_points = [key_points[i] for i in keep_keypoint_id]
    img = apply_radial_gaussian_multiple_points(img, select_key_points, base_grey=base_grey)
    return img

def convert_grey_img_to_rgb_video_list(img_list, n_frames=5):
    """
    Convert a list of grey images to an RGB video.
    
    Input:
    - img_list (list of PIL Images): List of grey images.
    - n_frames (int): Number of frames in the video.
    
    Output:
    - video_batch (numpy array): RGB video batch. value ranges from 0 to 1
    """
    input_pixel_format = _check_img_pixel_format(img_list[0])
    img_np_list = [np.asarray(img) for img in img_list]
    img_np = np.array(img_np_list)
    
    if input_pixel_format == '255':
        img_np = img_np / 255
    
    video_batch = np.repeat(img_np[:, np.newaxis], n_frames, axis=1)
    video_batch = np.repeat(video_batch[..., np.newaxis], 3, axis=-1)
    return video_batch

def square_generator_bo_batch(para_df):
    """
    Generate a batch of square images based on a DataFrame of parameters.
    
    Input:
    - para_df (Pandas DataFrame): DataFrame containing parameter values.
    
    Output:
    - img_list (list of PIL Images): List of generated square images.
    """
    img_list = [square_generator_bo(**row.to_dict()) for _, row in para_df.iterrows()]
    return img_list

def combine_parameters(para):
    """
    Combine parameter values to create a DataFrame of parameter combinations.
    
    Input:
    - para (dict): Dictionary of parameter lists.
    
    Output:
    - df (Pandas DataFrame): DataFrame of parameter combinations.
    """
    combinations = list(itertools.product(*para.values()))
    key_combinations = [{k: v for k, v in zip(para.keys(), combination)} for combination in combinations]
    df = pd.DataFrame(key_combinations)
    return df

def convert_neural_res_to_dataframe(neural_response):
    """
    Convert the final neural network results to a DataFrame.
    
    Input:
    - final_result (numpy array with shape [number of parameter combnations, number of time points, number of neurons]): Array of neural network results.
    
    Output:
    - neural_df (Pandas DataFrame): DataFrame of neural network results. It has three keys: parameter_id, neuron_id and response. Each value of response is a array of shape [number of time points]
    """
    n_param, n_time, n_neuron = neural_response.shape
    data = []
    
    for param_id in range(n_param):
        for neuron_id in range(n_neuron):
            response = neural_response[param_id, :, neuron_id]
            row = {'parameter_id': param_id, 'neuron_id': neuron_id, 'response': response}
            data.append(row)
    
    neural_df = pd.DataFrame(data)
    return neural_df

def merge_neural_res_with_para(neural_df, df):
    # Resetting the index of df to have a column for merging
    df_reset = df.reset_index().rename(columns={'index': 'parameter_id'})
    
    # Merging the neural_df with the original df based on parameter_id
    final_df = pd.merge(df_reset, neural_df, on='parameter_id')
    return final_df

def generate_square_and_neural_res(n_time, batch_size, para, prednet_json_file, prednet_weights_file, output_mode, batch_size_prednet=32):
    """
    Process data and generate a DataFrame.
    
    Input:
    - n_time (int): Number of time steps.
    - batch_size (int): Batch size.
    - para (dict): Dictionary of parameter lists.
    - prednet_json_file (str): Path to PredNet JSON file.
    - prednet_weights_file (str): Path to PredNet weights file.
    - output_mode (str): Output mode.
    - batch_size_prednet (int): Batch size for prednet.
    
    Output:
    - df (Pandas DataFrame): Generated DataFrame.
    """
    df = combine_parameters(para)
    groups = df.groupby(np.arange(len(df)) // batch_size)
    neural_res_all = []

    sub = Agent()
    sub.read_from_json(prednet_json_file, prednet_weights_file)
    for _, batch_df in groups:
        img_batch = square_generator_bo_batch(batch_df)
        video_batch = convert_grey_img_to_rgb_video_list(img_batch, n_frames=n_time)
        output_square = sub.output(video_batch, output_mode=output_mode, batch_size=batch_size_prednet, is_upscaled=False)
        output_square = {output_mode: output_square}
        output_square_center = keep_central_neuron(output_square)
        neural_res = output_square_center[output_mode]
        neural_res_all.append(neural_res)

    neural_res_all = np.concatenate(neural_res_all, axis=0)
    neural_res_df = convert_neural_res_to_dataframe(neural_res_all)

    df = merge_neural_res_with_para(neural_res_df, df)
    df = df.assign(module=output_mode)
    
    return df

def generate_square_and_neural_res_multiple_modes(n_time, batch_size, para, prednet_json_file, prednet_weights_file, output_mode_list):
    """
    Obtain neural responses to square stimuli with various of parameters specified by para.
    
    Input:
    - n_time (int): Number of time steps.
    - batch_size (int): Batch size.
    - para (dict): Dictionary of parameter lists.
    - prednet_json_file (str): Path to PredNet JSON file.
    - prednet_weights_file (str): Path to PredNet weights file.
    - output_mode (list of string): Output mode.
    
    Output:
    - df (Pandas DataFrame): Generated DataFrame.
    """
    df_list = []
    for output_mode in output_mode_list:
        df = generate_square_and_neural_res(n_time, batch_size, para, prednet_json_file, prednet_weights_file, output_mode)
        df_list.append(df)

    df = pd.concat(df_list)
    return df

def transform_batch_video_to_x_source(videos):
    '''
    input:
      videos: (n_videos, n_frames, height, width, n_channels)
    output:
        x: (n_videos * n_frames, height, width, n_channels)
        sources: list of strings corresponding to each video's name. One example is [v0, v0, v0, v1, v1, v1, ...]
    '''
    n_video, n_frames = videos.shape[0], videos.shape[1]
    x = videos.reshape(-1, *videos.shape[2:])
    sources = [f"v{i//n_frames}" for i in range(n_video * n_frames)]
    return x, sources
