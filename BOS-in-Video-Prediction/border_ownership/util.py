import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def compute_max_contour_distance_to_center(image, z_thresh=1):
    # Convert the image to binary
    binary_img = np.where(image > z_thresh, 1, 0)

    # Find the boundaries of the regions using gradient
    boundary = ndimage.morphological_gradient(binary_img, size=(3,3)).astype(bool)

    # Calculate the image center
    center_y, center_x = np.array(image.shape) / 2

    # Get the y,x coordinates of the boundary pixels
    y, x = np.where(boundary)

    # Calculate distances from boundary pixels to the center
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Compute average distance
    if len(distances) == 0:
        max_distance = np.maximum(*image.shape)
    else:
        max_distance = np.max(distances)
    return max_distance

def circular_average(bo_t, alpha, with_angle=False):
    # Convert angles to radians
    alpha_rad = np.deg2rad(alpha)
    
    # Convert BOI to x and y components
    x_components = bo_t * np.cos(alpha_rad)[:, None]
    y_components = bo_t * np.sin(alpha_rad)[:, None]
    
    # Sum x and y components separately
    sum_x = np.sum(x_components, axis=0)/ bo_t.shape[0]
    sum_y = np.sum(y_components, axis=0)/ bo_t.shape[0]
    
    # Convert summed components back to magnitude to get circular average
    circular_avg_magnitude = np.sqrt(sum_x**2 + sum_y**2)

    if with_angle:
        averaged_angle = np.arctan2(sum_y, sum_x)
        circular_avg_angle = np.rad2deg(averaged_angle) % 360
        return circular_avg_magnitude, circular_avg_angle

    return circular_avg_magnitude

def compute_group_medians(dataset, num_groups=10):
    # Get the dimensions of the dataset
    n_sample, n_neurons = dataset.shape

    # Calculate the group size (number of samples per group)
    group_size = n_sample // num_groups

    # Initialize a list to store the medians
    medians = []

    # Split the dataset into groups, calculate the median for each group, and store it
    for i in range(num_groups):
        start_index = i * group_size
        end_index = (i + 1) * group_size

        # Extract the group from the dataset
        group_data = dataset[start_index:end_index, :]

        # Flatten the group into a 1D array
        flat_group = group_data.flatten()

        # Compute the median for the group and add it to the list of medians
        group_median = np.median(flat_group)
        medians.append(group_median)

    return medians

def imshow_central_region(image, region_size=9):
    half_size = region_size // 2
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
    central_region = image[center_y - half_size:center_y + half_size + 1, 
                           center_x - half_size:center_x + half_size + 1]
    plt.imshow(central_region)
    plt.show()

def sci_notation(num, precision=2):
    if num == 0:
        return "0"

    exponent = int(np.floor(np.log10(abs(num))))
    mantissa = round(num / (10**exponent), precision)
    return f"${mantissa} \\times 10^{{{exponent}}}$"

def df_to_arr(df, value_name, condition_names=[]):
    '''
     Convert a DataFrame to a numpy array.
     inputs:
         df: DataFrame
         value_name: str, the name of the column that contains the values
         condition_names: list of str, the names of the columns that contain the conditions. Make sure that for specified a condition setting, there will only be one row (one value) sastifying the conditions
         value_type: str, the type of the values. 'scalar' or 'array'
     outputs:
         arr: numpy array. The shape of the array is [len_c[0], len_c[1], ..., n_values], where len_c[i] is the number of unique values in the i-th condition column, n_values is the number of values for each condition setting. If value_type is 'scalar', n_values = 1; if value_type is 'array', n_values is the length of the array in the value column.
         unique_values: list of numpy arrays. The i-th element is the unique values in the i-th condition column.
    '''
    # Verify input parameters
    if value_name not in df.columns:
        raise ValueError(f"{value_name} is not a column in the DataFrame.")
    if any(name not in df.columns for name in condition_names):
        raise ValueError("One or more condition names are not valid columns in the DataFrame.")
    
    # Get unique values and index mappings for each condition
    unique_values = [df[name].unique() for name in condition_names]
    index_maps = [{value: idx for idx, value in enumerate(values)} for values in unique_values]
    
    # Determine the shape of the output array
    sample_val = df[value_name].iloc[0]
    n_values = len(sample_val) if hasattr(sample_val, '__len__') else 1

    arr_shape = [len(values) for values in unique_values] + [n_values]
    arr = np.zeros(arr_shape, dtype=float)  # Default dtype is float, could be parameterized

    # Fill the array with values from the DataFrame
    for idx, row in df.iterrows():
        indices = [index_maps[i][row[col]] for i, col in enumerate(condition_names)]
        arr[tuple(indices)] = np.array(row[value_name])

    return arr, unique_values
