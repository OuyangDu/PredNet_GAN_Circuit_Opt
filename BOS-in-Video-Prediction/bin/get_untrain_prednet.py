import os
from keras import backend as K
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Flatten, TimeDistributed
from keras.callbacks import ModelCheckpoint
from prednet import PredNet
from kitti_settings import *

weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' 'prednet_kitti_weights_untrain_small.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model_untrain_small.json')

# Define PredNet architecture parameters (same as in your code)
n_channels, im_height, im_width = (3, 128, 160)
chs_first = K.image_data_format() == 'channels_first'
input_shape = (n_channels, im_height, im_width) if chs_first else (im_height, im_width, n_channels)
stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = stack_sizes
filt_width = 3
A_filt_sizes = (filt_width, filt_width, filt_width)
Ahat_filt_sizes = (filt_width, filt_width, filt_width, filt_width)
R_filt_sizes = (filt_width, filt_width, filt_width, filt_width)
stddev = 0.0  # internal noise

# Create an instance of PredNet model without training
prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True, stddev=stddev)

# Define input shape
nt = 11  # number of timesteps used for sequences in training

# Create input layer
inputs = Input(shape=(nt, ) + input_shape)

# Get errors from PredNet model
errors = prednet(inputs)

# Calculate errors by layer and by time
layer_loss_weights = np.array([1., 0., 0., 0.])
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
errors_by_time = TimeDistributed(Dense(1), trainable=False)(errors)
errors_by_time = Flatten()(errors_by_time)
final_errors = Dense(1)(errors_by_time)

# Create the model
model = Model(inputs=inputs, outputs=final_errors)

# Save the model architecture to a JSON file
json_string = model.to_json()
with open(json_file, "w") as f:
    f.write(json_string)
print("Untrained PredNet architecture saved to", json_file)

# Save the model weights to an HDF5 file
model.save_weights(weights_file)
print("Untrained PredNet weights saved to", weights_file)
