import os
import numpy as np

from keras import backend as K
from keras.models import Model, model_from_json
from numpy.random import default_rng
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *

class PredNet_Evaluator:
    def __init__(self, nt=10, batch_size=10):
        self.nt = nt
        self.batch_size = batch_size
        self.test_model = None

    def load_prototype(self, wights_dir=None, weights_file='tensorflow_weights/prednet_kitti_weights.hdf5', json_file='prednet_kitti_model.json'):
        if weights_dir is None:
            weights_dir = WEIGHTS_DIR
        self.proto = load_prototype(weights_dir=weight_dir, weights_file=weights_file, json_file=json_file)

    def load_model(self, r_mask_indices=None, e_mask_indices=None):
        # Create testing model (to output predictions)
        layer_config = self.proto.layers[1].get_config()
        layer_config['output_mode'] = 'prediction'
        layer_config['r_mask_indices'] = r_mask_indices; layer_config['e_mask_indices'] = e_mask_indices
        self.data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
        test_prednet = PredNet(weights=self.proto.layers[1].get_weights(), **layer_config)

        input_shape = list(self.proto.layers[0].batch_input_shape[1:])
        input_shape[0] = self.nt
        inputs = Input(shape=tuple(input_shape))
        predictions = test_prednet(inputs)

        self.test_model = Model(inputs=inputs, outputs=predictions)

    def load_test_data(self, kitti_data_dir=None, test_file='new_X_test.hkl', test_sources='new_sources_test.hkl', N_seq=None, bootstrap=True, data_format='channels_first'):
        '''
        please load the model first, since the arrangement of data depends on the model
        '''
        if kitti_data_dir is None:
            kitti_data_dir = KITTI_DATA_DIR

        test_file_path = os.path.join(kitti_data_dir, test_file)
        test_sources_path = os.path.join(kitti_data_dir, test_sources)

        self.test_generator = SequenceGenerator(test_file_path, test_sources_path, self.nt, sequence_start_mode='unique', data_format=data_format, shuffle=True, N_seq=N_seq)
        self.X_test = self.test_generator.create_all()
        if bootstrap: self.X_test = random_sample_rows(self.X_test, N_seq)

        # if self.data_format == 'channels_first':
        #     X_test = np.transpose(self.X_test, (0, 1, 3, 4, 2))
        return X_test

    def predict(self):
        if self.test_model is None:
            raise Exception("Model not loaded. Call load_model() before predict().")
        X_hat = self.test_model.predict(self.X_test, self.batch_size)

        if self.data_format == 'channels_first':
            X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))
        return X_hat

def compute_pred_mse(X_test, X_hat, mask=None):
    '''
    X_test: (n_samples, nt, nx, ny, n_channels)
    mask: (nx, ny)
    '''
    if mask is not None:
        X_test_mask = X_test * mask[..., np.newaxis]
        X_hat_mask = X_hat * mask[..., np.newaxis]
    else:
        X_test_mask = X_test
        X_hat_mask = X_hat
    mse_model = np.mean((X_test_mask[:, 1:] - X_hat_mask[:, 1:])**2)
    mse_prev = np.mean((X_test_mask[:, :-1] - X_test_mask[:, 1:])**2)
    return mse_model, mse_prev

def random_sample_rows(dataset, N_seq=None, seed_value=42):
    n_sample = dataset.shape[0]
    N_seq = N_seq or n_sample  # N_seq defaults to n_sample if None

    rng = default_rng(seed_value)  # Create a generator with a seed
    sampled_indices = rng.choice(n_sample, size=N_seq, replace=True)
    sampled_dataset = dataset[sampled_indices]
    
    return sampled_dataset

def load_prototype(weights_dir=None, weights_file='tensorflow_weights/prednet_kitti_weights.hdf5', json_file='prednet_kitti_model.json'):
    if weights_dir is None:
        weights_dir = WEIGHTS_DIR
    weights_file='tensorflow_weights/prednet_kitti_weights.hdf5'; json_file='prednet_kitti_model.json' # TODO

    weights_path = os.path.join(weights_dir, weights_file)
    json_path = os.path.join(weights_dir, json_file)

    # Load trained model
    with open(json_path, 'r') as f:
        json_string = f.read()
    train_model = model_from_json(json_string, custom_objects={'PredNet': PredNet})
    train_model.load_weights(weights_path)
    return train_model

def ablate_model(proto, r_mask_indices=None, e_mask_indices=None, nt=20):
    # Create testing model (to output predictions)
    layer_config = proto.layers[1].get_config()
    layer_config['output_mode'] = 'prediction'
    layer_config['r_mask_indices'] = r_mask_indices; layer_config['e_mask_indices'] = e_mask_indices
    data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
    test_prednet = PredNet(weights=proto.layers[1].get_weights(), **layer_config)

    input_shape = list(proto.layers[0].batch_input_shape[1:])
    input_shape[0] = nt
    inputs = Input(shape=tuple(input_shape))
    predictions = test_prednet(inputs)

    test_model = Model(inputs=inputs, outputs=predictions)
    return test_model, data_format
