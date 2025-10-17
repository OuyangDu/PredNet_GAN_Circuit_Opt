# this code credits to William Lotter et al. 2017, https://coxlab.github.io/prednet/
import hickle as hkl
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator

# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator):
    def __init__(self, data_file, source_file, nt, label_file=None, label_name_file=None,
                 batch_size=8, shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None,
                 data_format=K.image_data_format()):
        self.X = hkl.load(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels) = (n_images, 128, 160, 3). data type is np.ndarray

        self.sources = hkl.load(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video. Shape is (n_images,), type is np.ndarray
        if not (label_file is None):
            self.label = hkl.load(label_file)
        else:
            self.label = {}

        if not (label_name_file is None):
            self.label_name = np.array( hkl.load(label_name_file) )
        else:
            self.label_name = []

        self.nt = nt
        self.batch_size = batch_size
        self.data_format = data_format
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode
        self.source_name = []

        if self.data_format == 'channels_first':
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape

        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
            if bool(self.label):
                self.possible_label = [self.label[self.sources[i]] for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]]
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            possible_label = []
            while curr_location < self.X.shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    self.source_name.append(self.sources[curr_location]) # the source_name correspond to this location
                    possible_starts.append(curr_location)
                    if bool(self.label):
                        possible_label.append(self.label[self.source_name[-1]])
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts
            if bool(self.label):
                self.possible_label = possible_label

        if shuffle:
            idx = range(len(self.possible_starts))
            idx_permut = np.random.permutation(idx).astype(int)
            self.possible_starts = [self.possible_starts[i] for i in idx_permut]
            if bool(self.label):
                self.possible_label = [self.possible_label[i] for i in idx_permut]
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
            if bool(self.label):
                self.possible_label = self.possible_label[:N_seq]
        self.N_sequences = len(self.possible_starts)
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)

    def __getitem__(self, null):
        return self.next()

    def next(self):
        with self.lock:
            current_index = (self.batch_index * self.batch_size) % self.n
            index_array, current_batch_size = next(self.index_generator), self.batch_size
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        return X.astype(np.float32) / 255

    def create_all(self, out_label=False):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt])
        if out_label:
            if bool(self.label):
                label_all = self.possible_label.copy()
            return X_all, label_all, self.label_name.copy()
        else:
            return X_all

def convert_prednet_output(output, label, label_name, t0=0):
    '''
    the output label of create_all is the label for each video. Here we create the second label for the time step. So that each image has two labels: time and video label. We also cut out the first t0 time steps because in which the prediction of the prednet is bad
    features will be flatten
    output = {'X': [n_video, n_frames, im_width, im_length, RGB value], 'R1': [n_video, n_frames, other ranks]}
    label ( [n_video, n_label_video] )
    every video must have the same number of frames
    '''
    for key in output:
        output[key] = output[key][:, t0:]

    n_frames = output['X'].shape[1]
    n_videos = output['X'].shape[0]
    label_time_arr = np.arange(n_frames)
    label_time = np.empty((*output['X'].shape[0:2], 1)) # [n_video, n_frame, 1]
    for i in range(label_time.shape[0]): label_time[i, :, 0] = label_time_arr # repeat the same time label for every video

    label_repeat = np.expand_dims(label, axis=1)
    label_repeat = np.tile(label_repeat, (1, n_frames, 1)) # [n_video, n_frame, n_label_video]

    label_out = np.append(label_repeat, label_time, axis=-1)
    label_out = label_out.reshape( (-1, label_out.shape[-1]) )

    for key in output:
        output[key] = output[key].reshape((n_videos*n_frames, -1))

    label_name = [*label_name, 'time_step']
    return output, label_out, label_name

if __name__ == "__main__":
    n_video = 2
    n_frames = 2
    n_features = 2
    x = np.empty((n_video, n_frames, n_features))
    label = np.empty( (n_video, 1) )
    print('label: \n', label)

    output = {}
    output['X'] = x

    print('X before: \n', x)

    output, label = convert_prednet_output(output, label)
    print('label out: \n', label)

    print('X out: \n', output['X'])
