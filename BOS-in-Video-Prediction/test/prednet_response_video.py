# collect PredNet responses for a video
from data_utils import SequenceGenerator
import scipy.io
from border_ownership.ploter import plot_seq_prediction
from border_ownership.agent import Agent
import matplotlib.pyplot as plt
from kitti_settings import *

kitti_data_dir = None
test_kitti_file = 'new_X_test.hkl'
test_kitti_sources = 'new_sources_test.hkl'
nt = 20

test_file_path = os.path.join(KITTI_DATA_DIR, test_kitti_file)
test_sources_path = os.path.join(KITTI_DATA_DIR, test_kitti_sources)

test_generator = SequenceGenerator(test_file_path, test_sources_path, nt=nt, sequence_start_mode='unique', data_format='channels_last', shuffle=True, N_seq=None)
X_test = test_generator.create_all()

weights_file='tensorflow_weights/prednet_kitti_weights.hdf5';
json_file='prednet_kitti_model.json'
weights_path = os.path.join(WEIGHTS_DIR, weights_file)
json_path = os.path.join(WEIGHTS_DIR, json_file)
sub = Agent()
sub.read_from_json(json_path, weights_path)

# output_mode = ['prediction', 'E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
# output = sub.output_multiple(X_test, cha_first=False, output_mode=output_mode, is_upscaled=False)
# output['video'] = X_test
# scipy.io.savemat(os.path.join(DATA_DIR_HOME, 'prednet_response.mat'), output)

output = scipy.io.loadmat(os.path.join(DATA_DIR_HOME, 'prednet_response.mat'))
plot_seq_prediction(output['video'][0], output['prediction'][0])
plt.show()
print(output['prediction'].shape)

