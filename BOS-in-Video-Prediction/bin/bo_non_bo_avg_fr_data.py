import numpy as np
import os
import hickle as hkl
from scipy.stats import ranksums
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from border_ownership.ploter import plot_seq_prediction, plot_layer_boxplot_helper
from border_ownership.agent import Agent
from data_utils import SequenceGenerator
from kitti_settings import *

class VideoAnalysis:
    def load_video_paths(self, video_file, source_file):
        self.test_file_path = video_file
        self.test_sources_path = source_file
    
    def load_center_info(self, center_info_file):
        self.center_info = hkl.load(center_info_file)
    
    def initialize_agent(self, json_file, weights_file):
        self.agent = Agent()
        self.agent.read_from_json(json_file, weights_file)

    def process_video(self, nt=20, sequence_start_mode='unique', data_format='channels_first', shuffle=False, N_seq=None):
        test_generator = SequenceGenerator(self.test_file_path, self.test_sources_path, nt=nt,
                                           sequence_start_mode=sequence_start_mode, data_format=data_format,
                                           shuffle=shuffle, N_seq=N_seq)
        video = test_generator.create_all()
        video = video.transpose(0, 1, 3, 4, 2)  # Reformatting for certain visualizations or processing
        return video
    
    def compute_responses(self, video, output_mode, batch_size=32, is_upscaled=False, cha_first=True):
        if not self.agent:
            raise ValueError("Agent model is not initialized.")
        
        response_batch = self.agent.output_multiple(video, output_mode=output_mode, 
                                                    batch_size=batch_size, is_upscaled=is_upscaled, 
                                                    cha_first=cha_first)
        
        bo_avg_res = {}; nbo_avg_res = {};
        bo_id = {}; nbo_id = {}
        for module in output_mode:
            bo_avg_res[module] = []; nbo_avg_res[module] = []
            bo_id[module] = []; nbo_id[module] = []

            bo_info_module = self.center_info['bo_info'][module]
            bo_neuron_id = bo_info_module[bo_info_module['is_bo']]['neuron_id'].tolist()
            nbo_neuron_id = bo_info_module[~bo_info_module['is_bo']]['neuron_id'].tolist()

            for bni in bo_neuron_id:
                res_square = response_batch[module][:, :, bni[0], bni[1], bni[2]]**2
                bo_avg_res[module].append(res_square.mean())
                bo_id[module].append(bni)
            for nni in nbo_neuron_id:
                res_square = response_batch[module][:, :, nni[0], nni[1], nni[2]]**2
                nbo_avg_res[module].append(res_square.mean())
                nbo_id[module].append(nni)

        return {'bo_avg_res': bo_avg_res, 'nbo_avg_res': nbo_avg_res, 'bo_id': bo_id, 'nbo_id': nbo_id}

video_path = {
    'x_test':
    [
        os.path.join(DATA_DIR_HOME, 'square_bo_video_random_x.hkl'),
        os.path.join(DATA_DIR_HOME, 'square_bo_video_translating_x.hkl'),
        os.path.join(KITTI_DATA_DIR, 'new_X_test.hkl')
    ],
    'sources_test':
    [
        os.path.join(DATA_DIR_HOME, 'square_bo_video_random_sources.hkl'),
        os.path.join(DATA_DIR_HOME, 'square_bo_video_translating_sources.hkl'),
        os.path.join(KITTI_DATA_DIR, 'new_sources_test.hkl')
    ]
}
video_head = ['random', 'translating', 'kitti']

center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')
prednet_json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
prednet_weights_file = os.path.join(WEIGHTS_DIR, 'tensorflow_weights/' + 'prednet_kitti_weights.hdf5')

va = VideoAnalysis()
va.load_center_info(center_info_path)
va.initialize_agent(prednet_json_file, prednet_weights_file)
data = {}
for i in range(len(video_head)):
    va.load_video_paths(video_path['x_test'][i], video_path['sources_test'][i])
    video = va.process_video()
    response_data = va.compute_responses(video, ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3'])
    data[video_head[i]] = response_data

hkl.dump(data, os.path.join(DATA_DIR_HOME, 'bo_non_bo_response.hkl'))

# def print_dict_structure(d, indent=0):
#     for key, value in d.items():
#         print('    ' * indent + str(key))
#         if isinstance(value, dict):
#             print_dict_structure(value, indent+1)
# print_dict_structure(data)
# print(data)
