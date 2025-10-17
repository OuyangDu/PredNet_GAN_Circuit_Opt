import os
import hickle as hkl
import time
from border_ownership.prednet_performance import load_prototype, ablate_model, compute_pred_mse, PredNet_Evaluator
from data_utils import SequenceGenerator
from border_ownership.rf_finder import out_of_range
from border_ownership.bo_indices_masker import BO_Masker
from kitti_settings import *

class Ablation_Evaluator:
    def __init__(self, bo_info=None, pixel_mask_distance=20, nt=20, batch_size=10):
        if bo_info is None:
            center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')
            bo_info = hkl.load(center_info_path)['bo_info']

        self.bom = BO_Masker(bo_info)
        self.pixel_mask_distance = pixel_mask_distance
        self.nt = nt
        self.batch_size = batch_size
        self.num_units = None

    def ablate_unit(self, num_units, is_bo=False, sample_method='random', put_channel_first=True):
        '''
        if is_bo is None, both bo and non_bo will be computed in each boot
        '''
        self.num_units = num_units.copy()
        self.sample_method = sample_method
        self.put_channel_first = put_channel_first
        self.is_bo = is_bo

    def load_prototype(self):
        self.evaluator = PredNet_Evaluator(nt=self.nt)
        self.evaluator.load_prototype()

    def compute_performance(self, data_dir=None, test_file='new_X_test.hkl', test_sources='new_sources_test.hkl', N_seq=None, bootstrap_data=True):
        e_mask_indices, r_mask_indices, _ = self.bom.sample_bo_indices_modules(
            self.num_units, is_bo=self.is_bo, sample_method=self.sample_method, put_channel_first=self.put_channel_first
        )

        evaluator = PredNet_Evaluator(nt=self.nt)
        evaluator.load_model(r_mask_indices=r_mask_indices, e_mask_indices=e_mask_indices)
        X_test = evaluator.load_test_data(kitti_data_dir=data_dir, test_file=test_file, test_sources=test_sources, N_seq=N_seq, bootstrap=bootstrap_data)
        X_hat = evaluator.predict()

        if self.pixel_mask_distance is None:
            mse_model, mse_prev = compute_pred_mse(X_test, X_hat, mask=None)
        else:
            inv_mask = out_of_range(X_test.shape[-3], X_test.shape[-2], distance=self.pixel_mask_distance)
            mse_model, mse_prev = compute_pred_mse(X_test, X_hat, mask=~inv_mask)
        return mse_model

    def compute_performance_boot(self, num_runs, data_dir=None, test_file='new_X_test.hkl', test_sources='new_sources_test.hkl', N_seq=None, verbose=True):
        mse_models = []
        for _ in range(num_runs):
            start_time = time.time()

            mse_model = self.compute_performance(data_dir=data_dir, test_file=test_file, test_sources=test_sources, N_seq=N_seq)
            mse_models.append(mse_model)

            end_time = time.time()
            if verbose: print(f'Run time for one boot: {end_time - start_time:.2f}s')
        return mse_models


class AblationExperiment():
    def __init__(self, nt):
        self.nt = nt
        self.data = {}

    def load_bom(self, bo_info_path=None, center_info_path=None):
        if bo_info_path is not None:
            bo_info = hkl.load(bo_info_path)
        elif center_info_path is None:
            center_info_path = os.path.join(DATA_DIR_HOME, 'center_neuron_info.hkl')
            bo_info = hkl.load(center_info_path)['bo_info']
        else:
            bo_info = hkl.load(center_info_path)['bo_info']

        self.bom = BO_Masker(bo_info)
        self.min_units = {key: min(self.bom.n_bo_units[key], self.bom.n_non_bo_units[key]) for key in self.bom.n_bo_units.keys()}

    def load_data(self, kitti_data_dir, test_kitti_file, test_kitti_sources, N_seq=None):
        test_file_path = os.path.join(kitti_data_dir, test_kitti_file)
        test_sources_path = os.path.join(kitti_data_dir, test_kitti_sources)
        test_generator = SequenceGenerator(test_file_path, test_sources_path, nt=self.nt, sequence_start_mode='unique', data_format='channels_first', shuffle=True, N_seq=N_seq)
        self.X_test = test_generator.create_all()
        return self.X_test

    def load_prototype(self):
        self.proto = load_prototype()
        return self.proto

    def ablate_and_predict(self, module, n_ablation, is_bo):
        e_mask_indices_bo, r_mask_indices_bo, _ = self.bom.sample_bo_indices_modules(n_units_dict={module: n_ablation}, is_bo=is_bo)
        prednet, _ = ablate_model(self.proto, r_mask_indices=r_mask_indices_bo, e_mask_indices=e_mask_indices_bo, nt=self.nt)
        X_hat = prednet.predict(self.X_test)
        return compute_pred_mse(self.X_test, X_hat)

    def run_ablation_experiment(self, module_list, n_boot=1, n_ablation_start=None, n_ablation_end=None, is_bo_list=[True, False]):
        if (n_ablation_end is not None) and n_ablation_end<=0:
            self.run_no_ablation()
            return self.data

        for module in module_list:

            print(f"working on {module}")
            step_size = 20 if module in ['E2', 'E1'] else 4

            if n_ablation_start is None:
                n_ablation_start_i = self.min_units[module]
            else:
                n_ablation_start_i = n_ablation_start

            if n_ablation_end is None:
                n_ablation_end_i = self.min_units[module] + step_size
            else:
                n_ablation_end_i = n_ablation_end

            for n_ablation in range(n_ablation_start_i, n_ablation_end_i, step_size):
                for _ in range(n_boot):
                    for is_bo in is_bo_list:
                        start_time = time.time()
                        mse, _ = self.ablate_and_predict(module, n_ablation, is_bo)
                        key = f'{is_bo}_{module}_{n_ablation}'
                        self.data.setdefault(key, []).append(mse)
                        elapsed_time = time.time() - start_time
                        print(f"{key}: MSE = {mse}, Elapsed time: {elapsed_time} seconds per boot")
        return self.data

    def run_no_ablation(self):
        prednet, _ = ablate_model(self.proto)
        mse, _ = compute_pred_mse(self.X_test, prednet.predict(self.X_test))
        self.data.setdefault('no_ablation', []).append(mse)
