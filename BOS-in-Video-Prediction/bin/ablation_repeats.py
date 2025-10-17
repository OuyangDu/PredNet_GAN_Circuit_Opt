import os
import hickle as hkl
import copy
from data_utils import SequenceGenerator
from border_ownership.bo_indices_masker import BO_Masker
from border_ownership.ablation import AblationExperiment
from border_ownership.prednet_performance import random_sample_rows
from kitti_settings import *

def create_boot_list(n_boot, n_boot_sub):
    num_full = n_boot // n_boot_sub
    remainder = n_boot % n_boot_sub
    target_list = [n_boot_sub] * num_full
    if remainder > 0:
        target_list.append(remainder)
    assert sum(target_list) == n_boot
    return target_list

random_state = 0

experiment_default_params = {
    'random_state': random_state,
    'kitti_data_directory': KITTI_DATA_DIR,
    'bo_info_name': 'kitti_bo_info_subsampled.hkl',
    'test_file_name': 'new_X_test.hkl',
    'source_file_name': 'new_sources_test.hkl',
    'MOTHER_DATA': os.path.join(RESULTS_SAVE_DIR, f'kitti_ablate_diff_num_units_random{random_state}.hkl'),
    # 'MOTHER_DATA': os.path.join(f'./kitti_ablate_diff_num_units_random{random_state}_10boot.hkl'),
    'sequence_length': None,  # or specify the sequence length if needed
    'number_of_bootstraps': 10,
    'number_of_sub_bootstraps': 10,
    'nt': 20,
    'nb': None,
    'n_ablation_start': 1,
    'n_ablation_end': None,
    'N_seq': 10,
    'module_list': [],
    'is_bo_list': [True, False],
    'experiment_params_temp': os.path.join(RESULTS_SAVE_DIR, 'kitti_ablate_diff_num_units_para_temp.hkl'),
    'X_test_path': os.path.join(os.path.join(RESULTS_SAVE_DIR, 'X_test_temp.hkl'))
}

def execute(experiment_params):
    # load data
    test_file_path = os.path.join(experiment_params['kitti_data_directory'], experiment_params['test_file_name'])
    test_sources_path = os.path.join(experiment_params['kitti_data_directory'], experiment_params['source_file_name'])
    test_generator = SequenceGenerator(test_file_path, test_sources_path, nt=experiment_params['nt'], sequence_start_mode='unique', data_format='channels_first', shuffle=False, N_seq=None)
    X_test = test_generator.create_all()
    X_test = random_sample_rows(X_test, N_seq=experiment_params['N_seq'], seed_value=experiment_params['random_state'])
    hkl.dump(X_test, experiment_params['X_test_path'])

    bo_info_path = os.path.join(DATA_DIR_HOME, experiment_params['bo_info_name'])
    bo_info = hkl.load(bo_info_path)
    bom = BO_Masker(bo_info)
    min_units = {key: min(bom.n_bo_units[key], bom.n_non_bo_units[key]) for key in bom.n_bo_units.keys()}

    data = {}
    hkl.dump(data, experiment_params['MOTHER_DATA'])

    # no ablation
    experiment_params['n_ablation_end'] = -1
    hkl.dump(experiment_params, experiment_params['experiment_params_temp'])
    os.system(f"python ./bin/ablation_boot.py")
    data = hkl.load(experiment_params['MOTHER_DATA'])


    # ablation
    experiment_params['n_ablation_end'] = None
    boot_list = create_boot_list(experiment_params['number_of_bootstraps'], experiment_params['number_of_sub_bootstraps'])
    
    module_list = ['E0', 'E1', 'E2', 'E3', 'R0', 'R1', 'R2', 'R3']
    for module in module_list:
        experiment_params['module_list'] = [module]
        
        if module == 'E1': step_size = 20
        elif module == 'E2': step_size = 15
        else: step_size = 4
        for n in range(1, min_units[module] + step_size, step_size):
            if n > min_units[module]: n = min_units[module]

            experiment_params['n_ablation_start'] = n
            experiment_params['n_ablation_end'] = n + 1

            for nb in boot_list:
                experiment_params['nb'] = nb
                
                hkl.dump(experiment_params, experiment_params['experiment_params_temp'])
                os.system(f"python ./bin/ablation_boot.py")

for random_state in range(10): # random video samples
    experiment_default_params['random_state'] = random_state

    # KITTI
    experiment_params_kitti = copy.deepcopy(experiment_default_params)
    experiment_params_kitti['MOTHER_DATA'] = os.path.join(RESULTS_SAVE_DIR, f'kitti_ablate_diff_num_units_random{random_state}_test.hkl')
    execute(experiment_params_kitti)

    # Static Square
    # experiment_params_square = copy.deepcopy(experiment_default_params)
    # experiment_params_square['kitti_data_directory'] = DATA_DIR_HOME
    # experiment_params_square['test_file_name'] = 'square_bo_video_ori_x.hkl'
    # experiment_params_square['source_file_name'] = 'square_bo_video_ori_sources.hkl'
    # # experiment_params_square['MOTHER_DATA'] = os.path.join(RESULTS_SAVE_DIR, f'square_ablate_diff_num_units_random{random_state}.hkl')
    # execute(experiment_params_square)

    # experiment_params_trans = copy.deepcopy(experiment_default_params)
    # experiment_params_trans['kitti_data_directory'] = DATA_DIR_HOME
    # experiment_params_trans['bo_info_name'] = 'translating_bo_info_subsampled.hkl'
    # experiment_params_trans['test_file_name'] = 'square_bo_video_translating_x.hkl'
    # experiment_params_trans['source_file_name'] = 'square_bo_video_translating_sources.hkl'
    # experiment_params_trans['MOTHER_DATA'] = os.path.join(RESULTS_SAVE_DIR, f'trans_ablate_diff_num_units_random{random_state}.hkl')
    # execute(experiment_params_trans)

    # experiment_params_random = copy.deepcopy(experiment_default_params)
    # experiment_params_random['kitti_data_directory'] = DATA_DIR_HOME
    # experiment_params_trans['bo_info_name'] = 'random_bo_info_subsampled.hkl'
    # experiment_params_random['test_file_name'] = 'square_bo_video_random_x.hkl'
    # experiment_params_random['source_file_name'] = 'square_bo_video_random_sources.hkl'
    # experiment_params_random['MOTHER_DATA'] = os.path.join(RESULTS_SAVE_DIR, f'random_ablate_diff_num_units_random{random_state}.hkl')
    # execute(experiment_params_random)

data_kitti = hkl.load(experiment_params_kitti['MOTHER_DATA'])
print('data: ', data_kitti)
# data_square = hkl.load(experiment_params_square['MOTHER_DATA'])
# print('data: ', data_square)
# data_trans = hkl.load(experiment_params_trans['MOTHER_DATA'])
# print('data: ', data_trans)
# data_random = hkl.load(experiment_params_random['MOTHER_DATA'])
# print('data: ', data_random)
