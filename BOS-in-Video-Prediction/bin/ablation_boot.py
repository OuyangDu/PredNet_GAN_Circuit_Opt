import hickle as hkl
import os
import copy
from border_ownership.ablation import AblationExperiment
from kitti_settings import *

experiment_params = hkl.load(os.path.join(RESULTS_SAVE_DIR, 'kitti_ablate_diff_num_units_para_temp.hkl'))
data = hkl.load(experiment_params['MOTHER_DATA'])

# Create an instance of the class with the defined variables
experiment = AblationExperiment(nt=experiment_params['nt'])
experiment.data = copy.deepcopy(data)
experiment.X_test = hkl.load(experiment_params['X_test_path'])

bo_info_path = os.path.join(DATA_DIR_HOME, experiment_params['bo_info_name'])
experiment.load_bom(bo_info_path=bo_info_path)
experiment.load_prototype()
experiment.run_ablation_experiment(experiment_params['module_list'], n_boot=experiment_params['nb'], n_ablation_start=experiment_params['n_ablation_start'], n_ablation_end=experiment_params['n_ablation_end'], is_bo_list=experiment_params['is_bo_list'])

hkl.dump(experiment.data, experiment_params['MOTHER_DATA'])
