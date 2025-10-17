# go to colab and get new_X_test.npy and new_source_path.txt. Come here to get the hkl version
import hickle as hkl
import numpy as np
import os
from kitti_settings import *

ori_test_file = os.path.join(KITTI_DATA_DIR, 'new_X_test.npy')
ori_test_sources = os.path.join(KITTI_DATA_DIR, 'new_source_path.txt')

test_file = os.path.join(KITTI_DATA_DIR, 'new_X_test.hkl')
test_sources = os.path.join(KITTI_DATA_DIR, 'new_sources_test.hkl')

########## Load the data
with open(ori_test_sources, 'r') as file:
  source = [line.rstrip('\n') for line in file]
x = np.load(ori_test_file)

hkl.dump(x, test_file); hkl.dump(source, test_sources)

x=hkl.load(test_file)
