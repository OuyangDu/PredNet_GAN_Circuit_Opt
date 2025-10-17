# global parameters on the local computer
import os
# Where KITTI data will be saved if you run process_kitti.py
# If you directly download the processed data, change to the path of the data.
ERROR_TOL = 1e-10
DATA_DIR_HOME = './data/'
DATA_DIR = './data/test/data/rotating_square/'
KITTI_DATA_DIR = './data/kitti/'

FIGURE_DIR = './figure/'
# Where model weights and config will be saved if you run kitti_train.py
# If you directly download the trained weights, change to appropriate path.
WEIGHTS_DIR = './model_data_keras2/'

# Where results (prediction plots and evaluation file) will be saved.
RESULTS_SAVE_DIR = './kitti_results/'

# Where to store allen drive
ALLEN_DRIVE_PATH = './data/allen-brain-observatory/visual-coding-2p'

## turn off CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ALLEN_DRIFTING_GRATINGS_PARA = {'direction': [0, 45, 90, 135, 180, 225, 270, 315], 'temporal_frequency': [1, 2, 4, 8, 15]}
ALLEN_STATIC_GRATING_PAR = {'orientation': [0, 30, 60, 90, 120, 150]}

EQUAL_EPSILON = 1e-15 # a very small number. If the difference of two numbers smaller than this number we think they are equal
VISUAL_STIM_IMSHAPE = (200, 200) # default screen size for immaker/visual_stim_gen
RANDOM_SEED = 42


#################### Figure settings
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

# Colors
BLUE_PYTHON = "#0eb7eb"
BG_WHITE = "#fbf9f4"
GREY_LIGHT = "#b4aea9"
GREY50 = "#7F7F7F"
BLUE_DARK = "#1B2838"
BLUE = "#2a475e"
BLACK = "#282724"
GREY_DARK = "#747473"
RED_DARK = "#850e00"

# Colors taken from Dark2 palette in RColorBrewer R library
COLOR_SCALE = ["#1B9E77", "#D95F02", "#7570B3"]

# KITTI datasets name
KITTI_FILE_NAME={
    'city':
    [
       "2011_09_26_drive_0001",
       "2011_09_26_drive_0002",
       "2011_09_26_drive_0005",
       "2011_09_26_drive_0009",
       "2011_09_26_drive_0011",
       "2011_09_26_drive_0013",
       "2011_09_26_drive_0014",
       "2011_09_26_drive_0017",
       "2011_09_26_drive_0018",
       "2011_09_26_drive_0048",
       "2011_09_26_drive_0051",
       "2011_09_26_drive_0056",
       "2011_09_26_drive_0057",
       "2011_09_26_drive_0059",
       "2011_09_26_drive_0060",
       "2011_09_26_drive_0084",
       "2011_09_26_drive_0091",
       "2011_09_26_drive_0093",
       "2011_09_26_drive_0095",
       "2011_09_26_drive_0096",
       "2011_09_26_drive_0104",
       "2011_09_26_drive_0106",
       "2011_09_26_drive_0113",
       "2011_09_26_drive_0117",
       "2011_09_28_drive_0001",
       "2011_09_28_drive_0002",
       "2011_09_29_drive_0026",
       "2011_09_29_drive_0071",
    ],

'residential':
    [
       '2011_09_26_drive_0019',
       '2011_09_26_drive_0020',
       '2011_09_26_drive_0022',
       '2011_09_26_drive_0023',
       '2011_09_26_drive_0035',
       '2011_09_26_drive_0036',
       '2011_09_26_drive_0039',
       '2011_09_26_drive_0046',
       '2011_09_26_drive_0061',
       '2011_09_26_drive_0064',
       '2011_09_26_drive_0079',
       '2011_09_26_drive_0086',
       '2011_09_26_drive_0087',
       '2011_09_30_drive_0018',
       '2011_09_30_drive_0020',
       '2011_09_30_drive_0027',
       '2011_09_30_drive_0028',
       '2011_09_30_drive_0033',
       '2011_09_30_drive_0034',
       '2011_10_03_drive_0027',
       '2011_10_03_drive_0034',
    ],

'road':
    [
        '2011_09_26_drive_0015',
        '2011_09_26_drive_0027',
        '2011_09_26_drive_0028',
        '2011_09_26_drive_0029',
        '2011_09_26_drive_0032',
        '2011_09_26_drive_0052',
        '2011_09_26_drive_0070',
        '2011_09_26_drive_0101',
        '2011_09_29_drive_0004',
        '2011_09_30_drive_0016',
        '2011_10_03_drive_0042',
        '2011_10_03_drive_0047',
    ]

}
