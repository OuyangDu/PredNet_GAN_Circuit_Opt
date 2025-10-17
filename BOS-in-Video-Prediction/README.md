# Brain-like border ownership signals support prediction of natural videos

## Introduction
This repository contains the code for the paper "Brain-like border ownership signals support prediction of natural videos ".

## Main Repository Structure

- **bin/**: contains excutable files for reproducing paper's results
- **border_ownership/**: contains codes that being used by excutable files
- **kitti_settings.py**: Contains global hyperparameters and reads from `kitti_settings_lc.py`. If your GPU is available, comment out `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`. We found that running the PredNet model on GTX 3k+ is slower than on CPUs, while GTX 2k or below works better. This is probably because the PredNet model is a bit old (2017).
- **add_python_path**: add this repo to python path
- **model_data_keras2**: trained prednet models
- **env.yml**: suggested environment
- **paper.ipynb**: main file reproducing figures in the paper

## Environment Setup
    - Python 3.6
    - tensorflow-gpu==1.12.0
    - keras==2.2.4
    - hickle
    - mpi4py
    - seaborn, pandas, numpy, statsmodels, etc
    - Jupyter Notebook (optional for running experiment notebooks)

`env_yml` provides a simple suggestion of the environment. More comprehensive environment information (exported from Windows) can be found in `environment_detail.yaml`.

## Running the Experiment

### Method one: use the processed Data
1. We provided [key data files](https://wustl.box.com/s/ucum99kq03ucwb1hkqdta2okb6n3rcda). They include PredNet's prediction performance before/after ablation, BOS units' statistics and responses, and more. Place the data files in the folders indicated by `readme.docx`.

2. Add this repo to python path by running `add_python_path.bat` or something similar.

3. Run corresponding codes in `paper.ipynb` to reproduce main figures in the paper.

### Method two: Reproduce paper figure from scratch
1. This project relies on PredNet, a neural network trained for natural video prediction. PredNet training code is not included in this repo; it can be found in the [original PredNet repo](https://coxlab.github.io/prednet/) created by William Lotter. The trained PredNet model and processed KITTI videos were also available at the time of writing this README.

Trained models should be placed as `WEIGHTS_DIR/prednet_kitti_model.json` and `WEIGHTS_DIR/tensorflow_weights/prednet_kitti_weights.hdf5` where `WEIGHTS_DIR` is defined in `kitti_settings_lc.py`. The default value is `./model_data_keras2/`.

2. Run `add_python_path.bat` to add this repo to the python path.

3. Run `paper.ipynb`.

## Acknowledgement ##
This project is impossible without PredNet codes provided by [Lotter et al., 2017](https://coxlab.github.io/prednet/).

1. Lotter, W., Kreiman, G. & Cox, D. Deep predictive coding networks for video prediction and unsupervised learning. 5th Int. Conf. Learn. Represent. ICLR 2017 - Conf. Track Proc. 1–18 (2017).
