# create a dataset about prednet rsponses to orientation square
import numpy as np
import hickle as hkl
import os
import border_ownership.response_in_para as rip

from kitti_settings import *

class Prednet_Ori_Square_Dataset():
    def __init__(self, output_dark_path, output_light_path, label_path, output_mode='E2'):
        '''
        currently only support read in shape with shape (alpha, beta, gamma, time, neuron) if output_mode is not X. For X, (alpha, beta, gamma, time, width, height, 3)
        '''
        # load dataset
        output_dark = hkl.load(output_dark_path)
        output_light = hkl.load(output_light_path)
        angle = hkl.load(label_path)['angle']
        output_center, rf_para, self.alpha_rf_base, self.beta_rf_base, self.gamma_rf_base = rip.response_in_rf_paramemterization(output_dark, output_light, angle)
        self.time_rf_base = np.arange(output_center['X'].shape[3])

        # convert to shape (alpha, beta, gamma, time, neuron)
        self.output_module_base = output_center[output_mode] # (alpha, beta, gamma, time, neuron) if output_mode is not X. For X, (alpha, beta, gamma, time, width, height, 3)
        if output_mode == 'X':
            self.output_module_base = self.output_module_base[..., 0] # if X, one only of the RGB color will be maintained, so that the shape format of X is the same as other modules
            flat_shape = (*self.output_module_base.shape[:4], -1)
            self.output_module_base = self.output_module_base.reshape(flat_shape) # flat width and height
        self.label_name_base = ['alpha', 'beta', 'gamma', 'time']

    def _slice_time(self, time_cut):
        time_id = (time_cut[0] <= self.time_rf) * (self.time_rf < time_cut[1])
        self.output_module = self.output_module[..., time_id, :]
        self.time_rf = self.time_rf[time_id]
        return 

    def _average_time_window(self, time_cut):
        if self.time_rf is None:
            print('Data has been averaged on time')
            exit()
        self._slice_time(time_cut)
        self.output_module = np.mean(self.output_module, axis=3)
        self.time_rf = None

    def _make_copy_from_base(self):
        self.output_module = self.output_module_base.copy()
        self.alpha_rf, self.beta_rf, self.gamma_rf, self.time_rf = self.alpha_rf_base.copy(), self.beta_rf_base.copy(), self.gamma_rf_base.copy(), self.time_rf_base.copy()
        self.label_name = self.label_name_base.copy()

    def output_data(self, time_cut=None, time_processing=None):
        # make a copy of all attributes, then work on copies only
        self._make_copy_from_base()

        if time_processing == 'slice':
            self._slice_time(time_cut)
        elif time_processing == 'average':
            self._average_time_window(time_cut)

        # flat to format X, Y
        if self.time_rf is None:
            alpha_grid_rf, beta_grid_rf, gamma_grid_rf = np.meshgrid(self.alpha_rf, self.beta_rf, self.gamma_rf, indexing='ij')
            para = np.stack( (alpha_grid_rf, beta_grid_rf, gamma_grid_rf), axis=-1)
            label_name = ['alpha', 'beta', 'gamma']
        else:
            alpha_grid_rf, beta_grid_rf, gamma_grid_rf, time_grid_rf = np.meshgrid(self.alpha_rf, self.beta_rf, self.gamma_rf, self.time_rf, indexing='ij')
            para = np.stack( (alpha_grid_rf, beta_grid_rf, gamma_grid_rf, time_grid_rf), axis=-1)
            label_name = ['alpha', 'beta', 'gamma', 'time']

        n_neuron = self.output_module.shape[-1]
        X = self.output_module.reshape( (-1, n_neuron) )
        n_label = para.shape[-1]
        Y = para.reshape( (-1, n_label) )

        return X, Y, label_name
