from border_ownership.full_square_response import convert_grey_img_to_rgb_video_list, square_generator_bo
from border_ownership.grating_stimuli import create_grating
import numpy as np
import pandas as pd
import copy
import hickle as hkl
import matplotlib.pyplot as plt
from PIL import Image
from kitti_settings import *

def crop_center(image, width, height):
    img_width, img_height = image.size
    left = (img_width - width) / 2
    top = (img_height - height) / 2
    right = (img_width + width) / 2
    bottom = (img_height + height) / 2

    return image.crop((left, top, right, bottom))

class Switching_Stimuli():
    def generate_stimuli_prototype(self, t_len0, t_len1, *args, **kwargs):
        """
        Generate a square stimuli with the given parameters.

        :param t_len0: the length of the first image
        :param t_len1: the length of the second image
        :param args, kwargs: parameters about image generation
        return:
          para should contains t_len0 and t_len1 keys
          data = {'im0': im0, 'im1': im1, 'para': para} # im0 and im1 are pil image objects
        """
        NotImplementedError

    def generate_stimuli_prototype_batch(self, t_len0, t_len1, square_para_batch):
        '''
        Generate a square stimuli with the given parameters.
        input:
            t_len0: the time length of the first square. If it is a scalar, then it is the same for all the condition. If it is a numpy array, then it is the time length for each condition.
            t_len1: the time length of the second square. If it is a scalar, then it is the same for all the condition. If it is a numpy array, then it is the time length for each condition.
            square_para_batch: a list of dictionary containing the parameters of the square. The length of the list is the number of conditions.
        '''
        n_batch = len(square_para_batch)

        if not isinstance(t_len0, np.ndarray):
            t_len0 = np.ones(n_batch) * t_len0
            t_len1 = np.ones(n_batch) * t_len1

        self.data_list = []
        for i in range(n_batch):
            data = self.generate_stimuli_prototype(t_len0[i], t_len1[i], square_para_batch[i])
            self.data_list.append(data)
        # from list of dict into a dict with value as a list
        self.data_list = {k: [d[k] for d in self.data_list] for k in self.data_list[0]}


        return copy.deepcopy(self.data_list)

    @property
    def shape(self):
        return (len(self),)

    def dump(self, file_path):
        hkl.dump(self.data_list, file_path)

    def load(self, file_path):
        self.data_list = hkl.load(file_path)

    def __len__(self):
        return len(self.data_list['im0'])

    def _get_one_item(self, index):
        # index must be an integer
        if not isinstance(index, int):
            raise TypeError('index must be an integer')
        if index >= len(self):
            raise IndexError('index out of range')

        video0 = convert_grey_img_to_rgb_video_list([self.data_list['im0'][index]], n_frames=self.data_list['para'][index]['t_len0'])
        video1 = convert_grey_img_to_rgb_video_list([self.data_list['im1'][index]], n_frames=self.data_list['para'][index]['t_len1'])
        video = np.concatenate([video0, video1], axis=1)
        return video

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_one_item(index)
        elif isinstance(index, slice):
            video = []
            for i in range(*index.indices(len(self))):
                video.append(self._get_one_item(i))
            video = np.concatenate(video, axis=0)
            return video
        else:
            raise TypeError('index must be an integer or a slice')
        return video

    def export_para_df(self):
        '''
        export para as the format of df
        '''
        return pd.DataFrame(self.data_list['para'])

class Switching_Square_Stimuli(Switching_Stimuli):
    '''
    This class is used to generate switching square stimuli. Switching stimuli consists one square stimuli with t_len0 time steps and another half grey image with t_len1 time steps.
    To use this class. Initilize an instance say sss. Then generate data (not visuale to the user) by sss.generate_stimuli_prototype(t_len0, t_len1, square_para). where square_para is a list of parameter_dict. A good example is df_ori.to_dict('records') in the ./bin/generate_square_stim. Then you can treat sss as a list of videos parameters corresponding to the square_para. For example, sss[0] is the video corresponding to square_para[0]. sss[0:4] is the video corresponding to square_para[0:4]. You can also use sss.dump(path) to save the data to the path. sss.load(path) to load the data from the path.
    '''
    def generate_stimuli_prototype(self, t_len0, t_len1, square_para):
        """
        Generate a square stimuli with the given parameters.

        :param t_len0: the length of the first square
        :param t_len1: the length of the second square
        :param square_para: a dictionary containing the parameters of the square
        """
        t_len0 = int(t_len0)
        t_len1 = int(t_len1)
        square_para_copy = copy.deepcopy(square_para)
        im0 = square_generator_bo(**square_para_copy) # generate square image

        ## generate half plane image
        try: dwidth = square_para_copy['image_width'] * 2; dheight = square_para_copy['image_height'] * 2
        except: dwidth = 160 * 2; dheight = 128 * 2
        square_para_copy['size'] = 9999999; square_para_copy['image_width'] = dwidth; square_para_copy['image_height'] = dheight
        im1 = square_generator_bo(**square_para_copy)
        im1 = crop_center(im1, dwidth//2, dheight//2)

        para = copy.deepcopy(square_para)
        para['t_len0'] = t_len0; para['t_len1'] = t_len1;

        data = {'im0': im0, 'im1': im1, 'para': para} # im0 and im1 are pil image objects
        return data

class Switching_Square_to_Grey_Stimuli(Switching_Stimuli):
    def generate_stimuli_prototype(self, t_len0, t_len1, square_para):
        '''
        generate square stimuli with t length = t_len0, then generate a whole grey image with length equals to t_len1

        :param t_len0: the length of the first square
        :param t_len1: the length of the second square
        :param square_para: a dictionary containing the parameters of the square
        :return: a list of PIL Image object
        '''
        t_len0 = int(t_len0)
        t_len1 = int(t_len1)
        square_para_copy = copy.deepcopy(square_para)
        im0 = square_generator_bo(**square_para_copy) # generate square image

        try:
            light_grey = square_para_copy['light_grey']; dark_grey = square_para_copy['dark_grey'];
            base_grey = int((light_grey + dark_grey) / 2)
        except:
            base_grey = 128
        im1 = Image.new(mode='L', size=im0.size, color=base_grey)

        para = copy.deepcopy(square_para)
        para['t_len0'] = t_len0; para['t_len1'] = t_len1;

        data = {'im0': im0, 'im1': im1, 'para': para} # im0 and im1 are pil image objects
        return data

class Switching_Grey_to_Grey_Stimuli(Switching_Stimuli):
    def generate_stimuli_prototype(self, t_len0, t_len1, square_para):
        '''
        switch from darkgrey/lightgrey (beta = False, beta= True) to base grey (average of darkgrey and lightgrey)
        :param t_len0: the length of the the darkgrey/lightgrey
        :param t_len1: the length of the base grey
        :param square_para: a dictionary containing the parameters of the square. Only beta, size and grey values will be used
        '''
        t_len0 = int(t_len0)
        t_len1 = int(t_len1)
        square_para_copy = copy.deepcopy(square_para)
        try:
            light_grey = square_para_copy['light_grey']; dark_grey = square_para_copy['dark_grey'];
            base_grey = int((light_grey + dark_grey) / 2)
        except:
            light_grey = 255 * 2 // 3; dark_grey = 255 // 3;
            base_grey = 128

        first_grey = light_grey if square_para_copy['beta'] else dark_grey

        try: im0 = Image.new(mode='L', size=(square_para_copy['image_width'], square_para_copy['image_height']), color=first_grey)
        except: im0 = Image.new(mode='L', size=(160, 128), color=first_grey)

        im1 = Image.new(mode='L', size=im0.size, color=base_grey)

        para = copy.deepcopy(square_para)
        para['t_len0'] = t_len0; para['t_len1'] = t_len1;

        data = {'im0': im0, 'im1': im1, 'para': para} # im0 and im1 are pil image objects
        return data

class Switching_Pixel_Grey_Stimuli(Switching_Stimuli):
    def generate_stimuli_prototype(self, t_len0, t_len1, square_para):
        t_len0 = int(t_len0)
        t_len1 = int(t_len1)
        square_para_copy = copy.deepcopy(square_para)
        try:
            light_grey = square_para_copy['light_grey']; dark_grey = square_para_copy['dark_grey'];
            base_grey = int((light_grey + dark_grey) / 2)
        except:
            light_grey = 255 * 2 // 3; dark_grey = 255 // 3;
            base_grey = 128

        try: im1 = Image.new(mode='L', size=(square_para_copy['image_width'], square_para_copy['image_height']), color=base_grey)
        except: im1 = Image.new(mode='L', size=(160, 128), color=base_grey)

        im0 = im1.copy()
        pixel_grey = 255 if square_para_copy['beta'] else 0
        center = (im1.width // 2, im1.height // 2)
        im0.putpixel(center, pixel_grey)

        square_para_copy['t_len0'] = t_len0; square_para_copy['t_len1'] = t_len1;

        data = {'im0': im0, 'im1': im1, 'para': square_para_copy}
        return data

class Switching_Square_Flip(Switching_Stimuli):
    def generate_stimuli_prototype(self, t_len0, t_len1, square_para):
        t_len0 = int(t_len0)
        t_len1 = int(t_len1)
        square_para_copy = copy.deepcopy(square_para)
        im0 = square_generator_bo(**square_para_copy) # generate square image


        square_para_copy['beta'] = not square_para_copy['beta']
        im1 = square_generator_bo(**square_para_copy)

        para = copy.deepcopy(square_para)
        para['t_len0'] = t_len0; para['t_len1'] = t_len1;

        data = {'im0': im0, 'im1': im1, 'para': para} # im0 and im1 are pil image objects
        return data

class Switching_Ambiguous_Grey_Stimuli(Switching_Stimuli):
    def generate_stimuli_prototype(self, t_len0, t_len1, square_para):
        ## generate half plane image
        square_para_copy = copy.deepcopy(square_para)
        try: dwidth = square_para_copy['image_width'] * 2; dheight = square_para_copy['image_height'] * 2
        except: dwidth = 160 * 2; dheight = 128 * 2
        square_para_copy['size'] = 9999999; square_para_copy['image_width'] = dwidth; square_para_copy['image_height'] = dheight
        im0 = square_generator_bo(**square_para_copy)
        im0 = crop_center(im0, dwidth//2, dheight//2)

        try:
            light_grey = square_para_copy['light_grey']; dark_grey = square_para_copy['dark_grey'];
            base_grey = int((light_grey + dark_grey) / 2)
        except: base_grey = 128
        im1 = Image.new(mode='L', size=im0.size, color=base_grey)

        para = copy.deepcopy(square_para)
        para['t_len0'] = t_len0; para['t_len1'] = t_len1;

        data = {'im0': im0, 'im1': im1, 'para': para} # im0 and im1 are pil image objects
        return data

class Switching_Grating_Grey_Stimuli(Switching_Stimuli):
    def __init__(self):
        # unfortunately, the formatting of para has been used for quite a lot of classes. So here I can only set the default values indirectly
        self.default_val = {'s_period': 10, 'orientation': 0, 'phase': 0, 'wave': 'sin', 'image_height': 128, 'image_width': 160, 'light_grey': 255 * 2 // 3, 'dark_grey': 255 // 3, 'beta': False}

    def set_default_val(self, para):
        para_copy = copy.deepcopy(para)
        for key in self.default_val:
            if key not in para_copy:
                para_copy[key] = self.default_val[key]
        return para_copy

    def translate_gamma_to_phase(self, para):
        para['phase'] = 0 if para['gamma'] else 180
        return para

    def generate_stimuli_prototype(self, t_len0, t_len1, para, translate_gamma_to_phase=True):
        para_copy = self.set_default_val(para)

        if translate_gamma_to_phase:
            para_copy = self.translate_gamma_to_phase(para_copy)

        imsize = (para_copy['image_width'], para_copy['image_height'])
        im0 = create_grating(para_copy['s_period'], para_copy['orientation'], para_copy['phase'], para_copy['wave'], imsize)
        im0 = (im0 * 255).astype(np.uint8)
        im0 = Image.fromarray(im0)

        base_grey = int((para_copy['light_grey'] + para_copy['dark_grey']) / 2)
        im1 = Image.new(mode='L', size=im0.size, color=base_grey)

        para_copy['t_len0'] = t_len0; para_copy['t_len1'] = t_len1;

        data = {'im0': im0, 'im1': im1, 'para': para_copy} # im0 and im1 are pil image objects
        return data


class Oscillating_Square_Ambiguous():
    def __init__(self):
        self.ss_true = Switching_Square_Stimuli() # gamma = original
        self.ss_false = Switching_Square_Stimuli() # gamma = 1 - original

    def generate_stimuli_prototype_batch(self, t_len0, t_len1, square_para):
        square_para_copy = copy.deepcopy(square_para)
        self.ss_true.generate_stimuli_prototype_batch(t_len0, t_len1, square_para_copy)

        for one_para in square_para_copy:
            one_para['gamma'] = not one_para['gamma'] # invert gamma
        self.ss_false.generate_stimuli_prototype_batch(t_len0, t_len1, square_para_copy)
        self.data_list = [self.ss_true.data_list, self.ss_false.data_list]
        return copy.deepcopy(self.data_list)

    @property
    def shape(self):
        return (len(self),)

    def dump(self, file_path):
        hkl.dump(self.data_list, file_path)

    def load(self, file_path):
        self.data_list = hkl.load(file_path)
        self.ss_true.data_list = self.data_list[0]
        self.ss_false.data_list = self.data_list[1]

    def __len__(self):
        return len(self.ss_true)

    def _mix_videos(self, video_true, video_false, t_len0, t_len1):
        n_video, _, w, h, c = video_true.shape
    
        # New array to hold the mixed videos
        mixed_videos = np.empty((n_video, t_len0 + t_len1, w, h, c))
    
        for i in range(n_video):
            # Extract the first images from true and false videos
            vt_im0 = video_true[i, 0]
            vf_im0 = video_false[i, 0]

            # Mix the first subsequences (odd positions from vt_im0, even positions from vf_im0)
            mixed_first = np.vstack([[vt_im0, vf_im0] * (t_len0 // 2)])
            if t_len0 % 2 != 0:  # Handle if t_len0 is odd
                mixed_first = np.vstack((mixed_first, vt_im0[np.newaxis]))

            mixed_videos[i, :t_len0, ...] = mixed_first

            # Second subsequence is the second image from the true video
            vt_im1 = video_true[i, -1]
            mixed_videos[i, t_len0:t_len0+t_len1, ...] = vt_im1

        return mixed_videos

    def _get_one_item(self, index):
        # index must be an integer
        if not isinstance(index, int):
            raise TypeError('index must be an integer')
        if index >= len(self):
            raise IndexError('index out of range')

        video_true = self.ss_true._get_one_item(index)
        video_false = self.ss_false._get_one_item(index)

        t_len0 = self.ss_true.data_list['para'][index]['t_len0']
        t_len1 = self.ss_true.data_list['para'][index]['t_len1']

        video = self._mix_videos(video_true, video_false, t_len0, t_len1)
        return video

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._get_one_item(index)
        elif isinstance(index, slice):
            video = []
            for i in range(*index.indices(len(self))):
                video.append(self._get_one_item(i))
            video = np.concatenate(video, axis=0)
            return video
        else:
            raise TypeError('index must be an integer or a slice')
        return video

    def export_para_df(self):
        '''
        export para as the format of df
        '''
        return pd.DataFrame(self.ss_true.data_list['para'])
