import numpy as np
import math
from kitti_settings import *

def normalize_im_to_01(image, vmin=None, vmax=None):
    if vmin is None and vmax is None:
        vmin = np.min(image)
        vmax = np.max(image)
    return (image - vmin) / (vmax - vmin)

def create_grating(s_period, ori, phase, wave, imsize):
    """
    code modified from https://www.baskrahmer.nl/blog/neuro/generating-gratings-in-python-using-numpy/
    :param s_period: spatial period (in pixels). For example, imsize[0] = 160 and s_period = 16, there would be 10 white peaks in the image
    :param ori: wave orientation (in degrees, [0-360])
    :param phase: wave phase (in degrees, [0-360])
    :param wave: type of wave ('sqr' or 'sin')
    :param imsize: image size, tuple of int (integer)
    :return: numpy array of shape (imsize[0], imsize[1])
    """
    # Get x and y coordinates
    x, y = np.meshgrid(np.arange(imsize[0]), np.arange(imsize[1]))

    # Get the appropriate gradient
    gradient = np.cos(ori * math.pi / 180) * x - np.sin(ori * math.pi / 180) * y

    # Plug gradient into wave function
    if wave == 'sin':
        grating = np.cos((2 * math.pi * gradient ) / s_period - (phase * math.pi) / 180)
    elif wave == 'sqr':
        grating = signal.square((2 * math.pi * gradient) / s_period - (phase * math.pi) / 180)
    else:
        raise NotImplementedError

    # rescale to 0 to 1
    grating = normalize_im_to_01(grating, vmin=-1, vmax=1)

    return grating
