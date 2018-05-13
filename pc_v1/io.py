"""some IO related functions for running PC demos"""

import numpy as np
from scipy.signal import convolve2d


def fspecial(filter_type, *args):
    # based the code of special of MATLAB R2017b.
    # only cared `log` part.
    if filter_type == 'log':
        p2, p3 = args
        p2 = np.asarray([p2, p2])
        siz = (p2 - 1) // 2
        std2 = p3 ** 2

        [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
        arg = -(x * x + y * y) / (2 * std2)

        h = np.exp(arg)
        # https://stackoverflow.com/questions/19141432/python-numpy-machine-epsilon

        h[h < np.finfo(np.float64).eps * h.max()] = 0

        sumh = h.sum()
        if sumh != 0:
            h = h / sumh
        # now calculate Laplacian
        h1 = h * (x * x + y * y - 2 * std2) / (std2 ** 2)
        h = h1 - np.sum(h1) / np.product(p2)  # make the filter sum to zero
    else:
        raise NotImplementedError

    return h


def gabor(sigma, orient, wavel, phase, aspect, pxsize=None):
    """implements gabor.m from V1_ResponseProperties"""
    if pxsize is None:
        pxsize = int(np.fix(5 * sigma))

    radius = pxsize // 2
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))

    # rotation
    orient = -orient * np.pi / 180
    x_theta = x * np.cos(orient) + y * np.sin(orient)
    y_theta = -x * np.sin(orient) + y * np.cos(orient)

    phase = phase * np.pi / 180
    freq = 2 * np.pi / wavel

    gb = np.exp(-0.5 * (((x_theta ** 2) / (sigma ** 2)) + ((y_theta ** 2) / ((aspect * sigma) ** 2)))) * (
            np.cos(freq * y_theta + phase) - np.cos(phase) * np.exp(-0.25 * ((sigma * freq) ** 2)))

    return gb


def preprocess_image(im):
    """implements preprocess_image.m from V1_ResponseProperties"""
    # I think this image should be within the range of 0-1
    assert im.ndim == 2 and np.all(im <= 1) and np.all(im >= 0)

    log_f: np.ndarray = -fspecial('log', 9, 1)
    tmp = log_f.copy()
    # no idea why normalize this way.
    tmp[tmp < 0] = 0
    log_f = log_f / tmp.sum()

    x = convolve2d(im, log_f, mode='same')
    x = np.tanh(2 * np.pi * x)
    # split
    x_on = np.maximum(x, 0)
    x_off = np.maximum(-x, 0)
    return x_on, x_off


def image_circular_grating(csize, vsize, wavel, angle, phase, cont):
    """implements image_circular_grating.m from V1_ResponseProperties"""

    freq = 2 * np.pi / wavel
    angle = -angle * np.pi / 180
    phase = phase * np.pi / 180
    #
    # %define image size
    sz = int(np.fix(csize + 2 * vsize))
    if sz % 2 == 0:
        sz += 1

    # define mesh on which to draw sinusoids
    im_rad = sz // 2
    x, y = np.meshgrid(np.arange(-im_rad, im_rad + 1), np.arange(-im_rad, im_rad + 1))

    yg = -x * np.sin(angle) + y * np.cos(angle)
    #
    # %make sinusoids with values ranging from 0 to 1 (i.e. contrast is positive)
    grating = 0.5 + 0.5 * cont * np.cos(freq * yg + phase)
    #
    # %define radius from centre point
    radius = np.sqrt(x ** 2 + y ** 2)
    #
    # %put togeter image from components
    im = np.full((sz, sz), fill_value=0.5, dtype=np.float64)
    im[radius < csize / 2] = grating[radius < csize / 2]

    return im


def dim_conv_v1_filter_definitions():
    """implements dim_conv_V1_filter_definitions.m from V1_ResponseProperties"""
    wavel = 6
    sigma = wavel / 1.5
    aspect = 1 / np.sqrt(2)

    wff_on = []
    wff_off = []

    # %DEFINE feedforward WEIGHTS
    for phase in range(0, 360, 90):
        for angle in np.linspace(0, 180, 8, endpoint=False):
            # %ON and OFF channels (modelled by Gabor split into positive and negative parts)
            gb = gabor(sigma, angle, wavel, phase, aspect)
            norm = abs(gb).sum()
            gb = gb / norm
            wff_on.append(np.maximum(gb, 0))
            wff_off.append(np.maximum(-gb, 0))

    return np.asarray(wff_on), np.asarray(wff_off)
