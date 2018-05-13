import numpy as np
from scipy.signal import convolve, correlate


def dim_conv(w_ff_on: np.ndarray, w_ff_off: np.ndarray,
             x_on: np.ndarray, x_off: np.ndarray, iterations=50,
             psi=5000, epsilon1=1e-4, epsilon2=None, verbose=True, return_y_only=True):
    """ implements dim_conv_on_and_off.m from V1_ResponseProperties

    :param w_ff_on:
    :param w_ff_off:
    :param x_on:
    :param x_off:
    :param iterations:
    :param psi:
    :param epsilon1:
    :param epsilon2:
    :param verbose:
    :return:
    """
    if epsilon2 is None:
        epsilon2 = 100 * epsilon1 * psi

    # check shape
    dim_conv_check_shape(w_ff_on, w_ff_off, x_on, x_off)
    # normalize
    (w_ff_on_normed, w_ff_off_normed), (w_fb_on, w_fb_off) = dim_conv_normalize_weights(w_ff_on, w_ff_off, psi)

    *_, h_x, w_x = x_on.shape
    y = np.zeros((w_ff_on.shape[0], h_x, w_x))

    y_all = []
    e_on_all = []
    e_off_all = []

    for it in range(iterations):
        if verbose:
            print(f'iteration {it+1}')

        # update error units.
        # let's do convolution.
        r_on = np.sum(np.asarray([
            convolve(y_this, w_this, mode='same') for (y_this, w_this) in zip(
                y, w_fb_on
            )
        ]), axis=0)

        r_off = np.sum(np.asarray([
            convolve(y_this, w_this, mode='same') for (y_this, w_this) in zip(
                y, w_fb_off
            )
        ]), axis=0)

        e_on = x_on / (epsilon2 + r_on)
        e_off = x_off / (epsilon2 + r_off)

        # update outputs.
        y = np.asarray([
            (epsilon1 + y_this) * (
                    correlate(e_on, w_on_this, mode='same') + correlate(e_off, w_off_this, mode='same')) for
            (y_this, w_on_this, w_off_this) in zip(y, w_ff_on_normed, w_ff_off_normed)
        ])
        y = np.maximum(y, 0)

        y_all.append(y.copy())
        e_on_all.append(e_on.copy())
        e_off_all.append(e_off.copy())

    if not return_y_only:
        return np.asarray(y_all), np.asarray(e_on_all), np.asarray(e_off_all)
    else:
        return np.asarray(y_all)


def dim_conv_check_shape(w_ff_on, w_ff_off, x_on, x_off):
    assert w_ff_on.shape == w_ff_off.shape and w_ff_on.ndim == 3
    assert x_on.shape == x_off.shape
    # right now, let's focus on static image case.
    assert x_on.ndim == 2


def dim_conv_normalize_weights(w_ff_on: np.ndarray, w_ff_off: np.ndarray, psi: np.float,
                               epsilon=1e-9
                               ):
    """
    pp. 3533 of the 2010 paper
    The kernels wON,k and wOFF,k were normalized so that sum of all the weights in both the ON and OFF channel
    was equal to \psi, and wˆON,k and wˆOFF,k were normalized so that the
    maximum value across both the ON and OFF channel was equal to \psi.

    :param psi:
    :param w_ff_on:
    :param w_ff_off:
    :param epsilon:
    :return:
    """

    assert np.all(w_ff_on >= 0) and np.all(w_ff_off >= 0)
    # they should be of size [c, h, w] for easier pytorch translation later on.
    # for (w_ff_on_this, w_ff_off_this) in zip(w_ff_on, w_ff_off):
    norm_sum = w_ff_on.sum(axis=(1, 2), keepdims=True) + w_ff_off.sum(axis=(1, 2), keepdims=True)
    norm_max = np.maximum(np.max(w_ff_on, axis=(1, 2), keepdims=True),
                          np.max(w_ff_off, axis=(1, 2), keepdims=True))

    w_ff_on_normed = w_ff_on / (norm_sum / psi + epsilon)
    w_ff_off_normed = w_ff_off / (norm_sum / psi + epsilon)

    w_fb_on = w_ff_on / (norm_max / psi + epsilon)
    w_fb_off = w_ff_off / (norm_max / psi + epsilon)

    return (w_ff_on_normed, w_ff_off_normed), (w_fb_on, w_fb_off)
