import os

import numpy as np
from scipy.io import loadmat

from pc_v1 import dir_dictionary
from pc_v1 import io, core


def test_v1_orientation_tuning_contrast():
    # test `V1_orientation_tuning_contrast.m`
    ref_mat = loadmat(os.path.join(dir_dictionary['reference_V1_ResponseProperties'],
                                   'test_V1_orientation_tuning_contrast.mat'))

    grating_wavel = 6
    grating_angles = np.linspace(-22.5, 22.5, 7)
    contrasts = [0.05, 0.2, 0.8]
    patch_diam = 15
    phase = 0

    w_ff_on, w_ff_off = io.dim_conv_v1_filter_definitions()

    for j, contrast in enumerate(contrasts):
        for i, ga in enumerate(grating_angles):
            print(j, i)
            im_this = io.image_circular_grating(patch_diam, 20, grating_wavel, ga, phase, contrast)
            im_on, im_off = io.preprocess_image(im_this)

            y_init = core.dim_conv(w_ff_on, w_ff_off, im_on, im_off, iterations=1, verbose=False)
            y_full = core.dim_conv(w_ff_on, w_ff_off, im_on, im_off, iterations=12, verbose=False)
            y_init_ref = ref_mat['Y_init_all'][j, i]
            y_full_ref = ref_mat['Y_full_all'][j, i]

            # check init
            assert y_init.ndim == 4 and y_init.shape[0] == 1
            y_init = y_init[0].transpose((1, 2, 0))
            assert y_init.shape == y_init_ref.shape
            print(abs(y_init - y_init_ref).max())
            assert np.allclose(y_init, y_init_ref)

            # check full
            assert y_full.ndim == 4 and y_full.shape[0] == 12
            y_full = y_full.transpose((2, 3, 1, 0))
            assert y_full.shape == y_full_ref.shape
            print(abs(y_full - y_full_ref).max())
            # assert np.allclose(y_full, y_full_ref)
            assert abs(y_full - y_full_ref).max() < 5e-5


if __name__ == '__main__':
    test_v1_orientation_tuning_contrast()
