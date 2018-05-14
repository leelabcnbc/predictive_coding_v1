import os

import numpy as np
from scipy.io import loadmat

from pc_v1 import dir_dictionary
from pc_v1 import io


def test_v1_surround_suppression_dynamics_stim_b():
    ref_mat = loadmat(os.path.join(dir_dictionary['reference_V1_SuppressionMechanisms'],
                                   'test_v1_surround_suppression_dynamics_stim_b.mat'))

    grating_wavel = 6
    patch_diam = 11
    gap = 2
    image_size = 3 * patch_diam + 2 * gap
    contrast = 1
    phase = 0
    for test in range(2):
        for t in range(2):
            if t == 0:
                im_this = io.image_centre_surround(patch_diam, gap, 0.5 * (image_size - patch_diam),
                                                   grating_wavel, grating_wavel, 90, 90, phase, phase,
                                                   contrast, contrast)
            else:
                assert t == 1
                if test == 0:
                    im_this = io.image_centre_surround(patch_diam, gap, 0.5 * (image_size - patch_diam),
                                                       grating_wavel, grating_wavel, 90, 90, phase, phase,
                                                       contrast, contrast)
                else:
                    assert test == 1
                    im_this = io.image_centre_surround(patch_diam, gap, 0.5 * (image_size - patch_diam),
                                                       grating_wavel, grating_wavel, 0, 90, phase, phase,
                                                       contrast, contrast)
            im_this_ref = ref_mat['I_all'][test, t]
            assert im_this.shape == im_this_ref.shape
            # print(abs(im_this - im_this_ref).max())
            assert np.allclose(im_this, im_this_ref)


if __name__ == '__main__':
    test_v1_surround_suppression_dynamics_stim_b()
