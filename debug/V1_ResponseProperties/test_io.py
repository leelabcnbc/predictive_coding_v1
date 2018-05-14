import os

import numpy as np
from scipy.io import loadmat

from pc_v1 import dir_dictionary
from pc_v1 import io


def test_dim_conv_v1_filter_definitions():
    w_on, w_off = io.dim_conv_v1_filter_definitions()
    w_on = w_on.transpose((1, 2, 0))
    w_off = w_off.transpose((1, 2, 0))

    ref_mat = loadmat(os.path.join(dir_dictionary['reference_V1_ResponseProperties'],
                                   'test_dim_conv_V1_filter_definitions.mat'))
    w_on_ref, w_off_ref = ref_mat['wFFon'], ref_mat['wFFoff']
    assert w_on.shape == w_on_ref.shape
    assert w_off.shape == w_off_ref.shape
    # print(abs(w_on - w_on_ref).max())
    # print(abs(w_off - w_off_ref).max())
    assert np.allclose(w_on, w_on_ref)
    assert np.allclose(w_off, w_off_ref)


def test_circular_grating_and_process_image():
    ref_mat = loadmat(os.path.join(dir_dictionary['reference_V1_ResponseProperties'],
                                   'test_circular_grating_and_process_image.mat'))

    grating_wavel = 6
    grating_angles = np.linspace(-22.5, 22.5, 7)
    contrasts = [0.05, 0.2, 0.8]
    patch_diam = 15
    phase = 0

    for j, contrast in enumerate(contrasts):
        for i, ga in enumerate(grating_angles):
            # print(j, i)
            im_this = io.image_circular_grating(patch_diam, 20, grating_wavel, ga, phase, contrast)
            im_on, im_off = io.preprocess_image(im_this)
            im_this_ref = ref_mat['I_all'][j, i]
            im_on_ref = ref_mat['Ion_all'][j, i]
            im_off_ref = ref_mat['Ioff_all'][j, i]
            assert im_this.shape == im_this_ref.shape
            assert im_on.shape == im_on_ref.shape
            assert im_off.shape == im_off_ref.shape
            # print(abs(im_this - im_this_ref).max())
            # print(abs(im_on - im_on_ref).max())
            # print(abs(im_off - im_off_ref).max())
            assert np.allclose(im_this, im_this_ref)
            assert np.allclose(im_on, im_on_ref)
            assert np.allclose(im_off, im_off_ref)


def test_image_contextual_surround():
    ref_mat = loadmat(os.path.join(dir_dictionary['reference_V1_ResponseProperties'],
                                   'test_image_contextual_surround.mat'))

    grating_wavel = 6
    contrast = 0.25
    diams = np.arange(3, 32, 4)
    phase = 0

    for j in range(2):
        for i, diam in enumerate(diams):
            if j == 0:
                im_this = io.image_contextual_surround(diam, diams.max() - diam / 2, 0, grating_wavel, grating_wavel, 0,
                                                       phase, 0, contrast, 0)
            else:
                assert j == 1
                im_this = io.image_contextual_surround(0, diam / 2, diams.max() - diam / 2, grating_wavel,
                                                       grating_wavel, 0, 0, phase, 0, contrast)

            im_this_ref = ref_mat['I_all'][j, i]
            assert im_this.shape == im_this_ref.shape
            # print(abs(im_this - im_this_ref).max())
            assert np.allclose(im_this, im_this_ref)


def test_image_cross_ori_sine_and_square_grating():
    ref_mat = loadmat(os.path.join(dir_dictionary['reference_V1_ResponseProperties'],
                                   'test_image_cross_ori_sine_and_square_grating.mat'))

    grating_wavel = 6
    patch_diam = 51

    contrast = 0.5
    phase = 0

    grating_angles = np.arange(-60, 61, 20)

    for j in range(2):
        for i, angle in enumerate(grating_angles):
            if j == 0:
                im_this = io.image_cross_orientation_sine(patch_diam, grating_wavel, grating_wavel,
                                                          0, angle, phase, phase, contrast, contrast)
            else:
                assert j == 1
                im_this = io.image_square_grating(patch_diam, 0, grating_wavel, angle,
                                                  phase, contrast * 2)

            im_this_ref = ref_mat['I_all'][j, i]
            assert im_this.shape == im_this_ref.shape
            # print(abs(im_this - im_this_ref).max())
            assert np.allclose(im_this, im_this_ref)
            # also, I want to know, when angle is zero, whether two cases match.

            if angle == 0:
                im_this_ref_1 = ref_mat['I_all'][0, i]
                im_this_ref_2 = ref_mat['I_all'][1, i]
                assert im_this_ref_1.shape == im_this_ref_2.shape
                # print(abs(im_this_ref_1 - im_this_ref_2).max())
                assert np.allclose(im_this_ref_1, im_this_ref_2)


def test_v1_surround_suppression_orient_images():
    ref_mat = loadmat(os.path.join(dir_dictionary['reference_V1_ResponseProperties'],
                                   'test_V1_surround_suppression_orient_images.mat'))

    grating_wavel = 6
    phase = 0

    contrast_hi = 0.5
    context_angles = np.linspace(-90, 90, num=9, endpoint=True)

    maxdiam = 24
    patch_diam = 7
    for j in range(2):
        for i, ca in enumerate(context_angles):
            if j == 0:
                im_this_inner = io.image_contextual_surround(patch_diam, 0, patch_diam, grating_wavel,
                                                             grating_wavel, ca, phase, phase,
                                                             contrast_hi, contrast_hi)
            else:
                assert j == 1
                im_this_inner = io.image_circular_grating(patch_diam, patch_diam, grating_wavel, ca, phase,
                                                          contrast_hi)

            # %pad image so that they are the same size regardless of patch diameter
            # Yimeng: I use the way in my previous work. Not sure if equivalent to his.
            assert im_this_inner.shape == (im_this_inner.shape[0], im_this_inner.shape[0])
            diam_im_this_inner = im_this_inner.shape[0]
            im_this = np.full((maxdiam * 3, maxdiam * 3), fill_value=0.5, dtype=np.float64)
            box_t = int(np.floor((maxdiam * 3 - diam_im_this_inner) / 2.0))
            im_this[box_t:box_t + diam_im_this_inner, box_t:box_t + diam_im_this_inner] = im_this_inner

            im_this_ref = ref_mat['I_all'][j, i]
            assert im_this.shape == im_this_ref.shape
            # print(abs(im_this - im_this_ref).max())
            assert np.allclose(im_this, im_this_ref)


if __name__ == '__main__':
    test_dim_conv_v1_filter_definitions()
    test_circular_grating_and_process_image()
    test_image_contextual_surround()
    test_image_cross_ori_sine_and_square_grating()
    test_v1_surround_suppression_orient_images()
