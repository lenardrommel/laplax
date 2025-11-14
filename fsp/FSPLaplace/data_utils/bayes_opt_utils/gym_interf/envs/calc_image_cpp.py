from ctypes import *
import numpy as np
import os
import platform


def lib_path():
    dirname = os.path.dirname(__file__)

    system = platform.system()
    if system == 'Windows':
        lib_name = 'interf.dll'
    elif system == 'Darwin':
        lib_name = 'libinterf.dylib'
    else:
        lib_name = 'libinterf.so'

    return os.path.join(dirname, 'libs/' + lib_name)


libc = cdll.LoadLibrary(lib_path())

libc.calc_image.argtypes = [
    c_double, c_double, c_int, c_int, c_double,
    POINTER(c_double), POINTER(c_double), c_double, POINTER(c_double), c_double, c_int, c_double, c_double, c_double, c_double,
    POINTER(c_double), POINTER(c_double), c_double, POINTER(c_double), c_double, c_int, c_double, c_double, c_double, c_double,
    c_double, c_int, c_int, c_double, c_double, c_bool, c_double, c_double, c_double,
    c_int, POINTER(c_uint8), POINTER(c_double)
]


def calc_image(
        xstart, ystart, xpoints, ypoints, pixel_size,
        wave_vector1, center1, radius1, beam1_mask, length1, n_pixels1, sigma1x, sigma1y, beam1_ampl, beam1_rotation,
        wave_vector2, center2, radius2, beam2_mask, length2, n_pixels2, sigma2x, sigma2y, beam2_ampl, beam2_rotation,
        r_curvature, n_forward_frames, n_backward_frames, lamb, omega, has_interf,
        noise_coef, ampl_std, phase_std, use_beam_masks, n_threads=8):

    n_frames = n_forward_frames + n_backward_frames

    image = (c_uint8 * (n_frames * xpoints * ypoints))()
    total_intens = (c_double * n_frames)()

    def to_double_pointer(nparray):
        nparray = nparray.flatten()
        return nparray.ctypes.data_as(POINTER(c_double))

    if use_beam_masks:
        beam1_mask = to_double_pointer(beam1_mask)
        beam2_mask = to_double_pointer(beam2_mask)
    else:
        beam1_mask = None
        beam2_mask = None

    libc.calc_image(
        xstart, ystart, xpoints, ypoints, pixel_size,
        to_double_pointer(wave_vector1), to_double_pointer(center1), radius1, beam1_mask, length1, n_pixels1, sigma1x, sigma1y, beam1_ampl, beam1_rotation,
        to_double_pointer(wave_vector2), to_double_pointer(center2), radius2, beam2_mask, length2, n_pixels2, sigma2x, sigma2y, beam2_ampl, beam2_rotation,
        r_curvature, n_forward_frames, n_backward_frames, lamb, omega, has_interf, noise_coef, ampl_std, phase_std,
        n_threads, image, total_intens
    )

    result = np.ctypeslib.as_array(image)
    result = result.reshape(n_frames, xpoints, ypoints)

    result = result

    total_intens = np.ctypeslib.as_array(total_intens)

    return result, total_intens
