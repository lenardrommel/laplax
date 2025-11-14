import math
import numpy as np


def rotation_matrix2(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return np.asarray(vector) / np.linalg.norm(vector)


def cross_product(vector1, vector2):
    return np.cross(vector1, vector2)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = unit_vector(axis)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    one_minus_cos = 1 - cos_theta

    return np.array([
        [
            cos_theta + one_minus_cos * axis[0] ** 2,
            one_minus_cos * axis[0] * axis[1] - sin_theta * axis[2],
            one_minus_cos * axis[0] * axis[2] + sin_theta * axis[1]
        ],
        [
            one_minus_cos * axis[1] * axis[0] + sin_theta * axis[2],
            cos_theta + one_minus_cos * axis[1] ** 2,
            one_minus_cos * axis[1] * axis[2] - sin_theta * axis[0]
        ],
        [
            one_minus_cos * axis[2] * axis[0] - sin_theta * axis[1],
            one_minus_cos * axis[2] * axis[1] + sin_theta * axis[0],
            cos_theta + one_minus_cos * axis[2] ** 2
        ]
    ])


def rotate(vector, axis, angle):
    matrix = rotation_matrix(axis, angle)
    return np.dot(vector.T, matrix)


def rotate_x(vector, angle):
    return rotate(vector, [1, 0, 0], angle)


def rotate_y(vector, angle):
    return rotate(vector, [0, 1, 0], angle)


def rotate_euler_angles(vector, angles):
    axis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for angle, ax in zip(angles, axis):
        vector = rotate(vector, ax, angle)
    return vector


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            angle_between((1, 0, 0), (1, 0, 0))
            0.0
            angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def reflect(vector, normal):
    """
    'mirror' reflection
    :param vector:
    :param normal:
    :return:
    """
    rotation_axis = cross_product(normal, vector)
    alpha = math.pi - angle_between(vector, normal)

    assert alpha > 0 and alpha < math.pi / 2, alpha / math.pi * 180

    rotation_angle = math.pi - 2 * alpha

    rotated_vector = rotate(vector, rotation_axis, rotation_angle)
    return rotated_vector


def back_track(target, normal_vector, center):
    """
    :param target: projection point
    :param normal_vector: normal vector to the light source plane
    :param center: center of the light source plane
    :return: source vector
    """
    dot_scalar = np.dot(np.subtract(target, center), normal_vector)
    # now returning the position vector of the projection onto the plane
    return np.subtract(target, dot_scalar * normal_vector)


def project(source, source_normal, target_normal, target_center):
    """
    reverse to back_track
    :param source:
    :param source_normal:
    :param target_normal:
    :param target_center:
    :return:
    """
    normal_distance = np.dot(np.subtract(source, target_center), target_normal)
    sx = normal_distance / np.dot(-source_normal, target_normal)
    return source + source_normal * sx


def dist(a, b):
    """
    distance between two poins
    :param a:
    :param b:
    :return:
    """
    return np.linalg.norm(a - b)


def visibility(radius_top, radius_bottom, r_curvature, x, y, kx, ky, lamb):
    # works for interf without lenses and for interf with one telescope
    n = radius_bottom / radius_top
    n2 = n * n
    r2 = radius_top ** 2
    c2 = ((n2 + 1) / (n2 * r2)) ** 2 + (np.pi / (lamb * r_curvature)) ** 2

    exp1 = np.exp(
        -(x ** 2 + y ** 2) *
        (1 / (r2 * n2) - (n2 + 1) / (n2 ** 3 * r2 ** 3 * c2))
    )

    exp2 = np.exp(
        -(kx ** 2 + ky ** 2) *
        (n2 + 1) / (4 * c2 * n2 * r2)
    )

    exp3 = np.exp(
        (x * kx + y * ky) *
        np.pi / (lamb * r_curvature * n2 * r2 * c2)
    )

    return 4 / ((n2 + 1) * r2 * np.sqrt(c2)) * exp1 * exp2 * exp3


def visibility_for_telescopes(radius_top, radius_bottom, r_curvature, x, y, kx, ky, lamb):
    r_curvature = np.abs(r_curvature)
    num = -(radius_top ** 2 * radius_bottom ** 2 * r_curvature ** 2 * (radius_top ** 2 + radius_bottom ** 2) * (kx ** 2 + ky ** 2)
            - 4 * np.pi / lamb * radius_top ** 2 * radius_bottom ** 4 * (kx * x + ky * y) * r_curvature  +
            ((2 * np.pi /lamb * radius_top * radius_bottom ** 2) ** 2 + 4 * r_curvature ** 2 * (radius_bottom ** 2 + radius_top ** 2)) * (x ** 2 + y ** 2)


    )
    denom = (2 * np.pi / lamb * radius_bottom ** 2 * radius_top ** 2) ** 2 + (
            2 * r_curvature * (radius_top ** 2 + radius_bottom ** 2)) ** 2

    fact = 8 *  radius_top ** 2 * radius_bottom ** 2 * r_curvature / (denom ** (1 / 2) * (radius_top ** 2 + radius_bottom ** 2))
    return fact * np.exp(num / denom)
