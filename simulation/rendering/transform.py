import numpy as np

def rotate(axis_vector, angle):
    j = np.array(
        (
            (0, -axis_vector[2], axis_vector[1]),
            (axis_vector[2], 0, -axis_vector[0]),
            (-axis_vector[1], axis_vector[0], 0)
        ), dtype=float)

    rotation_matrix = np.sin(angle) * j + (1 - np.cos(angle)) * np.dot(j, j)

    homogeneous_rotation_matrix = np.identity(4, dtype=float)
    homogeneous_rotation_matrix[0:3, 0:3] += rotation_matrix

    return homogeneous_rotation_matrix

def translate(axis_vector):
    translation_matrix = np.identity(4)
    translation_matrix[:, 3] += axis_vector

    translation_matrix[3, 3] = 1

    return translation_matrix

def scale(scale_vector):
    scale_matrix = np.identity(4)

    for i in range(3):
        scale_matrix[i, i] = scale_vector[i]

    return scale_matrix