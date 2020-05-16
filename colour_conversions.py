import numpy as np


def hsv360_100_100_to_hsv180_255_255(colour_in):
    """
    Converts a standard 360° 100% 100% HSV colour into an OpenCV compatible 180º 255% 255% HSV colour
    :param colour_in: an arraylike object in HSV form
    :return: an OpenCV compatible HSV colour stored in a numpy uint8 ndarray
    """
    hsv_out = (
        colour_in[0] // 2,
        round(colour_in[1] / 100.0 * 255),
        round(colour_in[2] / 100.0 * 255)
    )
    return np.array(hsv_out, np.uint8)
