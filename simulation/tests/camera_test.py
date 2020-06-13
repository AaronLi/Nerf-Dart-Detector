import unittest
from simulation.rendering import camera
import numpy as np

class CameraTest(unittest.TestCase):

    def test_init(self):
        c = camera.Camera(10, 10, 90,  pos=np.array((0, 0, 0, 1)), gaze=np.array((1, 0, 0, 1)))

        np.testing.assert_array_equal(c.n, np.array((1, 0, 0, 0)))

        np.testing.assert_array_equal(c.u, np.array((0, 0, 1, 0)))

        np.testing.assert_array_equal(c.v, np.array((0, 1, 0, 0)))

        np.testing.assert_array_equal(c.viewing_matrix, np.array(
            (
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1)
            )
        ))