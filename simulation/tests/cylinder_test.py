import unittest
import numpy as np
from simulation.rendering.shape.cylinder import Cylinder

class CylinderTest(unittest.TestCase):

    def test_init(self):

        # Simple check that cylinders of various faces can be intialized
        for i in range(3, 100):
            c = Cylinder(1, 1, faces=i)

            c.translate_y(1)

            face_heights = np.hstack(c.get_world_faces()[:, :, 1].transpose())
            # verify all faces have been translated up
            np.testing.assert_array_equal(face_heights, [1 for j in range(i*2)] + [2 for j in range(i*2)])
            np.testing.assert_array_equal(np.hstack(c.get_world_faces()[:, :, 3].transpose()), [1 for j in range(i*4)])

        c = Cylinder(5, 5)

        self.assertTupleEqual(c.colour, (255, 255, 255))