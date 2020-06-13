import unittest
from simulation.rendering.shape.generic import GenericShape
import numpy as np

class GenericShapeTest(unittest.TestCase):
    EPSILON = 1.5e-16

    def test_translate(self):
        s = GenericShape()

        s.translate(np.array((1, 0, 0, 0)))

        s.translate(np.array((0, 1, 0, 0)))

        s.translate(np.array((0, 0, 1, 0)))

        np.testing.assert_array_equal(s.transform, np.array(
            (
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1)
            )
        ))

        np.testing.assert_array_equal(s.pos, np.array((1, 1, 1, 1)))

        s.translate(np.array((-1, -1, -1, 0)))

        np.testing.assert_array_equal(s.pos, np.array((0, 0, 0, 1)))

        s.translate(np.array((-1, 0, 0, 0)))

        np.testing.assert_array_equal(s.pos, np.array((-1, 0, 0, 1)))




    def test_scale(self):
        s = GenericShape()

        s.scale(np.array((2, 1, 1, 0)))

        np.testing.assert_array_equal(s.transform, np.array(
            (
                (2, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1)
            )))

        s.scale(np.array((1/2, 2, 2, 0)))

        np.testing.assert_array_equal(s.transform, np.array(
            (
                (1, 0, 0, 0),
                (0, 2, 0, 0),
                (0, 0, 2, 0),
                (0, 0, 0, 1)
            )
        ))

        s.scale_y(0.5)

        np.testing.assert_array_equal(s.transform, np.array(
            (
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 2, 0),
                (0, 0, 0, 1)
            )
        ))

        s.scale_z(0.5)

        np.testing.assert_array_equal(s.transform, np.array(
            (
                (1, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1)
            )
        ))

        s.scale_x(5)

        np.testing.assert_array_equal(s.transform, np.array(
            (
                (5, 0, 0, 0),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1)
            )
        ))

    def test_rotate(self):
        s = GenericShape()

        # verify identity matrix is correct
        self.assertTrue(np.array_equal(s.transform, np.identity(4)))

        # # translate the shape
        # s.translate(GenericShape.X)
        #
        # np.testing.assert_array_equal(s.transform, np.array(
        #     (
        #         (1, 0, 0, 0),
        #         (0, 1, 0, 0),
        #         (0, 0, 1, 0),
        #         (0, 0, 0, 1)
        #     )))
        #
        # # rotate the shape
        # s.rotate_z(np.radians(90))
        #
        # expected = np.array(
        #     (
        #         (0, -1, 0, 0),
        #         (1, 0, 0, 0),
        #         (0, 0, 1, 0),
        #         (0, 0, 0, 1)
        #     )
        # )
        # np.testing.assert_array_almost_equal(s.transform, expected, decimal=GenericShapeTest.EPSILON)
        #
        # # rotate in x
        # s.rotate_x(np.radians(90))
        #
        # np.testing.assert_array_almost_equal(s.transform, np.array(
        #     (
        #         (1, 0, 0, 0),
        #         (0, 1, 0, 0),
        #         (0, 0, 1, 0),
        #         (0, 0, 0, 1)
        #     )
        # ), decimal=GenericShapeTest.EPSILON)
        #
        # # try two dimensions at once
        #
        # s.translate(np.array((0, 1, 0, 0)))
        #
        # s.rotate_y(np.radians(90))
        #
        #
        # np.testing.assert_array_almost_equal(s.transform, np.array(
        #     (
        #         (1, 0, 0, 1),
        #         (0 ,1, 0, 1),
        #         (0, 0, 1, 0),
        #         (0, 0, 0, 1)
        #     )
        # ), decimal=GenericShapeTest.EPSILON)