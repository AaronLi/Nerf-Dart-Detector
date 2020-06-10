import unittest
import colour_conversions
import numpy as np

class ColourConversionTest(unittest.TestCase):

    def test_hsv_conversion(self):

        colour = (360, 100, 100)

        converted = colour_conversions.hsv360_100_100_to_hsv180_255_255(colour)

        self.assertSequenceEqual(converted.tolist(), (180, 255, 255))

        colour = (0, 0 ,0)

        converted = colour_conversions.hsv360_100_100_to_hsv180_255_255(colour)

        self.assertSequenceEqual(converted.tolist(), (0, 0, 0))

        colour = (180, 50, 50)

        converted = colour_conversions.hsv360_100_100_to_hsv180_255_255(colour)

        self.assertSequenceEqual(converted.tolist(), (90, 128, 128))

        colour = (1, 1, 1)

        converted = colour_conversions.hsv360_100_100_to_hsv180_255_255(colour)

        self.assertEqual(converted.dtype, np.uint8)
