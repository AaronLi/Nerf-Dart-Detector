import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List
from collections import namedtuple
import colour_conversions

import dart_profile
from multiprocessing import pool, cpu_count

import polar

DartResult = namedtuple("DartResult", ("score", "tip_position", "body_position", "bounding_rect"))
PotentialDart = namedtuple("PotentialDart", ("score", "tip_point", "body_point"))

# image and profile
DartSearchQuery = Tuple[np.ndarray, dart_profile.DartProfile]

# list of darts and image
DartSearchResult = Tuple[List[DartResult], np.ndarray]


def standard_body_blob_detector():
    """

    :return:
    """
    dart_detector_params = cv2.SimpleBlobDetector_Params()
    dart_detector_params.minThreshold = 30
    dart_detector_params.thresholdStep = 20
    dart_detector_params.maxThreshold = 250
    dart_detector_params.minRepeatability = 2
    dart_detector_params.filterByArea = True
    dart_detector_params.minArea = 50
    dart_detector_params.filterByColor = False
    dart_detector_params.filterByInertia = False
    dart_detector_params.filterByConvexity = True
    dart_detector_params.minConvexity = 0.8
    dart_detector_params.filterByCircularity = False
    return cv2.SimpleBlobDetector_create(dart_detector_params)


def standard_tip_blob_detector():
    """

    :return:
    """
    tip_detector_params = cv2.SimpleBlobDetector_Params()
    tip_detector_params.minThreshold = 30
    tip_detector_params.thresholdStep = 10
    tip_detector_params.maxThreshold = 180
    tip_detector_params.minRepeatability = 1
    tip_detector_params.filterByArea = True
    tip_detector_params.minArea = 16
    tip_detector_params.maxArea = 300
    tip_detector_params.filterByColor = False
    tip_detector_params.filterByInertia = True
    tip_detector_params.maxInertiaRatio = 0.7
    tip_detector_params.filterByConvexity = True
    tip_detector_params.minConvexity = 0.6
    tip_detector_params.filterByCircularity = False
    return cv2.SimpleBlobDetector_create(tip_detector_params)


class DartDetector:
    """
    Detects all the darts in an image
    load the profiles for each dart and then provide with images to detect darts from
    """

    def __init__(self, image_resolution=(640, 480)):
        """

        """
        super().__init__()
        self.image_resolution = image_resolution
        self.profiles = []
        self.dart_finding_pool = pool.Pool(cpu_count())

    def load_dart_profile(self, paths):
        """
        Loads a dart profile into the DartDetector
        you can pass in a list of profiles
        """
        for file_path in paths:
            with open(file_path) as f:
                loaded_profile = dart_profile.DartProfile.read_from_file(f)
                if loaded_profile:
                    self.profiles.append(loaded_profile)
        return self

    @staticmethod
    def get_parts_of_colour(image, isolation_spec: dart_profile.ColourIsolationSpec):
        """

        :param image:
        :param isolation_spec:
        :return:
        """
        threshold = cv2.inRange(image, isolation_spec.colour_low, isolation_spec.colour_high)
        # plt.imshow(threshold)
        # plt.show()
        eroding = cv2.erode(threshold, (2, 2), iterations=isolation_spec.erosions)
        # plt.imshow(eroding)
        # plt.show()
        dilating = cv2.dilate(eroding, (4, 4), iterations=isolation_spec.dilations)
        # plt.imshow(dilating)
        # plt.show()
        keyed_final = cv2.bitwise_and(image, image, mask=dilating)
        return keyed_final

    def find_darts(self, image: np.ndarray, visualize_results=False) -> DartSearchResult:
        """
        Finds all darts from the loaded profiles in the image

        :param image: Image containing the darts to find
        :param visualize_results: Whether the function should visualize the results
        :return: The list of results and the visualization as a tuple (results, image)
        """
        search_image = cv2.resize(image, self.image_resolution)
        task_pool = ((search_image, profile) for profile in self.profiles)
        results_out = []
        output_image = search_image.copy()
        for found_darts, profile in self.dart_finding_pool.imap_unordered(DartDetector.query_image_for_profile,
                                                                          task_pool):
            results_out += found_darts
            if visualize_results:
                DartDetector.render_dart_results(found_darts, output_image, profile)

        return sorted(results_out), output_image

    @staticmethod
    def query_image_for_profile(query: DartSearchQuery, show_stages=False):
        """
        Finds the darts of a single profile from an image
        Designed to be used in a separate thread to find all profiles simultaneously

        :param query:
        :param show_stages:
        :return:
        """

        dart_image = query[0]
        profile = query[1]

        image = cv2.blur(dart_image, (4, 4))
        dart_bodies = DartDetector.get_parts_of_colour(image, profile.body)

        dart_tips = DartDetector.get_parts_of_colour(image, profile.tip)
        dart_body_detector = standard_body_blob_detector()
        dart_tip_detector = standard_tip_blob_detector()
        detect_bodies = cv2.blur(cv2.cvtColor(cv2.cvtColor(dart_bodies, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY), (4, 4))
        body_points = dart_body_detector.detect(detect_bodies)
        detect_tips = cv2.blur(cv2.cvtColor(cv2.cvtColor(dart_tips, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY), (4, 4))
        tip_points = dart_tip_detector.detect(detect_tips)
        body_edges = cv2.Canny(detect_bodies, 150, 200, apertureSize=5)

        lines = cv2.HoughLinesP(body_edges, 1, np.pi / 360, 50)

        potential_darts = []
        all_combos = []
        used_combos = set()

        if lines is not None:
            lines = [line[0] for line in lines]
            lines.sort(key=lambda x: x[1])

            for dart in body_points:
                for tip in tip_points:
                    for line in lines:
                        dartiness = DartDetector.get_dartiness_points(line,
                                                                      (tip.pt[0], tip.pt[1], dart.pt[0], dart.pt[1]))
                        all_combos.append(PotentialDart(dartiness, tip, dart))

                cv2.circle(image, (int(dart.pt[0]), int(dart.pt[1])), int(dart.size * 2), (90, 255, 255), thickness=2)

            all_combos.sort()

        for dart in all_combos:

            if not dart[1] in used_combos and not dart[2] in used_combos and len(potential_darts) < min(len(tip_points),
                                                                                                        len(
                                                                                                            body_points)):
                # essentially a vector pointing from the tip to the body
                dart_tip_dx = dart.body_point.pt[0] - dart.tip_point.pt[0]
                dart_tip_dy = dart.body_point.pt[1] - dart.tip_point.pt[1]

                # add vector to point
                extended_end_of_dart_x = int(dart.body_point.pt[0] + dart_tip_dx * 1.2)
                extended_end_of_dart_y = int(dart.body_point.pt[1] + dart_tip_dy * 1.2)

                extended_dart_tip_x = int(dart.body_point.pt[0] - dart_tip_dx * 1.4)
                extended_dart_tip_y = int(dart.body_point.pt[1] - dart_tip_dy * 1.4)

                min_x = min(extended_dart_tip_x, extended_end_of_dart_x)
                min_y = min(extended_dart_tip_y, extended_end_of_dart_y)

                max_x = max(extended_dart_tip_x, extended_end_of_dart_x)
                max_y = max(extended_dart_tip_y, extended_end_of_dart_y)

                bounding_rect = (min_x, min_y, max_x - min_x, max_y - min_y)

                potential_darts.append(DartResult(dart.score, (dart.tip_point.pt[0], dart.tip_point.pt[1]),
                                                  (dart.body_point.pt[0], dart.body_point.pt[1]), bounding_rect))
                used_combos.add(dart[1])
                used_combos.add(dart[2])
        if show_stages:
            for tip in tip_points:
                cv2.circle(image, (int(tip.pt[0]), int(tip.pt[1])), int(tip.size), (0, 255, 255), thickness=2)
            plt.imshow(image)
            plt.show()
            plt.imshow(cv2.cvtColor(dart_bodies, cv2.COLOR_HSV2RGB))
            plt.show()
            plt.imshow(cv2.cvtColor(dart_tips, cv2.COLOR_HSV2RGB))
            plt.show()
            # plt.imshow(cv2.cvtColor(body_edges, cv2.COLOR_GRAY2RGB))
            # plt.show()
        return potential_darts, profile

    @staticmethod
    def render_dart_results(potential_darts, image, profile: dart_profile.DartProfile):
        """

        :param potential_darts:
        :param image:
        :param profile:
        :return:
        """
        for i, potent in enumerate(potential_darts):
            # print(int(potent[3].pt[0]))
            tip_x = int(potent.tip_position[0])
            tip_y = int(potent.tip_position[1])
            body_x = int(potent.body_position[0])
            body_y = int(potent.body_position[1])
            body_point = (body_x, body_y)
            tip_point = (tip_x, tip_y)
            # instead of drawing to the body and stopping, continue a bit
            dx = round((tip_x - body_x) * 0.75)
            dy = round((tip_y - body_y) * 0.75)
            center_point = ((body_x + tip_x) // 2, (body_y + tip_y) // 2)
            cv2.line(image, tip_point, (body_x - dx, body_y - dy), profile.identification_colour.tolist(), 5)
            print(potent.bounding_rect)
            cv2.rectangle(image,
                          (potent.bounding_rect[0], potent.bounding_rect[1]),
                          (potent.bounding_rect[0] + potent.bounding_rect[2],
                           potent.bounding_rect[1] + potent.bounding_rect[3]),
                          profile.identification_colour.tolist(), 2
                          )

            cv2.putText(image, str(int(potent.score)), body_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (179, 0, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(image, str(int(potent.score)), tip_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (179, 0, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(image, profile.dart_name, center_point, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1,
                        cv2.LINE_AA)
            # print('Potential Dart', polar.get_polar(potent[2].pt, potent[1].pt))
            # print('Line:',potent[1])
            # print('Distance', potent[0], '\n')

        return image

    @staticmethod
    def get_dartiness(line, dart_theta, dart_rho):
        """

        :param line:
        :param dart_rho:
        :param dart_theta:
        :return:
        """
        dartiness = (line[0] % 180 - dart_theta % 180) ** 2 + (line[1] - dart_rho) ** 2
        return dartiness

    @staticmethod
    def get_dartiness_points(line_points, dart_points):
        """

        :param dart_points:
        :param line_points:
        :return:
        """
        line_start_x, line_start_y, line_end_x, line_end_y = line_points

        dart_tip_x, dart_tip_y, dart_body_x, dart_body_y = dart_points

        # metric based on distance from dart points to line_points points
        line_start_point = np.array((line_start_x, line_start_y))
        line_end_point = np.array((line_end_x, line_end_y))

        dart_tip_point = np.array((dart_tip_x, dart_tip_y))
        dart_body_point = np.array((dart_body_x, dart_body_y))

        # try one way
        tip_start_distance = np.linalg.norm(line_start_point - dart_tip_point)
        body_end_distance = np.linalg.norm(line_end_point - dart_body_point)

        # swap points and try again, since our points may be backwards
        tip_end_distance = np.linalg.norm(line_end_point - dart_tip_point)
        body_start_distance = np.linalg.norm(line_start_point - dart_body_point)

        line_distance = min(np.hypot(tip_start_distance, body_end_distance),
                            np.hypot(tip_end_distance, body_start_distance))

        # metric based on line_points angle difference
        line_vector = np.array((line_end_x - line_start_x, line_end_y - line_start_y))
        dart_vector = np.array((dart_body_x - dart_tip_x, dart_body_y - dart_tip_y))

        denom = (np.linalg.norm(line_vector) * np.linalg.norm(dart_vector))

        deviation_score = 1

        if denom > 0:
            # a value from 0 to 1, where 1 is > than 90 degrees away
            deviation_angle = 1 - max((np.dot(line_vector, dart_vector) / denom), 0)
            # recalculated with a line pointing the other way, because both values are valid
            alt_deviation_angle = 1 - max((np.dot(-line_vector, dart_vector) / denom), 0)

            deviation_score = min(deviation_angle, alt_deviation_angle)

        combined_score = line_distance + deviation_score * 100

        return combined_score


if __name__ == '__main__':
    import math

    for i in range(int(math.pi * 100)):
        angle = i / 100
        print(angle, DartDetector.get_dartiness((0, 1), 1, angle))

    for i in range(int(math.pi * 100)):
        angle = i / 100
        print(angle, DartDetector.get_dartiness((0, 1), -1, angle))
