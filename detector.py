import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple, List

import dart_profile
from multiprocessing import pool, cpu_count

DartSearchQuery = Tuple[np.ndarray, dart_profile.DartProfile]
DartSearchResult = Tuple[Tuple[float, cv2.KeyPoint], np.ndarray]


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

    def load_dart_profile(self, *args):
        """
        Loads a dart profile into the DartDetector
        you can pass in a list of profiles
        """
        for file_path in args:
            with open(file_path) as f:
                self.profiles.append(dart_profile.DartProfile.read_from_file(f))
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
            results_out.append(found_darts)
            if visualize_results:
                DartDetector.render_dart_results(found_darts, output_image, profile)

        return results_out, output_image

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
        # body_edges = cv2.Canny(detect_bodies, 150, 200, apertureSize=5)
        '''
        lines = cv2.HoughLines(body_edges, 1, np.pi / 360, 50)
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
    
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            lines = [line[0] for line in lines]
            lines.sort(key=lambda x: x[1])
            # print(len(lines), lines)
        '''
        for tip in tip_points:
            cv2.circle(image, (int(tip.pt[0]), int(tip.pt[1])), int(tip.size), (0, 255, 255), thickness=2)

        potential_darts = []
        all_combos = []
        used_combos = []

        for dart in body_points:
            for tip in tip_points:
                '''rho, theta = polar.get_polar(dart.pt, tip.pt)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
    
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 1)
    
                for line in lines:
                    darty = dartiness(line, rho, theta)
                    all_combos.append([darty, line, tip, dart])
    
                
                #max_index = np.argmax(potential_darts[:, 0])
    
                # instead of swapping out largest and smallest values just append them to a list and use heapsort
                if darty[0] < potential_darts[max_index, 0]:
                    potential_darts[max_index] = [darty[0], lines[darty[1]], tip, dart]
                # print(potential_darts)'''
                distance = np.hypot((dart.pt[0] - tip.pt[0]), (dart.pt[1] - tip.pt[1]))
                # if distance in set(all_combos):
                # distance += 0.00001

                all_combos.append((distance, tip, dart))

            cv2.circle(image, (int(dart.pt[0]), int(dart.pt[1])), int(dart.size * 2), (90, 255, 255), thickness=2)

        all_combos.sort()

        for dart in all_combos:
            # print('Combo Dart', polar.get_polar(dart[3].pt, dart[2].pt))
            # print('Line:', dart[1])
            # print('Distance', dart[0], '\n')
            if not dart[1] in used_combos and not dart[2] in used_combos and len(potential_darts) < min(len(tip_points),
                                                                                                        len(
                                                                                                            body_points)):
                potential_darts.append((dart[0], (dart[1].pt[0], dart[1].pt[1]), (dart[2].pt[0], dart[2].pt[1])))
                used_combos.append(dart[1])
                used_combos.append(dart[2])
        if show_stages:
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
        for potent in potential_darts:
            # print(int(potent[3].pt[0]))
            tip_x = int(potent[1][0])
            tip_y = int(potent[1][1])
            body_x = int(potent[2][0])
            body_y = int(potent[2][1])
            body_point = (body_x, body_y)
            tip_point = (tip_x, tip_y)
            # instead of drawing to the body and stopping, continue a bit
            dx = round((tip_x - body_x) * 0.75)
            dy = round((tip_y - body_y) * 0.75)
            center_point = ((body_x + tip_x) // 2, (body_y + tip_y) // 2)
            cv2.line(image, tip_point, (body_x - dx, body_y - dy), profile.identification_colour.tolist(), 5)
            cv2.putText(image, str(int(potent[0])), body_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (179, 0, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(image, str(int(potent[0])), tip_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (179, 0, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(image, profile.dart_name, center_point, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1,
                        cv2.LINE_AA)
            # print('Potential Dart', polar.get_polar(potent[2].pt, potent[1].pt))
            # print('Line:',potent[1])
            # print('Distance', potent[0], '\n')

        return image

    @staticmethod
    def find_dartiness(line, dart_rho, dart_theta):
        """

        :param line:
        :param dart_rho:
        :param dart_theta:
        :return:
        """
        dartiness = (line[0] % 180 - dart_theta % 180) ** 2 + (line[1] - dart_rho) ** 2
        return dartiness


if __name__ == '__main__':
    a = DartDetector()
