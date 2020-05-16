import cv2
import numpy as np
from matplotlib import pyplot as plt

import dart_profile


def get_parts_of_colour(image, isolationSpec: dart_profile.ColourIsolationSpec):
    threshold = cv2.inRange(image, isolationSpec.colour_low, isolationSpec.colour_high)
    # plt.imshow(threshold)
    # plt.show()
    eroding = cv2.erode(threshold, (2, 2), iterations=isolationSpec.erosions)
    # plt.imshow(eroding)
    # plt.show()
    dilating = cv2.dilate(eroding, (4, 4), iterations=isolationSpec.dilations)
    # plt.imshow(dilating)
    # plt.show()
    keyed_final = cv2.bitwise_and(image, image, mask=dilating)
    return keyed_final


def get_dart_body_detector():
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


def get_dart_tip_detector():
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


def find_darts(image, profile: dart_profile.DartProfile, show_stages=False):
    image = cv2.blur(cv2.resize(image, (640, 480)), (4, 4))
    dart_bodies = get_parts_of_colour(image, profile.body)

    dart_tips = get_parts_of_colour(image, profile.tip)
    dart_body_detector = get_dart_body_detector()
    dart_tip_detector = get_dart_tip_detector()
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

    potential_darts = []  # np.full((min(len(tip_points), len(body_points)), 4),[np.inf, [0, 0], tip_points[0], body_points[0]])
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
                                                                                                    len(body_points)):
            potential_darts.append(dart)
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
    return potential_darts


def render_dart_results(potential_darts, image, profile):
    '''print(potential_darts[0, 3].pt)'''
    for potent in potential_darts:
        # print(int(potent[3].pt[0]))
        tipa = int(potent[1].pt[0])
        tipb = int(potent[1].pt[1])
        bodya = int(potent[2].pt[0])
        bodyb = int(potent[2].pt[1])
        bodypoint = (bodya, bodyb)
        tippoint = (tipa, tipb)
        cv2.putText(image, str(int(potent[0])), bodypoint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (179, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(int(potent[0])), tippoint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (179, 0, 255), 1, cv2.LINE_AA)

        center_point = ((bodya + tipa) // 2, (bodyb + tipb) // 2)
        cv2.putText(image, profile.dart_name, center_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

        # print('Potential Dart', polar.get_polar(potent[2].pt, potent[1].pt))
        # print('Line:',potent[1])
        # print('Distance', potent[0], '\n')

    return image


def find_dartiness(line, dart_rho, dart_theta):
    dartiness = (line[0] % 180 - dart_theta % 180) ** 2 + (line[1] - dart_rho) ** 2
    return dartiness