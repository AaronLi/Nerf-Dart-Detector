import cv2
import numpy as np
import matplotlib.pyplot as plt
import polar
import heapq


def get_parts_of_colour(low_colour, high_colour, image, erosions=0, dilations=0):
    threshold = cv2.inRange(image, low_colour, high_colour)
    # plt.imshow(threshold)
    # plt.show()
    eroding = cv2.erode(threshold, (2, 2), iterations=erosions)
    # plt.imshow(eroding)
    # plt.show()
    dilating = cv2.dilate(eroding, (4, 4), iterations=dilations)
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


# 2FB2FB
# [2, 68, 116] dart colour low rgb
# 103, 251, 116 dart colour low opencv hsv
# [60, 188, 253] dart colour high rgb
# 102, 195, 252.96 dart colour high opencv hsv
dart_colour_low = np.array((96, 100, 0), dtype=np.uint8)
dart_colour_high = np.array((108, 255, 255), dtype=np.uint8)

tip_colour_low = np.array((117, 90, 20), dtype=np.uint8)
tip_colour_high = np.array((150, 255, 255), dtype=np.uint8)


def find_darts(image, show_stages=False):
    image = cv2.blur(cv2.resize(image, (640, 480)), (4, 4))
    dart_bodies = get_parts_of_colour(dart_colour_low, dart_colour_high, image, erosions=5, dilations=3)

    dart_tips = get_parts_of_colour(tip_colour_low, tip_colour_high, image, erosions=2, dilations=5)
    dart_body_detector = get_dart_body_detector()
    dart_tip_detector = get_dart_tip_detector()
    detect_bodies = cv2.blur(cv2.cvtColor(cv2.cvtColor(dart_bodies, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY), (4, 4))
    body_points = dart_body_detector.detect(detect_bodies)
    detect_tips = cv2.blur(cv2.cvtColor(cv2.cvtColor(dart_tips, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY), (4, 4))
    tip_points = dart_tip_detector.detect(detect_tips)
    body_edges = cv2.Canny(detect_bodies, 150, 200, apertureSize=5)

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

    for tip in tip_points:
        cv2.circle(image, (int(tip.pt[0]), int(tip.pt[1])), int(tip.size), (0, 255, 255), thickness=2)

    potential_darts = []  # np.full((min(len(tip_points), len(body_points)), 4),[np.inf, [0, 0], tip_points[0], body_points[0]])
    all_combos = []
    used_combos = []

    for dart in body_points:
        for tip in tip_points:
            rho, theta = polar.get_polar(dart.pt, tip.pt)
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

            '''
            max_index = np.argmax(potential_darts[:, 0])

            # instead of swapping out largest and smallest values just append them to a list and use heapsort
            if darty[0] < potential_darts[max_index, 0]:
                potential_darts[max_index] = [darty[0], lines[darty[1]], tip, dart]
            # print(potential_darts)'''

        cv2.circle(image, (int(dart.pt[0]), int(dart.pt[1])), int(dart.size * 2), (90, 255, 255), thickness=2)


    all_combos.sort()
    for dart in all_combos:
        #print('Combo Dart', polar.get_polar(dart[3].pt, dart[2].pt))
        #print('Line:', dart[1])
        print('Dartiness', dart[0], '\n')
        if not dart[2] in used_combos and not dart[3] in used_combos and len(potential_darts) < min(len(tip_points), len(body_points)):
            potential_darts.append(dart)
            used_combos.append(dart[2])
            used_combos.append(dart[3])
    '''print(potential_darts[0, 3].pt)'''
    for potent in potential_darts:
        # print(int(potent[3].pt[0]))
        tipa = int(potent[2].pt[0])
        tipb = int(potent[2].pt[1])
        bodya = int(potent[3].pt[0])
        bodyb = int(potent[3].pt[1])
        bodypoint = (bodya, bodyb)
        tippoint = (tipa, tipb)
        cv2.putText(image, str(int(potent[0])), bodypoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(int(potent[0])), tippoint, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        print('Potential Dart',polar.get_polar(potent[3].pt,potent[2].pt))
        print('Line:',potent[1])
        print('Dartiness',potent[0],'\n')

    show_stages = True
    if show_stages:
        plt.imshow(image)
        plt.show()
        plt.imshow(cv2.cvtColor(dart_bodies, cv2.COLOR_HSV2RGB))
        plt.show()
        plt.imshow(cv2.cvtColor(dart_tips, cv2.COLOR_HSV2RGB))
        plt.show()
        plt.imshow(cv2.cvtColor(body_edges, cv2.COLOR_GRAY2RGB))
        plt.show()
    return image


def dartiness(line, dart_rho, dart_theta):
    '''temp = [np.inf, np.inf]
    for i in range(len(lines)):
        darty = (lines[i][1] - dart_rho) ** 2 * 3 + (lines[i][0] - dart_theta) ** 2
        if darty < temp[0]:
            temp[0] = darty
            temp[1] = i'''
    darty = (line[0] % 180 - dart_theta% 180 ) ** 2 + (line[1] - dart_rho) ** 2
    return darty


# #plt.imshow(cv2.cvtColor(threshold[1], cv2.COLOR_GRAY2RGB))
# #cv2.imshow("pic", cv2.cvtColor(filter_image, cv2.COLOR_HSV2BGR))
# #while True:
# #    if cv2.waitKey(1) == 113:
# #        cv2.destroyAllWindows()
# #        break
# print("opening camera")
# camera_in = cv2.VideoCapture("data/videoTest.avi")
# print("camera opened")
# running = True
# ret = True
# while running:
#     ret, image = camera_in.read()
#     if ret:
#         image = cv2.resize(image, (640, 480))
#         cv2.imshow("dart finder", cv2.cvtColor(find_darts(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)), cv2.COLOR_HSV2BGR))
#
#     if cv2.waitKey(1) == ord('q'):
#         running = False
# cv2.destroyAllWindows()
three_darts = cv2.cvtColor(cv2.imread("data/IMG_20200503_165208.jpg"), cv2.COLOR_BGR2HSV)
close_to_ground = cv2.cvtColor(cv2.imread("data/IMG_20200502_151557.jpg"), cv2.COLOR_BGR2HSV)
higher_from_ground = cv2.cvtColor(cv2.imread("data/IMG_20200503_185246.jpg"), cv2.COLOR_BGR2HSV)
medium_height = cv2.cvtColor(cv2.imread("data/IMG_20200503_191741.jpg"), cv2.COLOR_BGR2HSV)
single_dart = cv2.cvtColor(cv2.imread("data/IMG_20200503_202145.jpg"), cv2.COLOR_BGR2HSV)

plt.imshow(cv2.cvtColor(find_darts(higher_from_ground, True), cv2.COLOR_HSV2RGB))
plt.show()

# plt.imshow(cv2.cvtColor(find_darts(close_to_ground), cv2.COLOR_HSV2RGB))
# plt.show()
#
#
# plt.imshow(cv2.cvtColor(find_darts(higher_from_ground), cv2.COLOR_HSV2RGB))
# plt.show()
#
#
# plt.imshow(cv2.cvtColor(find_darts(medium_height), cv2.COLOR_HSV2RGB))
# plt.show()
#
#
# plt.imshow(cv2.cvtColor(find_darts(single_dart), cv2.COLOR_HSV2RGB))
# plt.show()
