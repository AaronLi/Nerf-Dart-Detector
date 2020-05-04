import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_parts_of_colour(low_colour, high_colour, image, erosions = 0, dilations = 0):
    threshold = cv2.inRange(image, low_colour, high_colour)
    eroding = cv2.erode(threshold, (2, 2), iterations=erosions)
    dilating = cv2.dilate(eroding, (4, 4), iterations=dilations)
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
#[2, 68, 116] dart colour low rgb
#103, 251, 116 dart colour low opencv hsv
#[60, 188, 253] dart colour high rgb
#102, 195, 252.96 dart colour high opencv hsv
dart_colour_low = np.array((96, 100, 0), dtype=np.uint8)
dart_colour_high = np.array((108,255, 255), dtype=np.uint8)

tip_colour_low = np.array((117, 90, 20), dtype=np.uint8)
tip_colour_high = np.array((150, 255, 255), dtype=np.uint8)
def find_darts(image, show_stages = False):
    image = cv2.blur(cv2.resize(image, (640, 480)), (4, 4))
    dart_bodies = get_parts_of_colour(dart_colour_low, dart_colour_high, image, erosions=5, dilations= 3)

    dart_tips = get_parts_of_colour(tip_colour_low, tip_colour_high, image, erosions=2, dilations=5)
    dart_body_detector = get_dart_body_detector()
    dart_tip_detector = get_dart_tip_detector()
    detect_bodies = cv2.blur(cv2.cvtColor(cv2.cvtColor(dart_bodies, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY), (4, 4))
    body_points = dart_body_detector.detect(detect_bodies)
    detect_tips = cv2.blur(cv2.cvtColor(cv2.cvtColor(dart_tips, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY), (4, 4))
    tip_points = dart_tip_detector.detect(detect_tips)
    body_edges = cv2.Canny(detect_bodies, 150, 200, apertureSize=5)


    lines = cv2.HoughLines(body_edges, 1,np.pi/360, 50)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(image,(x1,y1),(x2,y2),(0,0,255),1)
        lines = [line[0] for line in lines]
        lines.sort(key=lambda x: x[1])
        print(len(lines), lines)

    for tip in tip_points:
        cv2.circle(image, (int(tip.pt[0]), int(tip.pt[1])), int(tip.size), (0, 255, 255), thickness=2)
    potential_darts = []
    for dart in body_points:
        for tip in tip_points:
            dx = dart.pt[0] - tip.pt[0]
            dy = dart.pt[1] - tip.pt[1]
            slope = dy/dx
            rho = abs(-tip.pt[0] + slope * tip.pt[1]) / (slope**2 + 1)**0.5
            theta = np.arctan2(dy, dx)
            potential_darts.append((rho, theta))
            print(theta, rho, np.hypot(dx, dy), end=', ')
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 1)
        print()
        cv2.circle(image, (int(dart.pt[0]), int(dart.pt[1])), int(dart.size * 2), (90, 255, 255), thickness=2)

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

plt.imshow(cv2.cvtColor(find_darts(three_darts), cv2.COLOR_HSV2RGB))
plt.show()


plt.imshow(cv2.cvtColor(find_darts(close_to_ground), cv2.COLOR_HSV2RGB))
plt.show()


plt.imshow(cv2.cvtColor(find_darts(higher_from_ground), cv2.COLOR_HSV2RGB))
plt.show()


plt.imshow(cv2.cvtColor(find_darts(medium_height), cv2.COLOR_HSV2RGB))
plt.show()


plt.imshow(cv2.cvtColor(find_darts(single_dart), cv2.COLOR_HSV2RGB))
plt.show()