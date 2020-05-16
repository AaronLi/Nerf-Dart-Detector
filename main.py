import cv2
import matplotlib.pyplot as plt

import dart_profile

# 2FB2FB
# [2, 68, 116] dart colour low rgb
# 103, 251, 116 dart colour low opencv hsv
# [60, 188, 253] dart colour high rgb
# 102, 195, 252.96 dart colour high opencv hsv
from detector import find_darts, render_dart_results

# #I plt.imshow(cv2.cvtColor(threshold[1], cv2.COLOR_GRAY2RGB))
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
with open('dart_data/fortnite.json') as f:
    fortnite_dart_profile = dart_profile.DartProfile.read_from_file(f)

three_darts = cv2.resize(cv2.cvtColor(cv2.imread("data/IMG_20200503_165208.jpg"), cv2.COLOR_BGR2HSV), (640, 480))
close_to_ground = cv2.resize(cv2.cvtColor(cv2.imread("data/IMG_20200502_151557.jpg"), cv2.COLOR_BGR2HSV), (640, 480))
higher_from_ground = cv2.resize(cv2.cvtColor(cv2.imread("data/IMG_20200503_185246.jpg"), cv2.COLOR_BGR2HSV), (640, 480))
medium_height = cv2.resize(cv2.cvtColor(cv2.imread("data/IMG_20200503_191741.jpg"), cv2.COLOR_BGR2HSV), (640, 480))
single_dart = cv2.resize(cv2.cvtColor(cv2.imread("data/IMG_20200503_202145.jpg"), cv2.COLOR_BGR2HSV), (640, 480))

darts = find_darts(higher_from_ground, fortnite_dart_profile)

plt.imshow(cv2.cvtColor(render_dart_results(darts, higher_from_ground, fortnite_dart_profile), cv2.COLOR_HSV2RGB))
plt.show()

# plt.imshow(cv2.cvtColor(find_darts(close_to_ground, fortnite_dart_profile), cv2.COLOR_HSV2RGB))
# plt.show()
#
# plt.imshow(cv2.cvtColor(find_darts(higher_from_ground, fortnite_dart_profile), cv2.COLOR_HSV2RGB))
# plt.show()
#
# plt.imshow(cv2.cvtColor(find_darts(medium_height, fortnite_dart_profile), cv2.COLOR_HSV2RGB))
# plt.show()
#
# plt.imshow(cv2.cvtColor(find_darts(single_dart, fortnite_dart_profile), cv2.COLOR_HSV2RGB))
# plt.show()
