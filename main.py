import time

import cv2
import matplotlib.pyplot as plt

import dart_profile

from detector import DartDetector

if __name__ == '__main__':

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
    detector = DartDetector() \
        .load_dart_profile('dart_data/elite_standard.json', 'dart_data/fortnite.json', 'dart_data/nstrike.json')

    three_darts = cv2.cvtColor(cv2.imread("data/IMG_20200503_165208.jpg"), cv2.COLOR_BGR2HSV)
    close_to_ground = cv2.cvtColor(cv2.imread("data/IMG_20200502_151557.jpg"), cv2.COLOR_BGR2HSV)
    higher_from_ground = cv2.cvtColor(cv2.imread("data/IMG_20200503_185246.jpg"), cv2.COLOR_BGR2HSV)
    medium_height = cv2.cvtColor(cv2.imread("data/IMG_20200503_191741.jpg"), cv2.COLOR_BGR2HSV)
    single_dart = cv2.cvtColor(cv2.imread("data/IMG_20200503_202145.jpg"), cv2.COLOR_BGR2HSV)
    assorted_darts_medium_height = cv2.cvtColor(cv2.imread("data\\IMG_20200516_193812.jpg"), cv2.COLOR_BGR2HSV)
    assorted_darts_high_height = cv2.cvtColor(cv2.imread("data/IMG_20200516_193819.jpg"), cv2.COLOR_BGR2HSV)
    assorted_darts_low_height = cv2.cvtColor(cv2.imread("data/IMG_20200516_193829.jpg"), cv2.COLOR_BGR2HSV)
    assorted_darts_low_height_partially_shadowed = cv2.cvtColor(cv2.imread("data/IMG_20200516_193832.jpg"), cv2.COLOR_BGR2HSV)

    to_process = [assorted_darts_low_height_partially_shadowed, close_to_ground, higher_from_ground, three_darts,
                  assorted_darts_high_height, assorted_darts_medium_height]
    output_images = []
    start_time = time.time()
    for image in to_process:
        results, visualization = detector.find_darts(image, True)
        output_images.append(visualization)
    run_time = time.time() - start_time
    print(f'Finished {len(to_process)} images in {run_time:.2f} seconds ({len(to_process) / run_time: 0.2f} fps)')

    for image in output_images:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))
        plt.show()
