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
    detector = DartDetector()\
        .load_dart_profile('dart_data/elite_standard.json')\
        .load_dart_profile('dart_data/fortnite.json')\
        .load_dart_profile('dart_data/nstrike.json')

    three_darts = cv2.resize(cv2.cvtColor(cv2.imread("data/IMG_20200503_165208.jpg"), cv2.COLOR_BGR2HSV), (640, 480))
    close_to_ground = cv2.resize(cv2.cvtColor(cv2.imread("data/IMG_20200502_151557.jpg"), cv2.COLOR_BGR2HSV), (640, 480))
    higher_from_ground = cv2.resize(cv2.cvtColor(cv2.imread("data/IMG_20200503_185246.jpg"), cv2.COLOR_BGR2HSV), (640, 480))
    medium_height = cv2.resize(cv2.cvtColor(cv2.imread("data/IMG_20200503_191741.jpg"), cv2.COLOR_BGR2HSV), (640, 480))
    single_dart = cv2.resize(cv2.cvtColor(cv2.imread("data/IMG_20200503_202145.jpg"), cv2.COLOR_BGR2HSV), (640, 480))
    assorted_darts_medium_height = cv2.resize(cv2.cvtColor(cv2.imread("data\\IMG_20200516_193812.jpg"), cv2.COLOR_BGR2HSV), (640, 480))
    assorted_darts_high_height = cv2.resize(cv2.cvtColor(cv2.imread("data/IMG_20200516_193819.jpg"), cv2.COLOR_BGR2HSV), (640, 480))
    assorted_darts_low_height = cv2.resize(cv2.cvtColor(cv2.imread("data/IMG_20200516_193829.jpg"), cv2.COLOR_BGR2HSV), (640, 480))
    assorted_darts_low_height_partially_shadowed = cv2.resize(cv2.cvtColor(cv2.imread("data/IMG_20200516_193832.jpg"), cv2.COLOR_BGR2HSV), (640, 480))

    # darts = find_darts(higher_from_ground, fortnite_dart_profile)
    #
    # plt.imshow(cv2.cvtColor(render_dart_results(darts, higher_from_ground, fortnite_dart_profile), cv2.COLOR_HSV2RGB))
    # plt.show()
    to_process = [assorted_darts_low_height_partially_shadowed, close_to_ground, higher_from_ground, three_darts, assorted_darts_high_height, assorted_darts_medium_height]
    output_images = []
    start_time = time.time()
    for image in to_process:
        results, visualization = detector.find_darts(image, True)
        output_images.append(visualization)
    run_time = time.time() - start_time
    print(f'Finished {len(to_process)} images in {run_time:.2f} seconds ({len(to_process)/run_time: 0.2f} fps)')

    for image in output_images:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))
        plt.show()
    #
    # plt.imshow(cv2.cvtColor(find_darts(medium_height, fortnite_dart_profile), cv2.COLOR_HSV2RGB))
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(find_darts(single_dart, fortnite_dart_profile), cv2.COLOR_HSV2RGB))
    # plt.show()
