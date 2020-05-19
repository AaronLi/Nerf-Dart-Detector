from detector import DartDetector
import argparse
import cv2
import os
import atexit

if __name__ == '__main__':
    CAMERA = 'camera'
    PHOTO = 'photo'
    VIDEO = 'video'
    input_modes = (CAMERA, PHOTO, VIDEO)
    parser = argparse.ArgumentParser(description='Dart Detector CLI Interface')
    parser.add_argument('input_mode', metavar='I', choices=input_modes, help=f'Image source type {input_modes}')
    parser.add_argument('input_source', metavar='S', help='The image source (A file or camera index)')

    args = parser.parse_args()

    detector = DartDetector().load_dart_profile(*[os.path.join('dart_data', file) for file in os.listdir('dart_data')])
    # function for getting the image to show onscreen
    get_output = None

    if args.input_mode == CAMERA:

        print('reading camera')
        camera = cv2.VideoCapture(args.input_source)

        def read_camera_and_find_darts():
            ret, frame = camera.read()

            points, result_image = detector.find_darts(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), True)
            return cv2.cvtColor(result_image, cv2.COLOR_HSV2BGR)

        get_output = read_camera_and_find_darts
        atexit.register(camera.release)

    elif args.input_mode == PHOTO:
        image = cv2.cvtColor(cv2.imread(args.input_source), cv2.COLOR_BGR2HSV)

        points, result_image = detector.find_darts(image, True)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_HSV2BGR)
        def get_output():
            return result_image

    elif args.input_mode == VIDEO:
        video = cv2.VideoCapture(args.input_source)

        def read_video_and_find_darts():
            ret, frame = video.read()

            points, result_image = detector.find_darts(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), True)

            return cv2.cvtColor(result_image, cv2.COLOR_HSV2BGR)
        get_output = read_video_and_find_darts
        atexit.register(video.release)

    running = True
    while running:
        try:
            cv2.imshow('Dart Detector', get_output())
        except cv2.error:
            running = False

        key = cv2.waitKey(1)
        if key == -1:
            pass
        elif key == ord('q') or key == 27:
            running = False
        else:
            print(key)
    cv2.destroyAllWindows()