import os

import cv2
import detector


class VideoTracker:
    def __init__(self, tracker=cv2.TrackerCSRT_create) -> None:
        self.tracker_constructor = tracker
        self.tracker = None

        self.has_detection = False

    def track_frame(self, frame, new_bounding_box = None):
        """
        Starts or continues tracking. Frames are expected to be sequential in time.
        Pass in a new_bounding_box to start tracking it.
        :param frame: The frame of video to track the object in
        :param new_bounding_box: A bounding box for a new object
        :return: None if no object is tracked. A rectangle if there is an object tracked
        """
        if new_bounding_box:
            self.tracker = self.tracker_constructor()
            self.tracker.init(frame, new_bounding_box)
            self.has_detection = True
            print(new_bounding_box)

        if self.has_detection:
            (success, box) = self.tracker.update(frame)
            if success:
                print('.', end='')
                return box
            else:
                print('!')
                self.has_detection = False
        return None

class VideoDartTracker(VideoTracker):



    def __init__(self, profile_paths, tracker=cv2.TrackerCSRT_create) -> None:
        super().__init__(tracker)

        self.dart_detector = detector.DartDetector()

        self.dart_detector.load_dart_profile(profile_paths)

    def track_frame(self, frame, new_bounding_box=None):
        #if there is no given bounding box and no current detection, try to detect a new dart
        frame = cv2.resize(frame, self.dart_detector.image_resolution)
        if not new_bounding_box and not self.has_detection:
            darts = self.dart_detector.find_darts(frame, False)[0]
            # try:
            try:
                if darts[0].score <= 9999:
                    new_bounding_box = darts[0].bounding_rect

            except IndexError:
                pass

            # except:
            #     print(darts[0])
            #     pass
        tracked = super().track_frame(frame, new_bounding_box)
        if tracked:
            if (int(tracked[0]) not in range(0, self.dart_detector.image_resolution[0])) or (int(tracked[1]) not in range(0, self.dart_detector.image_resolution[1])):
                print("Out of range", tracked, self.dart_detector.image_resolution)
                self.has_detection = False
                return None
        return tracked


if __name__ == '__main__':
    profiles = [os.path.join('dart_data', i) for i in os.listdir("dart_data")]
    v = VideoDartTracker(profiles, tracker=cv2.TrackerCSRT_create)

    video_in = cv2.VideoCapture('data\\VID_20200606_195831.mp4')
    running = True
    new_bounding_box = None

    while running:
        ret, frame = video_in.read()

        if not ret:
            running = False
        else:
            new_frame = cv2.resize(frame, (640, 480))
            tracked = v.track_frame(cv2.cvtColor(new_frame, cv2.COLOR_BGR2HSV), new_bounding_box)

            if tracked:
                print('tracked')
                cv2.rectangle(new_frame, (int(tracked[0]), int(tracked[1])), (int(tracked[0] + tracked[2]), int(tracked[1] + tracked[3])), 0xFF0000, 4)

            cv2.imshow("tracker example", new_frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            running = False
        elif k == ord('s'):
            new_bounding_box = cv2.selectROI("tracker example", new_frame, True)
            print(new_bounding_box)
        else:
            new_bounding_box = None

    video_in.release()
    cv2.destroyAllWindows()
