import cv2
import mediapipe as mp
import time
import numpy as np
import os


class PoseLandmarker():
    def __init__(self, static_image_mode: bool = False, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        """
        Initialize class variables.

        Args:
            static_image_mode (bool): Whether to treat input images as static images or a continuous video stream.
                False: Treat input images as a continuous video stream (a.k.a. video mode). This mode will try to detect
                    poses in the first input image, and upon a successful detection, subsequent detections will be
                    made by attempting to track the pose from the previous frame. If tracking is successful, computation
                    for the frames after the first one should be faster than running pose detection on each individual
                    input image. Use this mode when you want to track poses across images in a video stream or for
                    live, real-time pose detection.
                True: Treat input images as static images (a.k.a. image mode). This mode will treat each input image as
                    an independent image and will not try to detect or track poses across images. Use this mode when you
                    want to run pose detection/pose landmark detection on a set of non-continuous, unrelated
                    input images.
            min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for pose detection to be considered
                successful.
            min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for pose tracking to be considered
                successful.
        """
        self.mp_pose = mp.solutions.pose # type: ignore
        self.pose = self.mp_pose.Pose(static_image_mode=static_image_mode,
                                      min_detection_confidence=min_detection_confidence,
                                      min_tracking_confidence=min_tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils # type: ignore
        self.drawing_spec_landmark = self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
        self.drawing_spec_connection = self.mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)

        return

    def get_landmarks(self, image, draw=False):
        """
        Detect pose and return pose landmarks for the given image.

        Args:
            image (np.ndarray): The input image in BGR format.
            draw (bool): Whether to draw the pose skeleton on the image.

        Returns:
            tuple[bool, np.ndarray]: A tuple containing:
                - pose_detected (bool): Indicates whether a pose was detected in the image.
                - landmarks_pixels (np.ndarray): An array of shape (33, 2) representing the pixel coordinates (x, y)
                  for each of the 33 total pose landmarks detected. The i-th row corresponds to the i-th landmark
                  (zero-indexed, so row 0 is landmark 1).

        Note:
            The input image should be in BGR format, as OpenCV loads images/videos in BGR format by default.
            The returned landmarks_pixels array contains the pixel coordinates (x, y) for each pose landmark.

        References:
            - Pose Landmarks Key: https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png
        """

        # Get image dimensions
        image_height, image_width, image_channels = image.shape

        # CV2 loads images/videos in BGR format, convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.pose.process(image_rgb)

        # landmarks_pixels is an array of shape (33, 2) with x, y coordinates (as pixels) for each landmark
        landmarks_pixels = np.zeros((33, 2), dtype="int")

        pose_detected = False

        if results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, self.drawing_spec_landmark,
                                                self.drawing_spec_connection)

        return pose_detected, landmarks_pixels


def main():
    # Get the directory where the running script is located
    script_directory = os.path.dirname(os.path.realpath(__file__))
    input_video = os.path.join(script_directory, "test_videos", "1_walking_toward.mp4")
    # input_video = "../Data/test_videos/1_walking_toward.mp4"
    video = cv2.VideoCapture(input_video)
    if not video.isOpened():
        print("Error opening video file:", input_video)
        exit(1)

    previous_time = 0
    start_time = time.time()

    detector = PoseLandmarker()

    # Loop through video frames
    while video.isOpened():
        # Get a frame of video
        ret, frame = video.read()
        if not ret:
            break

        # Get pixel locations of all pose landmarks
        pose_detected, landmarks_pixels = detector.get_landmarks(image=frame, draw=True)

        # if pose_detected:
        #     # Do something with the landmarks

        # Calculate and overlay FPS

        current_time = time.time()
        # FPS = (# frames processed (1)) / (# seconds taken to process those frames)
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Display frame

        cv2.imshow("Image", frame)
        cv2.waitKey(1)

    end_time = time.time()
    average_fps = (video.get(cv2.CAP_PROP_FRAME_COUNT) - 1) / (end_time - start_time)
    print(f"Average FPS: {average_fps}")

    video.release()


if __name__ == "__main__":
    main()
