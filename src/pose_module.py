import cv2
import mediapipe as mp
import time
import numpy as np
import os
from typing import Tuple
import math


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

    def get_landmarks(self, image: np.ndarray, draw: bool = False) -> Tuple[bool, bool, np.ndarray]:
        """
        Detect pose and return pose landmarks for the given image.

        Args:
            image (np.ndarray): The input image in BGR format.
            draw (bool): Whether to draw the pose skeleton on the image.

        Returns:
            tuple[bool, bool, np.ndarray]: A tuple containing:
                - pose_detected (bool): Indicates whether a pose was detected in the image.
                - contains_invalid_landmarks (bool): Indicates whether the returned landmarks_pixels array contains
                    invalid landmarks (i.e. landmarks that are outside of the image bounds or landmarks located at
                    negative-valued normalized pixel coordinates).
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
        contains_invalid_landmarks = False

        if results.pose_landmarks:
            # Pose detected
            pose_detected = True

            if draw:
                self.mp_draw.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, self.drawing_spec_landmark,
                                                self.drawing_spec_connection)
            
            # Loop through each landmark
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                # There are 33 landmarks in total, each with x, y, z normalized coordinates and a visibility value
                # print(id, landmark)

                # Convert normalized coordinates to pixel coordinates (NOTE: z is currently unused)
                # x, y = int(landmark.x * image_width), int(landmark.y * image_height)
                x, y = self._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_width, image_height)

                if (x, y) == (-1, -1):
                    # This landmark is invalid
                    contains_invalid_landmarks = True

                # print(id, x, y)

                # Overlay blue dots on each landmark to verify pixel coordinates
                # cv2.circle(image, (x, y), 5, (255, 0, 0), cv2.FILLED)

                # Store pixel coordinates in array
                landmarks_pixels[id] = (x, y)

        return pose_detected, contains_invalid_landmarks, landmarks_pixels
    
    def _is_valid_normalized_value(self, normalized_value: float) -> bool:
        """
        Checks if a value is a valid normalized value in the range [0, 1].
    
        Args:
            normalized_value (float): The value to be checked.
        
        Returns:
            bool: True if the value is within the range [0, 1], False otherwise.
        """
        return (normalized_value > 0 or math.isclose(0, normalized_value)) and (normalized_value < 1 or math.isclose(1, normalized_value))
    
    def _normalized_to_pixel_coordinates(self, normalized_x: float, normalized_y: float,
                                         image_width: int, image_height: int) -> Tuple[int, int]:
        """
        Converts normalized pixel coordinates to pixel coordinates in the image by scaling with image dimensions.

        Args:
            normalized_x (float): The x-coordinate in normalized units [0, 1].
            normalized_y (float): The y-coordinate in normalized units [0, 1].
            image_width (int): The width of the image in pixels.
            image_height (int): The height of the image in pixels.

        Returns:
            Tuple[int, int]: A tuple containing the pixel coordinates (x, y).
            
            If the provided normalized coordinates are valid, the function returns the corresponding pixel coordinates
            inside the image.
            
            If the provided normalized coordinates are invalid (negative or outside of the image bounds),
            the function returns (-1, -1).
        """

        if not (self._is_valid_normalized_value(normalized_x) and self._is_valid_normalized_value(normalized_y)):
            # At least one of the normalized coordinates is invalid
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return -1, -1
        
        x_px = min(math.floor(int(normalized_x * image_width)), image_width - 1)
        y_px = min(math.floor(int(normalized_y * image_height)), image_height - 1)
        return x_px, y_px


def main():
    # Get the directory where the running script is located
    script_directory = os.path.dirname(os.path.realpath(__file__))
    input_video = os.path.join(script_directory, "test_videos", "1_walking_toward.mp4")
    # input_video = os.path.join(script_directory, "test_videos", "6_two_people_walking_towards.mp4")
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
        pose_detected, contains_invalid_landmarks, landmarks_pixels = detector.get_landmarks(image=frame, draw=True)

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
