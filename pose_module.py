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
                                      min_tracking_confidence=min_tracking_confidence,model_complexity=1)
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
            If a landmark is not detected, its corresponding pixel coordinates will be (-1, -1).

        References:
            - Pose Landmarks Key: https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png
        """

        # Get image dimensions
        image_height, image_width, image_channels = image.shape

        # CV2 loads images/videos in BGR format, convert to RGB
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(np.ascontiguousarray(image, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        results = self.pose.process(image)

        # landmarks_pixels is an array of shape (36, 2) with x, y coordinates (as pixels) for each landmark
        # Initialize array with (-2, -2) for each landmark to indicate that the landmark has not been detected
        landmarks_pixels = np.full((36, 2), -2, dtype="int")
        world_coord = np.zeros((36, 3))

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
                world_coord[id] = (results.pose_world_landmarks.landmark[id].x, results.pose_world_landmarks.landmark[id].y, results.pose_world_landmarks.landmark[id].z)

            # Add shoulder_center landmark
            landmarks_pixels, world_coord = self._add_landmark_shoulder_center(landmarks_pixels, world_coord, image, draw)

            # Add hip_center landmark
            landmarks_pixels, world_coord = self._add_landmark_hip_center(landmarks_pixels, world_coord, image, draw)

            # Draw line between shoulder_center and hip_center landmarks
            if draw:
                image = self._draw_line_between_shoulder_center_and_hip_center(landmarks_pixels, image)

            # Add spine landmark
            landmarks_pixels, world_coord = self._add_landmark_spine(landmarks_pixels, world_coord, image, draw)

        return pose_detected, contains_invalid_landmarks, landmarks_pixels, world_coord, image
    
    
    def _add_landmark_shoulder_center(self, landmarks_pixels: np.ndarray, world_coord, image: np.ndarray, draw: bool) -> np.ndarray:
        # Get image dimensions
        image_height, image_width, image_channels = image.shape

        # Verify that landmarks 11 and 12 are not invalid
        if not self._is_valid_pixel_coordinate(landmarks_pixels[11], image_width, image_height) or not self._is_valid_pixel_coordinate(landmarks_pixels[12], image_width, image_height):
            # Left and right shoulder landmarks are invalid, so shoulder_center landmark is invalid
            # TODO: Maybe try to create shoulder_center landmark by using some other available landmarks
            landmarks_pixels[33] = (-1, -1)
            
            return landmarks_pixels, world_coord
        
        # Left and right shoulder landmarks are valid
        
        # Create shoulder_center landmark by averaging the left and right shoulder landmarks
        landmarks_pixels[33] = (int((landmarks_pixels[11][0] + landmarks_pixels[12][0]) / 2),
                                int((landmarks_pixels[11][1] + landmarks_pixels[12][1]) / 2))
        world_coord[33] = ((world_coord[11][0] + world_coord[12][0])/2,
                           (world_coord[11][1] + world_coord[12][1])/2,
                           (world_coord[11][2] + world_coord[12][2])/2)
        if draw:
            # Overlay orange dot on shoulder_center landmark to verify pixel coordinates
            cv2.circle(image, landmarks_pixels[33], 5, (0, 165, 255), cv2.FILLED)
        
        return landmarks_pixels, world_coord
    

    def _add_landmark_hip_center(self, landmarks_pixels: np.ndarray, world_coord, image: np.ndarray, draw: bool) -> np.ndarray:
        # Get image dimensions
        image_height, image_width, image_channels = image.shape

        # Verify that landmarks 23 and 24 are not invalid
        if not self._is_valid_pixel_coordinate(landmarks_pixels[23], image_width, image_height) or not self._is_valid_pixel_coordinate(landmarks_pixels[24], image_width, image_height):
            # Left and right hip landmarks are invalid, so hip_center landmark is invalid
            landmarks_pixels[34] = (-1, -1)

            return landmarks_pixels, world_coord
        
        # Left and right hip landmarks are valid
        
        # Create hip_center landmark by averaging the left and right hip landmarks
        landmarks_pixels[34] = (int((landmarks_pixels[23][0] + landmarks_pixels[24][0]) / 2),
                                int((landmarks_pixels[23][1] + landmarks_pixels[24][1]) / 2))
        world_coord[34] = ((world_coord[23][0] + world_coord[24][0])/2,
                            (world_coord[23][1] + world_coord[24][1])/2,
                            (world_coord[23][2] + world_coord[24][2])/2)
                
        if draw:
            # Overlay green dot on hip_center landmark to verify pixel coordinates
            cv2.circle(image, landmarks_pixels[34], 5, (0, 255, 0), cv2.FILLED)

        return landmarks_pixels, world_coord
    

    def _draw_line_between_shoulder_center_and_hip_center(self, landmarks_pixels: np.ndarray, image: np.ndarray) -> np.ndarray:
        # Get image dimensions
        image_height, image_width, image_channels = image.shape

        # Verify that landmarks 33 and 34 are not invalid
        if not self._is_valid_pixel_coordinate(landmarks_pixels[33], image_width, image_height) or not self._is_valid_pixel_coordinate(landmarks_pixels[34], image_width, image_height):
            # shoulder_center and hip_center landmarks are invalid, so don't draw line
            return image
        
        # shoulder_center and hip_center landmarks are valid

        # Draw line between shoulder_center and hip_center landmarks
        cv2.line(image, landmarks_pixels[33], landmarks_pixels[34], (255, 0, 255), 2)
        
        return image
    

    def _add_landmark_spine(self, landmarks_pixels: np.ndarray, world_coord, image: np.ndarray, draw: bool) -> np.ndarray:
        # Get image dimensions
        image_height, image_width, image_channels = image.shape
        
        # Verify that landmarks 33 and 34 are not invalid
        if not self._is_valid_pixel_coordinate(landmarks_pixels[33], image_width, image_height) or not self._is_valid_pixel_coordinate(landmarks_pixels[34], image_width, image_height):
            # shoulder_center and hip_center landmarks are invalid, so spine landmark is invalid
            landmarks_pixels[35] = (-1, -1)

            return landmarks_pixels, world_coord
        
        # shoulder_center and hip_center landmarks are valid

        # Calculate the coordinates of "spine" landmark that is 2/3 of the way down from the "shoulder_center" to the "hip_center"
        landmarks_pixels[35] = (int(landmarks_pixels[33][0] + (landmarks_pixels[34][0] - landmarks_pixels[33][0]) * 2 / 3),
                                int(landmarks_pixels[33][1] + (landmarks_pixels[34][1] - landmarks_pixels[33][1]) * 2 / 3))
        world_coord[35] = ((world_coord[33][0] + (world_coord[34][0] - world_coord[33][0]) * 2 / 3),
                            (world_coord[33][1] + (world_coord[34][1] - world_coord[33][1]) * 2 / 3),
                            (world_coord[33][2] + (world_coord[34][2] - world_coord[33][2]) * 2 / 3))
        if draw:
            # Overlay red dot on spine landmark to verify pixel coordinates
            cv2.circle(image, landmarks_pixels[35], 5, (0, 0, 255), cv2.FILLED)

        return landmarks_pixels, world_coord
    

    def _is_valid_pixel_coordinate(self, xy: Tuple[int, int], image_width: int, image_height: int) -> bool:
        """
        Checks if a pixel coordinate is valid (i.e. within the image bounds).
    
        Args:
            xy (Tuple[int, int]): The pixel coordinates (x, y) to be checked.
            image_width (int): The width of the image in pixels.
            image_height (int): The height of the image in pixels.
        
        Returns:
            bool: True if the pixel coordinates are within the image bounds, False otherwise.
        """
        return (xy[0] >= 0 and xy[0] < image_width) and (xy[1] >= 0 and xy[1] < image_height)
    
    
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
    # input_video = os.path.join(script_directory, "test_videos", "1_walking_toward.mp4")
    # input_video = os.path.join(script_directory, "test_videos", "6_two_people_walking_towards.mp4")
    input_video = os.path.join(os.path.dirname(script_directory), "Data", "ExampleData", "20190807T151230_001", "Baseline 1", "Recording_1.mp4")
    # input_video = os.path.join(os.path.dirname(script_directory), "Data", "ExampleData", "20190807T151230_001", "Baseline 1", "Recording_2.mp4")
    video = cv2.VideoCapture(input_video)
    if not video.isOpened():
        print("Error opening video file:", input_video)
        exit(1)

    # # Create VideoWriter object
    # output_video_path = "1_walking_toward_grayscale.mp4"  # Change the filename and path as per your requirement
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video
    # input_fps = video.get(cv2.CAP_PROP_FPS)  # Get the original FPS of the input video
    # frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # video_writer = cv2.VideoWriter(output_video_path, fourcc, input_fps, (frame_width, frame_height))

    detector = PoseLandmarker()

    previous_time = 0
    start_time = time.time()

    # Loop through video frames
    while video.isOpened():
        # Get a frame of video
        ret, frame = video.read()
        if not ret:
            break

        # # Convert frame to grayscale
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # Stack grayscale frame three times along the channel axis to create a 3-channel grayscale frame
        # frame = cv2.merge((frame, frame, frame))

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

        # Write frame to the output video
        # video_writer.write(frame)

    end_time = time.time()
    average_fps = (video.get(cv2.CAP_PROP_FRAME_COUNT) - 1) / (end_time - start_time)
    print(f"Average FPS: {average_fps}")

    video.release()

    # Release the video writer
    # video_writer.release()


if __name__ == "__main__":
    main()