import cv2
import mediapipe as mp
import time
import numpy as np
import os


class PoseDetector():
    def __init__(self) -> None:
        return
    
    def get_landmarks(self, image, draw=False):
        return False, None


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

    detector = PoseDetector()

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
