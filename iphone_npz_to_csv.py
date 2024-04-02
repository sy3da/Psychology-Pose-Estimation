import os
import numpy as np
import numpy.typing as npt
import time
import cv2
from typing import Tuple
import pandas as pd
from pose_module import PoseLandmarker

"""
This file loads in depth and rgb data from a .npz file, runs them through the pose estimation pipeline,
and outputs xyz for identified landmarks to a .csv file in the "Data/npz/csv" folder along with a video with 
the skeleton visualization in the "Data/npz/video" folder.

Runs on all .npz files in the "Data/npz" folder individually.
"""

class NpzToCsv():
    """
    NpzToCsv is a class that loads in xyz and rgb data from a .npz file, runs them through the pose estimation pipeline,
    and outputs xyz for identified landmarks to a csv file
    """

    def __init__(self, input_dir: str, image_width: int = 720, image_height: int = 960, 
                 left_participant_id: str = '00000L_', right_participant_id: str = '00000R_', draw_Pose: bool = True, 
                 visualize_Pose: bool = False, two_people: bool = False, landscape: bool = False):
        """
        Initialize NpzToCsv object

        Args:
            input_dir (str): absolute path to directory containing .npz files
            image_width (int): depends on the camera resolution
            image_height (int): depends on the camera resolution
            left_participant_id (str): 6-digit id number for left participant in collective condition
            right_participant_id (str): 6-digit id number for right participant in collective condition
            draw_Pose (bool): whether or not to draw pose skeleton
            visualize_Pose (bool): whether or not to display the vizualized pose skeletons while processing
            two_people (bool): whether or not there are two participants
            landscape (bool): whether or not the recording was taken in landscape

            NOTE: left and right participant is relative to the camera -- not the participant
        """

        # Directory containing input .npz files
        self.input_dir = input_dir
        
        if not os.path.exists(input_dir + '/csv/'): 
            os.mkdir(input_dir + '/csv/')
        
        if not os.path.exists(input_dir + '/video/'): 
            os.mkdir(input_dir + '/video/')
        
        # Whether or not to visualize pose skeleton
        self.draw_Pose = draw_Pose
        self.visualize_Pose = visualize_Pose
        # Whether or not there are two people
        self.two_people = two_people

        # Whether the picture needs to be landscape
        self.landscape = landscape

        if self.landscape == True:
            # Define image width and height (flipping height and width)
            self.image_width = image_height
            self.image_height = image_width
        else:
            # Define image width and height
            self.image_width = image_width
            self.image_height = image_height
    
        # If there are two participants the left and right subject ids
        self.left_participant_id = left_participant_id
        self.right_participant_id = right_participant_id

        self.cleaned_up = False
        
        return
    
    
    def clean_up(self):
        """
        Clean up class variables.

        NOTE: This should be called when the class is no longer needed. If this is not
        called, the garbage collector will attempt to call the destructor automatically, but
        it should not be relied on.
        """
        self.__del__()

    
    def __del__(self):
        """
        Destructor to clean up class variables.

        NOTE: This should be called automatically by the garbage collector when the object
        is no longer needed, however it may be unreliable. It is recommended to call
        self.clean_up() manually when the class is no longer needed.
        """
        if not self.cleaned_up:
            # Close the output csv file
            if self.two_people == True:
                self.output_csv_file_left.close()
                self.output_csv_file_right.close()
            else:
                self.output_csv_file.close()

            self.cleaned_up = True


    def _get_npz_files(self):
        """
        Get list of .npz files in input_dir

        Returns:
            filelist (list): list of .npz files in input_dir
        """

        # Get list of .npz files in input_dir
        filelist = []

        for filename in sorted(os.listdir(self.input_dir)):
            if filename.endswith(".npz"):
                # Remove the ".npz" suffix
                filename = filename[:-4]
                filelist.append(filename)

        return filelist
    
    def load_npz_file(self, filepath: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load in the npz file using np.load and outputs xyz and rgb arrays

        Args:
            filepath: The path to the npz file to be read.

        Returns:
            - xyz_all: An (rows, cols, 3, frame_num) array of spatial coordinate values
            - rgb_all: An (rows, cols, 3, frame_num) array of rgb intensity values
        """
        
        npz_file = np.load(filepath)
        xyz_all = npz_file['xyz_values']
        rgb_all = npz_file['rgb_values']

        return xyz_all, rgb_all
  
    def _is_valid_pixel_coordinate(self, xy: Tuple[int, int]) -> bool:
        """
        Checks if a pixel coordinate is valid (i.e. within the image bounds).
    
        Args:
            xy (Tuple[int, int]): The pixel coordinates (x, y) to be checked.
        
        Returns:
            bool: True if the pixel coordinates are within the image bounds, False otherwise.
        """
        return (xy[0] >= 0 and xy[0] < self.image_width) and (xy[1] >= 0 and xy[1] < self.image_height)

    def _process_pose_landmarks(
        self,
        landmarks_pixels: np.ndarray,
        frame_idx: int,
        frame_xyz: np.ndarray,
        frame_rgb: np.ndarray,
        filename: str,
        participant: str = 'none'
    ) -> None:
        """
        Process the pose landmarks for a single frame.

        Args:
            landmarks_pixels: An (34?, 2) array of landmark pixel coordinates.
            frame_idx: The index of the current frame.
            frame_xyz: An (rows, cols, 3) array of coordinate distances.
            frame_rgb: An (rows, cols, 3) array of RGB values.
            filename: The name of the file being processed.
            participant: Left or right if two people are in frame.
        """


        # For each landmark, get the x, y, and z values from their array and store them in a numpy array
        # Then, write the numpy array to a new row in the output csv file
        xyz_values = np.zeros((len(landmarks_pixels), 5))
        
        for landmark_idx in range(len(landmarks_pixels)):
            landmark_pixel_coord_x, landmark_pixel_coord_y = landmarks_pixels[landmark_idx]

            if not self._is_valid_pixel_coordinate((landmark_pixel_coord_x, landmark_pixel_coord_y)):
                # The pixel coordinates are invalid
                
                # Set the x, y, and z values to -int16.max (-32767)
                xyz_values[landmark_idx][0] = -np.iinfo(np.int16).max
                xyz_values[landmark_idx][1] = -np.iinfo(np.int16).max
                xyz_values[landmark_idx][2] = -np.iinfo(np.int16).max
                xyz_values[landmark_idx][3] = 0
                xyz_values[landmark_idx][4] = 0

                continue
            
            # The pixel coordinates are valid

            else:
            # Set the x, y, and z values to the values from world landmarks
                xyz_values[landmark_idx][0] = frame_xyz[landmark_pixel_coord_y, landmark_pixel_coord_x][0]
                xyz_values[landmark_idx][1] = frame_xyz[landmark_pixel_coord_y, landmark_pixel_coord_x][1]
                xyz_values[landmark_idx][2] = frame_xyz[landmark_pixel_coord_y, landmark_pixel_coord_x][2]
                xyz_values[landmark_idx][3] = landmark_pixel_coord_x
                xyz_values[landmark_idx][4] = landmark_pixel_coord_y
            
        # Landmarks of interest: (landmark_num: 'landmark_name_X,landmark_name_Y,landmark_name_Z,')
        # 34: 'Hip_Center_X,Hip_Center_Y,Hip_Center_Z,'
        # 35: 'Spine_X,Spine_Y,Spine_Z,'
        # 33: 'Shoulder_Center_X,Shoulder_Center_Y,Shoulder_Center_Z,'
        # 0: 'Head_X,Head_Y,Head_Z,'
        # 12: 'Shoulder_Right_X,Shoulder_Right_Y,Shoulder_Right_Z,'
        # 14: 'Elbow_Right_X,Elbow_Right_Y,Elbow_Right_Z,'
        # 16: 'Wrist_Right_X,Wrist_Right_Y,Wrist_Right_Z,'
        # 20: 'Hand_Right_X,Hand_Right_Y,Hand_Right_Z,'
        # 11: 'Shoulder_Left_X,Shoulder_Left_Y,Shoulder_Left_Z,'
        # 13: 'Elbow_Left_X,Elbow_Left_Y,Elbow_Left_Z,'
        # 15: 'Wrist_Left_X,Wrist_Left_Y,Wrist_Left_Z,'
        # 19: 'Hand_Left_X,Hand_Left_Y,Hand_Left_Z,'
        # 24: 'Hip_Right_X,Hip_Right_Y,Hip_Right_Z,'
        # 26: 'Knee_Right_X,Knee_Right_Y,Knee_Right_Z,'
        # 28: 'Ankle_Right_X,Ankle_Right_Y,Ankle_Right_Z,'
        # 32: 'Foot_Right_X,Foot_Right_Y,Foot_Right_Z,'
        # 23: 'Hip_Left_X,Hip_Left_Y,Hip_Left_Z,'
        # 25: 'Knee_Left_X,Knee_Left_Y,Knee_Left_Z,'
        # 27: 'Ankle_Left_X,Ankle_Left_Y,Ankle_Left_Z,'
        # 31: 'Foot_Left_X,Foot_Left_Y,Foot_Left_Z\n'

        if participant == 'Left':
            # Write the filename and frame_num to a new row in the output csv file
            self.output_csv_file_left.write(f"{filename},{frame_idx},")

            # Write the x, y, and z values for the landmarks of interest to the output csv file
            #landmark_idxs = [34, 35, 33, 0, 12, 14, 16, 20, 11, 13, 15, 19, 24, 26, 28, 32, 23, 25, 27, 31]
            self.output_csv_file_left.write(f"{xyz_values[34][0]},{xyz_values[34][1]},{xyz_values[34][2]},{xyz_values[34][3]},{xyz_values[34][4]},")
            self.output_csv_file_left.write(f"{xyz_values[35][0]},{xyz_values[35][1]},{xyz_values[35][2]},{xyz_values[35][3]},{xyz_values[35][4]},")
            self.output_csv_file_left.write(f"{xyz_values[33][0]},{xyz_values[33][1]},{xyz_values[33][2]},{xyz_values[33][3]},{xyz_values[33][4]},")
            self.output_csv_file_left.write(f"{xyz_values[0][0]}, {xyz_values[0][1]}, {xyz_values[0][2]}, {xyz_values[0][3]}, {xyz_values[0][4]}," )
            self.output_csv_file_left.write(f"{xyz_values[12][0]},{xyz_values[12][1]},{xyz_values[12][2]},{xyz_values[12][3]},{xyz_values[12][4]},")
            self.output_csv_file_left.write(f"{xyz_values[14][0]},{xyz_values[14][1]},{xyz_values[14][2]},{xyz_values[14][3]},{xyz_values[14][4]},")
            self.output_csv_file_left.write(f"{xyz_values[16][0]},{xyz_values[16][1]},{xyz_values[16][2]},{xyz_values[16][3]},{xyz_values[16][4]},")
            self.output_csv_file_left.write(f"{xyz_values[20][0]},{xyz_values[20][1]},{xyz_values[20][2]},{xyz_values[20][3]},{xyz_values[20][4]},")
            self.output_csv_file_left.write(f"{xyz_values[11][0]},{xyz_values[11][1]},{xyz_values[11][2]},{xyz_values[11][3]},{xyz_values[11][4]},")
            self.output_csv_file_left.write(f"{xyz_values[13][0]},{xyz_values[13][1]},{xyz_values[13][2]},{xyz_values[13][3]},{xyz_values[13][4]},")
            self.output_csv_file_left.write(f"{xyz_values[15][0]},{xyz_values[15][1]},{xyz_values[15][2]},{xyz_values[15][3]},{xyz_values[15][4]},")
            self.output_csv_file_left.write(f"{xyz_values[19][0]},{xyz_values[19][1]},{xyz_values[19][2]},{xyz_values[19][3]},{xyz_values[19][4]},")
            self.output_csv_file_left.write(f"{xyz_values[24][0]},{xyz_values[24][1]},{xyz_values[24][2]},{xyz_values[24][3]},{xyz_values[24][4]},")
            self.output_csv_file_left.write(f"{xyz_values[26][0]},{xyz_values[26][1]},{xyz_values[26][2]},{xyz_values[26][3]},{xyz_values[26][4]},")
            self.output_csv_file_left.write(f"{xyz_values[28][0]},{xyz_values[28][1]},{xyz_values[28][2]},{xyz_values[28][3]},{xyz_values[28][4]},")
            self.output_csv_file_left.write(f"{xyz_values[32][0]},{xyz_values[32][1]},{xyz_values[32][2]},{xyz_values[32][3]},{xyz_values[32][4]},")
            self.output_csv_file_left.write(f"{xyz_values[23][0]},{xyz_values[23][1]},{xyz_values[23][2]},{xyz_values[23][3]},{xyz_values[23][4]},")
            self.output_csv_file_left.write(f"{xyz_values[25][0]},{xyz_values[25][1]},{xyz_values[25][2]},{xyz_values[25][3]},{xyz_values[25][4]},")
            self.output_csv_file_left.write(f"{xyz_values[27][0]},{xyz_values[27][1]},{xyz_values[27][2]},{xyz_values[27][3]},{xyz_values[27][4]},")
            self.output_csv_file_left.write(f"{xyz_values[31][0]},{xyz_values[31][1]},{xyz_values[31][2]},{xyz_values[31][3]},{xyz_values[31][4]}\n")

               
        elif participant == 'Right':
            # Write the filename and frame_num to a new row in the output csv file
            self.output_csv_file_right.write(f"{filename},{frame_idx},")

            # Write the x, y, and z values for the landmarks of interest to the output csv file
            #landmark_idxs = [34, 35, 33, 0, 12, 14, 16, 20, 11, 13, 15, 19, 24, 26, 28, 32, 23, 25, 27, 31]
            self.output_csv_file_right.write(f"{xyz_values[34][0]},{xyz_values[34][1]},{xyz_values[34][2]},{xyz_values[34][3]},{xyz_values[34][4]},")
            self.output_csv_file_right.write(f"{xyz_values[35][0]},{xyz_values[35][1]},{xyz_values[35][2]},{xyz_values[35][3]},{xyz_values[35][4]},")
            self.output_csv_file_right.write(f"{xyz_values[33][0]},{xyz_values[33][1]},{xyz_values[33][2]},{xyz_values[33][3]},{xyz_values[33][4]},")
            self.output_csv_file_right.write(f"{xyz_values[0][0]}, {xyz_values[0][1]}, {xyz_values[0][2]}, {xyz_values[0][3]}, {xyz_values[0][4]}," )
            self.output_csv_file_right.write(f"{xyz_values[12][0]},{xyz_values[12][1]},{xyz_values[12][2]},{xyz_values[12][3]},{xyz_values[12][4]},")
            self.output_csv_file_right.write(f"{xyz_values[14][0]},{xyz_values[14][1]},{xyz_values[14][2]},{xyz_values[14][3]},{xyz_values[14][4]},")
            self.output_csv_file_right.write(f"{xyz_values[16][0]},{xyz_values[16][1]},{xyz_values[16][2]},{xyz_values[16][3]},{xyz_values[16][4]},")
            self.output_csv_file_right.write(f"{xyz_values[20][0]},{xyz_values[20][1]},{xyz_values[20][2]},{xyz_values[20][3]},{xyz_values[20][4]},")
            self.output_csv_file_right.write(f"{xyz_values[11][0]},{xyz_values[11][1]},{xyz_values[11][2]},{xyz_values[11][3]},{xyz_values[11][4]},")
            self.output_csv_file_right.write(f"{xyz_values[13][0]},{xyz_values[13][1]},{xyz_values[13][2]},{xyz_values[13][3]},{xyz_values[13][4]},")
            self.output_csv_file_right.write(f"{xyz_values[15][0]},{xyz_values[15][1]},{xyz_values[15][2]},{xyz_values[15][3]},{xyz_values[15][4]},")
            self.output_csv_file_right.write(f"{xyz_values[19][0]},{xyz_values[19][1]},{xyz_values[19][2]},{xyz_values[19][3]},{xyz_values[19][4]},")
            self.output_csv_file_right.write(f"{xyz_values[24][0]},{xyz_values[24][1]},{xyz_values[24][2]},{xyz_values[24][3]},{xyz_values[24][4]},")
            self.output_csv_file_right.write(f"{xyz_values[26][0]},{xyz_values[26][1]},{xyz_values[26][2]},{xyz_values[26][3]},{xyz_values[26][4]},")
            self.output_csv_file_right.write(f"{xyz_values[28][0]},{xyz_values[28][1]},{xyz_values[28][2]},{xyz_values[28][3]},{xyz_values[28][4]},")
            self.output_csv_file_right.write(f"{xyz_values[32][0]},{xyz_values[32][1]},{xyz_values[32][2]},{xyz_values[32][3]},{xyz_values[32][4]},")
            self.output_csv_file_right.write(f"{xyz_values[23][0]},{xyz_values[23][1]},{xyz_values[23][2]},{xyz_values[23][3]},{xyz_values[23][4]},")
            self.output_csv_file_right.write(f"{xyz_values[25][0]},{xyz_values[25][1]},{xyz_values[25][2]},{xyz_values[25][3]},{xyz_values[25][4]},")
            self.output_csv_file_right.write(f"{xyz_values[27][0]},{xyz_values[27][1]},{xyz_values[27][2]},{xyz_values[27][3]},{xyz_values[27][4]},")
            self.output_csv_file_right.write(f"{xyz_values[31][0]},{xyz_values[31][1]},{xyz_values[31][2]},{xyz_values[31][3]},{xyz_values[31][4]}\n")

        else:
            # Write the filename and frame_num to a new row in the output csv file
            self.output_csv_file.write(f"{filename},{frame_idx},")

            # Write the x, y, and z values for the landmarks of interest to the output csv file
            #landmark_idxs = [34, 35, 33, 0, 12, 14, 16, 20, 11, 13, 15, 19, 24, 26, 28, 32, 23, 25, 27, 31] 
            self.output_csv_file.write(f"{xyz_values[34][0]},{xyz_values[34][1]},{xyz_values[34][2]},{xyz_values[34][3]},{xyz_values[34][4]},")
            self.output_csv_file.write(f"{xyz_values[35][0]},{xyz_values[35][1]},{xyz_values[35][2]},{xyz_values[35][3]},{xyz_values[35][4]},")
            self.output_csv_file.write(f"{xyz_values[33][0]},{xyz_values[33][1]},{xyz_values[33][2]},{xyz_values[33][3]},{xyz_values[33][4]},")
            self.output_csv_file.write(f"{xyz_values[0][0]}, {xyz_values[0][1]}, {xyz_values[0][2]}, {xyz_values[0][3]}, {xyz_values[0][4]}," )
            self.output_csv_file.write(f"{xyz_values[12][0]},{xyz_values[12][1]},{xyz_values[12][2]},{xyz_values[12][3]},{xyz_values[12][4]},")
            self.output_csv_file.write(f"{xyz_values[14][0]},{xyz_values[14][1]},{xyz_values[14][2]},{xyz_values[14][3]},{xyz_values[14][4]},")
            self.output_csv_file.write(f"{xyz_values[16][0]},{xyz_values[16][1]},{xyz_values[16][2]},{xyz_values[16][3]},{xyz_values[16][4]},")
            self.output_csv_file.write(f"{xyz_values[20][0]},{xyz_values[20][1]},{xyz_values[20][2]},{xyz_values[20][3]},{xyz_values[20][4]},")
            self.output_csv_file.write(f"{xyz_values[11][0]},{xyz_values[11][1]},{xyz_values[11][2]},{xyz_values[11][3]},{xyz_values[11][4]},")
            self.output_csv_file.write(f"{xyz_values[13][0]},{xyz_values[13][1]},{xyz_values[13][2]},{xyz_values[13][3]},{xyz_values[13][4]},")
            self.output_csv_file.write(f"{xyz_values[15][0]},{xyz_values[15][1]},{xyz_values[15][2]},{xyz_values[15][3]},{xyz_values[15][4]},")
            self.output_csv_file.write(f"{xyz_values[19][0]},{xyz_values[19][1]},{xyz_values[19][2]},{xyz_values[19][3]},{xyz_values[19][4]},")
            self.output_csv_file.write(f"{xyz_values[24][0]},{xyz_values[24][1]},{xyz_values[24][2]},{xyz_values[24][3]},{xyz_values[24][4]},")
            self.output_csv_file.write(f"{xyz_values[26][0]},{xyz_values[26][1]},{xyz_values[26][2]},{xyz_values[26][3]},{xyz_values[26][4]},")
            self.output_csv_file.write(f"{xyz_values[28][0]},{xyz_values[28][1]},{xyz_values[28][2]},{xyz_values[28][3]},{xyz_values[28][4]},")
            self.output_csv_file.write(f"{xyz_values[32][0]},{xyz_values[32][1]},{xyz_values[32][2]},{xyz_values[32][3]},{xyz_values[32][4]},")
            self.output_csv_file.write(f"{xyz_values[23][0]},{xyz_values[23][1]},{xyz_values[23][2]},{xyz_values[23][3]},{xyz_values[23][4]},")
            self.output_csv_file.write(f"{xyz_values[25][0]},{xyz_values[25][1]},{xyz_values[25][2]},{xyz_values[25][3]},{xyz_values[25][4]},")
            self.output_csv_file.write(f"{xyz_values[27][0]},{xyz_values[27][1]},{xyz_values[27][2]},{xyz_values[27][3]},{xyz_values[27][4]},")
            self.output_csv_file.write(f"{xyz_values[31][0]},{xyz_values[31][1]},{xyz_values[31][2]},{xyz_values[31][3]},{xyz_values[31][4]}\n")
              
        return
    

    def _process_file(self, file_num: int, num_files_to_process: int, filename: str, pose_detector_1: PoseLandmarker, pose_detector_2: PoseLandmarker) -> None:
        """
        Load and process .npz file

        Args:
            file_num (int): number of file being processed
            num_files_to_process (int): total number of files to process
            filename (str): name of file to process
            pose_detector_1 (PoseLandmarker): mediapipe -- used to get the landmarks
            pose_detector_2 (PoseLandmarker): mediapipe -- used to get the landmarks

        Returns:
            None
        """

        print(f"Processing file {file_num}/{num_files_to_process}: {filename}...")

        # Load the file
        filepath = os.path.join(self.input_dir, filename + '.npz')
        xyz_all, rgb_all = self.load_npz_file(filepath)

        # Get number of frames in this video clip
        num_frames = np.shape(xyz_all)[3]

        # Used to calculate FPS
        previous_time = 0
        start_time = time.time()
        writer = cv2.VideoWriter(f'Data/npz/video/{self.output_filename}.avi', cv2.VideoWriter_fourcc(*'MJPG'), 12, (self.image_width, self.image_height))
        
        # Check if two people need to be tracked
        if self.two_people == True:
            # Loop through all frames
            for frame_idx in range(num_frames):
                # Split to left and right participants (relative to viewer)
                frame_xyz_left = xyz_all[:, 0:int((self.image_width/2)), :,  frame_idx]
                frame_xyz_right = xyz_all[:, int(self.image_width/2):(self.image_width), :, frame_idx]
                frame_rgb_left = rgb_all[:, 0:int((self.image_width/2)), :, frame_idx]
                frame_rgb_right = rgb_all[:, int(self.image_width/2):(self.image_width), :, frame_idx]

                # Get pixel locations of all pose landmarks for both skeletons
                # face_detected, landmarks_pixels = face_mesh_detector.find_face_mesh(image=frame_grayscale_rgb, draw=self.visualize_FaceMesh)
                pose_detected_left, contains_invalid_landmarks_left, landmarks_pixels_left, world_coord_left, frame_rgb_left = pose_detector_1.get_landmarks(image=frame_rgb_left, draw=self.draw_Pose)
                pose_detected_right, contains_invalid_landmarks_right, landmarks_pixels_right, world_coord_right, frame_rgb_right = pose_detector_2.get_landmarks(image=frame_rgb_right, draw=self.draw_Pose)

                # if pose_detected:
                #     # multithreading_tasks.append(self.thread_pool.submit(self._process_face_landmarks, landmarks_pixels, frame_idx, frame_x, frame_y, frame_z, frame_confidence, intensity_signal_current_file, depth_signal_current_file, ear_signal_current_file, frame_grayscale_rgb))
                #     self._process_pose_landmarks(landmarks_pixels, frame_idx, frame_x, frame_y, frame_z, frame_confidence, frame_grayscale_rgb, filename)
                
                self._process_pose_landmarks(landmarks_pixels_left, frame_idx, frame_xyz_left, frame_rgb_left, self.left_participant_id+filename, participant='Left')
                self._process_pose_landmarks(landmarks_pixels_right, frame_idx, frame_xyz_right, frame_rgb_right, self.right_participant_id+filename, participant='Right')

                # Combine frame_grayscale_rgb_left and _right
                frame_rgb = np.append(frame_rgb_left, frame_rgb_right, 1)
                
                # Calculate and overlay FPS
                current_time = time.time()

                # FPS = (# frames processed (1)) / (# seconds taken to process those frames)
                fps = 1 / (current_time - previous_time)
                previous_time = current_time
                
                cv2.putText(frame_rgb, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

                # Overlay frame number in top right corner
                text = f'{frame_idx + 1}'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
                text_x = frame_rgb.shape[1] - text_size[0] - 20  # Position text at the top right corner
                text_y = text_size[1] + 20
                cv2.putText(frame_rgb, text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

                writer.write(frame_rgb)

                if self.visualize_Pose == True:
                    # Display frame
                    cv2.imshow("Image", frame_rgb)
                    cv2.waitKey(1)
                    
                
        else:          
            # Loop through all frames
            for frame_idx in range(num_frames):
                frame_xyz = xyz_all[:, :, :, frame_idx]
                frame_rgb = rgb_all[:, :, :, frame_idx]

                # Get pixel locations of all pose landmarks
                # face_detected, landmarks_pixels = face_mesh_detector.find_face_mesh(image=frame_grayscale_rgb, draw=self.visualize_FaceMesh)
                pose_detected, contains_invalid_landmarks, landmarks_pixels, world_coord, frame_rgb = pose_detector_1.get_landmarks(image=frame_rgb, draw=self.draw_Pose)

                # if pose_detected:
                #     # multithreading_tasks.append(self.thread_pool.submit(self._process_face_landmarks, landmarks_pixels, frame_idx, frame_x, frame_y, frame_z, frame_confidence, intensity_signal_current_file, depth_signal_current_file, ear_signal_current_file, frame_grayscale_rgb))
                #     self._process_pose_landmarks(landmarks_pixels, frame_idx, frame_x, frame_y, frame_z, frame_confidence, frame_grayscale_rgb, filename)
                
                self._process_pose_landmarks(landmarks_pixels, frame_idx, frame_xyz, frame_rgb, filename)
                
                # Calculate and overlay FPS
                current_time = time.time()
                
                # FPS = (# frames processed (1)) / (# seconds taken to process those frames)
                fps = 1 / (current_time - previous_time)
                previous_time = current_time
                
                
                cv2.putText(frame_rgb, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

                # Overlay frame number in top right corner
                text = f'{frame_idx + 1}'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
                text_x = frame_rgb.shape[1] - text_size[0] - 20  # Position text at the top right corner
                text_y = text_size[1] + 20
                cv2.putText(frame_rgb, text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

                writer.write(frame_rgb) 
                 
                if self.visualize_Pose == True:
                    # Display frame
                    cv2.imshow("Image", frame_rgb)
                    cv2.waitKey(1)
                       
            
        # Calculate and print average FPS
        writer.release()
        end_time = time.time()
        average_fps = num_frames / (end_time - start_time)
        print(f"Average FPS: {average_fps}")

        return


    def run(self):
        """
        Run pose estimation pipeline on all .npz files.
        """

        # Get list of .npz files in input_dir
        npz_files = self._get_npz_files()

        # Load and process every input video file
        file_num = 0
        num_files_to_process = len(npz_files)

        # Define MediaPipe detectors
        pose_detector_1 = PoseLandmarker(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.9)
        pose_detector_2 = PoseLandmarker(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.9)

        # loop through all the files in the npz folder
        for filename in npz_files:
            file_num += 1
            self.output_filename = filename

            if self.two_people == True:
                # Create two output csv files (overwrite if it already exists)
                self.output_csv_filepath_left = os.path.join(self.input_dir + '/csv', self.left_participant_id + self.output_filename + '.csv')
                self.output_csv_file_left = open(self.output_csv_filepath_left, 'w')

                self.output_csv_filepath_right = os.path.join(self.input_dir + '/csv', self.right_participant_id + self.output_filename + '.csv')
                self.output_csv_file_right = open(self.output_csv_filepath_right, 'w')

                # Write header row
                self.output_csv_file_left.write('filename,frame_num,'
                                                'Hip_Center_X,Hip_Center_Y,Hip_Center_Z,Hip_Center_PixelX,Hip_Center_PixelY,'
                                                'Spine_X,Spine_Y,Spine_Z,Spine_PixelX,Spine_PixelY,'
                                                'Shoulder_Center_X,Shoulder_Center_Y,Shoulder_Center_Z,Shoulder_Center_PixelX,Shoulder_Center_PixelY,'
                                                'Head_X,Head_Y,Head_Z,Head_PixelX,Head_PixelY,'
                                                'Shoulder_Right_X,Shoulder_Right_Y,Shoulder_Right_Z,Shoulder_Right_PixelX,Shoulder_Right_PixelY,'
                                                'Elbow_Right_X,Elbow_Right_Y,Elbow_Right_Z,Elbow_Right_PixelX,Elbow_Right_PixelY,'
                                                'Wrist_Right_X,Wrist_Right_Y,Wrist_Right_Z,Wrist_Right_PixelX,Wrist_Right_PixelY,'
                                                'Hand_Right_X,Hand_Right_Y,Hand_Right_Z,Hand_Right_PixelX,Hand_Right_PixelY,'
                                                'Shoulder_Left_X,Shoulder_Left_Y,Shoulder_Left_Z,Shoulder_Left_PixelX,Shoulder_Left_PixelY,'
                                                'Elbow_Left_X,Elbow_Left_Y,Elbow_Left_Z,Elbow_Left_PixelX,Elbow_Left_PixelY,'
                                                'Wrist_Left_X,Wrist_Left_Y,Wrist_Left_Z,Wrist_Left_PixelX,Wrist_Left_PixelY,'
                                                'Hand_Left_X,Hand_Left_Y,Hand_Left_Z,Hand_Left_PixelX,Hand_Left_PixelY,'
                                                'Hip_Right_X,Hip_Right_Y,Hip_Right_Z,Hip_Right_PixelX,Hip_Right_PixelY,'
                                                'Knee_Right_X,Knee_Right_Y,Knee_Right_Z,Knee_Right_PixelX,Knee_Right_PixelY,'
                                                'Ankle_Right_X,Ankle_Right_Y,Ankle_Right_Z,Ankle_Right_PixelX,Ankle_Right_PixelY,'
                                                'Foot_Right_X,Foot_Right_Y,Foot_Right_Z,Foot_Right_PixelX,Foot_Right_PixelY,'
                                                'Hip_Left_X,Hip_Left_Y,Hip_Left_Z,Hip_Left_PixelX,Hip_Left_PixelY,'
                                                'Knee_Left_X,Knee_Left_Y,Knee_Left_Z,Knee_Left_PixelX,Knee_Left_PixelY,'
                                                'Ankle_Left_X,Ankle_Left_Y,Ankle_Left_Z,Ankle_Left_PixelX,Ankle_Left_PixelY,'
                                                'Foot_Left_X,Foot_Left_Y,Foot_Left_Z,Foot_Left_PixelX,Foot_Left_PixelY\n')
        
                self.output_csv_file_right.write('filename,frame_num,'
                                                'Hip_Center_X,Hip_Center_Y,Hip_Center_Z,Hip_Center_PixelX,Hip_Center_PixelY,'
                                                'Spine_X,Spine_Y,Spine_Z,Spine_PixelX,Spine_PixelY,'
                                                'Shoulder_Center_X,Shoulder_Center_Y,Shoulder_Center_Z,Shoulder_Center_PixelX,Shoulder_Center_PixelY,'
                                                'Head_X,Head_Y,Head_Z,Head_PixelX,Head_PixelY,'
                                                'Shoulder_Right_X,Shoulder_Right_Y,Shoulder_Right_Z,Shoulder_Right_PixelX,Shoulder_Right_PixelY,'
                                                'Elbow_Right_X,Elbow_Right_Y,Elbow_Right_Z,Elbow_Right_PixelX,Elbow_Right_PixelY,'
                                                'Wrist_Right_X,Wrist_Right_Y,Wrist_Right_Z,Wrist_Right_PixelX,Wrist_Right_PixelY,'
                                                'Hand_Right_X,Hand_Right_Y,Hand_Right_Z,Hand_Right_PixelX,Hand_Right_PixelY,'
                                                'Shoulder_Left_X,Shoulder_Left_Y,Shoulder_Left_Z,Shoulder_Left_PixelX,Shoulder_Left_PixelY,'
                                                'Elbow_Left_X,Elbow_Left_Y,Elbow_Left_Z,Elbow_Left_PixelX,Elbow_Left_PixelY,'
                                                'Wrist_Left_X,Wrist_Left_Y,Wrist_Left_Z,Wrist_Left_PixelX,Wrist_Left_PixelY,'
                                                'Hand_Left_X,Hand_Left_Y,Hand_Left_Z,Hand_Left_PixelX,Hand_Left_PixelY,'
                                                'Hip_Right_X,Hip_Right_Y,Hip_Right_Z,Hip_Right_PixelX,Hip_Right_PixelY,'
                                                'Knee_Right_X,Knee_Right_Y,Knee_Right_Z,Knee_Right_PixelX,Knee_Right_PixelY,'
                                                'Ankle_Right_X,Ankle_Right_Y,Ankle_Right_Z,Ankle_Right_PixelX,Ankle_Right_PixelY,'
                                                'Foot_Right_X,Foot_Right_Y,Foot_Right_Z,Foot_Right_PixelX,Foot_Right_PixelY,'
                                                'Hip_Left_X,Hip_Left_Y,Hip_Left_Z,Hip_Left_PixelX,Hip_Left_PixelY,'
                                                'Knee_Left_X,Knee_Left_Y,Knee_Left_Z,Knee_Left_PixelX,Knee_Left_PixelY,'
                                                'Ankle_Left_X,Ankle_Left_Y,Ankle_Left_Z,Ankle_Left_PixelX,Ankle_Left_PixelY,'
                                                'Foot_Left_X,Foot_Left_Y,Foot_Left_Z,Foot_Left_PixelX,Foot_Left_PixelY\n')
            
            else:
        
                # Create output csv file (overwrite if it already exists)
                self.output_csv_filepath = os.path.join(self.input_dir + '/csv', self.output_filename + '.csv')
                self.output_csv_file = open(self.output_csv_filepath, 'w')

                # Write header row
                self.output_csv_file.write('filename,frame_num,'
                                            'Hip_Center_X,Hip_Center_Y,Hip_Center_Z,Hip_Center_PixelX,Hip_Center_PixelY,'
                                            'Spine_X,Spine_Y,Spine_Z,Spine_PixelX,Spine_PixelY,'
                                            'Shoulder_Center_X,Shoulder_Center_Y,Shoulder_Center_Z,Shoulder_Center_PixelX,Shoulder_Center_PixelY,'
                                            'Head_X,Head_Y,Head_Z,Head_PixelX,Head_PixelY,'
                                            'Shoulder_Right_X,Shoulder_Right_Y,Shoulder_Right_Z,Shoulder_Right_PixelX,Shoulder_Right_PixelY,'
                                            'Elbow_Right_X,Elbow_Right_Y,Elbow_Right_Z,Elbow_Right_PixelX,Elbow_Right_PixelY,'
                                            'Wrist_Right_X,Wrist_Right_Y,Wrist_Right_Z,Wrist_Right_PixelX,Wrist_Right_PixelY,'
                                            'Hand_Right_X,Hand_Right_Y,Hand_Right_Z,Hand_Right_PixelX,Hand_Right_PixelY,'
                                            'Shoulder_Left_X,Shoulder_Left_Y,Shoulder_Left_Z,Shoulder_Left_PixelX,Shoulder_Left_PixelY,'
                                            'Elbow_Left_X,Elbow_Left_Y,Elbow_Left_Z,Elbow_Left_PixelX,Elbow_Left_PixelY,'
                                            'Wrist_Left_X,Wrist_Left_Y,Wrist_Left_Z,Wrist_Left_PixelX,Wrist_Left_PixelY,'
                                            'Hand_Left_X,Hand_Left_Y,Hand_Left_Z,Hand_Left_PixelX,Hand_Left_PixelY,'
                                            'Hip_Right_X,Hip_Right_Y,Hip_Right_Z,Hip_Right_PixelX,Hip_Right_PixelY,'
                                            'Knee_Right_X,Knee_Right_Y,Knee_Right_Z,Knee_Right_PixelX,Knee_Right_PixelY,'
                                            'Ankle_Right_X,Ankle_Right_Y,Ankle_Right_Z,Ankle_Right_PixelX,Ankle_Right_PixelY,'
                                            'Foot_Right_X,Foot_Right_Y,Foot_Right_Z,Foot_Right_PixelX,Foot_Right_PixelY,'
                                            'Hip_Left_X,Hip_Left_Y,Hip_Left_Z,Hip_Left_PixelX,Hip_Left_PixelY,'
                                            'Knee_Left_X,Knee_Left_Y,Knee_Left_Z,Knee_Left_PixelX,Knee_Left_PixelY,'
                                            'Ankle_Left_X,Ankle_Left_Y,Ankle_Left_Z,Ankle_Left_PixelX,Ankle_Left_PixelY,'
                                            'Foot_Left_X,Foot_Left_Y,Foot_Left_Z,Foot_Left_PixelX,Foot_Left_PixelY\n')

            # Set flag to indicate whether or not the class has been cleaned up
            # (either manually or automatically if the destructor was called by the garbage
            # collector)
            self.cleaned_up = False
            # Process the file
            self._process_file(file_num, num_files_to_process, filename, pose_detector_1, pose_detector_2)

        return
    

def main():
    # Get absolute path of directory where .npz files are located
    npzs_dir = os.path.join(os.getcwd(), 'Data', 'npz')
    print(npzs_dir)

    # Run pose estimation pipeline on all .npz files in npz_dir and save output to csvs_dir
    myNpzToCsv = NpzToCsv(input_dir=npzs_dir, visualize_Pose=False, two_people=False, landscape=False)  # , left_participant_id = '965142_', right_participant_id = '510750_'
    myNpzToCsv.run()

    return


if __name__ == "__main__":
    main()
        