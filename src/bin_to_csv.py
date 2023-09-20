import os
import numpy as np
import numpy.typing as npt
import time
import cv2
from typing import Tuple

from .pose_module import PoseLandmarker


class BinToCsv():
    """
    BinToCsv is a class that takes .bin files as input, runs them through the pose estimation pipeline,
    and saves the output to a .csv file.
    """
    def __init__(self, input_dir: str, output_filename_left: str, output_filename_right: str, image_width: int = 640, image_height: int = 480, visualize_Pose: bool = False):
        """
        Initialize BinToCsv object

        Args:
            input_dir (str): absolute path to directory containing .bin files
            output_filename (str): name of output .csv file
            visualize_Pose (bool): whether or not to visualize pose skeleton.
        """

        # Directory containing input .bin files
        self.input_dir = input_dir

        # Name of output .csv file
        self.output_filename_left = output_filename_left
        self.output_filename_right = output_filename_right

        # Define image width and height
        self.image_width = image_width
        self.image_height = image_height

        # Whether or not to visualize pose skeleton
        self.visualize_Pose = visualize_Pose

        # Create output csv file (overwrite if it already exists)
        self.output_csv_filepath_left = os.path.join(self.input_dir, self.output_filename_left + '.csv')
        self.output_csv_file_left = open(self.output_csv_filepath_left, 'w')

        self.output_csv_filepath_right = os.path.join(self.input_dir, self.output_filename_right + '.csv')
        self.output_csv_file_right = open(self.output_csv_filepath_right, 'w')
        # Write header row
        self.output_csv_file_right.write('filename,frame_num,'
                                   'Hip_Center_X,Hip_Center_Y,Hip_Center_Z,'
                                   'Spine_X,Spine_Y,Spine_Z,'
                                   'Shoulder_Center_X,Shoulder_Center_Y,Shoulder_Center_Z,'
                                   'Head_X,Head_Y,Head_Z,'
                                   'Shoulder_Left_X,Shoulder_Left_Y,Shoulder_Left_Z,'
                                   'Elbow_Left_X,Elbow_Left_Y,Elbow_Left_Z,'
                                   'Wrist_Left_X,Wrist_Left_Y,Wrist_Left_Z,'
                                   'Hand_Left_X,Hand_Left_Y,Hand_Left_Z,'
                                   'Shoulder_Right_X,Shoulder_Right_Y,Shoulder_Right_Z,'
                                   'Elbow_Right_X,Elbow_Right_Y,Elbow_Right_Z,'
                                   'Wrist_Right_X,Wrist_Right_Y,Wrist_Right_Z,'
                                   'Hand_Right_X,Hand_Right_Y,Hand_Right_Z,'
                                   'Hip_Left_X,Hip_Left_Y,Hip_Left_Z,'
                                   'Knee_Left_X,Knee_Left_Y,Knee_Left_Z,'
                                   'Ankle_Left_X,Ankle_Left_Y,Ankle_Left_Z,'
                                   'Foot_Left_X,Foot_Left_Y,Foot_Left_Z,'
                                   'Hip_Right_X,Hip_Right_Y,Hip_Right_Z,'
                                   'Knee_Right_X,Knee_Right_Y,Knee_Right_Z,'
                                   'Ankle_Right_X,Ankle_Right_Y,Ankle_Right_Z,'
                                   'Foot_Right_X,Foot_Right_Y,Foot_Right_Z\n')

        # Set flag to indicate whether or not the class has been cleaned up
        # (either manually or automatically if the destructor was called by the garbage
        # collector)
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
            self.output_csv_file_left.close()
            self.output_csv_file_right.close()

            self.cleaned_up = True
    
    
    def _get_bin_files(self):
        """
        Get list of .bin files in input_dir

        Returns:
            bin_files (list): list of .bin files in input_dir
        """

        # Get list of .bin files in input_dir
        filelist = []

        for filename in os.listdir(self.input_dir):
            if filename.endswith(".bin"):
                # Remove the ".bin" suffix
                filename = filename[:-4]
                filelist.append(filename)

        return filelist
    
    def _delete_trailing_black_frames(self, x_all: np.ndarray, y_all: np.ndarray, z_all: np.ndarray, confidence_all: np.ndarray):
        """
        Delete trailing black frames from x_all, y_all, z_all, and confidence_all

        Args:
            x_all (np.ndarray): (n,d) array of x-coordinates with shape = (height, width, num_frames) = (480, 640, num_frames)
            y_all (np.ndarray): (n,d) array of y-coordinates with shape = (height, width, num_frames) = (480, 640, num_frames)
            z_all (np.ndarray): (n,d) array of z-coordinates with shape = (height, width, num_frames) = (480, 640, num_frames)
            confidence_all (np.ndarray): (n,d) array of confidence values with shape = (height, width, num_frames) = (480, 640, num_frames)
        """
        # Find the index of the last non-black frame
        last_non_black_index = None

        for frame_idx in range(confidence_all.shape[2] - 1, -1, -1):
            if not np.all(confidence_all[:, :, frame_idx] == 0):
                last_non_black_index = frame_idx
                break
        
        if last_non_black_index is None:
            # All frames are black, return empty arrays
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Delete trailing black frames
        x_all = x_all[:, :, :last_non_black_index+1]
        y_all = y_all[:, :, :last_non_black_index+1]
        z_all = z_all[:, :, :last_non_black_index+1]
        confidence_all = confidence_all[:, :, :last_non_black_index+1]

        return x_all, y_all, z_all, confidence_all
    
    def _read_binary_file(self, filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Read a binary file containing x, y, z coordinates, and confidence values.

        Args:
            filepath: The path to the binary file to be read.

        Returns:
            A tuple containing four NumPy arrays: x_all, y_all, z_all, and confidence_all.
            - x_all: An (n,d) array of x-coordinates.
            - y_all: An (n,d) array of y-coordinates.
            - z_all: An (n,d) array of z-coordinates.
            - confidence_all: An (n,d) array of confidence values.

        This method reads a binary file and extracts x, y, z coordinates, and confidence values.
        The binary file is assumed to have a specific structure where each array is stored sequentially.

        Note: This method assumes that the binary file is properly formatted and contains valid data.

        Example usage:
            x, y, z, confidence = _read_binary_file('data.bin')
        """
        x_all, y_all, z_all, confidence_all = np.array([]), np.array([]), np.array([]), np.array([])

        NUM_FRAMES_PER_FILE = 600

        with open(filepath, 'rb') as binary_file:
            x_all = np.frombuffer(binary_file.read(NUM_FRAMES_PER_FILE * 307200 * 2), dtype=np.int16).reshape((NUM_FRAMES_PER_FILE, 307200)).transpose()
            y_all = np.frombuffer(binary_file.read(NUM_FRAMES_PER_FILE * 307200 * 2), dtype=np.int16).reshape((NUM_FRAMES_PER_FILE, 307200)).transpose()
            z_all = np.frombuffer(binary_file.read(NUM_FRAMES_PER_FILE * 307200 * 2), dtype=np.int16).reshape((NUM_FRAMES_PER_FILE, 307200)).transpose()
            confidence_all = np.frombuffer(binary_file.read(NUM_FRAMES_PER_FILE * 307200 * 2), dtype=np.int16).reshape((NUM_FRAMES_PER_FILE, 307200)).transpose()
        
        # Get number of frames (columns) in this video clip
        # num_frames = np.size(gray_all, 1)
        num_frames = np.shape(confidence_all)[1]

        # Each array is currently (height*width, num_frames) = (480*640, num_frames) = (307200, num_frames)
        # Reshape to (height, width, num_frames) = (480, 640, num_frames)
        x_all = x_all.reshape([self.image_height, self.image_width, num_frames])
        y_all = y_all.reshape([self.image_height, self.image_width, num_frames])
        z_all = z_all.reshape([self.image_height, self.image_width, num_frames])
        confidence_all = confidence_all.reshape([self.image_height, self.image_width, num_frames])

        x_all, y_all, z_all, confidence_all = self._delete_trailing_black_frames(x_all, y_all, z_all, confidence_all)

        return x_all, y_all, z_all, confidence_all
    
    
    def _convert_camera_confidence_to_grayscale(self, confidence_array: npt.NDArray[np.int16]) -> np.ndarray:
        """
        Convert the input confidence array to grayscale and scale down the brightness to help
        with face detection.

        Args:
            confidence_array: An (n, d) confidence image in the format outputted by the IMX520 camera.

        Returns:
            An (n, d) grayscale image containing grayscale intensity values in the range [0, 255].
        """

        brightness_scaling_factor = 4
        
        grayscale_img = confidence_array.astype(float)
        grayscale_img = grayscale_img * brightness_scaling_factor
        grayscale_img[np.where(grayscale_img > 255)] = 255
        grayscale_img = grayscale_img.astype('uint8')

        return grayscale_img

        # This is a new implementation that I believe should be more resilient to
        # changes in the lighting conditions of the scene.

        # # Normalize the confidence values to the range [0, 1]
        # min_val = np.min(confidence_array)
        # max_val = np.max(confidence_array)
        # difference = max_val - min_val

        # if difference != 0:
        #     normalized_data = (confidence_array - min_val) / difference

        #     # Map the normalized data to the range [0, 255]
        #     grayscale_image = (normalized_data * 255).astype(np.uint8)

        #     return grayscale_image
        # else:
        #     return np.zeros(confidence_array.shape, dtype=np.uint8)
    
    
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
    

    def _process_pose_landmarks(
        self,
        landmarks_pixels: np.ndarray,
        frame_idx: int,
        frame_x: np.ndarray,
        frame_y: np.ndarray,
        frame_z: np.ndarray,
        frame_confidence: np.ndarray,
        frame_grayscale_rgb: np.ndarray,
        filename: str,
        side: str
    ) -> None:
        """
        Process the pose landmarks for a single frame.

        Args:
            landmarks_pixels: An (n, 2) array of landmark pixel coordinates.
            frame_idx: The index of the current frame.
            frame_x: An (n, d) array of x-coordinates.
            frame_y: An (n, d) array of y-coordinates.
            frame_z: An (n, d) array of z-coordinates.
            frame_confidence: An (n, d) array of confidence values.
            frame_grayscale_rgb: An (n, d, 3) array of RGB values.
            filename: The name of the file being processed.
        """

        # # Print x, y, and z values of landmark 11 (left shoulder)
        # landmark_11_pixel_coord_x, landmark_11_pixel_coord_y = landmarks_pixels[11]

        # # Verify that the pixel coordinates are valid
        # if self._is_valid_pixel_coordinate((landmark_11_pixel_coord_x, landmark_11_pixel_coord_y), self.image_width, self.image_height):
        #     # Use cv2 to put a blue dot on the left shoulder
        #     cv2.circle(frame_grayscale_rgb, (landmark_11_pixel_coord_x, landmark_11_pixel_coord_y), 4, (255, 0, 0), -1)

        # landmark_11_x_value = frame_x[landmark_11_pixel_coord_x][landmark_11_pixel_coord_y]
        # landmark_11_y_value = frame_y[landmark_11_pixel_coord_x][landmark_11_pixel_coord_y]
        # landmark_11_z_value = frame_z[landmark_11_pixel_coord_x][landmark_11_pixel_coord_y]

        # print(f"Left shoulder frame #{frame_idx}: ({landmark_11_x_value}, {landmark_11_y_value}, {landmark_11_z_value})")




        # For each landmark, get the x, y, and z values from the frame_x, frame_y, and frame_z arrays and store them in a numpy array
        # Then, write the numpy array to a new row in the output csv file
        xyz_values = np.zeros((len(landmarks_pixels), 3))

        for landmark_idx in range(len(landmarks_pixels)):
            landmark_pixel_coord_x, landmark_pixel_coord_y = landmarks_pixels[landmark_idx]

            # Verify that the pixel coordinates are valid
            if not self._is_valid_pixel_coordinate((landmark_pixel_coord_x, landmark_pixel_coord_y), self.image_width, self.image_height):
                # The pixel coordinates are invalid
                
                # Set the x, y, and z values to -int16.max (-32767)
                xyz_values[landmark_idx][0] = -np.iinfo(np.int16).max
                xyz_values[landmark_idx][1] = -np.iinfo(np.int16).max
                xyz_values[landmark_idx][2] = -np.iinfo(np.int16).max

                continue
            
            # The pixel coordinates are valid

            # Set the x, y, and z values to the values from the frame_x, frame_y, and frame_z arrays
            xyz_values[landmark_idx][0] = frame_x[landmark_pixel_coord_x][landmark_pixel_coord_y]
            xyz_values[landmark_idx][1] = frame_y[landmark_pixel_coord_x][landmark_pixel_coord_y]
            xyz_values[landmark_idx][2] = frame_z[landmark_pixel_coord_x][landmark_pixel_coord_y]

        # Landmarks of interest: (landmark_num: 'landmark_name_X,landmark_name_Y,landmark_name_Z,')
        # 34: 'Hip_Center_X,Hip_Center_Y,Hip_Center_Z,'
        # 35: 'Spine_X,Spine_Y,Spine_Z,'
        # 33: 'Shoulder_Center_X,Shoulder_Center_Y,Shoulder_Center_Z,'
        # 0: 'Head_X,Head_Y,Head_Z,'
        # 12: 'Shoulder_Left_X,Shoulder_Left_Y,Shoulder_Left_Z,'
        # 14: 'Elbow_Left_X,Elbow_Left_Y,Elbow_Left_Z,'
        # 16: 'Wrist_Left_X,Wrist_Left_Y,Wrist_Left_Z,'
        # 20: 'Hand_Left_X,Hand_Left_Y,Hand_Left_Z,'
        # 11: 'Shoulder_Right_X,Shoulder_Right_Y,Shoulder_Right_Z,'
        # 13: 'Elbow_Right_X,Elbow_Right_Y,Elbow_Right_Z,'
        # 15: 'Wrist_Right_X,Wrist_Right_Y,Wrist_Right_Z,'
        # 19: 'Hand_Right_X,Hand_Right_Y,Hand_Right_Z,'
        # 24: 'Hip_Left_X,Hip_Left_Y,Hip_Left_Z,'
        # 26: 'Knee_Left_X,Knee_Left_Y,Knee_Left_Z,'
        # 28: 'Ankle_Left_X,Ankle_Left_Y,Ankle_Left_Z,'
        # 32: 'Foot_Left_X,Foot_Left_Y,Foot_Left_Z,'
        # 23: 'Hip_Right_X,Hip_Right_Y,Hip_Right_Z,'
        # 25: 'Knee_Right_X,Knee_Right_Y,Knee_Right_Z,'
        # 27: 'Ankle_Right_X,Ankle_Right_Y,Ankle_Right_Z,'
        # 31: 'Foot_Right_X,Foot_Right_Y,Foot_Right_Z\n'
        if side == 'left':
            # Write the filename and frame_num to a new row in the output csv file
            self.output_csv_file_left.write(f"{filename},{frame_idx},")

            # Write the x, y, and z values for the landmarks of interest to the output csv file
            self.output_csv_file_left.write(f"{xyz_values[34][0]},{xyz_values[34][1]},{xyz_values[34][2]},")
            self.output_csv_file_left.write(f"{xyz_values[35][0]},{xyz_values[35][1]},{xyz_values[35][2]},")
            self.output_csv_file_left.write(f"{xyz_values[33][0]},{xyz_values[33][1]},{xyz_values[33][2]},")
            self.output_csv_file_left.write(f"{xyz_values[0][0]},{xyz_values[0][1]},{xyz_values[0][2]},")
            self.output_csv_file_left.write(f"{xyz_values[12][0]},{xyz_values[12][1]},{xyz_values[12][2]},")
            self.output_csv_file_left.write(f"{xyz_values[14][0]},{xyz_values[14][1]},{xyz_values[14][2]},")
            self.output_csv_file_left.write(f"{xyz_values[16][0]},{xyz_values[16][1]},{xyz_values[16][2]},")
            self.output_csv_file_left.write(f"{xyz_values[20][0]},{xyz_values[20][1]},{xyz_values[20][2]},")
            self.output_csv_file_left.write(f"{xyz_values[11][0]},{xyz_values[11][1]},{xyz_values[11][2]},")
            self.output_csv_file_left.write(f"{xyz_values[13][0]},{xyz_values[13][1]},{xyz_values[13][2]},")
            self.output_csv_file_left.write(f"{xyz_values[15][0]},{xyz_values[15][1]},{xyz_values[15][2]},")
            self.output_csv_file_left.write(f"{xyz_values[19][0]},{xyz_values[19][1]},{xyz_values[19][2]},")
            self.output_csv_file_left.write(f"{xyz_values[24][0]},{xyz_values[24][1]},{xyz_values[24][2]},")
            self.output_csv_file_left.write(f"{xyz_values[26][0]},{xyz_values[26][1]},{xyz_values[26][2]},")
            self.output_csv_file_left.write(f"{xyz_values[28][0]},{xyz_values[28][1]},{xyz_values[28][2]},")
            self.output_csv_file_left.write(f"{xyz_values[32][0]},{xyz_values[32][1]},{xyz_values[32][2]},")
            self.output_csv_file_left.write(f"{xyz_values[23][0]},{xyz_values[23][1]},{xyz_values[23][2]},")
            self.output_csv_file_left.write(f"{xyz_values[25][0]},{xyz_values[25][1]},{xyz_values[25][2]},")
            self.output_csv_file_left.write(f"{xyz_values[27][0]},{xyz_values[27][1]},{xyz_values[27][2]},")
            self.output_csv_file_left.write(f"{xyz_values[31][0]},{xyz_values[31][1]},{xyz_values[31][2]}\n")

        elif side == 'right':
            # Write the filename and frame_num to a new row in the output csv file
            self.output_csv_file_right.write(f"{filename},{frame_idx},")

            # Write the x, y, and z values for the landmarks of interest to the output csv file
            self.output_csv_file_right.write(f"{xyz_values[34][0]},{xyz_values[34][1]},{xyz_values[34][2]},")
            self.output_csv_file_right.write(f"{xyz_values[35][0]},{xyz_values[35][1]},{xyz_values[35][2]},")
            self.output_csv_file_right.write(f"{xyz_values[33][0]},{xyz_values[33][1]},{xyz_values[33][2]},")
            self.output_csv_file_right.write(f"{xyz_values[0][0]},{xyz_values[0][1]},{xyz_values[0][2]},")
            self.output_csv_file_right.write(f"{xyz_values[12][0]},{xyz_values[12][1]},{xyz_values[12][2]},")
            self.output_csv_file_right.write(f"{xyz_values[14][0]},{xyz_values[14][1]},{xyz_values[14][2]},")
            self.output_csv_file_right.write(f"{xyz_values[16][0]},{xyz_values[16][1]},{xyz_values[16][2]},")
            self.output_csv_file_right.write(f"{xyz_values[20][0]},{xyz_values[20][1]},{xyz_values[20][2]},")
            self.output_csv_file_right.write(f"{xyz_values[11][0]},{xyz_values[11][1]},{xyz_values[11][2]},")
            self.output_csv_file_right.write(f"{xyz_values[13][0]},{xyz_values[13][1]},{xyz_values[13][2]},")
            self.output_csv_file_right.write(f"{xyz_values[15][0]},{xyz_values[15][1]},{xyz_values[15][2]},")
            self.output_csv_file_right.write(f"{xyz_values[19][0]},{xyz_values[19][1]},{xyz_values[19][2]},")
            self.output_csv_file_right.write(f"{xyz_values[24][0]},{xyz_values[24][1]},{xyz_values[24][2]},")
            self.output_csv_file_right.write(f"{xyz_values[26][0]},{xyz_values[26][1]},{xyz_values[26][2]},")
            self.output_csv_file_right.write(f"{xyz_values[28][0]},{xyz_values[28][1]},{xyz_values[28][2]},")
            self.output_csv_file_right.write(f"{xyz_values[32][0]},{xyz_values[32][1]},{xyz_values[32][2]},")
            self.output_csv_file_right.write(f"{xyz_values[23][0]},{xyz_values[23][1]},{xyz_values[23][2]},")
            self.output_csv_file_right.write(f"{xyz_values[25][0]},{xyz_values[25][1]},{xyz_values[25][2]},")
            self.output_csv_file_right.write(f"{xyz_values[27][0]},{xyz_values[27][1]},{xyz_values[27][2]},")
            self.output_csv_file_right.write(f"{xyz_values[31][0]},{xyz_values[31][1]},{xyz_values[31][2]}\n")
        return
    
    
    def _process_file(self, file_num: int, num_files_to_process: int, filename: str, pose_detector: PoseLandmarker) -> None:
        """
        Load and process .bin file

        Args:
            file_num (int): number of file being processed
            num_files_to_process (int): total number of files to process
            filename (str): name of file to process

        Returns:
            None
        """

        print(f"Processing file {file_num}/{num_files_to_process}: {filename}...")

        # Load the file
        filepath = os.path.join(self.input_dir, filename + '.bin')
        x_all, y_all, z_all, confidence_all = self._read_binary_file(filepath)

        # Get number of frames in this video clip
        num_frames = np.shape(confidence_all)[2]

        # Used to calculate FPS
        previous_time = 0
        start_time = time.time()

        # Loop through all frames
        for frame_idx in range(num_frames):
            frame_x = x_all[:, :, frame_idx]
            frame_y = y_all[:, :, frame_idx]
            frame_z = z_all[:, :, frame_idx]
            frame_confidence = confidence_all[:, :, frame_idx]

            # Track face and extract intensity and depth for all ROIs in this frame

            # Convert the frame's confidence values to a grayscale image (n,d)
            frame_grayscale = self._convert_camera_confidence_to_grayscale(frame_confidence)

            # # To improve performance, optionally mark the image as not writeable to
            # # pass by reference.
            # frame_grayscale.flags.writeable = False

            # Convert grayscale image to "RGB" (n,d,3)
            frame_grayscale_rgb = cv2.cvtColor(frame_grayscale, cv2.COLOR_GRAY2RGB)
            left_frame_grayscale_rgb = np.zeros_like(frame_grayscale_rgb)
            left_frame_grayscale_rgb[:, 0:320, :] = frame_grayscale_rgb[:, 0:320, :]
            right_frame_grayscale_rgb = np.zeros_like(frame_grayscale_rgb)
            right_frame_grayscale_rgb[:, 320:640, :] = frame_grayscale_rgb[:, 320:640, :]
          
            # Get pixel locations of all pose landmarks
            # face_detected, landmarks_pixels = face_mesh_detector.find_face_mesh(image=frame_grayscale_rgb, draw=self.visualize_FaceMesh)
            left_pose_detected, left_contains_invalid_landmarks, left_landmarks_pixels = pose_detector.get_landmarks(image=left_frame_grayscale_rgb, draw=self.visualize_Pose)
            right_pose_detected, right_contains_invalid_landmarks, right_landmarks_pixels = pose_detector.get_landmarks(image=right_frame_grayscale_rgb, draw=self.visualize_Pose)
            
            # if pose_detected:
            #     # multithreading_tasks.append(self.thread_pool.submit(self._process_face_landmarks, landmarks_pixels, frame_idx, frame_x, frame_y, frame_z, frame_confidence, intensity_signal_current_file, depth_signal_current_file, ear_signal_current_file, frame_grayscale_rgb))
            #     self._process_pose_landmarks(landmarks_pixels, frame_idx, frame_x, frame_y, frame_z, frame_confidence, frame_grayscale_rgb, filename)
            
            self._process_pose_landmarks(left_landmarks_pixels, frame_idx, frame_x, frame_y, frame_z, frame_confidence, frame_grayscale_rgb, filename, 'left')
            self._process_pose_landmarks(right_landmarks_pixels, frame_idx, frame_x, frame_y, frame_z, frame_confidence, frame_grayscale_rgb, filename, 'right')

            if self.visualize_Pose:
                # Calculate and overlay FPS

                current_time = time.time()
                # FPS = (# frames processed (1)) / (# seconds taken to process those frames)
                fps = 1 / (current_time - previous_time)
                previous_time = current_time
                cv2.putText(frame_grayscale_rgb, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

                # Overlay frame number in top right corner
                text = f'{frame_idx + 1}'
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
                text_x = frame_grayscale_rgb.shape[1] - text_size[0] - 20  # Position text at the top right corner
                text_y = text_size[1] + 20
                cv2.putText(frame_grayscale_rgb, text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

                # Display frame

                cv2.imshow("Image", frame_grayscale_rgb)
                cv2.waitKey(1)
        
        # Calculate and print average FPS
        end_time = time.time()
        average_fps = num_frames / (end_time - start_time)
        print(f"Average FPS: {average_fps}")

        return
    
    def run(self):
        """
        Run pose estimation pipeline on all .bin files.
        """

        # Get list of .bin files in input_dir
        bin_files = self._get_bin_files()

        # Load and process every input video file
        file_num = 0
        num_files_to_process = len(bin_files)

        # Define MediaPipe detectors
        pose_detector = PoseLandmarker(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        for filename in bin_files:
            file_num += 1

            # Process the file
            self._process_file(file_num, num_files_to_process, filename, pose_detector)

        return


def main():
    skvs_dir = os.path.join(os.path.dirname(os.getcwd()), 'Data', 'skvs')

    # Get absolute path of directory where .bin files are located
    bins_dir = os.path.join(skvs_dir, "bins")

    # Run pose estimation pipeline on all .bin files in bins_dir and save output to csvs_dir
    myBinToCsv = BinToCsv(input_dir=bins_dir, output_filename_left="pose_data_left", output_filename_right="pose_data_right", visualize_Pose=True)
    myBinToCsv.run()

    return


if __name__ == "__main__":
    main()