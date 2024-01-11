import os
import numpy as np
import numpy.typing as npt
import time
import cv2
from typing import Tuple
from scipy.io import loadmat
from scipy.optimize import fsolve
#from gekko import GEKKO

from pose_module import PoseLandmarker

class MatToCsv():
    """
    MatToCsv is a class that loads in depths and intensities from a .mat file, runs them through the pose estimation pipeline,
    converts depth to x,y,z, and outputs them to a csv file
    """

    def __init__(self, input_dir: str, image_width: int = 600, image_height: int = 804, image_fov: int = 77, left_participant_id: str = '000000_', left_part_demographics: str = '_M_21' , right_participant_id: str = '000000_', right_part_demographics: str = '_F_22', visualize_Pose: bool = False, two_people: bool = False, landscape: bool = False):
        """
        Initialize MatToCsv object

        Args:
            input_dir (str): absolute path to directory containing .mat files
            output_filename (str): name of output .csv file
            visualize_Pose (bool): whether or not to visualize pose skeleton.
        """

        # Directory containing input .mat files
        self.input_dir = input_dir

        # Name of output .csv file
        # self.output_filename = output_filename

        # Define image width, height, and fov
        self.image_width = image_width
        self.image_height = image_height
        self.image_fov = image_fov
        # Whether or not to visualize pose skeleton
        self.visualize_Pose = visualize_Pose
        # Whether or not there are two people
        self.two_people = two_people
        # Whether the picture needs to be landscape
        self.landscape = landscape

        # if there are two participants the left and right subject ids
        self.left_participant_id = left_participant_id
        self.right_participant_id = right_participant_id
        self.left_part_demographics = left_part_demographics
        self.right_part_demographics = right_part_demographics

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


    def _get_mat_files(self):
        """
        Get list of .mat files in input_dir

        Returns:
            mat_files (list): list of .mat files in input_dir
        """

        # Get list of .mat files in input_dir
        filelist = []

        for filename in os.listdir(self.input_dir):
            if filename.endswith(".mat"):
                # Remove the ".mat" suffix
                filename = filename[:-4]
                filelist.append(filename)

        return filelist
    

    def load_mat_file(self, filepath: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load in the mat file using scipy.io.loadmat and outputs depth and intensity

        Args:
            filepath: The path to the mat file to be read.

        Returns:
            A tuple containing two NumPy arrays: depth_all and intensity_all
            - depth_all: An (n,d) array of depth values
            - intensity_all: An (n,d) array of intensity values
        """
        
        mat_file = loadmat(filepath)
        depth_all = mat_file['D_values']
        intensity_all = mat_file['I_values']

        return depth_all, intensity_all
    
    def _convert_camera_intensity_to_grayscale(self, intensity_array: npt.NDArray[np.int16]) -> np.ndarray:
        """
        Convert the input intensity array to grayscale and scale down the brightness to help
        with face detection.

        Args:
            intensity_array: An (n, d) intensity image in the format outputted by the Thanos camera.

        Returns:
            An (n, d) grayscale image containing grayscale intensity values in the range [0, 255].
        """

        brightness_scaling_factor = 4
        
        grayscale_img = intensity_array.astype('float')
        grayscale_img = grayscale_img * brightness_scaling_factor
        grayscale_img[np.where(grayscale_img > 255)] = 255
        grayscale_img = grayscale_img.astype('uint8')

        return grayscale_img

    
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
    
    def fovFunction(self,angles):
        theta = angles[0]
        phi = angles[1]

        F = np.empty((2))
        F[0] = pow(theta,2) + pow(phi,2) - pow(self.image_fov,2)
        F[1] = phi - (self.image_height/self.image_width)*theta

        return F
 
    def convert_depth_to_xyz(self, depths, landmark_pixel_x, landmark_pixel_y, image_w, image_h, fov) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert depth values to x,y,z coordinates for one landmark in one frame.

        Args:
            depths: Depth data for one frame.
            landmark_pixel_x: One landmark pixel x coordinate for one frame
            landmark_pixel_y: One landmark pixel y coordinate for one frame
            image_w: Width of the image in pixels
            image_h: Height of the image in pixels
            fov: Field of view of the camera

        Returns:
            A tuple containing three integers: x_landmark, y_landmark, and z_landmark.
            - x_landmark: One x-coordinate.
            - y_landmark: One y-coordinate.
            - z_landmark: One z-coordinate.

        This takes in the depth values output by the camera for one frame and uses the field of view (77 degrees) to calculate x, y, z 
        coordinates at the landmark pixel locations.

        Example usage:
            frame_land_x, frame_land_y, frame_land_z = convert_depth_to_xyz(frame_depth, landmark_x, landmark_y)
        """

        # Solve system of equations defined in fovFunction to find horizontal and vertical FOV angles
        angleGuess = np.array([1,1])
        ang = fsolve(self.fovFunction, angleGuess)

        # assign to variables
        image_w_angle = ang[0]
        image_h_angle = ang[1]

        # Find theta, phi, and the depth at that landmark pixel
        # Convert depth to cm
        land_depth = depths[landmark_pixel_y][landmark_pixel_x]
        land_depth_cm = self.convert_unit_to_cm(land_depth)
        theta = (-image_w_angle/2) + ((image_w_angle/image_w)*landmark_pixel_x)
        phi = (image_h_angle/2) - ((image_h_angle/image_h)*landmark_pixel_y)

        x_landmark = land_depth_cm*np.sin(theta*(np.pi/180))*np.sin(phi*(np.pi/180))
        y_landmark = land_depth_cm*np.cos(theta*(np.pi/180))*np.sin(phi*(np.pi/180))
        z_landmark = land_depth_cm*np.cos(phi*(np.pi/180))

        return x_landmark,y_landmark,z_landmark

    def convert_unit_to_cm(self,depth_value):
        """
        A polyfit was done to convert the arbitrary units output by the thanos camera into cm. This was done in MATLAB,
        and now the equation calculated will be used to convert to real units.

        Args:
            depth_value: An integer depth value in arbitrary units
        
        Returns:
            depth_cm: An integer depth value in cm
        """

        depth_cm = (depth_value - 145.095238095238)/79.5742857142857

        return depth_cm

    def _process_pose_landmarks(
        self,
        landmarks_pixels: np.ndarray,
        frame_idx: int,
        frame_depth: np.ndarray,
        frame_intensity: np.ndarray,
        frame_grayscale_rgb: np.ndarray,
        filename: str,
        participant: str = 'none'
    ) -> None:
        """
        Process the pose landmarks for a single frame.

        Args:
            landmarks_pixels: An (n, 2) array of landmark pixel coordinates.
            frame_idx: The index of the current frame.
            frame_depth: An (n, d) array of depths.
            frame_intensity: An (n, d) array of confidence values.
            frame_grayscale_rgb: An (n, d, 3) array of RGB values.
            filename: The name of the file being processed.
            participant: Left or right if two people are in frame.
        """


        # For each landmark, get the x, y, and z values from the depth array and store them in a numpy array
        # Then, write the numpy array to a new row in the output csv file
        xyz_values = np.zeros((len(landmarks_pixels), 5))

        for landmark_idx in range(len(landmarks_pixels)):
            landmark_pixel_coord_x, landmark_pixel_coord_y = landmarks_pixels[landmark_idx]

            # Verify that the pixel coordinates are valid
            if self.lanscape == True:
                if not self._is_valid_pixel_coordinate((landmark_pixel_coord_x, landmark_pixel_coord_y), self.image_height, self.image_width):
                    # The pixel coordinates are invalid
                    
                    # Set the x, y, and z values to -int16.max (-32767)
                    xyz_values[landmark_idx][0] = -np.iinfo(np.int16).max
                    xyz_values[landmark_idx][1] = -np.iinfo(np.int16).max
                    xyz_values[landmark_idx][2] = -np.iinfo(np.int16).max

                    continue
            else:
                if not self._is_valid_pixel_coordinate((landmark_pixel_coord_x, landmark_pixel_coord_y), self.image_width, self.image_height):
                    # The pixel coordinates are invalid
                    
                    # Set the x, y, and z values to -int16.max (-32767)
                    xyz_values[landmark_idx][0] = -np.iinfo(np.int16).max
                    xyz_values[landmark_idx][1] = -np.iinfo(np.int16).max
                    xyz_values[landmark_idx][2] = -np.iinfo(np.int16).max

                    continue
            
            # The pixel coordinates are valid

            # Convert depth [cm] to x,y,z for landmark pixels
            if self.landscape == True:
                x_value,y_value,z_value = self.convert_depth_to_xyz(frame_depth,landmark_pixel_coord_x,landmark_pixel_coord_y, self.image_width, self.image_height, self.image_fov)
            else:
                x_value,y_value,z_value = self.convert_depth_to_xyz(frame_depth,landmark_pixel_coord_x,landmark_pixel_coord_y, self.image_height, self.image_width, self.image_fov)

            # Set the x, y, and z values to the values from convert depth to x,y,z
            xyz_values[landmark_idx][0] = x_value
            xyz_values[landmark_idx][1] = y_value
            xyz_values[landmark_idx][2] = z_value
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
        Load and process .mat file

        Args:
            file_num (int): number of file being processed
            num_files_to_process (int): total number of files to process
            filename (str): name of file to process

        Returns:
            None
        """

        print(f"Processing file {file_num}/{num_files_to_process}: {filename}...")

        # Load the file
        filepath = os.path.join(self.input_dir, filename + '.mat')
        depth_all, intensity_all = self.load_mat_file(filepath)

        # Get number of frames in this video clip
        num_frames = np.shape(intensity_all)[2]

        # Used to calculate FPS
        previous_time = 0
        start_time = time.time()
        if self.landscape == True:
            writer = cv2.VideoWriter(self.output_filename + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (self.image_height, self.image_width))
        else:
            writer = cv2.VideoWriter(self.output_filename + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (self.image_width, self.image_height))
        
        # Check if two people need to be tracked
        if self.two_people == True:
            # Loop through all frames
            for frame_idx in range(num_frames):
                # rotate image for landscape mode
                if self.landscape == True:
                    depth_rotate = np.flip(depth_all[:, :, frame_idx].transpose(),0)
                    intensity_rotate = np.flip(intensity_all[:,:, frame_idx].transpose(),0)
            
                    # Split to left and right participants (relative to viewer)
                    frame_depth_left = depth_rotate[:, 0:int((self.image_height/2))-1]
                    frame_depth_right = depth_rotate[:, int(self.image_height/2):(self.image_height-1)]
                    frame_intensity_left = intensity_rotate[:, 0:int((self.image_height/2))-1]
                    frame_intensity_right = intensity_rotate[:, int(self.image_height/2):(self.image_height-1)]
                else:
                    # Split to left and right participants (relative to viewer)
                    frame_depth_left = depth_all[:, 0:int((self.image_width/2))-1, frame_idx]
                    frame_depth_right = depth_all[:, int(self.image_width/2):(self.image_width-1), frame_idx]
                    frame_intensity_left = intensity_all[:, 0:int((self.image_width/2))-1, frame_idx]
                    frame_intensity_right = intensity_all[:, int(self.image_width/2):(self.image_width-1), frame_idx]

                # Track face and extract intensity and depth for all ROIs in each side of this frame

                # Convert each half of the frame's confidence values to a grayscale image (n,d)
                frame_grayscale_left = self._convert_camera_intensity_to_grayscale(frame_intensity_left)
                frame_grayscale_right = self._convert_camera_intensity_to_grayscale(frame_intensity_right)

                # # To improve performance, optionally mark the image as not writeable to
                # # pass by reference.
                # frame_grayscale.flags.writeable = False

                # Convert each half of the grayscale image to "RGB" (n,d,3)
                frame_grayscale_rgb_left = cv2.cvtColor(frame_grayscale_left, cv2.COLOR_GRAY2RGB)
                frame_grayscale_rgb_right = cv2.cvtColor(frame_grayscale_right, cv2.COLOR_GRAY2RGB)

                # Get pixel locations of all pose landmarks for both skeletons
                # face_detected, landmarks_pixels = face_mesh_detector.find_face_mesh(image=frame_grayscale_rgb, draw=self.visualize_FaceMesh)
                pose_detected_left, contains_invalid_landmarks_left, landmarks_pixels_left = pose_detector_1.get_landmarks(image=frame_grayscale_rgb_left, draw=self.visualize_Pose)
                pose_detected_right, contains_invalid_landmarks_right, landmarks_pixels_right = pose_detector_2.get_landmarks(image=frame_grayscale_rgb_right, draw=self.visualize_Pose)

                # if pose_detected:
                #     # multithreading_tasks.append(self.thread_pool.submit(self._process_face_landmarks, landmarks_pixels, frame_idx, frame_x, frame_y, frame_z, frame_confidence, intensity_signal_current_file, depth_signal_current_file, ear_signal_current_file, frame_grayscale_rgb))
                #     self._process_pose_landmarks(landmarks_pixels, frame_idx, frame_x, frame_y, frame_z, frame_confidence, frame_grayscale_rgb, filename)
                
                self._process_pose_landmarks(landmarks_pixels_left, frame_idx, frame_depth_left, frame_intensity_left, frame_grayscale_rgb_left, filename+'_left_participant', participant='Left')
                self._process_pose_landmarks(landmarks_pixels_right, frame_idx, frame_depth_right, frame_intensity_right, frame_grayscale_rgb_right, filename+'_right_participant', participant='Right')

                if self.visualize_Pose == True:
                    # Combine frame_grayscale_rgb_left and _right
                    frame_grayscale_rgb = np.append(frame_grayscale_rgb_left, frame_grayscale_rgb_right, 1)
                    
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
                    writer.write(frame_grayscale_rgb)
        else:          
            # Loop through all frames
            for frame_idx in range(num_frames):
                # rotate image for landscape mode
                if self.landscape == True:
                    frame_depth = np.flip(depth_all[:, :, frame_idx].transpose(),0)
                    frame_intensity = np.flip(intensity_all[:,:, frame_idx].transpose(),0)
                else:
                    frame_depth = depth_all[:, :, frame_idx]
                    frame_intensity = intensity_all[:, :, frame_idx]

                # Track face and extract intensity and depth for all ROIs in this frame

                # Convert the frame's confidence values to a grayscale image (n,d)
                frame_grayscale = self._convert_camera_intensity_to_grayscale(frame_intensity)

                # # To improve performance, optionally mark the image as not writeable to
                # # pass by reference.
                # frame_grayscale.flags.writeable = False

                # Convert grayscale image to "RGB" (n,d,3)
                frame_grayscale_rgb = cv2.cvtColor(frame_grayscale, cv2.COLOR_GRAY2RGB)

                # Get pixel locations of all pose landmarks
                # face_detected, landmarks_pixels = face_mesh_detector.find_face_mesh(image=frame_grayscale_rgb, draw=self.visualize_FaceMesh)
                pose_detected, contains_invalid_landmarks, landmarks_pixels = pose_detector_1.get_landmarks(image=frame_grayscale_rgb, draw=self.visualize_Pose)

                # if pose_detected:
                #     # multithreading_tasks.append(self.thread_pool.submit(self._process_face_landmarks, landmarks_pixels, frame_idx, frame_x, frame_y, frame_z, frame_confidence, intensity_signal_current_file, depth_signal_current_file, ear_signal_current_file, frame_grayscale_rgb))
                #     self._process_pose_landmarks(landmarks_pixels, frame_idx, frame_x, frame_y, frame_z, frame_confidence, frame_grayscale_rgb, filename)
                
                self._process_pose_landmarks(landmarks_pixels, frame_idx, frame_depth, frame_intensity, frame_grayscale_rgb, filename)

                if self.visualize_Pose == True:
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
                    writer.write(frame_grayscale_rgb)
            
        # Calculate and print average FPS
        writer.release()
        end_time = time.time()
        average_fps = num_frames / (end_time - start_time)
        print(f"Average FPS: {average_fps}")

        return


    def run(self):
        """
        Run pose estimation pipeline on all .mat files.
        """

        # Get list of .mat files in input_dir
        mat_files = self._get_mat_files()

        # Load and process every input video file
        file_num = 0
        num_files_to_process = len(mat_files)

        # Define MediaPipe detectors
        pose_detector_1 = PoseLandmarker(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.5)
        pose_detector_2 = PoseLandmarker(static_image_mode=False, min_detection_confidence=0.9, min_tracking_confidence=0.5)

        for filename in mat_files:
            file_num += 1
            self.output_filename = filename

            if self.two_people == True:
                # Create two output csv files (overwrite if it already exists)
                #self.output_csv_filepath_left = os.path.join(self.input_dir, '249800_'+ self.output_filename + '_F_22.csv')
                self.output_csv_filepath_left = os.path.join(self.input_dir + '/csv', self.left_participant_id + self.output_filename + self.left_part_demographics + '.csv')
                self.output_csv_file_left = open(self.output_csv_filepath_left, 'w')

                #self.output_csv_filepath_right = os.path.join(self.input_dir, '793320_' + self.output_filename + '_F_19.csv')
                self.output_csv_filepath_right = os.path.join(self.input_dir + '/csv', self.right_participant_id + self.output_filename + self.right_part_demographics + '.csv')
                self.output_csv_file_right = open(self.output_csv_filepath_right, 'w')

                # Write header row
                self.output_csv_file_left.write('filename,frame_num,'
                                            'Hip_Center_X,Hip_Center_Y,Hip_Center_Z,Hip_Center_RawX,Hip_Center_RawY,'
                                            'Spine_X,Spine_Y,Spine_Z,Spine_RawX,Spine_RawY,'
                                            'Shoulder_Center_X,Shoulder_Center_Y,Shoulder_Center_Z,Shoulder_Center_RawX,Shoulder_Center_RawY,'
                                            'Head_X,Head_Y,Head_Z,Head_RawX,Head_RawY,'
                                            'Shoulder_Right_X,Shoulder_Right_Y,Shoulder_Right_Z,Shoulder_Right_RawX,Shoulder_Right_RawY,'
                                            'Elbow_Right_X,Elbow_Right_Y,Elbow_Right_Z,Elbow_Right_RawX,Elbow_Right_RawY,'
                                            'Wrist_Right_X,Wrist_Right_Y,Wrist_Right_Z,Wrist_Right_RawX,Wrist_Right_RawY,'
                                            'Hand_Right_X,Hand_Right_Y,Hand_Right_Z,Hand_Right_RawX,Hand_Right_RawY,'
                                            'Shoulder_Left_X,Shoulder_Left_Y,Shoulder_Left_Z,Shoulder_Left_RawX,Shoulder_Left_RawY,'
                                            'Elbow_Left_X,Elbow_Left_Y,Elbow_Left_Z,Elbow_Left_RawX,Elbow_Left_RawY,'
                                            'Wrist_Left_X,Wrist_Left_Y,Wrist_Left_Z,Wrist_Left_RawX,Wrist_Left_RawY,'
                                            'Hand_Left_X,Hand_Left_Y,Hand_Left_Z,Hand_Left_RawX,Hand_Left_RawY,'
                                            'Hip_Right_X,Hip_Right_Y,Hip_Right_Z,Hip_Right_RawX,Hip_Right_RawY,'
                                            'Knee_Right_X,Knee_Right_Y,Knee_Right_Z,Knee_Right_RawX,Knee_Right_RawY,'
                                            'Ankle_Right_X,Ankle_Right_Y,Ankle_Right_Z,Ankle_Right_RawX,Ankle_Right_RawY,'
                                            'Foot_Right_X,Foot_Right_Y,Foot_Right_Z,Foot_Right_RawX,Foot_Right_RawY,'
                                            'Hip_Left_X,Hip_Left_Y,Hip_Left_Z,Hip_Left_RawX,Hip_Left_RawY,'
                                            'Knee_Left_X,Knee_Left_Y,Knee_Left_Z,Knee_Left_RawX,Knee_Left_RawY,'
                                            'Ankle_Left_X,Ankle_Left_Y,Ankle_Left_Z,Ankle_Left_RawX,Ankle_Left_RawY,'
                                            'Foot_Left_X,Foot_Left_Y,Foot_Left_Z,Foot_Left_RawX,Foot_Left_RawY\n')
        
                self.output_csv_file_right.write('filename,frame_num,'
                                            'Hip_Center_X,Hip_Center_Y,Hip_Center_Z,Hip_Center_RawX,Hip_Center_RawY,'
                                            'Spine_X,Spine_Y,Spine_Z,Spine_RawX,Spine_RawY,'
                                            'Shoulder_Center_X,Shoulder_Center_Y,Shoulder_Center_Z,Shoulder_Center_RawX,Shoulder_Center_RawY,'
                                            'Head_X,Head_Y,Head_Z,Head_RawX,Head_RawY,'
                                            'Shoulder_Right_X,Shoulder_Right_Y,Shoulder_Right_Z,Shoulder_Right_RawX,Shoulder_Right_RawY,'
                                            'Elbow_Right_X,Elbow_Right_Y,Elbow_Right_Z,Elbow_Right_RawX,Elbow_Right_RawY,'
                                            'Wrist_Right_X,Wrist_Right_Y,Wrist_Right_Z,Wrist_Right_RawX,Wrist_Right_RawY,'
                                            'Hand_Right_X,Hand_Right_Y,Hand_Right_Z,Hand_Right_RawX,Hand_Right_RawY,'
                                            'Shoulder_Left_X,Shoulder_Left_Y,Shoulder_Left_Z,Shoulder_Left_RawX,Shoulder_Left_RawY,'
                                            'Elbow_Left_X,Elbow_Left_Y,Elbow_Left_Z,Elbow_Left_RawX,Elbow_Left_RawY,'
                                            'Wrist_Left_X,Wrist_Left_Y,Wrist_Left_Z,Wrist_Left_RawX,Wrist_Left_RawY,'
                                            'Hand_Left_X,Hand_Left_Y,Hand_Left_Z,Hand_Left_RawX,Hand_Left_RawY,'
                                            'Hip_Right_X,Hip_Right_Y,Hip_Right_Z,Hip_Right_RawX,Hip_Right_RawY,'
                                            'Knee_Right_X,Knee_Right_Y,Knee_Right_Z,Knee_Right_RawX,Knee_Right_RawY,'
                                            'Ankle_Right_X,Ankle_Right_Y,Ankle_Right_Z,Ankle_Right_RawX,Ankle_Right_RawY,'
                                            'Foot_Right_X,Foot_Right_Y,Foot_Right_Z,Foot_Right_RawX,Foot_Right_RawY,'
                                            'Hip_Left_X,Hip_Left_Y,Hip_Left_Z,Hip_Left_RawX,Hip_Left_RawY,'
                                            'Knee_Left_X,Knee_Left_Y,Knee_Left_Z,Knee_Left_RawX,Knee_Left_RawY,'
                                            'Ankle_Left_X,Ankle_Left_Y,Ankle_Left_Z,Ankle_Left_RawX,Ankle_Left_RawY,'
                                            'Foot_Left_X,Foot_Left_Y,Foot_Left_Z,Foot_Left_RawX,Foot_Left_RawY\n')
            
            else:
        
                # Create output csv file (overwrite if it already exists)
                self.output_csv_filepath = os.path.join(self.input_dir + '/csv', self.output_filename + '.csv')
                self.output_csv_file = open(self.output_csv_filepath, 'w')

                # Write header row
                self.output_csv_file.write('filename,frame_num,'
                                        'Hip_Center_X,Hip_Center_Y,Hip_Center_Z,Hip_Center_RawX,Hip_Center_RawY,'
                                        'Spine_X,Spine_Y,Spine_Z,Spine_RawX,Spine_RawY,'
                                        'Shoulder_Center_X,Shoulder_Center_Y,Shoulder_Center_Z,Shoulder_Center_RawX,Shoulder_Center_RawY,'
                                        'Head_X,Head_Y,Head_Z,Head_RawX,Head_RawY,'
                                        'Shoulder_Right_X,Shoulder_Right_Y,Shoulder_Right_Z,Shoulder_Right_RawX,Shoulder_Right_RawY,'
                                        'Elbow_Right_X,Elbow_Right_Y,Elbow_Right_Z,Elbow_Right_RawX,Elbow_Right_RawY,'
                                        'Wrist_Right_X,Wrist_Right_Y,Wrist_Right_Z,Wrist_Right_RawX,Wrist_Right_RawY,'
                                        'Hand_Right_X,Hand_Right_Y,Hand_Right_Z,Hand_Right_RawX,Hand_Right_RawY,'
                                        'Shoulder_Left_X,Shoulder_Left_Y,Shoulder_Left_Z,Shoulder_Left_RawX,Shoulder_Left_RawY,'
                                        'Elbow_Left_X,Elbow_Left_Y,Elbow_Left_Z,Elbow_Left_RawX,Elbow_Left_RawY,'
                                        'Wrist_Left_X,Wrist_Left_Y,Wrist_Left_Z,Wrist_Left_RawX,Wrist_Left_RawY,'
                                        'Hand_Left_X,Hand_Left_Y,Hand_Left_Z,Hand_Left_RawX,Hand_Left_RawY,'
                                        'Hip_Right_X,Hip_Right_Y,Hip_Right_Z,Hip_Right_RawX,Hip_Right_RawY,'
                                        'Knee_Right_X,Knee_Right_Y,Knee_Right_Z,Knee_Right_RawX,Knee_Right_RawY,'
                                        'Ankle_Right_X,Ankle_Right_Y,Ankle_Right_Z,Ankle_Right_RawX,Ankle_Right_RawY,'
                                        'Foot_Right_X,Foot_Right_Y,Foot_Right_Z,Foot_Right_RawX,Foot_Right_RawY,'
                                        'Hip_Left_X,Hip_Left_Y,Hip_Left_Z,Hip_Left_RawX,Hip_Left_RawY,'
                                        'Knee_Left_X,Knee_Left_Y,Knee_Left_Z,Knee_Left_RawX,Knee_Left_RawY,'
                                        'Ankle_Left_X,Ankle_Left_Y,Ankle_Left_Z,Ankle_Left_RawX,Ankle_Left_RawY,'
                                        'Foot_Left_X,Foot_Left_Y,Foot_Left_Z,Foot_Left_RawX,Foot_Left_RawY\n')

            # Set flag to indicate whether or not the class has been cleaned up
            # (either manually or automatically if the destructor was called by the garbage
            # collector)
            self.cleaned_up = False
            # Process the file
            self._process_file(file_num, num_files_to_process, filename, pose_detector_1, pose_detector_2)

        return
    

def main():
    # Get absolute path of directory where .mat files are located
    mats_dir = os.path.join(os.getcwd(), 'Data', 'mat')
    print(mats_dir)

    # Run pose estimation pipeline on all .mat files in mats_dir and save output to csvs_dir
    myMatToCsv = MatToCsv(input_dir=mats_dir, visualize_Pose=True, two_people=True, landscape=False)
    myMatToCsv.run()

    return


if __name__ == "__main__":
    main()
        