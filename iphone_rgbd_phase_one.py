import cv2
import multiprocessing
from tqdm import tqdm
import numpy as np
import json
import os

"""
This file loads in depth and rgb data from a RGBD.mp4 file, converts depth to xyz data,
and outputs rgb and xyz data into a .npz file.

Runs on all .mp4 files in the "Data" folder individually.
"""

def process_frame(frame, max_depth, min_depth, num_rows, num_cols, fx, fy, cx, cy):
    """
    Takes in one frame and creates array for rgb data along with converting the colormap of depths
    into corresponding depth values in mm. Then converts depth to x,y,z data using the camera
    instrinsics.

    Args:
        frame: the xyz and rgb data for the frame being processed.
        max_depth: the maximum depth recorded by the LiDAR sensor
        min_depth: the minimum depth recorded by the LiDAR sensor
        num_rows: the number of rows in the array for the frame (should be the same as the resolution height: 960)
        num_cols: the number of columns in the array for the frame (should be twice the resolution width: 2*720)
        fx: x-axis focal length of the camera in pixels (part of the camera intrinsics)
        fy: y-axis focal length of the camera in pixels (part of the camera intrinsics)
        cx: x-axis optical center of the camera in pixels (part of the camera intrinsics)
        cy: y-axis optical center of the camera in pixels (part of the camera intrinsics)

    Returns:
        A Tuple containing:
        - xyz: An (rows, cols, 3) array of spatial coordinate values
        - rgb: An (rows, cols, 3) array of rgb intensity values
    """
    # select right side of the frame as rgb data
    rgb = cv2.cvtColor(frame[:, num_cols//2:, :], cv2.COLOR_BGR2RGB)
    
    # select left side of the frame as depth data and convert from hue to mm values
    depth_temp = cv2.cvtColor(frame[:, :num_cols//2, :], cv2.COLOR_BGR2HSV)[:,:,0]
    depth_temp = np.asarray(depth_temp, dtype=np.float32)/255
    depth_temp = min_depth + (depth_temp*(max_depth - min_depth))
    
    # use the camera intrinsics (fx,fy,cx,cy) to convert depth to xyz
    c, r = np.meshgrid(np.arange(num_cols//2), np.arange(num_rows), sparse=True)
    x = depth_temp*(c - cx)/fx
    y = depth_temp*(r - cy)/fy
    z = np.sqrt(depth_temp**2 - x**2 - y**2) ## need to check this
    
    xyz = np.dstack((x, y, z))
    
    return (rgb, xyz)

def worker(input_queue, output_queue, max_depth, min_depth, num_rows, num_cols, fx, fy, cx, cy):
    
    """
    !!! George can you fill out the description for this one? !!!

    Args:
        input_queue:
        output_queue: 
        max_depth: the maximum depth recorded by the LiDAR sensor
        min_depth: the minimum depth recorded by the LiDAR sensor
        num_rows: the number of rows in the array for the frame (should be the same as the resolution height: 960)
        num_cols: the number of columns in the array for the frame (should be twice the resolution width: 2*720)
        fx: x-axis focal length of the camera in pixels (part of the camera intrinsics)
        fy: y-axis focal length of the camera in pixels (part of the camera intrinsics)
        cx: x-axis optical center of the camera in pixels (part of the camera intrinsics)
        cy: y-axis optical center of the camera in pixels (part of the camera intrinsics)

    Returns:
        None
    """

    while True:
        frame = input_queue.get()  # Get a frame from input queue
        if frame is None:  # If None is received, break the loop
            break
        processed_frame = process_frame(frame, max_depth, min_depth, num_rows, num_cols, fx, fy, cx, cy)  # Process the frame
        output_queue.put(processed_frame)  # Put the processed frame into output queue

if __name__ == '__main__':
    
    # define the path to the .mp4 files
    pathname = 'Data/'
    
    # look within the path for mp4 files and create array with .mp4 file names
    mp4_files = []
    for file_name in sorted(os.listdir(pathname)):
        if file_name.endswith(".mp4"):
            # Remove the ".mp4" suffix
            file_name = file_name
            mp4_files.append(file_name)
    
    # loop through and run processing on each of the .mp4 files
    for file_num, mp4_name in enumerate(mp4_files):
        print(f'Processing file {file_num+1}/{len(mp4_files)}: {mp4_name}')

        file_name = pathname + mp4_name
        
        # define camera intrinsics (fx,fy,cx,cy), max_depth, and min_depth from meta data in .mp4 file
        with open(file_name, "rb") as file:
            file_content = file.read()
            meta = file_content[file_content.rindex(b'{"intrinsic'):]
            meta = meta[:-1]
            meta = meta.decode('UTF-8')
            metadata = json.loads(meta)
            intrinsicMatrix = metadata['intrinsicMatrix']
            fx = intrinsicMatrix[0]
            fy = intrinsicMatrix[4]
            cx = intrinsicMatrix[6]
            cy = intrinsicMatrix[7]
            
            depthInfo = metadata['rangeOfEncodedDepth']
            min_depth = depthInfo[0]
            max_depth = depthInfo[1]
        
        # establish frame queues (multiprocessing)
        input_queue = multiprocessing.Queue()  # Queue to hold incoming frames
        output_queue = multiprocessing.Queue()  # Queue to hold processed frames

        # Open the video stream
        video_capture = cv2.VideoCapture(file_name)  # Replace 'your_video.mp4' with your video file
        
        # Get the total number of frames in the video
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        num_rows = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_cols = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Start worker processes (multiprocessing)
        num_workers = multiprocessing.cpu_count()  # Get the number of CPU cores
        processes = []
        for _ in range(num_workers):
            p = multiprocessing.Process(target=worker, args=(input_queue, output_queue, 
                                                             max_depth, min_depth, 
                                                             num_rows, num_cols, 
                                                             fx, fy, cx, cy))
            p.start()
            processes.append(p)

        # create empty arrays to hold rgb and xyz data for each frame   
        rgb_values_list = []
        xyz_values_list = []
        
        # Use tqdm for progress bar (shows what frame you are on out of all frames in the clip)
        i = 0
        with tqdm(total=total_frames) as pbar:
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break

                input_queue.put(frame)  # Put the frame into the input queue (multiprocessing)
                rgb, xyz = output_queue.get()  # Get the processed frame from output queue (multiprocessing)

                # add rgb and xyz data for that frame to lists for respective data
                rgb_values_list.append(rgb)
                xyz_values_list.append(xyz)
                # Display the processed frame
                # cv2.imshow('Processed Frame', processed_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                pbar.update(1)  # Update progress bar

        # Send termination signal to worker processes (multiprocessing)
        for _ in range(num_workers):
            input_queue.put(None)

        # Wait for all processes to finish (multiprocessing)
        for p in processes:
            p.join()

        # Release resources
        video_capture.release()
        cv2.destroyAllWindows()

        # stack arrays so that they are (rows, cols, 3, num_frames)
        rgb_values = np.stack(rgb_values_list, axis=-1, dtype=np.uint8)
        xyz_values = np.stack(xyz_values_list, axis=-1, dtype=np.float32)
        
        # save data for this clip to .npz file in the "npz" folder within the "Data" folder
        np.savez(f'Data/npz/{mp4_name[:-4]}.npz', xyz_values=xyz_values, rgb_values=rgb_values)
        print()