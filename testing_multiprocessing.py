import cv2
import multiprocessing
from tqdm import tqdm
import numpy as np
import json
import os

def process_frame(frame, max_depth, min_depth, num_rows, num_cols, fx, fy, cx, cy):
    
    rgb = cv2.cvtColor(frame[:, num_cols//2:, :], cv2.COLOR_BGR2RGB)
    
    depth_temp = cv2.cvtColor(frame[:, :num_cols//2, :], cv2.COLOR_BGR2HSV)[:,:,0]
    depth_temp = np.asarray(depth_temp, dtype=np.float32)/255
    depth_temp = min_depth + (depth_temp*(max_depth - min_depth))
    
    c, r = np.meshgrid(np.arange(num_cols//2), np.arange(num_rows), sparse=True)
    x = depth_temp*(c - cx)/fx
    y = depth_temp*(r - cy)/fy
    z = np.sqrt(depth_temp**2 - x**2 - y**2) ## need to check this
    
    xyz = np.dstack((x, y, z))
    
    return (rgb, xyz)

def worker(input_queue, output_queue, max_depth, min_depth, num_rows, num_cols, fx, fy, cx, cy):
    while True:
        frame = input_queue.get()  # Get a frame from input queue
        if frame is None:  # If None is received, break the loop
            break
        processed_frame = process_frame(frame, max_depth, min_depth, num_rows, num_cols, fx, fy, cx, cy)  # Process the frame
        output_queue.put(processed_frame)  # Put the processed frame into output queue

if __name__ == '__main__':
    
    pathname = 'Data/'
    
    mp4_files = []
    for file_name in sorted(os.listdir(pathname)):
        if file_name.endswith(".mp4"):
            # Remove the ".mat" suffix
            file_name = file_name
            mp4_files.append(file_name)
            
    for file_num, mp4_name in enumerate(mp4_files):
        print(f'Processing file {file_num+1}/{len(mp4_files)}: {mp4_name}')

        file_name = pathname + mp4_name
        
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
        
        input_queue = multiprocessing.Queue()  # Queue to hold incoming frames
        output_queue = multiprocessing.Queue()  # Queue to hold processed frames

        # Open the video stream
        video_capture = cv2.VideoCapture(file_name)  # Replace 'your_video.mp4' with your video file
        
        # Get the total number of frames in the video
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        num_rows = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_cols = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Start worker processes
        num_workers = multiprocessing.cpu_count()  # Get the number of CPU cores
        processes = []
        for _ in range(num_workers):
            p = multiprocessing.Process(target=worker, args=(input_queue, output_queue, 
                                                             max_depth, min_depth, 
                                                             num_rows, num_cols, 
                                                             fx, fy, cx, cy))
            p.start()
            processes.append(p)
            
        rgb_values_list = []
        xyz_values_list = []
        
        # Use tqdm for progress bar
        i = 0
        with tqdm(total=total_frames) as pbar:
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break

                input_queue.put(frame)  # Put the frame into the input queue

                rgb, xyz = output_queue.get()  # Get the processed frame from output queue

                rgb_values_list.append(rgb)
                xyz_values_list.append(xyz)
                # Display the processed frame
                # cv2.imshow('Processed Frame', processed_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                pbar.update(1)  # Update progress bar

        # Send termination signal to worker processes
        for _ in range(num_workers):
            input_queue.put(None)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Release resources
        video_capture.release()
        cv2.destroyAllWindows()

        rgb_values = np.stack(rgb_values_list, axis=-1, dtype=np.uint8)
        xyz_values = np.stack(xyz_values_list, axis=-1, dtype=np.float32)
        
        np.savez(f'Data/mat/{mp4_name[:-4]}.npz', xyz_values, rgb_values)
        print()