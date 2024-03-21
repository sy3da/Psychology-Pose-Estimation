import numpy as np
import cv2 as cv
import json
from tqdm import tqdm
from scipy.io import savemat
import os
import matplotlib.pyplot as plt
from p_tqdm import p_map
from multiprocessing import cpu_count
import time

pathname = 'Data/'

def process_frame(args):
    frame, max_depth, min_depth, num_rows, num_cols = args
    
    rgb = cv.cvtColor(frame[:, num_cols//2:, :], cv.COLOR_BGR2RGB)
    
    depth_temp = cv.cvtColor(frame[:, :num_cols//2, :], cv.COLOR_BGR2HSV)[:,:,0]
    depth_temp = np.asarray(depth_temp, dtype=np.double)/255
    depth_temp = min_depth + (depth_temp*(max_depth - min_depth))
    
    x = depth_temp
    y = depth_temp
    z = depth_temp
    
    xyz = np.dstack((x, y, z))
    
    return (rgb, xyz)

mp4_files = []
for file_name in sorted(os.listdir(pathname)):
    if file_name.endswith(".mp4"):
        # Remove the ".mat" suffix
        file_name = file_name
        mp4_files.append(file_name)

for file_num, file_name in enumerate(mp4_files):
    print(f'Processing file {file_num+1}/{len(mp4_files)}: {file_name}')
    
    with open(pathname + file_name, "rb") as file:
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

    cap = cv.VideoCapture(pathname + file_name)
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    num_rows = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    num_cols = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    
    raw_data = np.zeros((num_rows, num_cols, 3, num_frames), dtype=np.uint8)
    
    i=0
    p_bar = tqdm(range(num_frames))
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            break
        
        p_bar.update(1)
        p_bar.refresh()
        
        # Our operations on the frame come here
        raw_data[:, :, :, i] = np.asarray(frame, dtype=np.uint8)
        i += 1
        
    cap.release()
    cv.destroyAllWindows()

    time.sleep(1)
    
    num_processes = 2
    args = [(raw_data[:, :, :, frame_index], max_depth, min_depth, num_rows, num_cols) for frame_index in range(num_frames)]
    results = p_map(process_frame, args)
    rgb_values_old, xyz_values_old = np.asarray(results)

    rgb_values = np.zeros((num_rows, num_cols//2, 3, num_frames), dtype=np.uint8)
    xyz_values = np.zeros((num_rows, num_cols//2, 3, num_frames), dtype=np.double)

    for i in range(num_frames):
            rgb_values[:, :, :, i] = rgb_values_old[i, :, :, :]
            xyz_values[:, :, :, i] = xyz_values_old[i, :, :, :]

    plt.imshow(xyz_values[:,:,0,0]/np.max(xyz_values[:,:,0,0]))
    plt.show()

    # savemat(f'Data/mat/{file_name[:-4]}.mat', {'xyz_values': xyz_values, 'rgb_values': rgb_values})
    