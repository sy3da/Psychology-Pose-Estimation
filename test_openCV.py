import numpy as np
import cv2 as cv
import json
from tqdm import tqdm
from scipy.io import savemat
from glob import glob as get_files

pathname = 'Data/'
mp4_files = get_files(pathname + '*.mp4')
mp4_files.sort()

for file_num, file_name in enumerate(mp4_files):
    print(f'Processing file {file_num+1}/{len(mp4_files)}: {file_name}')
    
    with open(file_name, "rb") as file:
        file_content = file.read()
        meta = file_content[file_content.rindex(b'{"intrinsic'):]
        meta = meta[:-1]
        meta = meta.decode('UTF-8')
        metadata = json.loads(meta)
        intrinsicMatrix = metadata['intrinsicMatrix']
        record3d_fx = intrinsicMatrix[0]
        record3d_fy = intrinsicMatrix[4]
        record3d_cx = intrinsicMatrix[6]
        record3d_cy = intrinsicMatrix[7]
        
        depthInfo = metadata['rangeOfEncodedDepth']
        min_depth = depthInfo[0]
        max_depth = depthInfo[1]

    cap = cv.VideoCapture(file_name)
    num_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    num_rows = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    num_cols = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

    rgb_values = np.zeros((num_rows, num_cols//2, 3, num_frames), dtype=np.uint8)
    xyz_values = np.zeros((num_rows, num_cols//2, 3, num_frames), dtype=np.float32)

    i = 0
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
        rgb_temp = frame[:, num_cols//2:, :]
        depth_temp = frame[:, :num_cols//2, :]

        rgb_temp = cv.cvtColor(rgb_temp, cv.COLOR_BGR2RGB)
        rgb_values[:, :, :, i] = np.asarray(rgb_temp, dtype=np.uint8)
        
        depth_array = cv.cvtColor(depth_temp, cv.COLOR_BGR2HSV)[:,:,0]
        depth_array = depth_array.astype(np.float32)
        depth_array = depth_array / 255.0
        depth_array = min_depth + (depth_array*(max_depth - min_depth))

        xyz_values[:,:,0,i] = depth_array
        xyz_values[:,:,1,i] = depth_array
        xyz_values[:,:,2,i] = depth_array
        
        i += 1
        
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

    savemat(f'Data/mat/{file_name}.mat', {'xyz_values': xyz_values, 'rgb_values': rgb_values})