"""
This code converts the outputs from the ST MOB-THANOSV3 into depth and intensity arrays, respectively.
The frame rate (fps) should be set to the correct value in this code and processHR.py
"""

import regex as re
import numpy as np
from PIL import Image
from p_tqdm import p_map
from tqdm import tqdm
from glob import glob as get_files
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from scipy.io import savemat
import os
from plyfile import PlyData, PlyElement

def read_ply(args):
    filename, img_width, img_height = args
    
    plydata = PlyData.read(filename)
    xyz_rgb = np.zeros((img_height, img_width, 6))
 
    if len(plydata['vertex']) == 480*640:
        row_idx = 0
        col_idx = 0
        for i in range(480*640):
            xyz_rgb[row_idx, col_idx, 0] = plydata['vertex'][i][0]
            xyz_rgb[row_idx, col_idx, 1] = plydata['vertex'][i][1]
            xyz_rgb[row_idx, col_idx, 2] = plydata['vertex'][i][2]
            xyz_rgb[row_idx, col_idx, 3] = plydata['vertex'][i][3]
            xyz_rgb[row_idx, col_idx, 4] = plydata['vertex'][i][4]
            xyz_rgb[row_idx, col_idx, 5] = plydata['vertex'][i][5]
            
            col_idx += 1
            if col_idx == 480:
                col_idx = 0
                row_idx += 1
                
    return xyz_rgb


def orientation_check(rgb_array, z_array):
    # Checks if the intensity and depth arrays are in the same orientation by displaying a specific frame  
    plt.subplot(1,2,1)
    plt.imshow(rgb_array)
    plt.subplot(1,2,2)
    sns.heatmap(z_array)
    plt.show()

# If any of the tests that require image data to be processed are to be run, run them
if __name__=="__main__":
    # Get list of folders in Data
    folders = next(os.walk('Data'))[1]
    if 'mat' in folders:
        folders.pop(folders.index('mat'))
    
    ctr = 1
    for filename in folders:
        # Choose the name of the folder of NIR images and file of depth data to be processed
        print(f'Processing file {ctr}/{len(folders)}: {filename}')
        pathname = f'Data/{filename}/'

        # Specifications for the measurement - image size, total frames, framerates
        img_width = 480
        img_height = 640

        # Get filenames for ply files
        ply_names = get_files(pathname + "*.ply*")
        ply_names.sort()

        # Run processing
        num_frames = len(ply_names)
        print(f'Number of Frames to Process: {num_frames}')

        xyz_values = np.zeros((img_height, img_width, 3, num_frames))
        rgb_values = np.zeros((img_height, img_width, 3, num_frames))

        num_processes = cpu_count()
        args = [(ply_names[frame_index], img_width, img_height) for frame_index in range(num_frames)]
        results = p_map(read_ply, args)
        xyz_rgb = np.asarray(results)
        
        i = 0
        while i<num_frames:
            if np.sum(xyz_rgb[i, :, :, :]) == 0:
                xyz_rgb = np.delete(xyz_rgb, i, 0)
                num_frames -= 1
            else:
                i += 1
                
        xyz_values_old = xyz_rgb[:, :, :, 0:3]
        rgb_values_old = xyz_rgb[:, :, :, 3:6]
        
        for i in range(num_frames):
            xyz_values[:, :, :, i] = xyz_values_old[i, :, :, :]
            rgb_values[:, :, :, i] = rgb_values_old[i, :, :, :]
            
        rgb_values = rgb_values.astype('uint8')
        
        #orientation_check(rgb_values[:, :, :, 0], xyz_values[:, :, 2, 0])
        savemat(f'Data/mat/{filename}.mat', {'xyz_values': xyz_values, 'rgb_values': rgb_values})
        ctr = ctr+1
        print()
    print('Done!')