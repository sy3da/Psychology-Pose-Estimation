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

def read_ply(filename, img_width, img_height, byteorder='>'):
    plydata = PlyData.read(filename)
    xyz = np.zeros((img_height, img_width, 3))
    rgb = np.zeros((img_height, img_width, 3))

    row_idx = 0
    col_idx = 0
    for i in range(640*480):
        xyz[row_idx, col_idx, 0] = plydata['vertex'][i][0]
        xyz[row_idx, col_idx, 1] = plydata['vertex'][i][1]
        xyz[row_idx, col_idx, 2] = plydata['vertex'][i][2]
        rgb[row_idx, col_idx, 0] = plydata['vertex'][i][3]
        rgb[row_idx, col_idx, 1] = plydata['vertex'][i][4]
        rgb[row_idx, col_idx, 2] = plydata['vertex'][i][5]
        
        col_idx += 1
        if col_idx == 480:
            col_idx = 0
            row_idx += 1
            
    return xyz, rgb


def orientation_check(I_array, D_array):
    # Checks if the intensity and depth arrays are in the same orientation by displaying a specific frame  
    plt.subplot(1,2,1)
    plt.imshow(I_array)
    plt.subplot(1,2,2)
    sns.heatmap(D_array)
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

        # Adjust frame rate accordingly
        RGB_framerate = 60
        Depth_framerate = 60

        # Get filenames for ply files
        ply_names = get_files(pathname + "*.ply*")
        ply_names.sort()

        # Run processing
        num_frames = len(ply_names)
        print(f'Number of Frames to Process: {num_frames}')

        x_values = np.zeros((img_height, img_width, num_frames))
        y_values = np.zeros((img_height, img_width, num_frames))
        z_values = np.zeros((img_height, img_width, num_frames))
        r_values = np.zeros((img_height, img_width, num_frames))
        g_values = np.zeros((img_height, img_width, num_frames))
        b_values = np.zeros((img_height, img_width, num_frames))
        i = 0
        for name in tqdm(ply_names, total=num_frames):
            XYZ, RGB = read_ply(name, img_width, img_height)
            x_values[:, :, i] = XYZ[:, :, 0]
            y_values[:, :, i] = XYZ[:, :, 1]
            z_values[:, :, i] = XYZ[:, :, 2]
            r_values[:, :, i] = RGB[:, :, 0]
            g_values[:, :, i] = RGB[:, :, 1]
            b_values[:, :, i] = RGB[:, :, 2]
            i = i + 1
        
        #orientation_check(I_values[:, :, 0], D_values[:, :, 0])

        savemat(f'Data/mat/{filename}.mat', {'x_values': x_values, 'y_values': y_values, 'z_values': z_values, 'r_values': r_values, 'g_values': g_values, 'b_values': b_values})
        ctr = ctr+1
        print()
    print('Done!')