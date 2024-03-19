"""
This code converts the outputs from the ST MOB-THANOSV3 into depth and intensity arrays, respectively.
The frame rate (fps) should be set to the correct value in this code and processHR.py
"""

import numpy as np
from p_tqdm import p_map
from tqdm import tqdm
from glob import glob as get_files
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from scipy.io import savemat
import os
import open3d as o3d
# import skvideo
# skvideo.setFFmpegPath(".venv/Lib/site-packages")
# import skvideo.io
import video2numpy as v2n

def orientation_check(rgb_array, z_array):
    # Checks if the intensity and depth arrays are in the same orientation by displaying a specific frame  
    plt.subplot(1,2,1)
    plt.imshow(rgb_array)
    plt.subplot(1,2,2)
    sns.heatmap(z_array)
    plt.show()

# If any of the tests that require image data to be processed are to be run, run them
if __name__=="__main__":
    # Get list of videos in Data
    base_pathname = f'Data/'
    rgbd_names = get_files(base_pathname + "*.mp4*")
    
    for num, filename in enumerate(rgbd_names):
        # Choose the name of the file to be processed
        print(f'Processing file {num+1}/{len(rgbd_names)}: {filename}')
        pathname = base_pathname + filename

        # Specifications for the measurement - image size, total frames, framerates
        img_width = 480
        img_height = 640

        # Open video file to process
        v2n.video2numpy(pathname, dest=None)
        
        # Run processing
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