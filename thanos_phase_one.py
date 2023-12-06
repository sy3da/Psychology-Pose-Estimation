"""
This code converts the outputs from the ST MOB-THANOSV3 into depth and intensity arrays, respectively.
The frame rate (fps) should be set to the correct value in this code and processHR.py
"""

import regex as re
import numpy as np
from PIL import Image
from p_tqdm import p_map
from glob import glob as get_files
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
import datetime
from scipy.io import savemat

def read_pfm(filename, byteorder='>'):
    """Return image data from a raw PFM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pfm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^Pf\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype=np.float32,
                            count=int(width)*int(height),
                            offset=len(header)
                            ).newbyteorder('>').reshape((int(height), int(width)))

# Loads all images in the specified folder into the I_values array
def load_and_process_nir_images(args):
    frame_index, amp_name, depth_name, img_width, img_height  = args
    data = np.zeros((img_height, img_width, 2))

    image = read_pfm(amp_name, byteorder='<')
    image = np.flipud(image)
    data[:, :, 0] = image

    f = depth_name
    image = Image.open(f)
    image = np.asarray(image)
    image = np.flipud(image)
    data[:, :, 1] = image
    
    return data


def orientation_check(I_array, D_array):
    # Checks if the intensity and depth arrays are in the same orientation by displaying a specific frame  
    plt.subplot(1,2,1)
    plt.imshow(I_array)
    plt.subplot(1,2,2)
    sns.heatmap(D_array)
    plt.show()

# If any of the tests that require image data to be processed are to be run, run them
if __name__=="__main__":
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Choose the name of the folder of NIR images and file of depth data to be processed
    filename = 'landscape'
    pathname = f'jarv3d 1.7.0/Records/User/{filename}/'

    # Specifications for the measurement - image size, total frames, framerates
    # num_frames = 100
    img_width = 600
    img_height = 804

    # Adjust frame rate accordingly
    NIR_framerate = 30
    Depth_framerate = 30

    # Get filenames for amplitude and depth data
    amp_names = get_files(pathname + "*amplitude*")
    amp_names.sort()

    depth_names = get_files(pathname + "*depth*")
    depth_names.sort()

    # Run processing
    num_frames = len(amp_names)
    print(f'Number of Frames to Process: {num_frames}')

    num_processes = cpu_count()
    args = [(frame_index, amp_names[frame_index], depth_names[frame_index], img_width, img_height) for frame_index in range(num_frames)]
    results = p_map(load_and_process_nir_images, args)
    
    I_and_D = np.asarray(results)
    I_values_old = I_and_D[:, :, :, 0]
    D_values_old = I_and_D[:, :, :, 1]

    I_values = np.zeros((img_height, img_width, num_frames))
    D_values = np.zeros((img_height, img_width, num_frames))
    for i in range(num_frames):
        I_values[:, :, i] = I_values_old[i, :, :]
        D_values[:, :, i] = D_values_old[i, :, :]
    
    #orientation_check(I_values[:, :, 0], D_values[:, :, 0])

    savemat(f'Data/mat/thanos_processed_{filename}.mat', {'I_values': I_values, 'D_values': D_values})