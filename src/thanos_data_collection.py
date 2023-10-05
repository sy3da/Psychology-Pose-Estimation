"""
This code converts the outputs from the L5/L8/L8 ultra and FoxAuto into depth and intensity arrays, respectively.
The orientation for the L5 and L8 should be set to mirror Y on their software programs in order for the
code below to work properly. The orientation for the L8 ultra and FoxAuto should be set to default on their
software programs. This ensures that both data arrays follow the same orientation, before processing begins.

The frame rate (fps) should be set to the correct value in this code and hr_script_test.py
"""

import regex as re
import os
#import paramiko
import numpy as np
#import hr_script_test
#import Testing
import csv
from PIL import Image
from tqdm import tqdm
from glob import glob as get_files

# Uncomment to display the IR and depth images
#import seaborn as sns
import matplotlib.pyplot as plt

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
def load_and_process_nir_images(num_frames, pathname):
    global I_values
    global D_values

    amp_names = get_files(pathname + "*amplitude*")
    amp_names.sort()

    depth_names = get_files(pathname + "*depth*")
    depth_names.sort()

    for x in tqdm(range(num_frames)):
        image = read_pfm(amp_names[x], byteorder='<')
        image = np.flipud(image)
        I_values[x] = image

        f = depth_names[x]
        image = Image.open(f)
        image = np.asarray(image)
        image = np.flipud(image)
        D_values[x] = image
    
    I_values = np.asarray(I_values)
    D_values = np.asarray(D_values)


    # Load each image, one at a time, and convert it into a numpy array
    #for file in os.listdir(directory):
    #    f = os.path.join(directory, file)
    #    image = Image.open(f)
    #    image = np.asarray(image)

    #    I_values.append(image)

    D_values = np.asarray(D_values)

def orientation_check(I_array, D_array):
    # Checks if the intensity and depth arrays are in the same orientation by displaying a specific frame
    plt.subplot(1,2,1)
    plt.imshow(I_array[2])
    plt.subplot(1,2,2)
    sns.heatmap(D_array[2])
    plt.show()

# If any of the tests that require image data to be processed are to be run, run them
if __name__=="__main__":

    # Choose the name of the folder of NIR images and file of depth data to be processed
    pathname = "jarv3d 1.7.0/Records/User/"
    
    # Toggle what tests you would like to run on the data
    #getFile = False         # Will retrieve a file of the given filename from a remote Raspberry Pi (use if Pi ran the L5)
    processData = False     # Runs the tablet code
    #roiTest = False         # Extracts depth and intensity signals for facial ROIs to visualize signal
    #fftTest = False # Extracts depth and intensity signals then graphs fourier transforms
    #depthTest = False       # Only plots depth and some attempts at cleaning the signal

    # Check if the intensity and depth arrays are in the same orientation
    orientation = True

    # Runs Kaiwen's algorithm on full face ROIs for direct-facing measurements
    # This is the most recent and (theoretically) correct algorithm for extracting HR
    #newProcessData = False

    # Select if depth data is coming from L5/L8 or L8 ultra.
    # Output data for L5/L8 follow the same format, while L8 ultra follows a different format.
    ultra = False # Change to False if using L5/L8

    # Specifications for the measurement - image size, total frames, framerates, bitcount
    num_frames = 400
    img_width = 600
    img_height = 804
    bitcount = 8 # keep images from FoxAuto at 8 bit

    # Adjust frame rate accordingly
    NIR_framerate = 30
    Depth_framerate = 30

    # Initialize empty arrays of pixel values to be populated later
    # 'distance' will be calculated as sqrt(X^2+D^2+Z^2), or depth in spherical coordinates
    # In this case depth from the L5 is already true distance, so X and Z should be 0
    #zeros = [[[np.single(0)]*img_width]*img_height]*total_frames
    #X_values = np.array(zeros)
    #Z_values = np.array(zeros)
    I_values = [0]*num_frames
    D_values = [0]*num_frames

    if processData or orientation:
        load_and_process_nir_images(num_frames, pathname)
        print("loaded nir")

        #if ultra:
        #    load_and_process_depth_data_ultra()
        #else:
        #    load_and_process_depth_data()
        #print("loaded depth")

    # Run each of the specified tests
    if orientation:
        print(len(I_values), " ", len(D_values))
        orientation_check(I_values, D_values)

    # if processData:
    #     hr_script_test.getHR(I_values, D_values, total_frames, filename)

    '''if roiTest:
        Testing.roi_test(I_values, X_values, D_values, Z_values)

    if fftTest:
        Testing.get_intensity_fft(I_values, X_values, D_values, Z_values)

    if newProcessData:
        Testing.kaiwen_test(I_values, X_values, D_values, Z_values)

    if depthTest:
        Testing.depth_test(I_values, X_values, D_values, Z_values)'''