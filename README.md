# iPhone - Body Posture Analysis

This repository contains all relevant scripts to run, process, and analyze rgb and depth data from studies using the LiDAR sensor in iPhones/iPads. All recordings are taken with the third party "Record3d" app availble on the Apple App Store.

## Getting Started

### Prerequisites

__Compatibility__:

- The recording "Record3d" app can __only__ be used on the iPhone Pro 12 or later, iPad Pro 11 inch 3rd generation or later, and iPad Pro 12.9 inch 5th generation or later (hasto have LiDAR scanner).

__Installation Instructions__:

Recording:
- Install the latest version of "Record3d" onto the iPhone/iPad that will be used for recording.

Processing:
- Download and install the most recent version of Python onto your operating system. This can be done at the following link: https://www.python.org/downloads/ 
- Download and install VSCode onot your operating system. This can be done at the following link: https://code.visualstudio.com/download 
- Link your GitHub account to your VSCode profile:  
    - Navigate to the “Extensions” tab on the left hand side of VS Code. Search and install “GitHub Pull Requests”.
    Relaunch VS Code.
    - Navigate to the “Accounts” tab in the lower left corner of VS Code. Select “Log into GitHub”. 
    - On the VS Code Home Page select “Clone Git Repository…”
    - Paste https://github.com/sy3da/Psychology-Pose-Estimation at the top of the window where it says “Provide repository URL or pick a repository source.”
- Install packages in VSCode and create virtual environment:
    - Navigate to the “Extensions” tab on the left hand side. Search and install “Python” v2024.0.1.
    - Within VS Code open the Command Palette (⇧⌘P) and search and select “Python: Create Environment”.
    - Select “Venv” for the environment type and the “Python {version that was installed onto operating system}” as the interpreter.
    - Select “requirements.txt” as the dependencies file to be installed in the virtual environment. 


The processing scripts are written in Python and run on Windows and Mac OS.

## Usage

### Step 1 - Recording
Follow the instructions outlined in the following powerpoint for setting up "Record3d" with the proper settings and for exporting files:
https://docs.google.com/presentation/d/1p2FjMWa6pQhxtKmbjKskehwqDIX7YroxsdVIMqZcKnU/edit?usp=sharing 


### Preparations for Step 2
After exporting the recordings to the RGBD.mp4 file format, place them in the "Data" folder. Remove any old .mp4 files that you do not wish to run the processing on.

Additionally, remove any old .npz files from "Data\npz" that you do not wish to run processing on.


### Step 2 - Inital Processing
Ensure the Pyschology-Pose-Estimation directory is selected.

Run the "iphone_rgbd_phase_one.py" processing script to get rgb and xyz data for each pixel in every frame from the .mp4 files. The output will be .npz files (one for each .mp4 file) located under "Data\npz".

Navigate to the "iphone_npz_to_csv.py" script.

Select the conditions for the trial where the class NpzToCsv() is called at the bottom of the script.
  Ex. for one participant:
  - NpzToCsv(input_dir=npzs_dir, visualize_Pose=True, two_people=False, landscape=False)
  Ex. for two participants:
  - NpzToCsv(input_dir=npzs_dir, visualize_Pose=True, two_people=True, , left_participant_id = '965142_', right_participant_id = '510750_, landscape=False)

Run the "phone_npz_to_csv.py" processing script to get a .csv file with xyz for identified landmarks saved in "Data/npz/csv" and a video with the skeleton visualization saved in "Data/npz/video".


### Step 3 - Processing and Data Analysis in R


## Additional Files
plot_rgb_frame.py
- This file can be used to plot/visualize rgb and xyz data for one frame within a .npz file.
- To run update the line with the path to the .npz file you would like to visualize along with the frame number.

pose_module.py
- This is an accessory file needed to call and run mediapipe on the videos within the "iphone_npz_to_csv.py" processing script.
- If changes need to be made to the conditions mediapipe is being run on, this is where it can be done.

calc_accuracy.py
- This can be used to compare real world distances between landmarks to pre-measured distances for a particular subject
- To use update the true_lengths variable with the pre-measured lengths in mm for [shoulder to shoulder, hip to hip, left shoulder to elbow, right shoulder to elbow, left hip to knee, right hip to knee] and place .csv files of interest in "Data\npz\csv"

## Support
If you are experiencing issues installing the hardware, or running the scripts, please contact <br/>
Lauren Terry, leterryy@umich.edu or <br/>
George Rabadi, grabadi@umich.edu

## Authors and Acknowledgment
Inital Processing Scripts:
- written by Lauren Terry and George Rabadi.

Additional File Scripts:
- written by Lauren Terry and George Rabadi.