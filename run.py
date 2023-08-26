import os
import sys
import subprocess
import shutil
import argparse
import time
from typing import Optional

from src.bin_to_csv import BinToCsv

def record_skv() -> Optional[str]:
    """
    Launches Automotive Suite and waits until the user exits it.
    Once the user is done recording, they can close the program.
    After it is closed, moves all of the newly recorded files from the
    Automotive Suite recordings directory to the './skvs/' directory.
    
    Returns:
        str: The absolute path to the './skvs/' directory if new files were recorded,
        otherwise script terminates.
    
    NOTE: If no new files were recorded, terminates the script.
    """
    # Find the folder path that matches the pattern "./automotive_suite_recorder_viewer*"
    # print(os.getcwd())
    folder_name = next(filter(lambda x: x.startswith('automotive_suite_recorder_viewer'), os.listdir()), None)
    # print(f"folder_name: {folder_name}")
    
    if not folder_name:
        print('Automotive Suite not found at automotive_suite_recorder_viewer*/automotive_suite.exe')
        sys.exit()
    else:
        folder_path = os.path.join(os.getcwd(), folder_name)
        exe_path = os.path.join(folder_path, 'automotive_suite.exe')
        # print(f"folder_path: {folder_path}")
        # print(f"exe_path: {exe_path}")
    
        # Get set of files in ./automotive_suite_recorder_viewer*
        #   - Put all *.skv filenames and the datetime they were created in a set of tuples {("filename.skv", datetime_created))}
        #   - Ignore folders

        recordings_path = os.path.join(folder_path, 'RecordedMovies')

        skvs_before_recording = set(filter(lambda x: x[0].endswith('.skv'), map(lambda x: (x, os.path.getctime(os.path.join(recordings_path, x))), os.listdir(recordings_path))))

        # skvs_before_recording = set()
        # for filename in os.listdir(recordings_path):
        #     print(f"filename: {filename}")
        #     if os.path.isfile(os.path.join(recordings_path, filename)) and filename.endswith('.skv'):
        #         # get the datetime the file was created
        #         datetime_created = os.path.getctime(os.path.join(recordings_path, filename))
        #         # add the filename and datetime_created to the set as a tuple
        #         skvs_before_recording.add((filename, datetime_created))




        # print("skvs before recording")
        # print(skvs_before_recording)
    
        # Launch the program and wait until the user exits it
        process = subprocess.run(exe_path, shell=True)
    
        skvs_after_recording = set(filter(lambda x: x[0].endswith('.skv'), map(lambda x: (x, os.path.getctime(os.path.join(recordings_path, x))), os.listdir(recordings_path))))
        # print("skvs after recording")
        # print(skvs_after_recording)
    
        # If sets are different, then new .skv file(s) were created
        # skvs_before_recording - skvs_after_recording = set of new .skv files
        if skvs_before_recording != skvs_after_recording:
            # Move the new .skv file(s) to ./skvs/
            new_skvs = skvs_after_recording - skvs_before_recording
            # print("new_skvs")
    
            # # Get absolute path to the new .skv file
            # new_skv_path = os.path.join(recordings_path, new_skv[0])

            for new_skv in new_skvs:
                # print(new_skv)
                new_skv_path = os.path.join(recordings_path, new_skv[0])
                # print(new_skv_path)
                shutil.move(new_skv_path, os.path.join(os.getcwd(), 'skvs'))
                print('Automotive Suite recorded new file: ' + new_skv_path)
            
            skv_dir = os.path.join(os.getcwd(), 'skvs')
    
            return skv_dir
        else:
            print('No new .skv file was recorded')
            sys.exit()

def check_for_skvs(skvs_dir: str):
    """
    Checks if there are any .skv files in skvs_dir.
    If there are no .skv files, terminates the script.

    Args:
        skvs_dir (str): absolute path to directory containing skv files

    Returns:
        None
    """
    skvs_before_recording = set(filter(lambda x: x[0].endswith('.skv'), map(lambda x: (x, os.path.getctime(os.path.join(skvs_dir, x))), os.listdir(skvs_dir))))

    # if skvs_before_recording empty, exit
    if not skvs_before_recording:
        print('No .skv files found in: ' + skvs_dir)
        print('Please record a .skv file using Automotive Suite.')
        sys.exit()
    
    return

def skv_to_bin(skvs_dir: str):
    """
    Convert each .skv file in skvs_dir to .bin file using imx520_sample.exe and save to bins_dir

    Args:
        skvs_dir (str): absolute path to directory containing skv files

    Returns:
        None
    """

    # Get absolute path of directory where .bin files will be saved to
    bins_dir = os.path.join(skvs_dir, "bins")

    # Convert all .skv video files in skvs_dir into .bin files using imx520_sample.exe and save to bins_dir

    # Get absolute path to imx520_sample.exe
    imx520_sample_exe_path = os.path.join(os.getcwd(), "src/skv_to_mat/r2_3_1/imx520_sample.exe")
    
    # Run imx520_sample.exe
    # ./imx520_sample.exe -i ./skvs/ -o ./skvs/mat/ -d
    process = subprocess.run([imx520_sample_exe_path, "-i", skvs_dir, "-o", bins_dir, "-d"], shell=True)

    return

def bin_to_csv(skvs_dir: str):
    """
    Take all .bin files in bins_dir, run through pose estimation pipeline, and save output to csvs_dir

    Args:
        skvs_dir (str): absolute path to directory containing skv files

    Returns:
        None
    """

    # Get absolute path of directory where .bin files are located
    bins_dir = os.path.join(skvs_dir, "bins")

    # Run pose estimation pipeline on all .bin files in bins_dir and save output to csvs_dir
    myBinToCsv = BinToCsv(input_dir=bins_dir, output_filename="my_csv", visualize_Pose=True)
    myBinToCsv.run()

    return

  
def process_args() -> argparse.Namespace:
    """
    Process command line arguments

    Returns:
        args (argparse.Namespace): parsed command line arguments

    Notes:
        If no command-line arguments are provided, all options are set to True by default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--skv_to_bin', action='store_true', help='Convert .skv files to .bin files')
    parser.add_argument('--bin_to_csv', action='store_true', help='Take all .bin files and generate output .csv file')
    args = parser.parse_args()

    # If no args are provided, set all 3 bools to True
    if not any(vars(args).values()):
        args.skv_to_bin = True
        args.bin_to_csv = True
    
    return args

if __name__ == '__main__':
    # TODO: Add checks to ensure that skvs_dir and bins_dir exist. If not, create them.
    
    main_start_time = time.time()
    
    args = process_args()

    # Get the path to the new .skv file
    # skvs_dir = record_skv()

    skvs_dir = os.path.join(os.getcwd(), 'Data', 'skvs')

    check_for_skvs(skvs_dir)

    if args.skv_to_bin:
        start_time = time.time()
        skv_to_bin(skvs_dir)
        end_time = time.time()
        print("skv_to_bin() took " + str(end_time - start_time) + " seconds to run")
    
    if args.bin_to_csv:
        start_time = time.time()
        bin_to_csv(skvs_dir)
        end_time = time.time()
        print("bin_to_csv() took " + str(end_time - start_time) + " seconds to run")

    main_end_time = time.time()

    print('Done!')

    print(f"run.py took {main_end_time - main_start_time} seconds to run")

