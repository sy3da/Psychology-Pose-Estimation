import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_csv(filename):
    data = pd.read_csv(filename)
    
    elim = data.query('Hip_Center_X == -32767')
    data = data.drop(elim.index)
    elim = data.query('shoulder_Center_X == -32767')
    data = data.drop(elim.index)
    
    return data

def _get_csv_files(directory):
    """
    Get list of .csv files in directory

    Returns:
        csv_files (list): list of .csv files in directory
    """

    # Get list of .csv files in input_dir
    filelist = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Remove the ".csv" suffix
            filename = filename[:-4]
            filelist.append(filename)

    return filelist

if __name__ == "__main__":
    # Get path to csv files and list of csv file names
    csv_dir = os.path.join(os.getcwd(), 'world landmarks test')
    files = _get_csv_files(csv_dir)
    
    # Determine number of files to process
    num_files_to_process = len(files)
    
    # Loop through files
    fig, (ax1, ax2) = plt.subplots(1, 2, layout='constrained', figsize=(9,5))
    for file_num, filename in enumerate(files):
        data = read_csv(csv_dir + '/' + filename + '.csv')
        diff = ((data['shoulder'] - data['hip'])/data['hip'])
        ratio = (data['shoulder'] / data['hip'])
        
        if file_num < 3:
            color='black'
        elif file_num <6:
            color='red'
        else:
            color='blue'
        
        # Plotting 'lens' data on the first subplot
        ax1.plot(data['hip'], diff, color=color)
        ax1.set_xlabel('Hip Distance')
        ax1.set_title('Shoulder - Hip')
        ax1.invert_xaxis()
        
        
        # Plotting 'errors' data on the second subplot
        ax2.plot(data['hip'], ratio, color=color)
        ax2.set_xlabel('Hip Distance')
        ax2.set_title('Shoulder / Hip')
        ax2.invert_xaxis()

    # Creating a shared legend for both subplots    
    ax1.set_xlim(left=17500, right=9000)
    ax2.set_xlim(left=17500, right=9000)
    plt.show()