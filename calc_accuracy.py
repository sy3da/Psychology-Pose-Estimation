import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_csv(filename):
    data = pd.read_csv(filename)
    data_trimmed = data[['frame_num', 
                         'Shoulder_Right_X', 'Shoulder_Right_Y', 'Shoulder_Right_Z', 
                         'Elbow_Right_X', 'Elbow_Right_Y', 'Elbow_Right_Z', 
                         'Shoulder_Left_X', 'Shoulder_Left_Y', 'Shoulder_Left_Z', 
                         'Elbow_Left_X', 'Elbow_Left_Y', 'Elbow_Left_Z', 
                         'Hip_Right_X', 'Hip_Right_Y', 'Hip_Right_Z', 
                         'Knee_Right_X', 'Knee_Right_Y', 'Knee_Right_Z', 
                         'Hip_Left_X', 'Hip_Left_Y', 'Hip_Left_Z', 
                         'Knee_Left_X', 'Knee_Left_Y', 'Knee_Left_Z']]
    
    elim = data_trimmed.query('Shoulder_Right_X == -32767')
    data_trimmed.drop(elim.index, inplace=True)
    
    return data_trimmed

def calc_shoulder_to_elbow(shoulder_x, shoulder_y, shoulder_z, elbow_x, elbow_y, elbow_z):
    distance = np.sqrt(((shoulder_x - elbow_x)**2 + (shoulder_y - elbow_y)**2 + (shoulder_z - elbow_z)**2))
    return distance

def calc_shoulder_to_shoulder(lshoulder_x, lshoulder_y, lshoulder_z, rshoulder_x, rshoulder_y, rshoulder_z):
    distance = np.sqrt(((lshoulder_x - rshoulder_x)**2 + (lshoulder_y - rshoulder_y)**2 + (lshoulder_z - rshoulder_z)**2))
    return distance

def calc_hip_to_knee(hip_x, hip_y, hip_z, knee_x, knee_y, knee_z):
    distance = np.sqrt(((hip_x - knee_x)**2 + (hip_y - knee_y)**2 + (hip_z - knee_z)**2))
    return distance

def calc_hip_to_hip(lhip_x, lhip_y, lhip_z, rhip_x, rhip_y, rhip_z):
    distance = np.sqrt(((lhip_x - rhip_x)**2 + (lhip_y - rhip_y)**2 + (lhip_z - rhip_z)**2))
    return distance

data = read_csv('blank_testing_two_people2_left_participant.csv')
lens = np.zeros((len(data), 6))

lens[:, 0] = calc_shoulder_to_shoulder(data['Shoulder_Left_X'], data['Shoulder_Left_Y'], data['Shoulder_Left_Z'],
                                       data['Shoulder_Right_X'], data['Shoulder_Right_Y'], data['Shoulder_Right_Z'])

lens[:, 1] = calc_hip_to_hip(data['Hip_Left_X'], data['Hip_Left_Y'], data['Hip_Left_Z'], 
                             data['Hip_Right_X'], data['Hip_Right_Y'], data['Hip_Right_Z'])

lens[:, 2] = calc_shoulder_to_elbow(data['Shoulder_Left_X'], data['Shoulder_Left_Y'], data['Shoulder_Left_Z'],
                                    data['Elbow_Left_X'], data['Elbow_Left_Y'], data['Elbow_Left_Z'])

lens[:, 3] = calc_shoulder_to_elbow(data['Shoulder_Right_X'], data['Shoulder_Right_Y'], data['Shoulder_Right_Z'],
                                    data['Elbow_Right_X'], data['Elbow_Right_Y'], data['Elbow_Right_Z'])

lens[:, 4] = calc_hip_to_knee(data['Hip_Left_X'], data['Hip_Left_Y'], data['Hip_Left_Z'], 
                              data['Knee_Left_X'], data['Knee_Left_Y'], data['Knee_Left_Z'])

lens[:, 5] = calc_hip_to_knee(data['Hip_Right_X'], data['Hip_Right_Y'], data['Hip_Right_Z'], 
                              data['Knee_Right_X'], data['Knee_Right_Y'], data['Knee_Right_Z'])

plt.plot(lens)
plt.xlabel('Frame Number')
plt.ylabel('Length (cm)')
plt.legend(['Shoulder-Shoulder', 'Hip-Hip', 'Left Arm', 'Right Arm', 'Left Leg', 'Right Leg'])
plt.show()
    