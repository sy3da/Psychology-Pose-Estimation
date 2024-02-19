from plyfile import PlyData, PlyElement
import numpy as np
import matplotlib.pyplot as plt
import cv2

plydata = PlyData.read('0000000.ply')
data = np.zeros((640, 480, 6))

row_idx = 0
col_idx = 0
for i in range(640*480):
    data[row_idx, col_idx, 0] = plydata['vertex'][i][0]
    data[row_idx, col_idx, 1] = plydata['vertex'][i][1]
    data[row_idx, col_idx, 2] = plydata['vertex'][i][2]
    data[row_idx, col_idx, 3] = plydata['vertex'][i][3]
    data[row_idx, col_idx, 4] = plydata['vertex'][i][4]
    data[row_idx, col_idx, 5] = plydata['vertex'][i][5]
    
    col_idx += 1
    if col_idx == 480:
        col_idx = 0
        row_idx += 1
        
#plt.imshow(data[:,:,3:6]/255)
plt.imshow(data[:,:,1], cmap='gray')
plt.show()