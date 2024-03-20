from plyfile import PlyData, PlyElement
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
#import cv2

plydata = PlyData.read('Data/wall_test_3-10-24/0000120.ply')
data = np.zeros((640, 480, 6))
print(plydata['vertex'][0])
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
plt.subplot(1,3,1)
plt.imshow(data[:,:,0], cmap="PiYG", norm=colors.TwoSlopeNorm(vcenter=0))
plt.subplot(1,3,2)
plt.imshow(data[:,:,1], cmap="PiYG", norm=colors.TwoSlopeNorm(vcenter=0))
plt.subplot(1,3,3)
plt.imshow(data[:,:,2], cmap="PiYG", norm=colors.TwoSlopeNorm(vcenter=0))
plt.show()