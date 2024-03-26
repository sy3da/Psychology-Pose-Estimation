import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

data = np.load('Data/npz/lauren_baseline_test.npz')
frame_num = 100

xyz = data['xyz_values']
rgb = data['rgb_values']


plt.subplot(1,4,1)
plt.imshow(xyz[:,:,0, frame_num], cmap="PiYG", norm=colors.TwoSlopeNorm(vcenter=0))
plt.title('X')
plt.subplot(1,4,2)
plt.imshow(xyz[:,:,1, frame_num], cmap="PiYG", norm=colors.TwoSlopeNorm(vcenter=0))
plt.title('Y')
plt.subplot(1,4,3)
plt.imshow(xyz[:,:,2, frame_num], cmap="PiYG", norm=colors.TwoSlopeNorm(vcenter=0))
plt.title('Z')

plt.subplot(1,4,4)
plt.imshow(rgb[:, :, :, frame_num])
plt.title('RGB')

plt.show()