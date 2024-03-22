import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

data = np.load('Data/npz/george_b1.npz')
xyz = data['xyz_values']
rgb = data['rgb_values']

frame_num = 200

plt.subplot(1,4,1)
plt.imshow(xyz[:,:,0, frame_num], cmap="PiYG", norm=colors.TwoSlopeNorm(vcenter=0))
plt.subplot(1,4,2)
plt.imshow(xyz[:,:,1, frame_num], cmap="PiYG", norm=colors.TwoSlopeNorm(vcenter=0))
plt.subplot(1,4,3)
plt.imshow(xyz[:,:,2, frame_num], cmap="PiYG", norm=colors.TwoSlopeNorm(vcenter=0))

plt.subplot(1,4,4)
plt.imshow(rgb[:, :, :, frame_num])

plt.show()