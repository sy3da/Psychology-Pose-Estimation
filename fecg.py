import wfdb 
import matplotlib.pyplot as plt

signal, dict = wfdb.rdsamp('sub01_snr00dB_l1_c0_fecg1', pn_dir='fecgsyndb/sub01/snr00dB')
print(signal.shape)

plt.plot(signal[:, 0])
plt.xlim([0, 1000])
plt.show()
