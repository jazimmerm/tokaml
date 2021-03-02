import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelmax, argrelmin
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os

B1 = open('../data/B1/174828.txt')
# prmtan = open('data/prmtan_neped/174828.txt')
B1_lines = B1.readlines()

B1.close()
# prmtan.close()



B1_t = [line.strip('\n').split(' ')[0] for line in B1_lines]
B1_mag = [line.strip('\n').split(' ')[1] for line in B1_lines]

B1_t = list(map(float, B1_t))

print(B1_t[0], type(B1_t[0]))
exit()
fig, (ax1, ax2) = plt.subplots(nrows=2)

Pxx, freqs, bins, im = ax2.specgram(B1_mag[5000:10000])

ax1.plot(B1_t[5000:10000], B1_mag[5000:10000])
plt.show()