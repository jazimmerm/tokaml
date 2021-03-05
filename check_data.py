import numpy as np
import matplotlib.pyplot as plt

arr = np.load('data/unmasked_array.npy', allow_pickle=True)

fig = plt.figure()

heatmap = plt.imshow(arr[1],
                extent=[arr[0][0], arr[0][-1], arr[2][0], arr[2][-1]],
                origin='lower',
                vmin=0, vmax=0.05,
                interpolation='none')
plt.gca().set_xlabel('Time (ms)')
plt.gca().set_ylabel('Frequency (kHz)')
plt.show()