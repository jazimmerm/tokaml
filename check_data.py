import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
arr = np.load('data/unmasked_array.npy', allow_pickle=True)
=======
arr = np.load('unmasked_array.npy', allow_pickle=True)
>>>>>>> 5c3ec884656cde95a75c959d6f2ace9df1a26e50

fig = plt.figure()

heatmap = plt.imshow(arr[1],
                extent=[arr[0][0], arr[0][-1], arr[2][0], arr[2][-1]],
                origin='lower',
                vmin=0, vmax=0.05,
                interpolation='none')
plt.gca().set_xlabel('Time (ms)')
plt.gca().set_ylabel('Frequency (kHz)')
plt.show()