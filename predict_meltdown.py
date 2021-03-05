import torch
import numpy as np

arr = np.load('data/unmasked_array.npy', allow_pickle=True)

print(len(arr))
print(arr[0].shape)
print(arr[1].shape)
print(arr[2].shape)