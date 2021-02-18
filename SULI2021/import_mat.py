import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelmax, argrelmin
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os


# extract values for specific shot from above list of shots
def get_shot(shot=None):

    cwd = os.getcwd()
    file = cwd + '/SULI2021/data/RESULTS_ECH_EFFECTS_SPECTRA_B3.mat'
    # import data from .mat file into python numpy arrays
    mat = h5py.File(file, 'r')
    dat_ech = mat['DAT_ECH']
    shlst = mat['shn'][:]

    if shot:
        shindex = [np.where(shlst == s) for s in shlst if shot in s][0][0][0]

        tds = dat_ech.get('TIME')[shindex][0]
        sds = dat_ech.get('SPECTRUM')[shindex][0]
        fds = dat_ech.get('FREQ')[shindex][0]

        time = mat[tds][0][:]
        spectrum = mat[sds][:]
        freq = mat[fds][0][:]

        return np.array([time, spectrum, freq], dtype=object)
    else:
        return [str(i) for i in shlst]


# make the heatmap
def heatmap2d(arr, ax=None):

    if ax is None:
        ax = plt.gca()
    heatmap = ax.imshow(arr[1],
                    extent=[arr[0][0], arr[0][-1], arr[2][0], arr[2][-1]],
                    origin='lower',
                    vmin=0, vmax=0.05,
                    interpolation='none')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (kHz)')
    return heatmap


# make the plot of the spectral data for given time
def slice1d(arr, time, smooth=False, ax=None):

    if ax is None:
        ax = plt.gca()
    index = np.argmin(np.abs(np.array(arr[0]) - time))  # finds index of time entered. Kind of slow.
    if index < 0:
        raise ValueError('Selected time is out of bounds of run time for shot.')
    slce = arr[1][:, index]
    ax.plot(arr[2], slce)
    ax.set_xlabel('Frequency (kHz)')
    if smooth:
        smooth_arr = gaussian_filter1d(slce, 10)
        peaks, _ = find_peaks(smooth_arr, prominence=(0.005, None), distance=75)
        ax.plot(arr[2], smooth_arr, 'y')
        ax.plot(arr[2][peaks], smooth_arr[peaks], 'ro')
    else:
        peaks, _ = find_peaks(slce, prominence=(0.03, None), distance=75)
        ax.plot(arr[2][peaks], slce[peaks], 'ro')
    return peaks


# Function to sweep through spectrogram
def sweep(arr):

    peaks_map = np.zeros_like(arr[1].T)
    peaks_index = []
    for i in range(len(arr[1][0])):
        slce = arr[1][:, i]
        smooth_arr = gaussian_filter1d(slce, 8)
        peaks, properties = find_peaks(smooth_arr, prominence=(0.005, None), distance=75, height=0, width=0)
        peaks_index.append(peaks.tolist())
        for e, w in enumerate(zip(properties['left_ips'], properties['right_ips'])):
            peaks_map[i][round(w[0]):round(w[1])] = properties['width_heights'][e]
        peaks_map[i][peaks] = properties['peak_heights']
    peaks_index = np.asarray(peaks_index, dtype=object)
    return peaks_index, peaks_map


# This is just to re-use the plots for different figures.
def plot(arr, time):

    fig, (ax1, ax2) = plt.subplots(2, 1)
    heatmap2d(arr, ax=ax1)
    ax1.axvline(time, c='r')
    # smooth bool switches between gaussian filtered 1d slice or raw data.
    peaks = slice1d(arr, time, smooth=True, ax=ax2)
    ax1.plot(np.full_like(peaks, time), arr[2][peaks], 'rx')
    plt.show()


def make_mask(arr, plot=False):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    heatmap2d(arr, ax=ax1)
    peaks_index, peaks_map = sweep(arr)
    # creates array of the values of peaks and null values everywhere else
    mask = np.ma.masked_where(peaks_map != 0, peaks_map)
    mask = mask.filled(fill_value=1)
    mask = np.ma.masked_where(mask == 0, mask)
    mask = np.transpose(mask)

    if plot:
        ax1.imshow(mask,
                   extent=[arr[0][0], arr[0][-1], arr[2][0], arr[2][-1]],
                   origin='lower',
                   cmap='Set1',
                   interpolation='none')
        ax1.set_title('Spectrogram With Modes Overlaid (1D Gaussian Filter Applied)')
        ax2.imshow(mask,
                   extent=[arr[0][0], arr[0][-1], arr[2][0], arr[2][-1]],
                   origin='lower',
                   cmap='Set1',
                   interpolation='none')
        ax2.set_title('Modes Only (1D Gaussian Filter Applied)')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Frequency (kHz)')
        plt.show()
        return

    return mask


# returns dict with all elm cycles of a particular shot. Elements are transposed. Before plotting, transpose back.
def split(arr, plot=False):

    pi, pm = sweep(arr)
    stop = 1500
    pm_norm = (pm[:, stop:] - np.amin(pm[:, stop:])) / (np.amax(pm[:, stop:]) - np.amin(pm[:, stop:]))
    sums = np.sum(pm_norm, axis=1)
    sums[sums <= 25] = 0
    l_elms = argrelmax(np.gradient(sums), order=5)
    r_elms = argrelmin(np.gradient(sums), order=5)

    if plot:
        fig, ax = plt.subplots(1, 1)
        heatmap2d(arr, ax=ax)
        for i in l_elms[0]:
            ax.axvline(arr[0][i], ymin=stop/arr[1].shape[0], c='red', linewidth=4)
        for i in r_elms[0]:
            ax.axvline(arr[0][i], ymin=stop/arr[1].shape[0], c='green', linewidth=4)
        plt.show()

    # make dict with keys, values, times
    elm_cycles = {}
    for i in range(len(r_elms[0])-1):
        for j in arr[0][r_elms[0][i]:l_elms[0][i+1]]:
            k = np.argwhere(arr[0] == j)[0][0]
            elm_cycles[('{}'.format(i), j, k)] = arr[1].T[np.argwhere(arr[0] == j)][0][0]

    index = pd.MultiIndex.from_tuples(elm_cycles.keys(), names=['ELM_No', 'Time(ms)', 'Index'])
    elmdf = pd.DataFrame(elm_cycles.values(), index=index)

    return elmdf

if __name__ == '__main__':

    '''
        Shots: 
         [174828.]
         [174829.]
         [174830.]
         [174833.]
         [174860.]
         [174870.]
    '''


    # sh = get_shot(174830)
    # # time = 1900
    # # plot(sh, time)
    # # plot_mask(sh, plot=True)
    # mplot = make_mask(sh)

    masked = np.load('SULI2021/data/masked_array.pickle', allow_pickle=True)

    split_df = pd.read_pickle('SULI2021/data/split_df.pickle')

    single_df = split_df.xs('21', level=0)

    # print(single_df.index.get_level_values('Index'))

    elm_id = {}
    for i in single_df.index.get_level_values('Index'):
        if i == 0:
            list1 = np.ma.flatnotmasked_contiguous(masked[0])
            continue
        else:
            list2 = np.ma.flatnotmasked_contiguous(masked[i])
            
        for index in range(len(list2)):
            if index not in elm_id:
                elm_id[index] = []
            try:
                if list2[index].start <= list1[index].stop:
                    elm_id[index].append(list2[index])
                elif list2[index].stop >= list1[index].start:
                    elm_id[index].append(list2[index])
            except:
                pass

        list1 = list2


    print(elm_id[20])


    exit()

    fig, ax1 = plt.subplots(1, 1)
    split_list = split_df.index.get_level_values('Index').to_numpy()

    msplit = mplot[split_list]

    ax1.imshow(np.transpose(msplit),
               origin='lower',
               vmin=0, vmax=1,
               cmap='Reds',
               interpolation='none'
               )

    plt.show()