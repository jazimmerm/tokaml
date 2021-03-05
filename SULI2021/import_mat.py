import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks, argrelmax, argrelmin, spectrogram
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os
import pyarrow

# extract values for specific shot from above list of shots
# this is for the HDF5 files my mentor gave me, I'll switch it up for the parquet files
def get_shot_from_mat(shot):
    cwd = os.getcwd()
    file = cwd + '/data/RESULTS_ECH_EFFECTS_SPECTRA_B3.mat'
    # import data from .mat file into python numpy arrays
    mat = h5py.File(file, 'r')
    dat_ech = mat['DAT_ECH']
    shlst = mat['shn'][:]

    shindex = [np.where(shlst == s) for s in shlst if shot in s][0][0][0]

    tds = dat_ech.get('TIME')[shindex][0]
    sds = dat_ech.get('SPECTRUM')[shindex][0]
    fds = dat_ech.get('FREQ')[shindex][0]

    time = mat[tds][0][:]
    spectrum = mat[sds][:]
    freq = mat[fds][0][:]

    return np.array([time, spectrum, freq], dtype=object)

# This is the one we can use for the parquet files.
def get_shot(shot):

    # change this to where your parquet files are stored
    pq_dir = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'
    file = pq_dir + str(shot) + '.pq'

    raw = pd.read_parquet(file, engine='pyarrow')
    raw_values = raw['amplitude'].values


    y_height = 8192  # IMPORTANT: A power of 2 is most efficient
    noverlap = (y_height*2) - int(np.floor(((len(raw_values))/10201))) + 1

    freq, time, spectrum, = spectrogram(raw['amplitude'].values,
                                        nperseg=y_height*2,
                                        noverlap=noverlap)

    print(raw['time'].values[0], raw['time'].values[-1])
    time = 1000*raw['time'].values[-1]*(time-time[0])/(time[-1]-time[0])
    # TODO: Normalize data between start start(raw['time'].values[0]) and stop (raw['time'].values[0]). Currently between zero and stop.
    print(time[0], time[-1])
    return np.array([time, spectrum, freq], dtype=object)


# make the heatmap
def heatmap2d(arr, ax=None):
    '''
    This basically just plots the spectrogram.
    '''
    if ax is None:
        ax = plt.gca()

    heatmap = ax.imshow(arr[1],
                        norm=LogNorm(),
                        origin='lower',
                        extent=[arr[0][0], arr[0][-1], 0, len(arr[2])],
                        interpolation='none',
                        aspect='auto'
                        )

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (kHz)')
    return heatmap


# make the plot of the spectral data for given time
def slice1d(arr, time, smooth=False, ax=None):
    '''
    This plots the 1D slice in plot slice. Not super necessary as an extra function, but I kinda wanted to see if
    I could move matplotlib objects between functions.
    '''
    if ax is None:
        ax = plt.gca()
    index = np.argmin(np.abs(np.array(arr[0]) - time))  # finds index of time entered. Kind of slow?
    print(index, arr[0][index], time)
    if len(arr[0]) < index < 0:
        raise ValueError('Selected time is out of bounds of run time for shot.')
    slce = arr[1][:, index]
    ax.plot(arr[2], slce)
    ax.set_xlabel('Frequency (kHz)')
    if smooth:
        smooth_arr = gaussian_filter1d(slce, 10)
        peaks, _ = find_peaks(smooth_arr, prominence=(0.05, None), distance=75)
        ax.plot(arr[2], smooth_arr, 'y')
        ax.plot(arr[2][peaks], smooth_arr[peaks], 'ro')
    else:
        peaks, _ = find_peaks(slce, prominence=(0.03, None), distance=75)
        ax.plot(arr[2][peaks], slce[peaks], 'ro')
    return peaks


# Function to find peaks through spectrogram
def get_peaks(arr):
    '''
    finds and labels the peaks of each 1D slice of spectrogram. These are the locations of the modes.
    '''
    peaks_map = np.zeros_like(arr[1].T)
    peaks_index = []
    for i in range(len(arr[1][0])):
        slce = arr[1][:, i]
        # These properties can be tweaked to fine-tune the results.
        smooth_arr = gaussian_filter1d(slce, 10)
        peaks, properties = find_peaks(smooth_arr, prominence=(0.05, None), distance=75, height=0, width=0)
        peaks_index.append(peaks.tolist())
        for e, w in enumerate(zip(properties['left_ips'], properties['right_ips'])):
            peaks_map[i][round(w[0]):round(w[1])] = properties['width_heights'][e]
        peaks_map[i][peaks] = properties['peak_heights']
    peaks_index = np.asarray(peaks_index, dtype=object)
    return peaks_index, peaks_map


# This is just to re-use the plots for different figures.
def plot_slice(arr, tme):
    '''
    I made a few different plotting functions. This one plots the spectrogram and a cross section. For testing
    smoothing functions.
    '''
    fig, (ax1, ax2) = plt.subplots(2, 1)
    heatmap2d(arr, ax=ax1)
    ax1.axvline(tme, c='r')
    # smooth bool switches between gaussian filtered 1d slice or raw data.
    peaks = slice1d(arr, tme, smooth=True, ax=ax2)
    ax1.plot(np.full_like(peaks, tme), arr[2][peaks], 'rx')
    plt.show()


def make_mask(arr, plot=False):

    '''
    make_mask returns a np.ma.masked object that is an array of equal shape to the original array.
    This array is masked for all data points not belonging to the modes (the horizontal squigglies)
    Currently, it sets all points belonging to modes to 1, but their real values can be used.
    '''
    fig, (ax1, ax2) = plt.subplots(2, 1)
    heatmap2d(arr, ax=ax1)
    peaks_index, peaks_map = get_peaks(arr)
    # creates array of the values of peaks and null values everywhere else
    # to return real amplitudes of modes, uncomment below and comment out the next section marked by '''
    # mask = np.ma.masked_where(peaks_map == 0, peaks_map)
    # '''
    mask = np.ma.masked_where(peaks_map != 0, peaks_map)
    mask = mask.filled(fill_value=1)
    mask = np.ma.masked_where(mask == 0, mask)
    # '''
    mask = np.transpose(mask)

    if plot:
        ax1.imshow(mask,
                   extent=[arr[0][0], arr[0][-1], 0, len(arr[2])],
                   norm=LogNorm(),
                   origin='lower',
                   cmap='Set1',
                   interpolation='none',
                   aspect='auto')
        ax1.set_title('Spectrogram With Modes Overlaid (1D Gaussian Filter Applied)')
        ax2.imshow(mask,
                   extent=[arr[0][0], arr[0][-1], 0, len(arr[2])],
                   norm=LogNorm(),
                   origin='lower',
                   cmap='Set1',
                   interpolation='none',
                   aspect='auto')
        ax2.set_title('Modes Only (1D Gaussian Filter Applied)')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Frequency (kHz)')
        plt.show()
        return

    return mask


# returns dict with all elm cycles of a particular shot. Elements are transposed. Before plotting, transpose back.
def split(arr, plot=False):
    '''
    split returns a dataframe of the original array which excludes all arrays belonging to ELM's.
    It also returns a hot array with 0 for all intra-elm indices and 1 for all elm indices (the ones that were excluded
    from the dataframe.
    '''

    pi, pm = get_peaks(arr)
    stop = 1500
    pm_norm = (pm[:, stop:] - np.amin(pm[:, stop:])) / (np.amax(pm[:, stop:]) - np.amin(pm[:, stop:]))
    sums = np.sum(pm_norm, axis=1)
    sums[sums <= 25] = 0
    l_elms = argrelmax(np.gradient(sums), order=5)[0]
    r_elms = argrelmin(np.gradient(sums), order=5)[0]

    hot = np.zeros_like(arr[0])
    for i in np.column_stack((l_elms, r_elms)):
        hot[i[0]:i[1]] = 1

    if plot:
        fig, ax = plt.subplots(1, 1)
        heatmap2d(arr, ax=ax)
        for i, num in enumerate(hot):
            if num == 1:
                ax.axvline(arr[0][i], c='orange')
        for i in l_elms:
            ax.axvline(arr[0][i], ymin=stop/arr[1].shape[0], c='red', linewidth=4)
        for i in r_elms:
            ax.axvline(arr[0][i], ymin=stop/arr[1].shape[0], c='green', linewidth=4)
        plt.show()

    # make dict with keys, values, times
    elm_cycles = {}
    for i in range(len(r_elms)-1):
        for j in arr[0][r_elms[i]:l_elms[i+1]]:
            k = np.argwhere(arr[0] == j)[0][0]
            elm_cycles[('{}'.format(i), j, k)] = arr[1].T[np.argwhere(arr[0] == j)][0][0]

    index = pd.MultiIndex.from_tuples(elm_cycles.keys(), names=['ELM_No', 'Time(ms)', 'Index'])
    elmdf = pd.DataFrame(elm_cycles.values(), index=index)

    return elmdf, hot

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

    sh = get_shot(174828)

    # make_mask(sh, plot=True)
    # split(sh, plot=True)
    plot_slice(sh, 3500)
    print(sh[0], sh[2], len(sh[0]), len(sh[2]))
    # make_mask(sh, plot=True)
    # # exit()
    #
    # fig, ax = plt.subplots(1,1)
    # ax.imshow(spec, norm=LogNorm(), origin='lower', interpolation=None)
    # plt.show()
    # exit()

    masked = np.load('SULI2021/data/masked_array.pickle', allow_pickle=True)
    split_df = pd.read_pickle('SULI2021/data/split_df.pickle')

