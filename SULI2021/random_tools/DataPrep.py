import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks, argrelmax, argrelmin, spectrogram
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import pandas as pd
import os

try:
    from tools import index_match
except:
    from .tools import index_match
import pyarrow


class DataPrep:

    def __init__(self, shot_no, dir):

        self.shot_no = shot_no
        self.dir = dir

        # variables that can be changed to tweak data output
        # Minimum time between ELMs
        self.min_elm_window = 50
        # spectrogram parameters
        self.spectrogram_height = 8192
        self.spectrogram_width = 10201
        # Whether to return binary masked array or encode amplitude into mask
        self.set_mask_binary = False
        # height of highest mode
        self.stop_height = 1500
        self.blur = 3

        self.arr = self.get_shot()

    @staticmethod
    def peakomatic(slce):
        '''Returns peak properties.'''
        # These properties can be tweaked to fine-tune the results.
        smooth_arr = gaussian_filter1d(slce, 10)
        peaks, properties = find_peaks(smooth_arr, prominence=(np.mean(abs(smooth_arr)), None), distance=100,
                                       height=0,
                                       width=0)

        return peaks, \
               properties['peak_heights'], \
               list(zip(properties['left_ips'], properties['right_ips'])), \
               properties['width_heights']

    def id_band(self, xsec):
        '''Used in Max Curie's work. Unfinished here.'''
        ielm_df = self.peak_properties().xs(xsec, level=0)
        widths = ielm_df['Left/Right'].to_numpy()
        peaks = ielm_df['Peak'].to_numpy()
        band_id = []

        for i, width_list in enumerate(reversed(widths)):
            for band, bandwidth in enumerate(width_list):
                if band not in band_id:
                    band_id.append(band)
        print(band_id)
        return

    # extract values for specific shot from above list of shots
    # this is for the HDF5 files my mentor gave me, I'll switch it up for the parquet files
    @staticmethod
    def get_shot_from_mat(shot_no):
        '''Retrieves shots from .mat files.'''
        file = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/RESULTS_ECH_EFFECTS_SPECTRA_B3.mat'
        # import data from .mat file into python numpy arrays
        mat = h5py.File(file, 'r')
        dat_ech = mat['DAT_ECH']
        shlst = mat['shn'][:]

        shindex = [np.where(shlst == s) for s in shlst if shot_no in s][0][0][0]

        tds = dat_ech.get('TIME')[shindex][0]
        sds = dat_ech.get('SPECTRUM')[shindex][0]
        fds = dat_ech.get('FREQ')[shindex][0]

        time = mat[tds][0][:]
        spectrum = mat[sds][:]
        freq = mat[fds][0][:]

        return np.array([time, spectrum, freq], dtype=object)

    # This is the one we can use for the parquet files.
    def get_shot(self):
        '''Retrieve shots from parquet files.'''
        # change this to where your parquet files are stored
        file = self.dir + str(self.shot_no) + '.pq'

        raw = pd.read_parquet(file, engine='pyarrow')
        raw_values = raw['amplitude'].values
        sampling_frequency = (len(raw['time'])) / (raw['time'].values[-1] - raw['time'].values[0])
        y_height = self.spectrogram_height  # Default = 8192. IMPORTANT: A power of 2 is most efficient

        freq, time, spectrum, = spectrogram(raw['amplitude'].values,
                                            fs=sampling_frequency,
                                            nperseg=y_height * 2,
                                            noverlap=(y_height * 2) - int(
                                                np.floor(((len(raw_values)) / self.spectrogram_width))) + 1
                                            )

        time = 1000 * (raw['time'].values[0] + (raw['time'].values[-1] - raw['time'].values[0]) * (time - time[0]) / (
                time[-1] - time[0]))
        # spectrum = spectrum/np.linalg.norm(spectrum)

        return np.array([time, spectrum, freq], dtype=object)

    def elm_loc(self, plot=False):
        '''Locates ELMs from raw data'''
        pq_dir = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'
        file = pq_dir + str(self.shot_no) + '.pq'

        raw = pd.read_parquet(file, engine='pyarrow')
        raw_values = raw['amplitude'].values
        raw_time = raw['time'].values
        peaks, props = find_peaks(np.absolute(raw_values), prominence=(None, None), distance=1000,
                                  height=(np.median(abs(raw_values)) * 20, None),
                                  width=(None, None), rel_height=1.0)
        peaks_time = 1000 * raw_time[peaks]

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            fig.subplots_adjust(hspace=0)

            self.heatmap2d(ax=ax1)
            ax2.plot(peaks_time, props['peak_heights'], 'ro')
            ax2.plot(1000 * raw_time, raw_values)
            ax1.set_title(None)
            ax2.set_title(None)
            ax2.set_xlabel(None)
            ax1.text(0.5, 0.9, 'Spectrogram',
                     horizontalalignment='center',
                     verticalalignment='top',
                     color='white', fontsize=16,
                     transform=ax1.transAxes)
            ax2.text(0.5, 0.9, 'Raw B-Dot Data',
                     horizontalalignment='center',
                     verticalalignment='top',
                     color='black', fontsize=16,
                     transform=ax2.transAxes)
            ax2.set_xlabel('Time (ms)', fontsize=14)
            ax2.set_ylabel('Amplitude', fontsize=14)
            fig.suptitle(f'Locations of ELMs in shot {self.shot_no}', y=0.93, fontsize=20)
            plt.show()

        return peaks_time

    # make the heatmap
    def heatmap2d(self, ax=None):
        '''
        This basically just plots the spectrogram.
        '''
        if ax is None:
            ax = plt.gca()

        heatmap = ax.imshow(self.arr[1],
                            norm=LogNorm(vmin=np.min(self.arr[1])*3*10e7, vmax=np.max(self.arr[1])/500),
                            origin='lower',
                            extent=[self.arr[0][0], self.arr[0][-1], self.arr[2][0], self.arr[2][-1]],
                            interpolation='none',
                            cmap='viridis',
                            aspect='auto'
                            )
        ax.set_title(f'Spectrogram For Shot {self.shot_no}')
        ax.set_xlabel('Time (ms)', fontsize=14)
        ax.set_ylabel('Frequency (Hz)', fontsize=14)
        return heatmap

    # make the plot of the spectral data for given time
    def slice1d(self, time, smooth=False, ax=None):
        '''
        This plots the 1D slice in plot slice. Not super necessary as an extra function, but I kinda wanted to see if
        I could move matplotlib objects between functions.
        '''
        if ax is None:
            ax = plt.gca()
        index = index_match(self.arr[0], time)  # finds index of time entered. Kind of slow?
        if len(self.arr[0]) < index < 0:
            raise ValueError('Selected time is out of bounds of run time for shot.')
        slce = self.arr[1][:, index]
        ax.plot(self.arr[2], slce)
        ax.set_xlabel('Frequency (Hz)')
        if smooth:
            smooth_arr = gaussian_filter1d(slce, 10)
            peaks, _ = find_peaks(smooth_arr, prominence=(np.mean(abs(smooth_arr)), None), distance=50)
            ax.plot(self.arr[2], smooth_arr, 'y')
            ax.plot(self.arr[2][peaks], smooth_arr[peaks], 'ro')
        else:
            peaks, _ = find_peaks(slce, prominence=(np.mean(abs(slce)), None), distance=50)
            ax.plot(self.arr[2][peaks], slce[peaks], 'ro')
        return peaks

    # helper function

    # Function to find peaks through spectrogram
    def get_peaks(self):
        '''
        finds and labels the peaks of each 1D slice of spectrogram. These are the locations of the modes.
        '''

        if hasattr(self, 'peaks_index'):
            return self.peaks_index, self.peaks_map

        self.peaks_map = np.zeros_like(self.arr[1].T)
        peaks_index = []
        for i, _ in enumerate(self.arr[1][0]):
            slce = self.arr[1][:, i]

            peaks, peak_heights, width, width_heights = self.peakomatic(slce)

            peaks_index.append(peaks)
            for e, w in enumerate(width):
                self.peaks_map[i][round(w[0]):round(w[1])] = width_heights[e]
            self.peaks_map[i][peaks] = peak_heights
        self.peaks_index = np.asarray(peaks_index, dtype=object)
        return self.peaks_index, self.peaks_map

    def plot_slice(self, tme):
        '''
        I made a few different plotting functions. This one plots the spectrogram and a cross section. For testing
        smoothing functions.
        '''
        fig, (ax1, ax2) = plt.subplots(2, 1)
        self.heatmap2d(ax=ax1)
        ax1.axvline(tme, c='r')
        # smooth bool switches between gaussian filtered 1d slice or raw data.
        peaks = self.slice1d(tme, smooth=True, ax=ax2)
        ax1.plot(np.full_like(peaks, tme), self.arr[2][peaks], 'rx', markersize=15, mew=5)
        ax2.text(0.5, 0.9, f'Cross Section at Time t={tme}ms',
                 horizontalalignment='center',
                 verticalalignment='top',
                 color='black', fontsize=16,
                 transform=ax2.transAxes)
        ax2.set_ylabel('Amplitude', fontsize=14)
        ax2.set_xlabel('Frequency (Hz)', fontsize=14)
        plt.tight_layout()
        plt.show()

    def make_mask(self, plot=False):
        '''
        returns a np.ma.masked object that is an array of equal shape to the original array.
        This array is masked for all data points not belonging to the modes (the horizontal squigglies)
        Currently, it sets all points belonging to modes to 1, but their real values can be used.
        '''

        if hasattr(self, 'mask') and plot == True:

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
            fig.subplots_adjust(hspace=0)

            self.heatmap2d(ax=ax1)
            ax1.imshow(self.mask.T,
                       extent=[self.arr[0][0], self.arr[0][-1], self.arr[2][0], self.arr[2][-1]],
                       norm=LogNorm(),
                       origin='lower',
                       cmap='Set1',
                       interpolation='none',
                       aspect='auto')
            # )

            ax2.imshow(self.mask.T,
                       extent=[self.arr[0][0], self.arr[0][-1], self.arr[2][0], self.arr[2][-1]],
                       norm=LogNorm(),
                       origin='lower',
                       cmap='Set1',
                       interpolation='none',
                       aspect='auto')
            # )

            ax2.set_xlabel('Time (ms)', fontsize=14)
            ax2.set_ylabel('Frequency (Hz)', fontsize=14)

            ax1.text(0.5, 0.9, 'Spectrogram with Mask Overlaid',
                     horizontalalignment='center',
                     verticalalignment='top',
                     color='white', fontsize=16,
                     transform=ax1.transAxes)
            ax2.text(0.5, 0.9, 'Masked Array with Modes Only',
                     horizontalalignment='center',
                     verticalalignment='top',
                     color='black', fontsize=16,
                     transform=ax2.transAxes)
            plt.show()

            return self.mask

        elif hasattr(self, 'mask') and plot == False:
            return self.mask

        peaks_index, peaks_map = self.get_peaks()
        # creates array of the amplitude values of modes and null values everywhere else
        # to return only location of modes, set self.set_mask_binary to True
        if self.set_mask_binary:
            mask = np.ma.masked_where(peaks_map != 0, peaks_map)
            mask = mask.filled(fill_value=1)
            mask = np.ma.masked_where(mask == 0, mask)
        else:
            mask = np.ma.masked_where(peaks_map == 0, peaks_map)

        mask = np.transpose(mask)
        self.mask = mask.T
        return self.make_mask(plot=plot)

    def split(self, plot=False, ax=None):
        '''
        split returns a dataframe of the original array which excludes all arrays belonging to ELM's.
        It also returns a hot array with 0 for all intra-elm indices and 1 for all elm indices (the ones that were excluded
        from the dataframe.
        '''

        if hasattr(self, 'elmdf'):
            return self.elmdf

        pm = self.arr[1].T
        stop = self.stop_height  # default 1500
        pm_norm = (pm[:, stop:] - np.amin(pm[:, stop:])) / (np.amax(pm[:, stop:]) - np.amin(pm[:, stop:]))
        sums = np.sum(pm_norm, axis=1)

        peaks, props = find_peaks(sums, distance=10, prominence=(0.5, None), width=(10, 50), rel_height=0.95)
        l_elms = props['left_ips']
        r_elms = props['right_ips']
        r_elms = np.asarray([*map(np.ceil, r_elms)]).astype(int)
        l_elms = np.asarray([*map(np.floor, l_elms)]).astype(int)
        l_elms_time = self.arr[0][l_elms]
        r_elms_time = self.arr[0][r_elms]


        # make dict with keys, values, times
        elm_cycles = {}
        elms = list(zip(l_elms_time, r_elms_time))
        for elm_no, _ in enumerate(elms[:-1]):
            if elms[elm_no + 1][0] - elms[elm_no][1] <= self.min_elm_window:  # default min_elm_window = 50
                continue

            start_ielm = index_match(self.arr[0], elms[elm_no][1])
            stop_ielm = index_match(self.arr[0], elms[elm_no + 1][0])

            for ielm_time in self.arr[0][start_ielm:stop_ielm]:
                ielm_index = np.argwhere(self.arr[0] == ielm_time)[0][0]
                elm_cycles[(elm_no, ielm_index, ielm_time, ielm_time - self.arr[0][start_ielm],
                            self.arr[0][stop_ielm] - ielm_time)] = self.arr[1].T[
                    ielm_index]
                # elm_cycles[(elm_no, ielm_index, ielm_time, (ielm_time - self.arr[0][start_ielm]) /
                #             (self.arr[0][stop_ielm] - self.arr[0][start_ielm]))] = self.arr[1].T[ielm_index]
        index = pd.MultiIndex.from_tuples(elm_cycles.keys(),
                                          names=['ELM_No', 'Index', 'Time (ms)', 't_since_elm', 't_to_elm'])
        # index = pd.MultiIndex.from_tuples(elm_cycles.keys(), names=['ELM_No', 'Index', 'Time (ms)', '% ELM'])
        self.elmdf = pd.DataFrame(elm_cycles.values(), index=index)

        if plot:
            if ax == None:
                ax = plt.gca()
            # fig, ax = plt.subplots(1, 1)
            self.heatmap2d(ax=ax)
            for t in l_elms_time:
                ax.axvline(t, ymin=stop / self.arr[1].shape[0], c='red', linewidth=1.)
            for t in r_elms_time:
                ax.axvline(t, ymin=stop / self.arr[1].shape[0], c='green', linewidth=1.)

            return ax, self.elmdf

        return self.elmdf

    def split_from_raw(self):
    '''This is used for testing purposes. self.split() is used in practice.'''
        if hasattr(self, 'elmdf'):
            return self.elmdf

        self.elms = self.elm_loc()
        elm_cycles = {}
        for elm_no, elm_time in enumerate(self.elms[:-1]):

            if self.elms[elm_no + 1] - self.elms[elm_no] <= self.min_elm_window:  # default min_elm_window = 50
                continue

            start_ielm = index_match(self.arr[0], elm_time)
            stop_ielm = index_match(self.arr[0], self.elms[elm_no + 1])

            for ielm_time in self.arr[0][start_ielm:stop_ielm]:
                '''MAX: Uncomment the lines below to get % of ELM'''
                ielm_index = np.argwhere(self.arr[0] == ielm_time)[0][0]
                # elm_cycles[(elm_no, ielm_index, ielm_time, self.arr[0][stop_ielm] - ielm_time)] = self.arr[1].T[ielm_index]
                elm_cycles[(elm_no, ielm_index, ielm_time, (ielm_time - self.arr[0][start_ielm]) / (
                        self.arr[0][stop_ielm] - self.arr[0][start_ielm]))] = self.arr[1].T[ielm_index]
        # index = pd.MultiIndex.from_tuples(elm_cycles.keys(), names=['ELM_No', 'Index', 'Time (ms)', 't_to_elm'])
        index = pd.MultiIndex.from_tuples(elm_cycles.keys(), names=['ELM_No', 'Index', 'Time (ms)', '% ELM'])
        elmdf = pd.DataFrame(elm_cycles.values(), index=index)

        return elmdf

    def peak_properties(self, blur=None, original=False, plot=False):

        if hasattr(self, 'props'):
            return self.props
        if not hasattr(self, 'elmdf'):
            self.split()
        if not hasattr(self, 'mask'):
            if self.set_mask_binary:
                mask_bin = self.make_mask()
            else:
                self.set_mask_binary = True
                if blur is not None:
                    self.blur = blur
                self.peak_properties(blur=self.blur, original=original, plot=plot)
                return self.props

        mask = mask_bin[self.elmdf.index.get_level_values(level='Index').to_numpy()]
        mask_blur = gaussian_filter(mask, blur)
        if plot:
            plt.rc('font', size=15)
            plt.imshow(mask_blur.T,
                       extent=[self.arr[0][0], self.arr[0][-1], self.arr[2][0], self.arr[2][-1]],
                       norm=LogNorm(),
                       origin='lower',
                       cmap='Reds',
                       interpolation='none',
                       aspect='auto')

            plt.show()

        maskdf = pd.DataFrame(data=mask_blur, index=self.elmdf.index)
        props = maskdf.apply(
            lambda x: pd.Series(self.peakomatic(x),
                                index=['Peak_Freq', 'Peak_Amp', 'Left/Right', 'Width Height']),
            axis=1)

        props['width'] = props['Left/Right'].apply(lambda x: [*map(lambda y: y[1] - y[0], x)])

        if original:
            self.props = props
            return self.props

        props.reset_index(inplace=True)
        rows = []
        np.seterr(divide='ignore')
        _ = props.apply(lambda row: [rows.append([row['ELM_No'],  # Index
                                                  row['Index'],  # Index
                                                  row['Time (ms)'],  # Index
                                                  row['t_since_elm'],
                                                  row['t_to_elm'],
                                                  freq,
                                                  row['Peak_Amp'][i],
                                                  row['Left/Right'][i],
                                                  row['Width Height'][i],
                                                  row['width'][i]])
                                     for i, freq in enumerate(row.Peak_Freq)], axis=1)
        self.props = pd.DataFrame(rows, columns=props.columns).set_index(['ELM_No', 'Index', 'Time (ms)'])
        return self.props


if __name__ == '__main__':

    dir = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'
    sh = DataPrep(174830, dir)
    sh.set_mask_binary = True
    print((sh.arr[0][-1]-sh.arr[0][0])/len(sh.arr[0]))
    sh.plot_slice(2950)
    exit()
    window_spec = sh.split().xs(7, level=0)
    window = sh.peak_properties(blur=3)
    # print(window_spec.iloc[0:10])
    # exit()
    plt.imshow(window_spec.T,
               extent=[window_spec.index.get_level_values(level='Time (ms)')[0],
                       window_spec.index.get_level_values(level='Time (ms)')[-1],
                       0,
                       len(window_spec.values[0])],
               norm=LogNorm(vmin=np.min(self.arr[1])*3*10e8, vmax=np.max(self.arr[1])/500),
               cmap='Set1',
               origin='lower',
               interpolation='none',
               aspect='auto')

    for time in window.xs(7, level=0).iterrows():
        widths = np.array(time[1][2]).T
        widths[0] = np.subtract(time[1][0], widths[0])
        widths[1] = np.subtract(widths[1], time[1][0])
        plt.errorbar(np.full_like(time[1][0], time[0][1]), time[1][0], yerr=widths,
                     color='blue', fmt='o',
                     solid_capstyle='projecting', capsize=5)
    plt.ylim((0, 1500))
    plt.show()
    exit()
