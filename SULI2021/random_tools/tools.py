from numpy import argmin, abs, array
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import SULI2021.random_tools.DataPrep as dp

def index_match(arr1, time):
    return argmin(abs(array(arr1) - time))


def plot_t_to_elm(sh_obj):

    print(type(sh_obj))
    elmdf = sh_obj.split()
    print(elmdf.index)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.imshow(elmdf.to_numpy().T,
               norm=LogNorm(),
               origin='lower',
               interpolation='none',
               aspect='auto')
    ax2.plot(range(len(elmdf.index)), [i[3] for i in elmdf.index])
    plt.show()

def plot_split(sh_obj):

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    fig.subplots_adjust(hspace=0)
    fig.suptitle(f'Filtered and Masked Spectrogram for Shot {sh_obj.shot_no}', fontsize=20)

    elmdf = sh_obj.split()
    sh_obj.set_mask_binary = True
    mask = sh_obj.make_mask()
    elm_loc = sh_obj.elm_loc()
    ielm_index = np.array([i[2] for i in elmdf.index])
    ielm_time = np.array([i[1] for i in elmdf.index])

    ax2.imshow(mask[ielm_index].T,
               vmin=0, vmax=1,
               origin='lower',
               cmap='Reds',
               alpha=1,
               interpolation='none',
               aspect='auto')
    ax1.imshow(elmdf.to_numpy().T,
               norm=LogNorm(),
               origin='lower',
               interpolation='none',
               aspect='auto')

    ax1.text(0.5, 0.9, 'Spectrogram with ELMs Demarcated',
             horizontalalignment='center',
             verticalalignment='top',
             color='black', fontsize=16,
             transform=ax1.transAxes)
    ax2.text(0.5, 0.9, 'Masked Array with Modes Only',
             horizontalalignment='center',
             verticalalignment='top',
             color='black', fontsize=16,
             transform=ax2.transAxes)

    for i, elm in enumerate(elm_loc[:-1]):
        if elm_loc[i+1] - elm <= 50:
            continue
        elm_index = np.argmin(np.abs(np.array(ielm_time - elm)))
        ax1.axvline(elm_index, c='r', alpha=0.1)
        ax2.axvline(elm_index, c='r', alpha=0.1)

    plt.show()
