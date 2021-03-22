# from random_tools.DataAnalysis import *
from random_tools.DataPrep import *
from random_tools.tools import *

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

    plot_t_to_elm(170867)
    exit()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    sh178430 = DataPrep(170874)

    fig.subplots_adjust(hspace=0)
    fig.suptitle('Filtered and Masked Spectrogram for Shot 170874', fontsize=20)

    elmdf = sh178430.split()

    sh178430.set_mask_binary = True
    mask = sh178430.make_mask()
    elm_loc = sh178430.elm_loc()
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
