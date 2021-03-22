from numpy import argmin, abs, array
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import SULI2021.random_tools.DataPrep as dp

def index_match(arr1, time):
    return argmin(abs(array(arr1) - time))

def plot_t_to_elm(shot):

    sh = dp.DataPrep(shot)
    elmdf = sh.split()
    print(elmdf.index)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.imshow(elmdf.to_numpy().T,
               norm=LogNorm(),
               origin='lower',
               interpolation='none',
               aspect='auto')
    ax2.plot(range(len(elmdf.index)), [i[3] for i in elmdf.index])
    plt.show()