from torch import nn
import torch
import SULI2021.random_tools.DataPrep as dp

if __name__ == '__main__':

    sh = dp.DataPrep(174828)
    sh.make_mask(plot=True)

