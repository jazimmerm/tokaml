from torch import nn
import torch
import torchvision
import torchvision.transforms as transforms
import SULI2021.random_tools.DataPrep as dp



if __name__ == '__main__':

    dir = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'
    sh = dp.DataPrep(174828, dir)
    elmdf = sh.split()


