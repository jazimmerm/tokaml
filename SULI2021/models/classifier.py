from typing import List

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from SULI2021.random_tools.DataPrep import DataPrep
import os


if __name__ == '__main__':

    dir = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'
    sh = dp.DataPrep(174828, dir)
    elmdf = sh.split()


