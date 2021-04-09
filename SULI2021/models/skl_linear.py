from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from SULI2021.random_tools.DataPrep import DataPrep
import os


def get_data():
    pq_dir = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'
    shots = sorted(os.listdir(pq_dir))
    shots = [i.split('.')[0] for i in shots if '153' not in i]
    print(shots[0])
    sh = DataPrep(shots[0], pq_dir)
    props = sh.peak_properties()

    # for shot in shots:
    #     print(shot)
    #     sh = DataPrep(shot, pq_dir)
    #     props_new = sh.peak_properties()
    #     props = pd.concat((props, props_new), ignore_index=True)

    return props


def to_parquet():
    all_dir = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/'
    all_df = get_data()
    all_df.drop(columns=['Left/Right'], inplace=True)
    all_df.to_parquet(f'{all_dir}all.pq', engine='pyarrow')

def lin_model(shot):
    # set hyperparameters #
    epochs = 10  # how many times through the data
    poly_degree = 4  # max degree of polynomial features to create during data augmentation

    # get data #
    pq_dir = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'
    data = DataPrep(shot, pq_dir).peak_properties()
    # data = pd.read_parquet('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/all.pq', engine='pyarrow')
    # going to assume the fields are amp, freq, width, and t_elm

    X = data[['Peak_Amp', 'Peak_Freq', 'width', 't_since_elm']]
    Y = data['t_to_elm']

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.33,
                                                        random_state=42)

    ### train the first layer model ###
    # this model learns the best polynomial to predict t_elm from
    # the other data fields

    # augment the data by creating higher-dimensional polynomial features
    poly_transformer = PolynomialFeatures(degree=poly_degree)
    lin_model = linear_model.LinearRegression()
    X_train_aug = poly_transformer.fit_transform(X_train)

    # fit a linear model to the higher-order data
    lin_model.fit(X_train_aug, y_train)

    ### test the first layer model ###
    X_test_aug = poly_transformer.fit_transform(X_test)
    y_pred = lin_model.predict(X_test_aug)

    MSE = mean_squared_error(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    EXP = explained_variance_score(y_test, y_pred)

    print(f'Shot {shot}')
    print(f'\tMean Squared Error: {np.sqrt(MSE)}')
    print(f'\tMean Absolute Error: {MAE}')
    print(f'\tExplained Variance: {EXP}')
    print(f'\tCoefficient of Determination (R2): {r2}')

    fig, axs = plt.subplots(2, 2)
    x, y = (list(t) for t in zip(*sorted(zip(X_test['Peak_Amp'].tolist(), y_pred))))
    axs[0, 0].set_title('Amplitude')
    axs[0, 0].scatter(X_test['Peak_Amp'].tolist(), y_test, c='black')
    axs[0, 0].plot(x, y, c='blue')
    axs[0, 0].set_xlabel('Amplitude')
    axs[0, 0].set_ylabel('Time to ELM')

    x, y = (list(t) for t in zip(*sorted(zip(X_test['Peak_Freq'].tolist(), y_pred))))
    axs[0, 1].set_title('Frequency')
    axs[0, 1].scatter(X['Peak_Freq'].tolist(), Y, c='black')
    axs[0, 1].plot(x, y, c='blue')
    axs[0, 1].set_xlabel('Frequency')
    axs[0, 1].set_ylabel('Time to ELM')

    x, y = (list(t) for t in zip(*sorted(zip(X_test['width'].tolist(), y_pred))))
    axs[1, 1].set_title('Width')
    axs[1, 1].scatter(X['width'].tolist(), Y, c='black')
    axs[1, 1].plot(x, y, c='blue')
    axs[1, 1].set_xlabel('width')
    axs[1, 1].set_ylabel('Time to ELM')

    x, y = (list(t) for t in zip(*sorted(zip(X_test['t_since_elm'].tolist(), y_pred))))
    axs[1, 0].set_title('Time since Last ELM')
    axs[1, 0].scatter(X['t_since_elm'].tolist(), Y, c='black')
    axs[1, 0].plot(x, y, c='blue')
    axs[1, 0].set_xlabel('t_since_elm')
    axs[1, 0].set_ylabel('Time to ELM')


    plt.show()

    return MSE, MAE, r2




if __name__ == '__main__':

    shots = sorted(os.listdir('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet'))
    shots = [i.split('.')[0] for i in shots if '153' not in i]

    MSE_list, MAE_list, r2_list = [], [], []

    # data = pd.read_parquet('/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/all.pq', engine='pyarrow')

    MSE, MAE, r2 = lin_model(170891)
    MSE_list.append(MSE)
    MAE_list.append(MAE)
    r2_list.append(r2)

    print(f'Root Mean Square Error: {np.mean(np.sqrt(MSE_list))}')
    print(f'Average mean absolute error: {np.mean(MAE_list)}')
    print(f'Average coefficient of determination (R2): {np.mean(r2_list)}')

    print(f'Highest/lowest coefficient of determination (R2): {np.max(r2_list)}/{np.min(r2_list)}')