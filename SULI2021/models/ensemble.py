import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from SULI2021.random_tools.DataPrep import DataPrep
import os

def get_data():

    pq_dir = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/parquet/'
    shots = sorted(os.listdir(pq_dir))
    shots = [i.split('.')[0] for i in shots if '153' not in i]

    sh = DataPrep(shots[0], pq_dir)
    props = sh.peak_properties()
    props.reset_index(inplace=True)
    props['width'] = props['Left/Right'].apply(lambda x: [*map(lambda y: y[1] - y[0], x)])

    # for shot in shots:
    #     sh = DataPrep(shot, pq_dir)
    #     props_new = sh.peak_properties().reset_index(inplace=False)
    #     props_new['width'] = props_new['Left/Right'].apply(lambda x: [*map(lambda y: y[1] - y[0], x)])
    #     props = pd.concat((props, props_new), ignore_index=True)

    return props

def to_parquet():

    all_dir = '/home/jazimmerman/PycharmProjects/SULI2021/SULI2021/data/B3/'
    all_df = get_data()
    all_df.to_parquet(f'{all_dir}all.pq', engine='pyarrow')



if __name__ == '__main__':

    ### set hyperparameters ###
    epochs = 10 # how many times through the data
    poly_degree = 4 # max degree of polynomial features to create during
                    # data augmentation

    ### get data ###
    data = get_data() # Jeff plz insert DataFrame of tuples. one row = one sample
    # going to assume the fields are amp, freq, width, and t_elm

    X = data[['amp', 'freq', 'width', 't_since_elm']]
    Y = data['t_elm']

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
    print(y_pred)