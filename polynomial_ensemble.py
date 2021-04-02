import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

### set hyperparameters ###
epochs = 10 # how many times through the data
poly_degree = 4 # max degree of polynomial features to create during
                # data augmentation

### get data ###
data = None # Jeff plz insert DataFrame of tuples. one row = one sample
# going to assume the fields are amp, freq, width, and t_elm
X = data[['amp', 'freq', 'width', 'elm']]
Y = data['t_elm']

# split the data
X_train, X_test, y_train, y_test = \
    train_test_split(
                        data,
                        X,
                        y,
                        test_size=0.33,
                        random_state=42
                    )

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