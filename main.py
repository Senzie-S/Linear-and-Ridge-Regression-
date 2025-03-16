# Homework 3 - main file
# COMP.4220 Machine Learning

import itertools, functools
import numpy as np
import matplotlib.pyplot as plt
from regression import LinearRegression, RidgeRegression
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge as skRidge
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler as skStandardScalar


def train_test_split(X, t, test_size=0.2, random_state=None):
    """Splits data into training and testing sets using only NumPy."""

    if random_state:
        np.random.seed(random_state)

    # ---- Part (d) ---- #

    # 1. Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # Use shuffled indices to reorder data
    X_shuffled = X[indices]
    t_shuffled = t[indices]
    
    # 2. Split the data
    split_index = int(X.shape[0] * (1 - test_size))
    X_train = X[:split_index]
    X_test = X[split_index:]
    t_train = t[:split_index]
    t_test = t[split_index:]

    return X_train, X_test, t_train, t_test


def standardscalar(x: np.ndarray):
    # ---- Part (b) ---- #
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


class PolynomialFeature(object):
    def __init__(self, degree=2):
        assert isinstance(degree, int)
        self.degree = degree

    def transform(self, x):
        if x.ndim == 1:
            x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for items in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda x, y: x * y, items))
        return np.asarray(features).transpose()


def main():
    # ---- Part (a) ---- #
    housing = fetch_california_housing()
    X = housing.data
    t = housing.target

    print(housing.data.shape, housing.target.shape)
    print(housing.feature_names[0:6])
    print(housing.DESCR)

    # ---- Part (b) ---- #
    Xs = standardscalar(X)

    # ---- Part (c) ---- #
    standardizer = skStandardScalar()
    Xss = standardizer.fit_transform(X)

    
    # ---- Part (d) ---- #
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=42)


    # ---- Part (k) ---- #
    X_train_biased = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
    X_test_biased = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)


    # ---- Part (g, h) ---- #
    # Model building
    lr = LinearRegression()
    lr.fit(X_train_biased, t_train)
    y_lr = lr.predict(X_test_biased)
    rmse_lr = root_mean_squared_error(t_test, y_lr)
    r2_lr = r2_score(t_test, y_lr)

    print('Linear Regression results')
    print(f'RMSE: {rmse_lr}')
    print(f'R2: {r2_lr}')


    rr = RidgeRegression(lambd=1.0)
    rr.fit(X_train_biased, t_train)
    y_rr = rr.predict(X_test_biased)
    rmse_rr = root_mean_squared_error(t_test, y_rr)
    r2_rr = r2_score(t_test, y_rr)

    print('Ridge Regression results')
    print(f'RMSE: {rmse_rr}')
    print(f'R2: {r2_rr}')


    # ---- Part (i) ---- #
    lr_sk = skLinearRegression()
    lr_sk.fit(X_train, t_train)
    y_lr_sk = lr_sk.predict(X_test)
    rmse_lr_sk = root_mean_squared_error(t_test, y_lr_sk)
    r2_lr_sk = r2_score(t_test, y_lr_sk)

    print('Sklearn Linear Regression results')
    print(f'RMSE: {rmse_lr_sk}')
    print(f'R2: {r2_lr_sk}')


    rr_sk = skRidge(alpha=1.0)
    rr_sk.fit(X_train, t_train)
    y_rr_sk = rr_sk.predict(X_test)
    rmse_rr_sk = root_mean_squared_error(t_test, y_rr_sk)
    r2_rr_sk = r2_score(t_test, y_rr_sk)

    print('Sklearn Ridge Regression results')
    print(f'RMSE: {rmse_rr_sk}')
    print(f'R2: {r2_rr_sk}')


    # ---- Part (j) ---- #
    # Plot the results
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 2, 1)
    plt.scatter(t_test, y_lr)
    plt.plot([min(t_test), max(t_test)], [min(t_test), max(t_test)], color='red')  # Line for reference
    plt.xlabel('Actual Target')
    plt.ylabel('Predicted Target')
    plt.title('Linear Regression')

    plt.subplot(2, 2, 2)
    plt.scatter(t_test, y_rr)
    plt.plot([min(t_test), max(t_test)], [min(t_test), max(t_test)], color='red')
    plt.xlabel('Actual Target')
    plt.ylabel('Predicted Target')
    plt.title('Ridge Regression')

    plt.subplot(2, 2, 3)
    plt.scatter(t_test, y_lr_sk)
    plt.plot([min(t_test), max(t_test)], [min(t_test), max(t_test)], color='red')
    plt.xlabel('Actual Target')
    plt.ylabel('Predicted Target')
    plt.title('Linear Regression (Scikit-learn)')

    plt.subplot(2, 2, 4)
    plt.scatter(t_test, y_rr_sk)
    plt.plot([min(t_test), max(t_test)], [min(t_test), max(t_test)], color='red')
    plt.xlabel('Actual Target')
    plt.ylabel('Predicted Target')
    plt.title('Ridge Regression (Scikit-learn)')

    plt.tight_layout()
    plt.show()


if __name__=='__main__':
    main()
