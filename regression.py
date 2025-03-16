# Linear Regression models
# COMP.4220 Machine Learning

import numpy as np

class Regression(object):
    pass

class LinearRegression(Regression):
    """Linear regression"""
    def fit(self, x_train: np.ndarray, t_train: np.ndarray):
        """Perform least squares fitting.
        Parameters
        ----------
        x_train : np.ndarray
            training independent variable (N, D)
        t_train : np.ndarray
            training dependent variable (N,)
        """
        self.w = np.linalg.pinv(x_train) @ t_train
        self.var = np.mean(np.square(x_train @ self.w - t_train))

    def predict(self, x: np.ndarray, return_std: bool = False):
        """Return prediction given input.
        Parameters
        ----------
        x : np.ndarray
            samples to predict their output (N, D)
        return_std : bool, optional
            returns standard deviation of each predition if True

        Returns
        -------
        y : np.ndarray
            prediction of each sample (N,)
        y_std : np.ndarray
            standard deviation of each predition (N,)
        """
        y = x @ self.w
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y)
            return y, y_std
        return y


class RidgeRegression(Regression):
    """Ridge regression"""
    def __init__(self, lambd: float = 1.):
        """
        Parameters
        ----------
        lambda : float, optional
            Regularization Coefficient
        """
        self.lambd = lambd

    def fit(self, x_train: np.ndarray, t_train: np.ndarray):
        """Maximum A Posteriori (MAP) estimation.
        Parameters
        ----------
        x_train : np.ndarray
            training data independent variable (N, D)
        y_train : np.ndarray
            training data dependent variable (N,)
        """
        # ---- Part (e) ---- #
        N, D = x_train.shape
        A = x_train.T @ x_train + self.lambd * np.eye(D)
        b = x_train.T @ t_train
        self.w = np.linalg.solve(A, b)

    def predict(self, x: np.ndarray):
        """Return prediction.
        Parameters
        ----------
        x : np.ndarray
            samples to predict their output (N, D)

        Returns
        -------
        np.ndarray
            prediction of each input (N,)
        """
        # ---- Part (f) ---- #
        return x @ self.w
