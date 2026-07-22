#!/usr/bin/env python3
""" Module containing a function to instantiate a linear regression model. """
from sklearn import linear_model


def Linear_Regression():
    """ Creates and returns an untrained LinearRegression model from scikit-learn.

    Returns:
        LinearRegression: An instance of sklearn.linear_model.
    """
    return linear_model.LinearRegression()