#!/usr/bin/env python3
""" Module containing a function to instantiate a Ridge regression model. """
from sklearn import linear_model


def ridge_regression(random_state):
    """ Creates and returns an untrained Ridge Regression model.

    Args:
        random_state: An integer used to set the random seed
            for reproducibility.

    Returns:
        model: An untrained Ridge regression model instance
    """
    return linear_model.Ridge(random_state=random_state)
