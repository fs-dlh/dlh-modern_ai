#!/usr/bin/env python3
"""Module containing a function to instantiate a logistic regression model."""
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """     Creates and returns an untrained LogisticRegression model.

    Args:
        random_state: An integer used to set the random seed
            for reproducibility.

    Returns:
        model: An untrained LogisticRegression instance.
    """
    return linear_model.LogisticRegression(random_state=random_state)
