#!/usr/bin/env python3
""" Module containing a function to create SVM classifier using Scikit-learn
with the specified kernel.
"""
from sklearn import svm


def get_SVM_model(name, random_state):
    """
    Creates and returns an untrained SVM classifier with the specified kernel.

    Args:
        name: A string indicating the type of model to return.
        Accepted values are:
            'linear': returns a SVM model with a linear kernel
            'poly': returns a SVM model with a polynomial kernel
            'rbf': returns a SVM model with a radial basis function kernel
        random_state: The seed used by the random number generator
            for reproducibility.

    Returns:
        An untrained instance of SVC
   """
    kernel = {'linear', 'poly', 'rbf'}
    if name not in kernel:
        raise ValueError(f"Invalid kernel. Expected one of {kernel}.")

    return svm.SVC(kernel=name, random_state=random_state)
