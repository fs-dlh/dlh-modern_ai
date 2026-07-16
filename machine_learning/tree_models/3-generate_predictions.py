#!/usr/bin/env python3
""" Module to generate predictions using a trained classifier. """


def generate_predictions(clf, X):
    """
    Generate class predictions from a trained tree-based classifier.

    Args:
        clf: A trained Scikit-learn classifier instance.
        X: Feature matrix (NumPy array or pandas DataFrame).

    Returns:
        numpy.ndarray: Predicted class labels for each sample in X.
    """
    return clf.predict(X)
