#!/usr/bin/env python3
""" Module containing a function to train a tree-based classifier."""


def train_tree(clf, X, y):
    """ Train a tree-based classifier using Scikit-learn.

    Args:
        clf: A Scikit-learn classifier instance (e.g., DecisionTreeClassifier).
        X: Input features, shape (n_samples, n_features).
        y: Target labels, shape (n_samples,).

    Returns:
        None: The classifier is trained in-place.
    """
    clf.fit(X, y)
