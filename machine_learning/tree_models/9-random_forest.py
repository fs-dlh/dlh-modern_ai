#!/usr/bin/env python3
""" Module to create a Random Forest classifier using Scikit-learn. """

from sklearn import ensemble


def random_forest(n_estimators, random_state):
    """
    Create a Random Forest classifier.

    Args:
        n_estimators : Number of trees in the forest.
        random_state : Seed for reproducibility.

    Returns:
        RandomForestClassifier: An RFC instance.
    """
    return ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
