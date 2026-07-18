#!/usr/bin/env python3
""" Module to perform grid search for pre-pruning. """
from sklearn import model_selection


def prepruning(X, y, clf):
    """ Perform Grid Search for optimal pre-pruning.

    The search explores:
    - criterion: 'gini' or 'entropy'
    - max_depth: integers in [2, 5) → 2, 3, 4
    - min_samples_leaf: integers in [2, 5) → 2, 3, 4
    - min_samples_split: integers in [2, 5) → 2, 3, 4

    Args:
        X : Input features.
        y : Target labels.
        clf: An untrained DecisionTreeClassifier instance.

    Returns:
        dict: Best hyperparameter combination found by GridSearchCV.
    """
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': list(range(2, 5)),
        'min_samples_leaf': list(range(2, 5)),
        'min_samples_split': list(range(2, 5))
    }

    grid_search = model_selection.GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy'
    )
    grid_search.fit(X, y)

    return grid_search.best_params_
