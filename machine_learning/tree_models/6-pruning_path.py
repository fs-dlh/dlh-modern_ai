#!/usr/bin/env python3
""" Module to retrieve the cost-complexity pruning path of a decision tree."""


def get_pruning_path(clf, X, y):
    """ Retrieve the cost-complexity pruning path.

    Args:
        clf: A DecisionTreeClassifier instance (not yet fitted).
        X: Input features.
        y: Target labels.

    Returns:
        tuple: (ccp_alphas, impurities)
            ccp_alphas : Effective alpha values used for pruning.
            impurities : Total leaf impurity at each corresponding alpha.
    """
    clf.fit(X, y)
    path = clf.cost_complexity_pruning_path(X, y)
    return path.ccp_alphas, path.impurities
