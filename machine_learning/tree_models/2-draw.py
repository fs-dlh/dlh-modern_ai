#!/usr/bin/env python3
""" Module to display the textual structure of a trained decision tree. """
from sklearn import tree


def draw(clf, feature_names, class_names):
    """
    Print a human-readable textual representation of a trained decision tree.

    Args:
        clf: A trained DecisionTreeClassifier instance from Scikit-learn.
        feature_names : List of feature names.
        class_names : List of target class names.

    Returns:
        None: The function prints the tree structure to stdout.
    """
    tree_text = tree.export_text(
        clf,
        feature_names=list(feature_names),
        class_names=list(class_names)
        )
    print(tree_text)
