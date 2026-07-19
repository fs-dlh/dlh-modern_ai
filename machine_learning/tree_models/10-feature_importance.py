#!/usr/bin/env python3
""" Module to compute feature importances from a trained Random Forest. """

import numpy as np


def feature_importance(rf):
    """ Compute feature importances from a trained Random Forest classifier.

    Args:
        rf : Trained RandomForestClassifier instance.

    Returns:
        tuple: (importances, indices)
            importances : Feature importance scores.
            indices : Indices of features sorted from least to most important.
    """
    importances = rf.feature_importances_
    indices = np.argsort(importances)
    return importances, indices
