#!/usr/bin/env python3
""" Module to evaluate classifier performance using classification report. """

from sklearn import metrics


def evaluate(true_labels, predicted_labels, class_names):
    """ Generate a detailed classification report.

    Args:
        true_labels : Ground truth labels.
        predicted_labels : Predicted labels.
        class_names : List of class names corresponding to label indices.

    Returns:
        str: Classification report as a formatted string.
    """
    return metrics.classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
        )
