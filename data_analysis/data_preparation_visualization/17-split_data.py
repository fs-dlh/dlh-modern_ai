#!/usr/bin/env python3
""" Module for splitting data into train/test sets with stratification. """

from sklearn import model_selection


def split_data(df, target='Churn', test_size=0.2, random_state=42):
    """ Split data into train and test sets using stratified sampling.

    Args:
        df : Input DataFrame containing features and target.
        target : Name of the target column.
        test_size : Proportion of data for test set.
        random_state : Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test
