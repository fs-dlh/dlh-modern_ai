#!/usr/bin/env python3
""" Module for scaling numeric features using StandardScaler. """
from sklearn import preprocessing


def scale_numeric(df):
    """ Standardize MonthlyCharges and TotalCharges using StandardScaler.

    Args:
        df : Input DataFrame with columns
                               'MonthlyCharges' and 'TotalCharges'.
    Returns:
        df : DataFrame with scaled columns.
    """
    scaler = preprocessing.StandardScaler()
    df[['MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
        df[['MonthlyCharges', 'TotalCharges']]
    )
    return df
