#!/usr/bin/env python3
""" Module for handling missing values in the TotalCharges column.
"""


def clean_total_charges(df, method='drop'):
    """
    Handle missing values in the TotalCharges column.

    Args:
        df : DataFrame containing 'TotalCharges' column.

    Returns:
        rdf: Returned DataFrame.
    """
    rdf = df.copy()

    if method == 'drop':
        rdf = df.dropna(subset=['TotalCharges'])

    elif method == 'median':
        median_val = df['TotalCharges'].median()
        rdf['TotalCharges'] = df['TotalCharges'].fillna(median_val)

    elif method == 'impute':
        values = df['MonthlyCharges'] * df['tenure']
        rdf['TotalCharges'] = df['TotalCharges'].fillna(values)

    return rdf
