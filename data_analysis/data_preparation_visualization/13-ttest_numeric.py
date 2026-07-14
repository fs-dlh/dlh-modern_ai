#!/usr/bin/env python3
""" Module for performing Welch's t-tests between numeric features and target.
"""
from scipy import stats


def ttest_numeric(df):
    """     Perform Welch's t-test (unequal variance) for each numeric feature.

    Args:
        df : DataFrame containing a 'Churn' column.

    Returns:
        dict: {feature_name: p_value} for each numeric feature.
    """

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    results = {}

    for col in numeric_cols:
        yes = df[df['Churn'] == 'Yes'][col]
        no = df[df['Churn'] == 'No'][col]
        _, p = stats.ttest_ind(yes, no, equal_var=False)
        results[col] = p

    return results
