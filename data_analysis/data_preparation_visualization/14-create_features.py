#!/usr/bin/env python3
""" Module for creating new features from the dataset. """
import pandas as pd


def create_features(df):
    """ Engineer new features: NumServices and TenureGroup.

    Args:
        df : Input DataFrame containing service columns and tenure.

    Returns:
        df : DataFrame with new columns and original used columns dropped.
    """

    service_cols = [
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]

    df['NumServices'] = 0
    for col in service_cols:
        if col == 'InternetService':
            df['NumServices'] += df[col].isin(
                ['DSL', 'Fiber optic']).astype(int)
        else:
            df['NumServices'] += (df[col] == 'Yes').astype(int)

    df['TenureGroup'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 60, float('inf')],
        labels=['0-12', '13-24', '25-48', '49-60', '60+']
        )

    df = df.drop(columns=service_cols + ['tenure'])

    return df
