#!/usr/bin/env python3
""" Module for converting column types. """

import pandas as pd


def convert_columns(df):
    """ Convert TotalCharges to numeric and map SeniorCitizen to strings.

    Args:
        df : DataFrame containing columns 'TotalCharges'
            and 'SeniorCitizen'.

    Returns:
        pandas.DataFrame: Modified DataFrame with converted columns.
    """
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce')
    df['SeniorCitizen'] = df['SeniorCitizen'].map(
        {0: 'No',
         1: 'Yes'})
    return df
