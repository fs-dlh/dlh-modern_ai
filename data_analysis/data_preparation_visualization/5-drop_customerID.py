#!/usr/bin/env python3
""" Module for dropping the customerID column """


def drop_customerID(df):
    """ Remove the customerID column from the DataFrame.

    Args:
        df : DataFrame containing a 'customerID' column.

    Returns:
        df : DataFrame with the 'customerID' column removed.
    """
    return df.drop('customerID', axis=1)
