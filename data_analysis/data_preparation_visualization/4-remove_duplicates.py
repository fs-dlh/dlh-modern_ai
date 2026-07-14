#!/usr/bin/env python3
""" Module for removing duplicate rows from a DataFrame. """


def remove_duplicates(df):
    """ Remove all duplicate rows from the DataFrame.

    Args:
        df : DataFrame to deduplicate.

    Returns:
        df : DataFrame with duplicate rows removed.
    """
    return df.drop_duplicates()
