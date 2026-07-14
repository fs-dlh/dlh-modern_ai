#!/usr/bin/env python3
""" Module for visualizing missing values in a DataFrame
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """ Plot missing values in a DataFrame as a scatter plot.
    """
    plt.figure(figsize=(12, 8))


    missing = df.isnull().values
    rows, cols = np.where(missing)
    plt.scatter(rows, cols, marker='|')
    plt.yticks(range(len(df.columns)), df.columns)
    plt.title('Missingness Plot')


    plt.tight_layout()
    plt.show()
