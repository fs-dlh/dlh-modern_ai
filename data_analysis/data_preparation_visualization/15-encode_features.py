#!/usr/bin/env python3
""" Module for encoding categorical features for modeling. """
import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """     Encode categorical features for modeling.

    Args:
        df : Input DataFrame with features.

    Returns:
        tuple: (encoded DataFrame, fitted LabelEncoder for Churn,
                fitted OrdinalEncoder for binary columns,
                fitted OrdinalEncoder for TenureGroup)
    """
    df_enc = df.copy()

    bin_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'PaperlessBilling']
    ten_col = 'TenureGroup'
    churn_col = 'Churn'
    contract_col = 'Contract'
    payment_col = 'PaymentMethod'

    churn_le = preprocessing.LabelEncoder()
    df_enc[churn_col] = churn_le.fit_transform(df_enc[churn_col])

    binary_oe = preprocessing.OrdinalEncoder(
        categories=[['No', 'Yes']] * len(bin_cols)
    )
    df_enc[bin_cols] = binary_oe.fit_transform(df_enc[bin_cols]).astype(int)

    tenure_categories = sorted(df_enc[ten_col].unique())
    tenure_oe = preprocessing.OrdinalEncoder(categories=[tenure_categories])
    df_enc[ten_col] = tenure_oe.fit_transform(
        df_enc[[ten_col]]
    ).astype(int)

    df_enc = pd.get_dummies(
        df_enc,
        columns=[contract_col, payment_col],
        drop_first=True,
        dtype=int
    )

    return df_enc, churn_le, binary_oe, tenure_oe
