"""
feature_engineering.py
-----------------------
Creates predictive features from the cleaned dataset.

Features derived from the 'date' column:
  hour         - hour of day (0-23), captures intra-day patterns
  day          - day of month (1-31)
  day_of_week  - 0=Monday … 6=Sunday, captures weekday patterns
  month        - month (1-12), captures seasonal patterns
  year         - year (numeric)
  is_weekend   - binary flag (1 if Saturday/Sunday, else 0)

Note on lag/rolling features:
  Lag features are NOT included in this MVP to keep the pipeline simple and
  avoid any risk of subtle data leakage. The current feature set is sufficient
  for a well-performing baseline and is easier to defend in interviews.

All features are numeric. No categorical encoding is required for this dataset.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Appliances"

# Columns excluded from the feature matrix
# - 'date'   : used only for feature extraction, not a direct predictor
# - 'lights' : can be considered part of total energy and may cause target leakage
#              depending on interpretation; kept here as a feature (it is a separate
#              measurement in Wh that contributes to building energy but is NOT
#              the target 'Appliances')
EXCLUDE_FROM_FEATURES = ["date", TARGET_COLUMN]


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive time-based features from the 'date' column.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with a parsed 'date' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional time-based columns.
    """
    df = df.copy()
    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    logger.info("Datetime features added: hour, day, day_of_week, month, year, is_weekend")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of columns to use as model features (X).

    Excludes 'date' and the target column 'Appliances'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after feature engineering.

    Returns
    -------
    List[str]
        Ordered list of feature column names.
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)
    return feature_cols


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Full feature preparation pipeline:
    1. Add datetime features.
    2. Select feature columns.
    3. Return X (feature DataFrame) and y (target Series).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        (X, y)
    """
    df = add_datetime_features(df)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    logger.info("Feature matrix shape: %s | Target shape: %s", X.shape, y.shape)
    return X, y


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessing import clean_data, chronological_split

    raw = load_raw_data()
    cleaned = clean_data(raw)
    train_df, test_df = chronological_split(cleaned)

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"\nFeatures:\n{list(X_train.columns)}")
    print(f"\nSample features:\n{X_train.head()}")
