"""
preprocessing.py
----------------
Cleans and prepares the raw Appliances Energy dataset for model training.

Key decisions:
- 'date' column is parsed as datetime and used as the chronological index.
- 'random1' and 'random2' columns (NSM-based dummy variables from the original
  dataset paper) are dropped — they add noise and no physical meaning.
- No rows are dropped (dataset has no missing values per UCI documentation).
- Chronological 80/20 train/test split is used to respect time-series ordering
  and prevent temporal data leakage.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

# Columns that carry no predictive value for energy consumption
COLUMNS_TO_DROP = ["rv1", "rv2"]  # random variables in the dataset

TARGET_COLUMN = "Appliances"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame:
    - Parse 'date' as datetime
    - Sort chronologically
    - Drop random/useless columns
    - Verify no duplicates exist
    - Report missing value summary

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset from data_loader.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame sorted by date.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # 1. Parse date column
    # ------------------------------------------------------------------
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        logger.info("'date' column parsed as datetime.")
    else:
        raise ValueError("Expected a 'date' column in the dataset.")

    # ------------------------------------------------------------------
    # 2. Sort chronologically (important for time-series integrity)
    # ------------------------------------------------------------------
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Data sorted chronologically.")

    # ------------------------------------------------------------------
    # 3. Check and handle duplicates
    # ------------------------------------------------------------------
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.warning("Found %d duplicate rows — dropping them.", n_duplicates)
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        logger.info("No duplicate rows found.")

    # ------------------------------------------------------------------
    # 4. Report missing values
    # ------------------------------------------------------------------
    missing = df.isnull().sum()
    if missing.any():
        logger.warning("Missing values detected:\n%s", missing[missing > 0])
        df = df.fillna(df.median(numeric_only=True))
        logger.info("Missing values filled with column median.")
    else:
        logger.info("No missing values detected.")

    # ------------------------------------------------------------------
    # 5. Drop random/noise columns
    # ------------------------------------------------------------------
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if cols_present:
        df = df.drop(columns=cols_present)
        logger.info("Dropped noise columns: %s", cols_present)

    logger.info("Cleaned dataset shape: %s", df.shape)
    return df


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and test sets chronologically.

    Why chronological splitting?
    ------------------------------
    This dataset is a time series. A random split would allow the model to
    'see' future timestamps during training, causing data leakage and
    artificially inflated metrics. Chronological splitting keeps the temporal
    ordering intact and gives an honest evaluation of out-of-sample performance.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, chronologically sorted DataFrame.
    train_ratio : float
        Fraction of data used for training (default 0.80).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    logger.info(
        "Chronological split: train=%d rows (%.0f%%), test=%d rows (%.0f%%)",
        len(train_df),
        100 * train_ratio,
        len(test_df),
        100 * (1 - train_ratio),
    )
    logger.info(
        "Train period: %s → %s",
        train_df["date"].min(),
        train_df["date"].max(),
    )
    logger.info(
        "Test period:  %s → %s",
        test_df["date"].min(),
        test_df["date"].max(),
    )
    return train_df, test_df


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    raw = load_raw_data()
    cleaned = clean_data(raw)
    train, test = chronological_split(cleaned)
    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print(cleaned.describe())
