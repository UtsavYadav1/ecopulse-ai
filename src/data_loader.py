"""
data_loader.py
--------------
Handles downloading and loading the UCI Appliances Energy Prediction Dataset.

Dataset:
    UCI ML Repository - Appliances Energy Prediction
    https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction

The dataset is downloaded programmatically and cached locally in the data/ directory.
"""

import os
import io
import zipfile
import logging
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_CSV_PATH = os.path.join(DATA_DIR, "energydata_complete.csv")

# ---------------------------------------------------------------------------
# Primary download URL (direct CSV from UCI ML Repo via GitHub mirror)
# The UCI API zip download is used first; CSV fallback used if that fails.
# ---------------------------------------------------------------------------
UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
)
CSV_FALLBACK_URL = (
    "https://raw.githubusercontent.com/LuisM78/Appliances-energy-prediction-data/"
    "master/energydata_complete.csv"
)


def _download_from_zip(url: str, target_path: str) -> bool:
    """Download the UCI zip archive and extract the CSV."""
    try:
        logger.info("Attempting to download dataset zip from UCI repository…")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_names = [n for n in z.namelist() if n.endswith(".csv")]
            if not csv_names:
                logger.warning("No CSV found inside the zip archive.")
                return False
            # Pick the largest CSV (the main data file)
            csv_name = max(csv_names, key=lambda n: z.getinfo(n).file_size)
            logger.info("Extracting '%s' from zip…", csv_name)
            with z.open(csv_name) as src, open(target_path, "wb") as dst:
                dst.write(src.read())
        logger.info("Dataset saved to: %s", target_path)
        return True
    except Exception as exc:
        logger.warning("ZIP download failed: %s", exc)
        return False


def _download_csv_direct(url: str, target_path: str) -> bool:
    """Download the CSV directly from a fallback URL."""
    try:
        logger.info("Downloading dataset CSV directly from fallback URL…")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            f.write(response.content)
        logger.info("Dataset saved to: %s", target_path)
        return True
    except Exception as exc:
        logger.warning("Direct CSV download failed: %s", exc)
        return False


def download_dataset() -> str:
    """
    Download the dataset if not already present locally.

    Returns
    -------
    str
        Absolute path to the local CSV file.

    Raises
    ------
    RuntimeError
        If the dataset cannot be downloaded from any source.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.isfile(RAW_CSV_PATH):
        logger.info("Dataset already present at: %s", RAW_CSV_PATH)
        return RAW_CSV_PATH

    # Try UCI zip first, then direct CSV fallback
    if _download_from_zip(UCI_ZIP_URL, RAW_CSV_PATH):
        return RAW_CSV_PATH
    if _download_csv_direct(CSV_FALLBACK_URL, RAW_CSV_PATH):
        return RAW_CSV_PATH

    raise RuntimeError(
        "Could not download the dataset automatically.\n"
        "Please download it manually from:\n"
        "  https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction\n"
        f"and place 'energydata_complete.csv' in: {DATA_DIR}"
    )


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw dataset into a pandas DataFrame.

    Downloads the dataset automatically if not already cached locally.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all original columns.
    """
    csv_path = download_dataset()
    logger.info("Loading dataset from: %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Dataset loaded — shape: %s", df.shape)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nColumns:\n{list(df.columns)}")
